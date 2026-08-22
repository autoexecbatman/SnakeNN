#include <torch/torch.h>

#include <cmath>

#include "az_parameters.h"
#include "snake_env.h"
#include "training_batch.h"

BatchBuffers makeBatchBuffers(const trainer::Settings& settings)
{
    const size_t batch = static_cast<size_t>(settings.batch_size);
    const size_t cells = static_cast<size_t>(settings.cellCount());
    BatchBuffers buffers;
    buffers.planes.resize(batch * SnakeEnv::PLANE_COUNT * cells);
    buffers.policies.resize(batch * SnakeEnv::ACTION_COUNT);
    buffers.values.resize(batch);
    buffers.steps.resize(batch);
    buffers.death_risks.resize(batch * SnakeEnv::ACTION_COUNT);
    buffers.death_mask.resize(batch);
    return buffers;
}

void fillBatch(const trainer::Settings& settings, const ReplayWindow& replay, BatchBuffers& buffers)
{
    const size_t cells = static_cast<size_t>(settings.cellCount());
    for (int item = 0; item < settings.batch_size; item++)
    {
        // Uniform over the window, with replacement.
        const size_t pick = static_cast<size_t>(
            torch::randint(0, static_cast<int64_t>(replay.size()), { 1 }).item<int64_t>());
        const TrainingRecord& record = replay[pick];
        SnakeEnv::encodeSnapshot(
            settings.board, settings.board, record.position,
            buffers.planes.data() + static_cast<size_t>(item) * SnakeEnv::PLANE_COUNT * cells);
        for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++)
        {
            buffers.policies[static_cast<size_t>(item) * SnakeEnv::ACTION_COUNT + action] =
                record.policy[action];
            buffers.death_risks[static_cast<size_t>(item) * SnakeEnv::ACTION_COUNT + action] =
                record.death_risk_target[action];
        }
        buffers.values[item] = record.value_target;
        buffers.steps[item] = record.steps_target;
        buffers.death_mask[item] = record.death_risk_usable ? 1.0f : 0.0f;
    }
}

BatchTensors toTensors(const trainer::Settings& settings, BatchBuffers& buffers,
                       torch::Device device)
{
    BatchTensors batch;
    batch.input =
        torch::from_blob(buffers.planes.data(), { settings.batch_size, SnakeEnv::PLANE_COUNT,
                                                  settings.board, settings.board })
            .to(device);
    batch.policy_target =
        torch::from_blob(buffers.policies.data(), { settings.batch_size, SnakeEnv::ACTION_COUNT })
            .to(device);
    batch.value_target =
        torch::from_blob(buffers.values.data(), { settings.batch_size, 1 }).to(device);
    batch.steps_target =
        torch::from_blob(buffers.steps.data(), { settings.batch_size, 1 }).to(device);
    batch.death_target = torch::from_blob(buffers.death_risks.data(),
                                          { settings.batch_size, SnakeEnv::ACTION_COUNT })
                             .to(device);
    batch.death_weight =
        torch::from_blob(buffers.death_mask.data(), { settings.batch_size, 1 }).to(device);
    return batch;
}

Losses computeLosses(const Prediction& prediction, const BatchTensors& batch)
{
    Losses losses;
    const torch::Tensor log_policy = torch::log_softmax(prediction.policy_logits, 1);
    losses.policy = -(batch.policy_target * log_policy).sum(1).mean();
    // Measured on the normalised scale, which is the loss the squashed version produced up
    // to a constant - so the balance against the policy loss, and every learning rate
    // chosen under it, carries over unchanged.
    losses.value =
        torch::mse_loss(prediction.value / az::VALUE_SCALE, batch.value_target / az::VALUE_SCALE);
    // Undiscounted, unlike the value: the only estimate here that can see the deadline.
    losses.steps = torch::mse_loss(prediction.steps_to_go, batch.steps_target);
    // Cross entropy rather than squared error, because the head is a sigmoid and the target
    // is a probability.
    const torch::Tensor elementwise = torch::binary_cross_entropy(
        prediction.death_risk, batch.death_target, {}, at::Reduction::None);
    // Averaged over the records whose label survived the mask: with a fixed denominator a
    // batch of mostly unusable labels reports a small loss and takes a correspondingly
    // small step, which reads like a head that has already learned its target.
    losses.usable = batch.death_weight.sum().clamp_min(1.0f);
    losses.death = (elementwise.mean(1, true) * batch.death_weight).sum() / losses.usable;
    losses.total = losses.policy + losses.value + az::STEPS_LOSS_WEIGHT * losses.steps +
                   az::DEATH_LOSS_WEIGHT * losses.death;
    return losses;
}

trainer::LossTotals trainOnReplay(const trainer::Settings& settings, AlphaZeroNet& network,
                                  torch::optim::Adam& optimizer, const ReplayWindow& replay,
                                  BatchBuffers& buffers, torch::Device device, int iteration)
{
    trainer::LossTotals totals;
    // Too little to draw a batch from; the caller reports zero batches rather than a loss.
    if (static_cast<int>(replay.size()) < settings.batch_size)
    {
        return totals;
    }
    for (int batch_index = 0; batch_index < settings.batches_per_iteration; batch_index++)
    {
        fillBatch(settings, replay, buffers);
        const BatchTensors batch = toTensors(settings, buffers, device);
        const Losses losses = computeLosses(network->forward(batch.input), batch);

        // A non-finite loss trains every weight into NaN and the run carries on printing
        // plausible-looking iterations afterwards, so it stops here instead.
        TORCH_CHECK(std::isfinite(losses.total.item<double>()), "loss is not finite at iteration ",
                    iteration, " batch ", batch_index, " - policy ", losses.policy.item<double>(),
                    " value ", losses.value.item<double>(), " steps ", losses.steps.item<double>(),
                    " death ", losses.death.item<double>(), " usable labels ",
                    losses.usable.item<double>());

        optimizer.zero_grad();
        losses.total.backward();
        optimizer.step();

        totals.policy += losses.policy.item<double>();
        totals.value += losses.value.item<double>();
        // The death loss is read every batch rather than only on failure, because it is
        // meaningless without the label count beside it.
        totals.death += losses.death.item<double>();
        totals.usable_labels += losses.usable.item<double>();
        totals.batches_run++;
    }
    return totals;
}
