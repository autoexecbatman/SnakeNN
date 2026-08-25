#include <torch/torch.h>

#include <cmath>

#include "az_parameters.h"
#include "snake_env.h"
#include "replay_sampling.h"
#include "training_batch.h"

namespace
{
// How many draws a biased pick may take before keeping whatever it has. Four gives a
// window that is one percent decisive a better than one-in-thirty chance of finding one,
// and costs three extra draws at worst.
constexpr int DECISIVE_TRIES = 4;
}  // namespace

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
    buffers.death_mask.resize(batch * SnakeEnv::ACTION_COUNT);
    // One label per cell, which is where the ownership head gets its density: a batch
    // of 256 at 12x12 carries 36,864 labels against 256 for the value.
    buffers.ownership.resize(batch * cells);
    buffers.policy_mask.resize(batch);
    return buffers;
}

void fillBatch(const trainer::Settings& settings, const ReplayWindow& replay, BatchBuffers& buffers)
{
    const size_t cells = static_cast<size_t>(settings.cellCount());
    // One draw from the window, with replacement. Randomness lives here rather than in
    // the sampler, which keeps that unit testable without a generator.
    const auto draw = [&replay]
    {
        return static_cast<size_t>(
            torch::randint(0, static_cast<int64_t>(replay.size()), { 1 }).item<int64_t>());
    };
    const auto isDecisive = [&replay](size_t index) { return replay[index].decisive; };
    for (int item = 0; item < settings.batch_size; item++)
    {
        // Whether this item hunts for a decisive position. Drawn per item, so the share
        // is met in expectation rather than by filling a block of the batch with them.
        const bool prefer = torch::rand({ 1 }).item<float>() < settings.decisive_share;
        const size_t pick = sampling::pickBiased(draw, isDecisive, prefer, DECISIVE_TRIES);
        const TrainingRecord& record = replay[pick];
        SnakeEnv::encodeSnapshot(
            settings.board, settings.board, record.position,
            buffers.planes.data() + static_cast<size_t>(item) * SnakeEnv::PLANE_COUNT * cells);
        for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++)
        {
            const size_t slot = static_cast<size_t>(item) * SnakeEnv::ACTION_COUNT + action;
            buffers.policies[slot] = record.policy[action];
            if (settings.doom_label_from_trajectory)
            {
                // Only the action played has a known outcome; the other two are masked
                // out rather than guessed at.
                const bool played = action == record.played_action;
                buffers.death_risks[slot] = played ? record.doom_target : 0.0f;
                buffers.death_mask[slot] = played ? 1.0f : 0.0f;
            }
            else
            {
                buffers.death_risks[slot] = record.death_risk_target[action];
                buffers.death_mask[slot] = record.death_risk_usable ? 1.0f : 0.0f;
            }
        }
        buffers.values[item] = record.value_target;
        buffers.steps[item] = record.steps_target;
        buffers.policy_mask[item] = record.policy_usable ? 1.0f : 0.0f;
        // A record whose game predates this target carries an empty mask; those cells read
        // as never visited rather than as a shape mismatch.
        for (size_t cell = 0; cell < cells; cell++)
        {
            buffers.ownership[static_cast<size_t>(item) * cells + cell] =
                cell < record.future_cells.size() ? static_cast<float>(record.future_cells[cell])
                                                  : 0.0f;
        }
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
        torch::from_blob(buffers.death_mask.data(), { settings.batch_size, SnakeEnv::ACTION_COUNT })
            .to(device);
    // Shaped like the head's output - one channel at board resolution - so the loss is a
    // straight comparison rather than a reshape at the call site.
    batch.policy_weight =
        torch::from_blob(buffers.policy_mask.data(), { settings.batch_size, 1 }).to(device);
    batch.ownership_target =
        torch::from_blob(buffers.ownership.data(),
                         { settings.batch_size, 1, settings.board, settings.board })
            .to(device);
    return batch;
}

Losses computeLosses(const Prediction& prediction, const BatchTensors& batch)
{
    Losses losses;
    const torch::Tensor log_policy = torch::log_softmax(prediction.policy_logits, 1);
    // Averaged over the records whose visits came from a full search. Divided by what
    // survived rather than by the batch: with a fixed denominator a batch of mostly cheap
    // searches reports a small loss and takes a correspondingly small step, which reads
    // like a policy that has already converged.
    const torch::Tensor policy_kept = batch.policy_weight.sum().clamp_min(1.0f);
    losses.policy = (-(batch.policy_target * log_policy).sum(1, true) * batch.policy_weight).sum() /
                    policy_kept;
    // Measured on the normalised scale, which is the loss the squashed version produced up
    // to a constant - so the balance against the policy loss, and every learning rate
    // chosen under it, carries over unchanged.
    const torch::Tensor scaled_target = batch.value_target / az::VALUE_SCALE;
    losses.value = torch::mse_loss(prediction.value / az::VALUE_SCALE, scaled_target);
    // The variance of the target itself, which is what a head predicting the batch mean
    // would score. Unbiased is not worth the argument at batch sizes in the hundreds.
    losses.value_variance = torch::var(scaled_target, /*unbiased=*/false);
    // Undiscounted, unlike the value: the only estimate here that can see the deadline.
    losses.steps = torch::mse_loss(prediction.steps_to_go, batch.steps_target);
    // Cross entropy rather than squared error, because the head is a sigmoid and the target
    // is a probability.
    const torch::Tensor elementwise = torch::binary_cross_entropy(
        prediction.death_risk, batch.death_target, {}, at::Reduction::None);
    // Averaged over the records whose label survived the mask: with a fixed denominator a
    // batch of mostly unusable labels reports a small loss and takes a correspondingly
    // small step, which reads like a head that has already learned its target.
    // Counted in labels rather than in records, since a trajectory label marks one
    // action of three and a search label marks all three or none.
    losses.usable = batch.death_weight.sum().clamp_min(1.0f);
    losses.death = (elementwise * batch.death_weight).sum() / losses.usable;
    // Averaged over every cell of the batch, so the term does not grow with board size.
    losses.ownership = torch::binary_cross_entropy(prediction.ownership, batch.ownership_target);
    losses.total = losses.policy + losses.value + az::STEPS_LOSS_WEIGHT * losses.steps +
                   az::DEATH_LOSS_WEIGHT * losses.death +
                   az::OWNERSHIP_LOSS_WEIGHT * losses.ownership;
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
        totals.value_variance += losses.value_variance.item<double>();
        totals.ownership += losses.ownership.item<double>();
        // The death loss is read every batch rather than only on failure, because it is
        // meaningless without the label count beside it.
        totals.death += losses.death.item<double>();
        totals.usable_labels += losses.usable.item<double>();
        totals.batches_run++;
    }
    return totals;
}
