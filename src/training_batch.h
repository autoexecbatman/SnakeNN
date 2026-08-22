#pragma once

#include <torch/torch.h>

#include <vector>

#include "az_network.h"
#include "iteration_report.h"
#include "replay_window.h"
#include "trainer_options.h"

// One iteration's gradient steps: drawing a batch out of the replay window, encoding it,
// running the four heads and stepping the optimizer.
//
// Usage - allocate the buffers once for the run, then call once per iteration:
//
//     BatchBuffers buffers = makeBatchBuffers(settings);          // sized for board/batch
//     network->train();                                           // caller owns the mode
//     const trainer::LossTotals totals =
//         trainOnReplay(settings, network, optimizer, replay, buffers, device, iteration);
//     totals.policy / totals.batches_run;    // mean policy loss, when batches_run > 0
//
// trainOnReplay returns zeros without training when the window holds fewer records than
// one batch. The pieces below it are separate so each is readable on its own; the only one
// a trainer needs is trainOnReplay.
//
// Every check here is a TORCH_CHECK rather than an assert. LibTorch ships release-only
// libraries, so a debug build of anything linking it dies before main, and a release build
// compiles an assert away - an assert in this file would be unreachable in both
// configurations.

// Scratch arrays one training batch is assembled into, allocated once for the run.
struct BatchBuffers
{
    // Encoded positions: batch x planes x cells.
    std::vector<float> planes;
    // Visit-share targets: batch x actions.
    std::vector<float> policies;
    // Return targets, one per record.
    std::vector<float> values;
    // Steps-to-go targets, one per record.
    std::vector<float> steps;
    // Death-risk targets: batch x actions.
    std::vector<float> death_risks;
    // One per record: whether its risk label is worth learning from. A batch can be almost
    // entirely masked out, which is why the death loss divides by what survived.
    std::vector<float> death_mask;
};

// One assembled batch, on the device the network lives on.
struct BatchTensors
{
    // Encoded positions, the network's input.
    torch::Tensor input;
    // Visit shares the policy head is trained against.
    torch::Tensor policy_target;
    // Returns the value head is trained against.
    torch::Tensor value_target;
    // Steps-to-go the steps head is trained against.
    torch::Tensor steps_target;
    // Per-action death risk the death head is trained against.
    torch::Tensor death_target;
    // 1 where the death label is usable, 0 where it is not.
    torch::Tensor death_weight;
};

// The four heads' losses and the weighted sum trained on.
struct Losses
{
    // Cross entropy against the search's visit shares.
    torch::Tensor policy;
    // Squared error on the normalised value scale.
    torch::Tensor value;
    // Squared error on steps-to-go.
    torch::Tensor steps;
    // Cross entropy on death risk, over usable labels only.
    torch::Tensor death;
    // How many labels survived the mask, floored at one so the division is safe.
    torch::Tensor usable;
    // The weighted sum backward() is called on.
    torch::Tensor total;
};

// Buffers sized for this run's batch and board.
//
//     BatchBuffers buffers = makeBatchBuffers(settings);
//     buffers.values.size();   // == settings.batch_size
BatchBuffers makeBatchBuffers(const trainer::Settings& settings);

// Draws settings.batch_size records from `replay` and encodes them into `buffers`.
//
//     fillBatch(settings, replay, buffers);
//
// Drawn with replacement, as in the paper's "minibatches drawn from the last 2,000 games".
// The caller guarantees the window holds at least one batch.
void fillBatch(const trainer::Settings& settings, const ReplayWindow& replay,
               BatchBuffers& buffers);

// Wraps `buffers` as tensors on `device`.
//
//     const BatchTensors batch = toTensors(settings, buffers, device);
//
// from_blob does not copy, so `buffers` must outlive the call; the copy happens in .to().
BatchTensors toTensors(const trainer::Settings& settings, BatchBuffers& buffers,
                       torch::Device device);

// The losses for one forward pass.
//
//     const Losses losses = computeLosses(network->forward(batch.input), batch);
//     losses.total.backward();
//
// The value target is the return itself: the head is bounded at VALUE_SCALE rather than at
// 1, so nothing is squashed and the search receives a value in the same units as the
// rewards it adds to it.
Losses computeLosses(const Prediction& prediction, const BatchTensors& batch);

// Takes this iteration's gradient steps and returns what they cost.
//
//     const trainer::LossTotals totals =
//         trainOnReplay(settings, network, optimizer, replay, buffers, device, 331);
//
// Returns zeros without training when the window holds fewer records than one batch.
// `iteration` is used only to name the iteration in a non-finite-loss message.
trainer::LossTotals trainOnReplay(const trainer::Settings& settings, AlphaZeroNet& network,
                                  torch::optim::Adam& optimizer, const ReplayWindow& replay,
                                  BatchBuffers& buffers, torch::Device device, int iteration);
