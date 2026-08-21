#pragma once

#include <torch/torch.h>

#include <string>
#include <string_view>

#include "az_network.h"

// Getting a network onto a device, shared by every program that reads a checkpoint.
//
// Both of these were written out separately in each program - device selection five times
// and the checkpoint load four - so a change to either had to be made in every copy and
// was made in some. They live here so there is one of each.
//
// Usage, from az_evaluate.cpp:
//
//     const Compute compute = chooseDevice();
//
//     AlphaZeroNet network(board, board, channels, blocks);
//     loadCheckpoint(network, settings.checkpoint, "");   // throws if the file will not read
//     network->to(compute.device);
//     network->eval();
//
// Moving to the device and choosing train or eval mode stay with the caller: the trainer
// needs train(), everything else needs eval(), and the trainer moves its network before
// building an optimiser over the parameters.

// Where a run puts its tensors.
struct Compute
{
    // The device itself.
    torch::Device device;
    // Whether that device is a GPU. Programs print it, so it is returned rather than
    // recomputed by asking Torch a second time.
    bool cuda;
};

// CUDA when it is available, the CPU otherwise.
//
//     const Compute compute = chooseDevice();
//     std::cout << (compute.cuda ? "cuda" : "cpu");
Compute chooseDevice();

// Loads a checkpoint into an already-constructed network, widening a narrower stem.
//
//     loadCheckpoint(network, settings.checkpoint, "");
//     loadCheckpoint(positions, other_path, "positions: ");   // two networks in one run
//
// Widening rather than loading flat is the only path on purpose. A checkpoint saved before
// the clock plane has an 8-plane stem, and torch::load on that mismatch does not throw -
// it takes the process down - so there is no failure to fall back from.
//
// Prints every parameter and buffer the file did not carry, each on its own line prefixed
// by `report_prefix`. Pass "" for the network under test. They are printed rather than
// swallowed because a mistyped module name looks exactly like a head added after the
// checkpoint was written.
//
// Throws std::runtime_error naming the path when the file cannot be read. Callers decide
// what that means: the evaluator marks its ledger row failed, the probe just exits.
void loadCheckpoint(AlphaZeroNet& network, const std::string& checkpoint_path,
                    std::string_view report_prefix);
