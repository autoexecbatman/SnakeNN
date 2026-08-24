#include <format>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "network_setup.h"

// Implementation of chooseDevice and loadCheckpoint. What they are for, and how to call
// them, are in network_setup.h.

Compute chooseDevice()
{
    // Asked once and carried, so the caller's header line and the caller's tensors cannot
    // disagree about which device the run used.
    const bool cuda = torch::cuda::is_available();
    return { cuda ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU), cuda };
}

void loadCheckpoint(AlphaZeroNet& network, const std::string& checkpoint_path,
                    std::string_view report_prefix)
{
    try
    {
        for (const std::string& name : network->loadNarrowerStem(checkpoint_path))
        {
            std::cout << std::format("  {}fresh, absent from {}: {}\n", report_prefix,
                                     checkpoint_path, name);
        }
    }
    catch (const std::exception& error)
    {
        // Rethrown as runtime_error so a caller can tell an unreadable file from a bad
        // flag, which arrives as invalid_argument from the parsers.
        throw std::runtime_error(
            std::format("could not load {}: {}", checkpoint_path, error.what()));
    }
}

AlphaZeroNet loadForEvaluation(int board, int channels, int blocks,
                               const std::string& checkpoint_path, std::string_view report_prefix,
                               torch::Device device)
{
    // Constructed at the caller's board size; the 4x4 pooling means a checkpoint saved on
    // another board still fits.
    AlphaZeroNet network(board, board, channels, blocks);
    loadCheckpoint(network, checkpoint_path, report_prefix);
    // After the load, so the restored tensors travel with it.
    network->to(device);
    // Batch norm uses its running statistics rather than the batch's own.
    network->eval();
    return network;
}
