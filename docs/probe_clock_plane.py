"""How much of the stem's first layer the clock plane actually occupies.

The clock plane is input channel 8 of `stem_conv.weight`, and `widenStemWeight`
sets it to exactly zero when an 8-plane checkpoint is widened. So its norm after
training is the amount of signal training chose to put there, measured against the
eight board planes it competes with - and against zero, which is where it started.

Run: python docs/probe_clock_plane.py <checkpoint.pt> [<checkpoint.pt> ...]
"""

import sys

import torch

# src/snake_env.h. Planes 0..7 are the board; plane 8 is budgetRemaining().
CLOCK_PLANE = 8


def report(path):
    module = torch.jit.load(path, map_location="cpu")
    weights = dict(module.named_parameters())["stem_conv.weight"].detach()
    plane_count = weights.size(1)
    print(f"{path}: stem_conv.weight {list(weights.size())}")
    if plane_count <= CLOCK_PLANE:
        print(f"  no clock plane - this checkpoint has {plane_count} input planes")
        return

    norms = [weights[:, plane].norm().item() for plane in range(plane_count)]
    board_norms = norms[:CLOCK_PLANE]
    clock_norm = norms[CLOCK_PLANE]
    board_mean = sum(board_norms) / len(board_norms)

    for plane, norm in enumerate(norms):
        label = "clock" if plane == CLOCK_PLANE else "board"
        print(f"  plane {plane} ({label}): L2 {norm:.5f}")
    print(f"  board planes: mean {board_mean:.5f}, "
          f"min {min(board_norms):.5f}, max {max(board_norms):.5f}")
    print(f"  clock plane is {100.0 * clock_norm / board_mean:.1f} percent of the board mean, "
          f"and {'above' if clock_norm > min(board_norms) else 'below'} the weakest board plane")

    # The clock is constant across the board, so its whole contribution to a cell is
    # the sum of its kernel - a spatial pattern on a constant input cancels. This is
    # the number that says whether the plane changes any activation at all.
    kernel_sums = weights[:, CLOCK_PLANE].sum(dim=(1, 2))
    print(f"  clock kernel sums: mean abs {kernel_sums.abs().mean().item():.5f}, "
          f"max abs {kernel_sums.abs().max().item():.5f}")
    board_kernel_sums = weights[:, :CLOCK_PLANE].sum(dim=(2, 3)).abs().mean().item()
    print(f"  board kernel sums: mean abs {board_kernel_sums:.5f}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    for path in sys.argv[1:]:
        report(path)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
