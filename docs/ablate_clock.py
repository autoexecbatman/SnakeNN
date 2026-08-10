"""Why the clock cannot be frozen by editing the checkpoint. Kept as the evidence.

This refuses to write anything, and the refusal is the result. The real ablation is
`SnakeEnv::freezeClockForAblation`, reached by `--freeze-clock-percent`.

The idea below is sound in the interior of the board and false at its border: zero
padding truncates the stem kernel there, so the clock's contribution is not the same
constant everywhere and a per-channel running mean cannot absorb it. On 10x10 with a
3x3 stem that is 36 of 100 cells, and the residual is 1.087 rather than 0. The
exactness check at the bottom is what catches it.

The original claim, left standing so the reasoning can be read:


The ablation this supports: is the network's time awareness load-bearing, or has
training put weight on plane 8 that changes nothing it does? Fine-tuning cannot
answer that, because a fine-tune changes the weights as well as the input.

The clock plane is constant across the board, so its whole contribution to the stem
convolution's output channel j is `kernel_sum_j * budget`. Replacing the varying
budget with a fixed `value` is therefore a constant shift of the pre-normalisation
activation, and batch normalisation subtracts a fixed running mean - so the shift is
absorbed exactly by moving that mean. The result is the same network with the clock
frozen, not an approximation of it.

    stem_conv.weight[:, 8] := 0
    stem_norm.running_mean -= kernel_sum * value

Run: python docs/ablate_clock.py <in.pt> <out.pt> <value>
"""

import sys

import torch

CLOCK_PLANE = 8


def check_the_shift_is_exact(original_weight, ablated_weight, shift, value):
    """The two stems agree when the clock reads `value`, and disagree when it does not.

    Both halves matter. Agreement alone would also hold if the clock channel were
    already zero, in which case the ablation is vacuous and there is nothing to
    measure - the disagreement check is what rules that out.
    """
    torch.manual_seed(0)
    planes = torch.rand(4, original_weight.size(1), 10, 10)

    planes[:, CLOCK_PLANE] = value
    at_value = torch.nn.functional.conv2d(planes, original_weight, padding=1)
    ablated = torch.nn.functional.conv2d(planes, ablated_weight, padding=1)
    # The normalisation subtracts the running mean, so the compensated activation is
    # the ablated one plus the shift that was taken out of it.
    compensated = ablated + shift.view(1, -1, 1, 1)
    agreement = (at_value - compensated).abs().max().item()

    planes[:, CLOCK_PLANE] = 0.0 if value != 0.0 else 1.0
    elsewhere = torch.nn.functional.conv2d(planes, original_weight, padding=1)
    difference = (elsewhere - at_value).abs().max().item()

    print(f"  at budget {value}: max difference {agreement:.3e} (must be ~0)")
    print(f"  at another budget: max difference {difference:.3e} (must not be 0)")
    return agreement < 1e-4 and difference > 1e-4


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        return 2
    source_path, target_path, value = sys.argv[1], sys.argv[2], float(sys.argv[3])

    # Onto the host: the trainer saves from the device it trained on, and the C++
    # loader copies into a network that is moved to its own device afterwards.
    module = torch.jit.load(source_path, map_location="cpu")
    parameters = dict(module.named_parameters())
    buffers = dict(module.named_buffers())

    weight = parameters["stem_conv.weight"]
    if weight.size(1) <= CLOCK_PLANE:
        print(f"{source_path} has {weight.size(1)} input planes and no clock to freeze")
        return 1

    with torch.no_grad():
        original_weight = weight.detach().clone()
        # Summed over the kernel, because a spatial pattern applied to a constant
        # input contributes only its total.
        shift = original_weight[:, CLOCK_PLANE].sum(dim=(1, 2)) * value

        ablated_weight = original_weight.clone()
        ablated_weight[:, CLOCK_PLANE] = 0.0

        if not check_the_shift_is_exact(original_weight, ablated_weight, shift, value):
            print("the compensation is not exact - refusing to write a checkpoint")
            return 1

        weight.copy_(ablated_weight)
        buffers["stem_norm.running_mean"].sub_(shift)

    torch.jit.save(module, target_path)
    print(f"wrote {target_path}: clock frozen at {value}, "
          f"running mean shifted by max {shift.abs().max().item():.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
