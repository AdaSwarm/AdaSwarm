"""General-purpose AdaSwarm loss for arbitrary (elementwise) loss functions.

This is the regression-friendly, vectorised counterpart to the classification
losses in :mod:`adaswarm.nn`. It applies the AdaSwarm mechanism to *any* loss:

1. Run a particle swarm over candidate model **outputs** ``p`` to minimise
   ``loss_fn(p, target)`` — finding the swarm global best ``gbest``.
2. Set the gradient w.r.t. the model output to ``(output - gbest)`` instead of the
   analytic ``dL/doutput``.

Because the descent direction comes from the swarm's global best rather than the
local loss slope, it can escape deceptive local minima in non-convex output losses
where plain gradient descent stalls.

The ``loss_fn`` must be **elementwise** (return a tensor the same shape as its
inputs, i.e. unreduced); ``SwarmLoss`` handles the reduction. Example::

    import torch, adaswarm.nn
    pinball = lambda p, y, tau=0.5: torch.maximum(tau * (y - p), (tau - 1) * (y - p))
    criterion = adaswarm.nn.SwarmLoss(pinball)
    loss = criterion(model(x), y)
    loss.backward()
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from adaswarm.utils.options import get_device

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _swarm_best(
    target: torch.Tensor,
    loss_fn: LossFn,
    *,
    swarm_size: int,
    iterations: int,
    span: float,
    inertia: float,
    c_1: float,
    c_2: float,
    generator: torch.Generator | None,
    device: torch.device,
) -> torch.Tensor:
    """Vectorised PSO over candidate outputs. Returns ``gbest`` (same shape as ``target``).

    Each particle holds a full candidate output tensor; fitness is the mean loss of
    that candidate against ``target``.
    """
    base_shape = target.shape
    reduce_dims = tuple(range(1, target.dim() + 1))  # everything except the particle axis

    def spread() -> torch.Tensor:
        noise = torch.rand(swarm_size, *base_shape, device=device, generator=generator)
        return (noise - 0.5) * 2.0 * span + target.unsqueeze(0)

    positions = spread()
    velocities = torch.zeros_like(positions)
    targets_b = target.unsqueeze(0).expand_as(positions)

    def fitness(pos: torch.Tensor) -> torch.Tensor:
        per_element = loss_fn(pos, targets_b)  # (swarm, *shape)
        if reduce_dims:
            return per_element.mean(dim=reduce_dims)  # (swarm,)
        return per_element

    pbest = positions.clone()
    pbest_val = fitness(positions)
    gbest = pbest[int(pbest_val.argmin())].clone()

    view = (swarm_size, *([1] * target.dim()))
    for _ in range(iterations):
        r_1 = torch.rand(positions.shape, device=device, generator=generator)
        r_2 = torch.rand(positions.shape, device=device, generator=generator)
        velocities = (
            inertia * velocities
            + c_1 * r_1 * (pbest - positions)
            + c_2 * r_2 * (gbest.unsqueeze(0) - positions)
        )
        positions = positions + velocities
        val = fitness(positions)
        improved = val < pbest_val
        pbest = torch.where(improved.view(view), positions, pbest)
        pbest_val = torch.where(improved, val, pbest_val)
        gbest = pbest[int(pbest_val.argmin())].clone()

    return gbest.detach()


class _SwarmLossFunction(torch.autograd.Function):
    """Autograd bridge: report ``loss_fn`` but back-propagate the swarm surrogate."""

    @staticmethod
    def forward(ctx, output, target, loss_fn, config):  # pylint: disable=arguments-differ
        gbest = _swarm_best(target.detach(), loss_fn, **config)
        ctx.save_for_backward(output, gbest)
        return loss_fn(output, target).mean()

    @staticmethod
    def backward(ctx, grad_output):  # pylint: disable=arguments-differ
        output, gbest = ctx.saved_tensors
        # Pull the output toward the swarm's global best (mean-reduced scale).
        grad_input = (output - gbest) / output.shape[0]
        return grad_input * grad_output, None, None, None


def SwarmLoss(  # noqa: N802 (factory mirrors torch.nn loss naming)
    loss_fn: LossFn,
    *,
    swarm_size: int = 20,
    iterations: int = 40,
    span: float = 6.0,
    inertia: float = 0.7,
    c_1: float = 1.5,
    c_2: float = 1.5,
    seed: int | None = None,
    device: torch.device | None = None,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Create an AdaSwarm loss around an arbitrary elementwise ``loss_fn``.

    Args:
        loss_fn: Elementwise (unreduced) loss returning a tensor shaped like its inputs.
        swarm_size: Number of particles.
        iterations: Swarm iterations per forward pass.
        span: Half-width of the initial particle spread around the target.
        inertia, c_1, c_2: PSO inertia and acceleration coefficients.
        seed: Optional seed for reproducible swarms.
        device: Torch device (defaults to :func:`adaswarm.utils.options.get_device`).

    Returns:
        A callable ``criterion(output, target) -> scalar loss`` usable in a standard
        training loop with any optimiser.
    """
    resolved_device = device or get_device()
    generator: torch.Generator | None = None
    if seed is not None:
        generator = torch.Generator(device=resolved_device)
        generator.manual_seed(seed)

    config = {
        "swarm_size": swarm_size,
        "iterations": iterations,
        "span": span,
        "inertia": inertia,
        "c_1": c_1,
        "c_2": c_2,
        "generator": generator,
        "device": resolved_device,
    }

    def criterion(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probe = loss_fn(output, target)
        if probe.shape != output.shape:
            raise ValueError(
                "SwarmLoss expects an elementwise (unreduced) loss returning a tensor "
                f"the same shape as its inputs {tuple(output.shape)}, but loss_fn returned "
                f"shape {tuple(probe.shape)}. Remove any .mean()/.sum() reduction from loss_fn."
            )
        return _SwarmLossFunction.apply(output, target, loss_fn, config)

    return criterion
