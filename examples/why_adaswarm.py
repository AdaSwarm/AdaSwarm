#!/usr/bin/env python3
"""
Why AdaSwarm? Two things plain gradient descent cannot do.
==========================================================

AdaSwarm (via ``adaswarm.nn.SwarmLoss``) runs a particle swarm over candidate model
*outputs* and sets the gradient to ``output - gbest`` instead of the analytic loss
slope. Two consequences that a standard optimiser cannot replicate:

  ACT 1 — escapes deceptive LOCAL MINIMA in a non-convex output loss.
  ACT 2 — trains on a NON-DIFFERENTIABLE loss (zero gradient everywhere),
          because the swarm only ever *evaluates* the loss, never differentiates it.

Run:
    uv run python examples/why_adaswarm.py
"""

from __future__ import annotations

import torch
import torch.nn as nn

import adaswarm.nn


def mse(p, y):
    return ((p - y) ** 2).mean()


def make_model(seed: int) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 1))


def data():
    x = torch.linspace(-3, 3, 200).unsqueeze(1)
    y = 2.0 * x - 1.0 + 8.0  # shifted far from the model's init outputs
    return x, y


# --- ACT 1: deceptive local minima -------------------------------------------
def multiwell(p, y):
    """Elementwise, non-convex: global min at p==y, regular deceptive local minima."""
    r = p - y
    return 0.1 * r**2 + (1 - torch.cos(3 * r))


# --- ACT 2: non-differentiable (piecewise-constant) --------------------------
def step_loss(p, y):
    """Elementwise, non-differentiable: mismatch of integer levels (zero grad a.e.)."""
    return (p.round() - y.round()).abs()


def train(
    loss_fn,
    mode: str,
    seed: int = 0,
    epochs: int = 250,
    lr: float = 0.05,
    span: float = 6.0,
):
    x, y = data()
    model = make_model(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = adaswarm.nn.SwarmLoss(loss_fn, seed=seed, span=span) if mode == "adaswarm" else None
    for _ in range(epochs):
        opt.zero_grad()
        out = model(x)
        loss = criterion(out, y) if mode == "adaswarm" else loss_fn(out, y).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return mse(model(x), y).item()


def run_act(title, loss_fn, span, seeds=range(4)):
    print(f"\n{title}")
    print(f"  {'seed':>4}{'standard Adam':>16}{'AdaSwarm':>12}")
    std, ada = [], []
    for s in seeds:
        a = train(loss_fn, "adaswarm", seed=s, span=span)
        d = train(loss_fn, "standard", seed=s, span=span)
        std.append(d)
        ada.append(a)
        print(f"  {s:>4}{d:>16.3f}{a:>12.3f}")
    mean_std = sum(std) / len(std)
    mean_ada = sum(ada) / len(ada)
    print(f"  {'mean':>4}{mean_std:>16.3f}{mean_ada:>12.3f}   (final MSE, lower is better)")


def main():
    run_act("ACT 1 — deceptive local minima (multi-well loss)", multiwell, span=6.0)
    run_act("ACT 2 — non-differentiable loss (zero gradient everywhere)", step_loss, span=8.0)
    print(
        "\nTakeaway: on both losses, standard Adam stalls (local minimum / no gradient) "
        "while AdaSwarm reaches the target — because it optimises via a swarm over outputs, "
        "not the analytic loss slope."
    )


if __name__ == "__main__":
    main()
