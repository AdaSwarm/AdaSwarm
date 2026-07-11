"""Tests for adaswarm.nn.SwarmLoss (general custom-loss AdaSwarm).

Written test-first: these tests specify the contract for SwarmLoss.
"""

import pytest
import torch
import torch.nn as nn

import adaswarm.nn


def mse_elem(p, y):
    """Elementwise squared error."""
    return (p - y) ** 2


def multiwell_elem(p, y):
    """Non-convex, multi-modal output loss: global min at p == y, deceptive local minima."""
    r = p - y
    return 0.1 * r**2 + (1 - torch.cos(3 * r))


def _make_model(seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 1))


def _data():
    x = torch.linspace(-3, 3, 128).unsqueeze(1)
    y = 2.0 * x - 1.0 + 8.0  # shifted far from the model's init outputs
    return x, y


# --- contract -----------------------------------------------------------------


def test_forward_returns_scalar_equal_to_mean_loss():
    criterion = adaswarm.nn.SwarmLoss(mse_elem, seed=0)
    output = torch.rand(8, 1, requires_grad=True)
    target = torch.rand(8, 1)
    loss = criterion(output, target)
    assert loss.dim() == 0
    assert torch.isclose(loss, mse_elem(output, target).mean())


def test_backward_produces_gradient_of_output_shape():
    criterion = adaswarm.nn.SwarmLoss(mse_elem, seed=0)
    output = torch.rand(8, 1, requires_grad=True)
    target = torch.rand(8, 1)
    criterion(output, target).backward()
    assert output.grad is not None
    assert output.grad.shape == output.shape


def test_deterministic_with_seed():
    output = torch.rand(8, 1, requires_grad=True)
    target = torch.rand(8, 1)
    g = []
    for _ in range(2):
        out = output.detach().clone().requires_grad_(True)
        adaswarm.nn.SwarmLoss(multiwell_elem, seed=42)(out, target).backward()
        g.append(out.grad.clone())
    assert torch.allclose(g[0], g[1])


def test_rejects_reduced_loss_function():
    """A loss that returns a scalar (already reduced) must be rejected clearly."""
    reduced = lambda p, y: ((p - y) ** 2).mean()  # noqa: E731
    criterion = adaswarm.nn.SwarmLoss(reduced, seed=0)
    with pytest.raises(ValueError, match="elementwise"):
        criterion(torch.rand(8, 1, requires_grad=True), torch.rand(8, 1))


# --- the headline claim -------------------------------------------------------


def test_escapes_local_minimum_where_gradient_descent_stalls():
    """On a multi-well output loss, AdaSwarm reaches a far lower fit than plain Adam."""
    x, y = _data()

    def train(mode, seed=0, epochs=200, lr=0.05):
        model = _make_model(seed)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = adaswarm.nn.SwarmLoss(multiwell_elem, seed=seed)
        for _ in range(epochs):
            opt.zero_grad()
            out = model(x)
            loss = criterion(out, y) if mode == "adaswarm" else multiwell_elem(out, y).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            return ((model(x) - y) ** 2).mean().item()

    standard_mse = train("standard")
    adaswarm_mse = train("adaswarm")
    # AdaSwarm should be at least 2x better (in practice ~10-100x on this loss).
    assert adaswarm_mse < standard_mse * 0.5
