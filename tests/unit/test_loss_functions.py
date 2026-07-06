"""Tests for the AdaSwarm loss functions (nn module)."""

import pytest
import torch

import adaswarm.nn


@pytest.fixture(autouse=True)
def _iris_dataset(monkeypatch):
    """Pin the dataset context so tests are isolated from global env state.

    The loss relies on ``ADASWARM_DATASET_NAME`` (read via ``options.dataset_name``)
    to decide how particle positions are initialised; without pinning it, the
    result depends on test execution order.
    """
    monkeypatch.setenv("ADASWARM_DATASET_NAME", "Iris")


def _make_batch(dimension=8, classes=3, seed=0):
    torch.manual_seed(seed)
    predictions = torch.rand(dimension, classes)
    targets = torch.rand(dimension, classes, requires_grad=True)
    return targets, predictions


def test_bce_loss_forward_returns_scalar():
    criterion = adaswarm.nn.BCELoss()
    targets, predictions = _make_batch()
    loss = criterion(targets, predictions)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_cross_entropy_loss_forward_returns_scalar():
    criterion = adaswarm.nn.CrossEntropyLoss()
    targets, predictions = _make_batch()
    loss = criterion(targets, predictions)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_bce_loss_backward_produces_gradients():
    criterion = adaswarm.nn.BCELoss()
    targets, predictions = _make_batch()
    targets = targets.clone().detach().requires_grad_(True)
    loss = criterion(targets, predictions)
    loss.backward()
    assert targets.grad is not None
    assert targets.grad.shape == targets.shape
