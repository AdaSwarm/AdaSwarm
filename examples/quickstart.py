#!/usr/bin/env python3
"""
AdaSwarm Quickstart (2026)
==========================

A single, self-contained script that shows the *one thing* you need to know:

    AdaSwarm is a drop-in LOSS FUNCTION. You keep your normal optimiser
    (here, Adam) and your normal training loop — you only swap the criterion.

This script trains the same tiny network on the Iris dataset twice:
  1. with a standard ``torch.nn.BCELoss``
  2. with AdaSwarm's swarm-based ``adaswarm.nn.BCELoss``
and prints the final test accuracy of each so you can see AdaSwarm plug in.

Run it:
    uv run python examples/quickstart.py
    # or, once installed:  python examples/quickstart.py
"""

from __future__ import annotations

import torch

import adaswarm.nn
from adaswarm.data import DataLoaderFetcher


def train_and_evaluate(use_adaswarm: bool, epochs: int = 40, seed: int = 0) -> float:
    """Train the Iris model once and return final test accuracy.

    The ONLY difference between the two runs is the ``criterion`` line below.
    """
    torch.manual_seed(seed)

    # 1. Data — Iris is the default; no environment variables required.
    fetcher = DataLoaderFetcher(name="Iris")
    train_loader = fetcher.train_loader()
    test_loader = fetcher.test_loader()

    # 2. Model — a plain 2-layer network (sigmoid output for BCE).
    model = fetcher.model()

    # 3. Optimiser — standard Adam, unchanged whether or not we use AdaSwarm.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # 4. Loss — this is the whole point. Swap the criterion, nothing else.
    if use_adaswarm:
        criterion = adaswarm.nn.BCELoss()      # <-- swarm-intelligence loss
    else:
        criterion = torch.nn.BCELoss()         # <-- ordinary PyTorch loss

    # 5. A completely ordinary training loop.
    for _ in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.float(), targets.float()
            optimizer.zero_grad()
            outputs = model(inputs).float()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    # 6. Evaluate.
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs.float())
            predicted = outputs.argmax(dim=1)
            actual = targets.argmax(dim=1)
            correct += (predicted == actual).sum().item()
            total += targets.size(0)
    return correct / total


def main() -> None:
    print("Training with standard BCELoss ...")
    baseline_acc = train_and_evaluate(use_adaswarm=False)

    print("Training with AdaSwarm BCELoss ...")
    adaswarm_acc = train_and_evaluate(use_adaswarm=True)

    print("\n" + "=" * 40)
    print(f"Standard BCELoss   test accuracy: {baseline_acc:.2%}")
    print(f"AdaSwarm  BCELoss   test accuracy: {adaswarm_acc:.2%}")
    print("=" * 40)


if __name__ == "__main__":
    main()
