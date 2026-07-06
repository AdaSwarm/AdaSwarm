"""AdaSwarm — swarm-intelligence loss functions for PyTorch.

AdaSwarm augments a standard gradient-based optimiser (e.g. ``torch.optim.Adam``)
with Particle Swarm Optimisation. In practice you use it as a **drop-in loss
function**: keep your usual optimiser and training loop, and swap your criterion
for one of AdaSwarm's swarm-based losses.

Quickstart
----------
>>> import torch
>>> import adaswarm.nn
>>> criterion = adaswarm.nn.BCELoss()          # swarm-based loss
>>> optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
>>> loss = criterion(model(inputs), targets)   # use like any nn loss
>>> loss.backward()
>>> optimizer.step()

See ``examples/quickstart.py`` and ``examples/quickstart.ipynb`` for full,
runnable end-to-end examples.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("adaswarm")
except PackageNotFoundError:  # package is not installed (e.g. running from source)
    __version__ = "0.0.0.dev0"

__all__ = ["__version__"]
