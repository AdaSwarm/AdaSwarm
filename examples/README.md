# AdaSwarm examples

| Example | What it shows | Start here if… |
|---|---|---|
| [`why_adaswarm.py`](why_adaswarm.py) / [`why_adaswarm.ipynb`](why_adaswarm.ipynb) | **Why AdaSwarm exists** — head-to-head vs plain Adam on (1) a deceptive multi-well loss and (2) a non-differentiable loss. | You want to see where AdaSwarm genuinely beats gradient descent. |
| [`quickstart.py`](quickstart.py) / [`quickstart.ipynb`](quickstart.ipynb) | **Basic API** — using `adaswarm.nn.BCELoss()` with Adam on the Iris dataset. | You just want the minimal usage pattern on a classic dataset. |
| [`main.py`](main.py) | Longer training-loop example on Iris. | You want a fuller training script. |

Install the example dependencies first:

```bash
uv sync --extra examples
```

See [`docs/use-cases.md`](../docs/use-cases.md) for real-world problem classes where `SwarmLoss` fits.
