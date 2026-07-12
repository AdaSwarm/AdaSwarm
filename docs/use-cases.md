# Real-world use cases for AdaSwarm `SwarmLoss`

AdaSwarm's general loss, [`adaswarm.nn.SwarmLoss`](../adaswarm/swarm_loss.py), optimises a model by
running a **particle swarm over candidate outputs** and pulling the model toward the swarm's global
best, instead of following the analytic loss gradient. That gives it two capabilities a standard
optimiser does not have:

1. **Escapes deceptive local minima** in a non-convex / multi-modal *output* loss.
2. **Trains through non-differentiable losses** — it only ever *evaluates* the loss, never
   differentiates it, so the loss can be discrete, black-box, or piecewise-constant.

This document collects concrete problem classes where one or both of those properties matter. Each
entry notes **why standard gradient descent struggles**, **how `SwarmLoss` helps**, and an honest
**maturity** flag:

- ✅ *demonstrated* — shown to work in this repo (`examples/why_adaswarm.ipynb`)
- 🧪 *plausible* — strong theoretical fit, needs a benchmark
- ⚠️ *caveat* — may work but competes with strong specialised methods

> **Scope reminder.** `SwarmLoss` is a niche tool. For smooth convex losses (MSE, cross-entropy) a
> standard optimiser is cheaper and just as good. Reach for `SwarmLoss` when the *output* loss is
> non-convex or non-differentiable and normal training stalls.

---

## A. Periodic / wrapped / multi-modal outputs

These problems have loss landscapes with regularly spaced local minima (the `1 − cos` shape in the
demo). Gradient descent slides into the nearest trough; the swarm can jump between basins.

### A1. Phase retrieval 🧪
Reconstruct a signal/image from magnitude-only measurements (`|F x| = y`) — ptychography, X-ray
crystallography, coherent diffraction imaging, holography.
- **Why GD struggles:** the magnitude loss is famously non-convex with many local minima; classical
  solvers rely on alternating projections and careful initialisation.
- **How SwarmLoss helps:** the swarm explores multiple phase hypotheses per step instead of committing
  to the nearest local basin.

### A2. Phase unwrapping / interferometry ✅ (multi-frequency) · 🧪 (single-frequency)
Recover absolute phase from values wrapped into `(−π, π]` — InSAR (satellite ground deformation),
optical metrology, MRI phase.
- **Why GD struggles:** `2π` ambiguities create a periodic, multi-modal objective.
- **How SwarmLoss helps:** with **two wavelengths** the per-pixel objective is globally identifiable
  (Chinese-remainder-style disambiguation) but riddled with alias local minima —
  `SwarmLoss(..., per_sample=True)` recovers the deformation where Adam gets trapped (>100× lower error).
  **Demonstrated** in [`examples/insar_phase_unwrapping.ipynb`](../examples/insar_phase_unwrapping.ipynb).
- **Caveat:** dense *single-frequency* unwrapping needs a spatial-smoothness prior; a per-pixel swarm
  alone will not solve it.

### A3. Audio pitch / fundamental-frequency estimation 🧪
Predict F0 for music/speech.
- **Why GD struggles:** *octave errors* — half/double-frequency give strong local minima.
- **How SwarmLoss helps:** the swarm can "jump octaves" to the global best rather than settling on a
  harmonic.

### A4. Direction-of-arrival / beamforming 🧪
Estimate source angles from sensor arrays.
- **Why GD struggles:** angular ambiguities and side-lobes are local minima.
- **How SwarmLoss helps:** global exploration over the angular output.

### A5. Rotation / 6-DoF pose (SO(3)) 🧪
Regress orientations (robotics, AR, structure-from-motion).
- **Why GD struggles:** geodesic/angular losses are periodic and multi-modal (gimbal-like ambiguity).
- **How SwarmLoss helps:** treats the periodic output loss globally.

---

## B. Non-differentiable / black-box losses

Here the loss has **no usable gradient at all** — standard backprop simply cannot be applied to it.
This is `SwarmLoss`'s most distinctive capability (see Act 2 of the demo). ✅

### B1. Directly optimising discrete evaluation metrics ✅🧪
Train against the metric you actually care about instead of a smooth surrogate: **F1, precision/recall
at k, AUC, IoU/Dice, word-error-rate, BLEU, exact-match**.
- **Why GD struggles:** these are piecewise-constant (zero gradient) or rank-based (non-differentiable).
  Teams normally hand-craft differentiable surrogates that only approximate the target metric.
- **How SwarmLoss helps:** pass the metric itself as an elementwise loss; the swarm optimises it directly.
  The demo's Act 2 (a piecewise-constant step loss where Adam cannot move) is a minimal proof of this.

### B2. Simulator-in-the-loop training 🧪
The "loss" is the output of a non-differentiable simulator — physics engines, SPICE circuit
simulators, ray/renderer pipelines, climate or market simulators.
- **Why GD struggles:** you cannot backprop through the simulator.
- **How SwarmLoss helps:** differentiate the *network*, not the simulator — the swarm only needs to
  evaluate the simulator on candidate outputs.

### B3. Quantised / discrete-output fitting ✅
Targets live on discrete levels (codebooks, symbol constellations, quantisation-aware training).
- **Why GD struggles:** rounding/quantisation kills the gradient.
- **How SwarmLoss helps:** demonstrated on the integer-level step loss in `why_adaswarm.ipynb`.

### B4. Rank / sort / assignment losses 🧪
Learning-to-rank, permutation and matching objectives.
- **Why GD struggles:** sorting/argmax are non-differentiable.
- **How SwarmLoss helps:** score candidate orderings directly.

---

## C. Multi-modal scientific model fitting

Domains where global optimisers (GA/PSO/basin-hopping) are already the norm — `SwarmLoss` brings that
into a neural training loop.

### C1. Spectral / peak fitting 🧪
Fit sums of Gaussians/Lorentzians to spectra — NMR, mass spectrometry, Raman/IR, chromatography.
- **Why GD struggles:** overlapping peaks create many local optima; results are sensitive to
  initialisation.
- **How SwarmLoss helps:** swarm exploration over the fit output.

### C2. Physics-informed neural networks (PINNs) for stiff PDEs 🧪
Residual losses for Schrödinger, Navier–Stokes, reaction–diffusion, etc. (the original AdaSwarm paper's
motivation).
- **Why GD struggles:** PINN residual landscapes are notoriously ill-conditioned with local minima.
- **How SwarmLoss helps:** applies swarm exploration at the residual/output level.

### C3. Robust regression with redescending estimators ⚠️
Non-convex robust losses (Tukey biweight, Welsch) that down-weight outliers.
- **Caveat:** in our spike, plain Adam already coped with a mild saturating (Welsch) loss, so this is
  **not** a guaranteed win — only losses with genuine *multiple minima* are likely to benefit. Benchmark
  before relying on it.

---

## When **not** to use `SwarmLoss`

- **Smooth convex losses** (MSE, MAE, cross-entropy, hinge, pinball): standard (sub)gradient methods are
  optimal and far cheaper. `SwarmLoss` is parity-at-best and slower.
- **Very high-dimensional outputs** (large-vocab softmaxes, dense pixel maps): swarm cost scales with
  output dimension × swarm size × iterations. Best for **low-dimensional per-sample outputs**.
- **When you have not benchmarked it.** Every 🧪 above is a hypothesis, not a promise.

---

## Try one in ~10 lines

```python
import torch
import adaswarm.nn

# Any ELEMENTWISE (unreduced) loss works — here a wrap-aware phase loss (A2):
def phase_loss(pred, target):
    return 1 - torch.cos(pred - target)      # periodic -> multi-modal

criterion = adaswarm.nn.SwarmLoss(phase_loss)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

for x, y in loader:
    optimizer.zero_grad()
    loss = criterion(model(x), y)            # swarm runs internally
    loss.backward()                          # surrogate gradient -> standard step
    optimizer.step()
```

See [`examples/why_adaswarm.ipynb`](../examples/why_adaswarm.ipynb) for runnable head-to-head demos.

---

## Help validate a use case

Most entries above are marked 🧪 *plausible* — we would love reproducible benchmarks turning them into
✅ *demonstrated*. If you try `SwarmLoss` on one of these (or a new one), please open an issue or PR with
your setup and results.
