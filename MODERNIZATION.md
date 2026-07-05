# AdaSwarm Modernization Manifesto (2022 → 2026)

> A Principal-Engineer review and staged upgrade plan.
> Each item is independently checkable. Work top-to-bottom; phases are ordered so
> that later work builds on a green baseline established earlier.

**Legend:** 🔴 blocker / correctness · 🟠 health / maintainability · 🟢 polish · 🔗 closes a tracked issue

> **North Star:** a newcomer with *zero* prior context can install, run, and understand
> AdaSwarm in under 5 minutes — via a copy-paste quickstart and a runnable notebook.
> Every phase is judged against that bar.

---

## Phase 0 — Baseline & Safety Net
*Goal: be able to build, run, and test the project as-is before changing anything.*

- [ ] 🔴 Capture a "before" snapshot: `git status`, current Python (3.12 local), record that `main` is level with `origin/main`
- [ ] 🔴 Confirm the project currently builds/installs in a throwaway env (document the failure mode if it doesn't — see #80)
- [ ] 🟠 Run the existing `unittest` suite and record pass/fail baseline
- [ ] 🟠 Create a working branch (e.g. `modernize/2026`) so `main` stays clean
- [ ] 🟢 Note current `.git` size (~36 MiB) as the baseline for repo-hygiene work

---

## Phase 1 — Repository Hygiene 🔗 #81
*Goal: stop shipping binaries and generated artifacts; shrink the repo.*

- [x] 🔴 🔗 #81 Remove committed binary/generated data from tracking:
  - [x] `data/MNIST/raw/*.ubyte` (regenerated on download) — *already untracked; kept ignored*
  - [x] `checkpoint/ckpt.pth` — *already untracked; kept ignored*
  - [x] `mnist_performance*/` TensorBoard event dumps — *already untracked*
  - [x] `lightning_logs/`, `profile*.austin`, `report*/` — *already untracked*
  - [x] `dist/*.whl` / `dist/*.tar.gz` build artifacts — **untracked (`git rm --cached`)**
- [x] 🟠 Tighten `.gitignore` to cover all of the above (rewritten into clear sections)
- [ ] 🟠 Decide history strategy: `.git` is ~36 MiB (dominated by `docs/paper/*.pdf` ≈13 MB + previously-committed data). **Deferred — needs team agreement before a `git filter-repo` + force-push.**
- [x] 🟢 Remove dead code: `adaswarm/resnet.py` (no imports reference it) — **removed**
- [x] 🟢 Remove committed `__pycache__` / `.pyc` artifacts — *none tracked; cleaned working tree*

---

## Phase 2 — Toolchain & Packaging 🔗 #80
*Goal: modern, resolvable dependency management on current Python.*

- [x] 🔴 🔗 #80 Relax the impossible version pins that break `pip install adaswarm`:
  - [x] `torch` `^1.10` → `>=2.2` (resolved to 2.12.1)
  - [x] `torchvision` `^0.11.1` (`<0.12`) → `>=0.17` (moved to `examples` extra; resolved to 0.27.1)
  - [x] `pandas`/`scipy`/`tabulate`/`visidata`/`seaborn` — **removed (unused)**; `scikit-learn` `>=1.3`, `matplotlib` `>=3.8` (examples extra)
- [x] 🔴 Widen Python support: `>=3.9,<3.11` → `>=3.10` (matrix targets 3.10–3.13)
- [x] 🟠 Migrate packaging to **PEP 621** (`[project]` table) with **hatchling** backend + `[dependency-groups]` for dev
- [x] 🟠 Generate and commit a fresh lockfile (`uv.lock`, 165 pkgs); dropped stale `poetry.lock`
- [x] 🟢 Drop unused deps if confirmed unused (`visidata`, `tabulate`, `pandas`, `scipy`, `seaborn`)
- [x] 🟢 Bump package version (0.1.0) and refresh metadata (URLs, classifiers)

---

## Phase 3 — Code Correctness & Deprecated APIs 🔴
*Goal: remove PyTorch deprecations that will break on modern versions.*

- [x] 🔴 Replace `torch.autograd.Variable` (deprecated since 0.4) with plain tensors in `examples/main.py`
- [x] 🔴 Replace `F.sigmoid(...)` with `torch.sigmoid(...)` in `adaswarm/model.py`
- [x] 🟠 Modernize `super(Model, self).__init__()` → `super().__init__()`
- [x] 🟠 Replace `numpy.core.fromnumeric` import in tests with public API (dead import removed)
- [ ] 🟠 Audit `get_device()` used as a **default argument** (evaluated at import time) — move to call-time resolution *(deferred to Phase 4 config refactor)*
- [ ] 🟢 Replace hand-rolled `to_categorical` (copied from Keras) with `torch.nn.functional.one_hot` where feasible *(deferred to Phase 4)*

> ✅ Baseline green: **17 passed** on torch 2.12.1 + numpy 2.x (was untested on modern stack).

---

## Phase 4 — Architecture & Design 🟠
*Goal: reduce coupling, kill duplication, improve testability.*

- [x] 🟠 De-duplicate the velocity-update logic — was implemented **twice** (verbatim) in `RotatedEMParticle.update_velocity` and `ParticleSwarm.update_velocities`; swarm now delegates to the particle method (single source of truth)
- [ ] 🟠 Replace global env-var config (`options.py`) with an injectable, typed config object *(deferred — API-changing; needs maintainer sign-off)*
- [ ] 🟠 Reconsider `class ParticleSwarm(list)` — favor composition *(deferred — API-changing)*
- [ ] 🟢 Address the standing `# TODO: Vectorize this operation` — batch the per-particle loop *(deferred — perf work, needs benchmark harness)*
- [ ] 🟢 Pull hardcoded hyperparameters (`swarm_size=10`, `beta=0.1`, `run_iteration(number=40)`) into config *(pairs with config-object work)*
- [ ] 🟢 Add reproducibility: centralized seeding for `torch` / `numpy` / `random` *(deferred to Phase 6 test utilities)*

---

## Phase 5 — API & Documentation Alignment 🔗 #46
*Goal: make the README match reality, or make reality match the README.*

- [x] 🔴 🔗 #46 Resolve the headline mismatch: README promises `optim.AdaSwarm()` (drop-in optimizer); actual API is a **loss function** (`adaswarm.nn.BCELoss()`) used with `torch.optim.Adam`
  - [x] Documented the true loss-function API clearly (TL;DR + FAQ + mental-model diagram)
  - [ ] *(Future)* provide a real `optim.AdaSwarm` optimizer wrapper to honor the originally promised API
- [x] 🟠 Rewrite the "How to run" section for the new toolchain (uv)
- [x] 🟠 Fix citation typos in README bibtex ("daSwarm", "201")
- [x] 🟢 Add a minimal, copy-pasteable usage example that runs standalone (addresses the #46 confusion about global `dataset` vars)
- [x] 🟢 Populate the empty root `adaswarm/__init__.py` with public exports + `__version__`

---

## Phase 5.5 — Developer Experience & Onboarding 🟢🔴
*Goal: a stranger to the project is productive in < 5 minutes. This is the North Star.*

- [x] 🔴 **60-second Quickstart** at the top of the README: install → import → train a model in one copy-paste block that actually runs
- [x] 🔴 **Runnable Jupyter notebook** (`examples/quickstart.ipynb`) — narrated end-to-end: load data → build model → train with AdaSwarm loss → plot loss/accuracy vs. plain Adam *(executed clean, plot renders)*
  - [x] "Open in Colab" badge with a single-cell `pip install` so no local setup is needed
  - [x] Self-contained: no reliance on hidden env vars or global `dataset_name()` state
- [x] 🟠 **Standalone script example** (`examples/quickstart.py`) mirroring the notebook for non-notebook users
- [ ] 🟠 Clear, task-oriented example set: (1) tabular/Iris ✅, (2) image/MNIST, (3) "bring your own model & data" *(Iris done; MNIST/BYO deferred)*
- [x] 🟠 Document the mental model up front: *AdaSwarm is a drop-in **loss function** used with a standard optimizer* — with a diagram of where it plugs into the training loop
- [x] 🟠 `uv run` one-liners so users can try it without cloning
- [x] 🟢 Troubleshooting/FAQ section seeded from real issues (#46, #80) — "it installed but how do I use it?"
- [x] 🟢 Inline, well-commented example code (explain *why*, not just *what*)
- [x] 🟢 Add a plotted result showing AdaSwarm vs Adam convergence *(in the notebook)*

---

## Phase 6 — Testing & Quality Gates 🟠
*Goal: a trustworthy, fast, modern feedback loop.*

- [x] 🟠 Migrate `unittest` → **pytest** (runner + config in `pyproject.toml`; existing `unittest.TestCase` classes run unchanged)
- [x] 🟠 Add `pytest` coverage reporting (`pytest-cov`; 73% and rising)
- [x] 🟠 Add tests around the loss function forward/backward (`test_loss_functions.py` — was 0% covered, now exercised)
- [x] 🟢 Fixed latent **test-pollution** bug (global `ADASWARM_DATASET_NAME` leaked across tests) via isolated fixture
- [ ] 🟢 Add seeded determinism tests for the swarm *(deferred — pairs with Phase 4 seeding utility)*
- [x] 🟢 Replace `pylint` + `black 21.x` with **ruff** (format + lint) and add **mypy** type checking (removed `.pylintrc`)

---

## Phase 7 — CI/CD Modernization 🟠
*Goal: current, multi-version, trustworthy pipeline.*

- [x] 🟠 Rename workflow from "GitHub Actions Demo" to `CI`
- [x] 🔴 Bump actions: `checkout@v2` → `@v4`, dropped `setup-python@v2`
- [x] 🟠 Replace the pinned third-party Poetry action with **`astral-sh/setup-uv`**
- [x] 🟠 Test matrix across Python 3.10 / 3.11 / 3.12 / 3.13
- [x] 🟢 Add ruff + mypy + pytest-coverage as CI steps
- [ ] 🟢 Add Dependabot / uv update automation *(deferred — optional follow-up)*

---

## Phase 8 — Close the Loop
*Goal: land it and update the community.*

- [ ] 🟠 Verify a clean `pip install .` / `uv pip install .` in a fresh env
- [ ] 🟠 Full green run: ruff, types, tests, example script end-to-end
- [ ] 🔗 Comment on / close #80 (deps), #81 (LFS/binaries), #46 (API docs), and review stale #2
- [ ] 🟢 Tag a new release and publish updated package
- [ ] 🟢 Update `README` badges / status

---

### Tracked GitHub Issues → Phase mapping
| Issue | Summary | Addressed in |
|---|---|---|
| [#80](https://github.com/AdaSwarm/AdaSwarm/issues/80) | Dependency conflict on install | Phase 2 |
| [#81](https://github.com/AdaSwarm/AdaSwarm/issues/81) | Use Git LFS / binaries in repo | Phase 1 |
| [#46](https://github.com/AdaSwarm/AdaSwarm/issues/46) | `optim.AdaSwarm()` API not working | Phase 5 |
| [#2](https://github.com/AdaSwarm/AdaSwarm/issues/2)  | Stale `torch.clamp(torch.exp(...))` question | Phase 8 (review/close) |
