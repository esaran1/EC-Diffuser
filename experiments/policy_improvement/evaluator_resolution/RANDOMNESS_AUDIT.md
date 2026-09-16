# Sources of evaluator realization variability — audit

**Finding: every randomness source under Python control is explicitly seeded and
identical between repeats.** The observed repeat variability must therefore
originate below that layer — in GPU-level nondeterminism — but this audit does
**not** isolate which component, and the conservative umbrella term
**END-TO-END EVALUATOR REALIZATION VARIABILITY** is used throughout.

## The evaluator, traced

`experiments/policy_improvement/phase2_diagnostics.py::control()`:

```python
for bs in range(0, n_ep, nE):
    idx = list(range(bs, min(bs + nE, n_ep))); na = len(idx)
    idx += [idx[-1]] * (nE - na)
    torch.manual_seed(NOISE_SEED + bs)                 # re-seeded EVERY batch
    obs = env.reset(set_init_states=..., set_goal_states=...)   # from frozen .pkl
    for _ in range(env.horizon):
        ...
        s = diff.conditional_sample(c, n_steps=4)      # NFE=4
        obs, _, _, infos = env.step(...)
    d = np.linalg.norm(st[:, 1:, :2] - gp, axis=-1).max(axis=1)
    succ.extend((d[:na] <= SUCCESS_THRESH).astype(float).tolist())
```

## Source table

| source | fixed? | seeded? | same seed between repeats? | deterministic given seed? | likely contribution | evidence |
|---|---|---|---|---|---|---|
| scenario init/goal states | **yes** | n/a | **yes** — loaded from frozen `.pkl` per E set | yes | **none** | `ep["init"]`/`ep["goal"]` read from `experiments/isaacgym_episode_sets/*.pkl`; never regenerated at eval time |
| scenario ordering / batching | **yes** | n/a | yes — deterministic `range(0, n_ep, nE)` | yes | **none** | index construction is a pure function of `n_ep`, `nE` |
| episode padding | **yes** | n/a | yes — last index repeated, `succ` truncated to `na` | yes | **none** | `idx += [idx[-1]] * (nE - na)`; only `d[:na]` scored |
| policy diffusion noise | no | **yes** | **yes** — `torch.manual_seed(NOISE_SEED + bs)` before each batch | yes on CPU; see CUDA row | **none from the draw itself** | same constant `NOISE_SEED = 20260902` in every run |
| PyTorch global RNG | no | **yes** | yes | see CUDA row | none from the draw | re-seeded per batch, so drift cannot accumulate |
| NFE / solver | **yes** | n/a | yes — `n_steps=4`, fixed-step Euler | yes | **none** | `kw["n_steps"] = 4` hardcoded in `control()` |
| success criterion | **yes** | n/a | yes — `max_obj_dist <= 0.04` | yes | **none** | `SUCCESS_THRESH` constant |
| horizon / env count | **yes** | n/a | yes | yes | **none** | from frozen `Args` |
| DLP encoder | **yes** (frozen weights) | n/a | yes | deterministic modulo CUDA | none from weights | encoder not retrained or reseeded |
| checkpoint weights | **yes** | n/a | yes — SHA-verified identical | yes | **none** | registry records SHA256 per evaluation |
| **Isaac Gym GPU physics** | no | **not seeded by us** | n/a | **no** — GPU contact solving is order/precision sensitive | **plausible primary** | only unfixed component in the loop; known nondeterminism in GPU-pipeline PhysX |
| **CUDA kernel nondeterminism** | no | n/a | n/a | **no** — atomics/reduction order | **plausible contributor** | `torch.use_deterministic_algorithms` NOT enabled; cuDNN benchmark not pinned |
| rendering | n/a | n/a | n/a | n/a | **none** | control loop does not render; imagination panels are separate |
| asynchronous stepping | no | n/a | n/a | no | unknown | not separately instrumented |

## What this licenses, and what it does not

**Licensed:** stating that the variability is *not* attributable to scenario
selection, policy noise draws, ordering, the success threshold, NFE, or the
checkpoint — all of those are provably fixed and SHA- or seed-verified identical
across repeats.

**Not licensed:** attributing the variance specifically to "Isaac Gym physics."
Two unfixed candidates remain (GPU physics and CUDA kernel nondeterminism) and
this study does not separate them. Doing so would require e.g. forcing
deterministic CUDA algorithms or a CPU-physics comparison — both of which would
*change the evaluator*, which §16 of the directive explicitly forbids for this
study.

Throughout the report the measured quantity is therefore called
**end-to-end evaluator realization variability**: the reproducibility of the
production evaluation pipeline as it actually runs, with everything we control
held fixed.

## Note on a tempting shortcut

Because every Python-level seed is fixed, it is tempting to conclude the
evaluator "should" be deterministic and that any variation indicates a bug. It
does not: GPU physics with contact-rich manipulation is well known to diverge
from bitwise-identical initial conditions, and this benchmark's success criterion
(`max_obj_dist <= 0.04`) turns small trajectory divergence into discrete
success/failure flips. A measured 1.31 pp pooled SD over 96-episode sets is
consistent with a modest number of borderline episodes flipping — though with no
episode-level logging in the existing artifacts, that mechanism is **inferred,
not demonstrated**.
