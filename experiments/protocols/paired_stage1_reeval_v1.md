# Protocol: paired Stage-1 re-evaluation (PREDECLARED, NOT RUN)

Status: **predeclared, awaiting GPU availability and review.** No training.
No retraining. Existing checkpoints only.

## 1. Why this is needed

The current Stage-1 table came from **one unseeded evaluation of ~32 episodes
per arm**. `plan_pandapush_flow_final_single_gpu.py` sets `seed: None`, and
`ArgsParser.set_seed` then draws `random.randint(0, 1000000)`. So **each arm
was evaluated on a different, unrecorded set of episodes.** The arms are not
comparable to each other, and the non-monotonicity (16 NFE scoring below
1 NFE) may be entirely between-arm episode variation.

**What this protocol does and does not establish.** It removes between-arm
episode variance so the NFE curve is measured on identical task instances. It
estimates **evaluation/environment variance only**. It does **not** license
algorithm-level claims: those require independent *training* seeds, which this
protocol deliberately does not provide. Any conclusion drawn here is
conditional on this one trained checkpoint.

## 2. Arms (6)

Single fixed checkpoint per method; no retraining.

| Arm | Method | NFE | Checkpoint |
|---|---|--:|---|
| A | GaussianDiffusion | 100 | `ecdiffuser-data/pretrained_models/panda_push/diffusion/3C_adalnpintlarge_dlp_randcolor_H5_T100/state_1200000.pt` |
| B | ConditionalFlowMatching | 1 | `data/panda_push/flow/3C_dlp_adalnpint_randcolor_H5_T4_seed42/state_400000.pt` (EMA, embedded step 499000) |
| C | " | 2 | same |
| D | " | 4 | same |
| E | " | 8 | same |
| F | " | 16 | same |

Checkpoint SHA256 to be recorded in the results file at run time.

## 3. Pairing mechanism

`SB3VecEnvAdapter.seed()` is a **no-op**, so the simulator cannot be re-seeded
after construction. Pairing is therefore achieved by:

1. **One fresh process per (arm, evaluation seed).** Never reuse a process.
2. `--seed S` passed explicitly, so `set_seed(S)` runs *before* Isaac Gym
   construction and before any cube/goal sampling. Cube placement, goal
   sampling, and random-color selection all draw from the global torch/NumPy
   RNG seeded at that point.
3. Identical `S` across all six arms ⇒ identical initial states and goals.

**Pairing must be verified, not assumed.** Each run writes a SHA256 over the
concatenated per-episode initial object positions and goal positions. Arms
sharing a seed must produce identical hashes. If hashes differ, the pairing
has failed and the results are void — this check gates the whole analysis.

## 4. Evaluation seeds and episode count — one common core set

**Corrected from the previous draft.** The earlier version gave Flow 96
episodes × 5 seeds and Diffusion 48 × 3, which meant the headline
Flow-vs-Diffusion comparison was computed on *different episodes* — the exact
defect this protocol exists to remove.

### Core set (defines every primary comparison)

| Parameter | Value |
|---|---|
| Core evaluation seeds | **101, 202, 303** (3 seeds) |
| Episodes per seed | **48** |
| Core episodes per arm | **144** |
| Arms | **all six**, identically |

Every arm — Diffusion@100 and Flow@1/2/4/8/16 — is evaluated on **exactly
this same set of (seed, episode-index) pairs**. All primary results, all
hypothesis tests, and all reported aggregates come from this core set and
nothing else.

The pairing hash of §3 must match across all six arms for every core seed.
Any mismatch voids the analysis.

### Secondary Flow-only extension (excluded from the aggregate)

Flow arms are cheap (20–319 ms/plan vs 1989 ms). After the core set completes,
Flow@1/2/4/8/16 may additionally run seeds **404, 505** at 48 episodes.

Rules, binding:

- Secondary results are reported in a **separate table**, never merged.
- They **must not** enter the paired aggregate, the hypothesis tests, or any
  mean/std that includes Diffusion.
- Their only role is a consistency check: if the Flow NFE ordering on seeds
  404/505 contradicts the core set, that indicates 3 seeds are insufficient
  and is reported as such.

This keeps the primary comparison exactly paired while still using spare
cheap capacity.

### Power

48 episodes/seed gives a per-seed binomial SE of ~4.3% at p≈0.9. Across 3
paired seeds the mean has ~2.5% SE. Crucially, the tests in §7 are **paired**:
the relevant quantity is the SE of the per-episode *difference*, which is much
smaller than the SE of either arm because episode difficulty is shared.
For a ~16-point effect (the observed 8→16 NFE drop), 144 paired episodes is
ample; for the ~6-point 8-NFE advantage it is adequate but not generous, which
is stated in advance as a limitation rather than discovered afterwards.

## 5. Fixed across all arms

Task (`3C_randcolor`, 3 entities); success definition; `exe_steps=1`;
`horizon=5`; episode length 100; observation/goal representation; normalizer;
`measure_planning_latency=True`; `planning_warmup_calls=10`;
`count_denoiser_calls=True`; timing boundary (CUDA-synchronized
`perf_counter` around trajectory generation only, per
`diffuser/diffuser/sampling/policies.py`); GPU; driver; conda env.

Only **method** and **`--n_diffusion_steps`** vary.

## 6. Metrics (per seed, raw, then aggregated)

Per episode: success; `goal_success_fraction`; `average_object_goal_distance`;
`maximum_object_goal_distance`; `average_reward`.

Per (arm, seed): the above aggregated; `denoiser_calls`;
`mean_ms` / `p50_ms` / `p95_ms`; peak allocated inference VRAM.

Aggregate across the **3 core seeds**: mean, sample standard deviation, and
standard error, computed over **seed-level means** (n=3), not pooled episodes.
Paired per-episode differences are additionally reported (see §7). Secondary
Flow-only seeds are aggregated separately and never pooled with the core set.

Raw per-episode records are written per run and retained; no aggregate is
reported without its raw file.

## 7. Preregistered analysis

Because arms are paired on identical episodes, use **paired** tests:

- **H1 (8-NFE advantage).** Flow@8 vs Flow@4 and vs Diffusion@100, paired by
  (seed, episode) over the core set. Report the paired mean difference with a
  95% CI computed over the 144 paired per-episode differences (bootstrap,
  clustered by seed). *Confirmed if* the CI excludes 0 in favour of @8.
- **H2 (16-NFE degradation).** Flow@16 vs Flow@8, paired. *Confirmed if* the
  paired CI excludes 0 with @16 worse. *Refuted if* the CI covers 0 — which
  would mean the observed drop was between-arm episode variance.
- **H3 (NFE monotonicity).** Report the full paired curve. A genuine
  non-monotonicity is itself a scientific finding about the integration path;
  an apparent one that vanishes under pairing is a sampling artifact.

Both outcomes of H2 are informative and are recorded either way.

**Interpretation ceiling, stated in advance:** all conclusions are conditional
on one training seed. No claim of the form "Flow beats diffusion" will be made
from this protocol — only "at this checkpoint, on identical episodes, NFE
affects success as follows."

## 8. Exact command shape

```bash
# one process per (arm, seed); never reuse a process
env CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline \
  python diffuser/scripts/eval_agent.py \
  --config config.plan_pandapush_flow_final_single_gpu \
  --num_entity 3 --planning_only \
  --seed ${SEED} \
  --num_eval_episodes 48 \
  --n_diffusion_steps ${NFE} \
  --exp_note paired_nfe${NFE}_seed${SEED}
```

Arm A uses `config.plan_pandapush_pint_timed_single_gpu` with
`--n_diffusion_steps 100`.

`plan_pandapush_flow_final_single_gpu.py` currently hardcodes `seed: None`;
`--seed` must be honored on the command line. Verify this before launching —
if the config overrides the flag, the config needs a one-line change to
`seed: None → from args`, which is a source change requiring its own commit.

## 9. Cost

Planner-time cost is `ms/plan x episodes x seeds x 100 steps`.

### Core set — all six arms, 48 episodes x 3 seeds (144 episodes each)

| Arm | ms/plan | plans | planner GPU-h |
|---|--:|--:|--:|
| A Diffusion@100 | 1989 | 14,400 | 7.96 |
| B Flow@1 | 20.2 | 14,400 | 0.08 |
| C Flow@2 | 40.2 | 14,400 | 0.16 |
| D Flow@4 | 80.2 | 14,400 | 0.32 |
| E Flow@8 | 160.0 | 14,400 | 0.64 |
| F Flow@16 | 319.2 | 14,400 | 1.28 |
| **Core total** | | | **10.44** |

### Optional secondary Flow-only extension (seeds 404, 505)

| Arms | plans | planner GPU-h |
|---|--:|--:|
| Flow @1/2/4/8/16, 48 ep x 2 seeds | 9,600 each | 1.65 |

**Core: 10.44 GPU-h. Core + secondary: 12.09 GPU-h.**

Arm A is 76% of the core cost. It cannot be reduced further without breaking
the exact pairing that is the whole point of this protocol, so the episode
count is held at 48 rather than trading pairing for episodes.

### Deferral

**This protocol is DEFERRED and must not be launched yet.** The 3-cube task
currently looks like a weak discriminator: Flow@1 already reaches 82% goal
achievement and Flow@8 reaches 97%, so there is little dynamic range in which
an NFE effect could be scientifically decisive. Spending 10.4 GPU-h to
replicate a result on a near-saturated task is not justified until we know it
supports a central paper claim.

Revisit if and only if 3-cube becomes load-bearing — for example if it is
needed as the entity-structured control arm against cube-triple.

Episode and seed counts are fixed here **before launch** and must not be
adjusted after seeing results.
