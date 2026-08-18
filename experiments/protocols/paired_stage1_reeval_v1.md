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

## 4. Evaluation seeds and episode count

- Evaluation seeds: **`101, 202, 303, 404, 505`** (5 seeds, from the
  predeclared `experiments/fast_policy_protocol.json`).
- Episodes per seed: **96** (protocol value; 3× the current 32).
- Total per arm: **480 episodes**, vs ~32 today (15×).
- Total evaluations: 6 arms × 5 seeds = **30 processes**.

96 episodes/seed gives a per-seed binomial standard error of ~2.2% at p≈0.9;
across 5 paired seeds the mean has ~1% standard error — small enough to
resolve the ~6-point 8-NFE effect and the ~16-point 16-NFE drop, if real.

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

Aggregate across the 5 seeds: mean, sample standard deviation, and standard
error, computed over **seed-level means** (n=5), not pooled episodes.

Raw per-episode records are written per run and retained; no aggregate is
reported without its raw file.

## 7. Preregistered analysis

Because arms are paired on identical episodes, use **paired** tests:

- **H1 (8-NFE advantage).** Flow@8 vs Flow@4 and vs Diffusion@100, paired by
  (seed, episode). Report paired mean difference with a 95% CI over the 5
  seed-level differences. *Confirmed if* the CI excludes 0 in favour of @8.
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
  --num_eval_episodes 96 \
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

30 processes. Current 32-episode timings scale linearly in episodes and NFE.
Dominated by arm A (Diffusion, 1989 ms/plan × 100 steps × 96 episodes ≈ 5.1 h
of planner time alone across 5 seeds).

Planner-time cost is `ms/plan × episodes × seeds × 100 steps`. At the naive
96 episodes × 5 seeds for every arm:

| Arm | ms/plan | plans | planner GPU-h |
|---|--:|--:|--:|
| A Diffusion@100 | 1989 | 48,000 | 26.52 |
| B Flow@1 | 20.2 | 48,000 | 0.27 |
| C Flow@2 | 40.2 | 48,000 | 0.54 |
| D Flow@4 | 80.2 | 48,000 | 1.07 |
| E Flow@8 | 160.0 | 48,000 | 2.13 |
| F Flow@16 | 319.2 | 48,000 | 4.26 |
| **Total** | | | **34.79** |

**34.8 GPU-h exceeds the 24 GPU-h cap.** Arm A alone is 76% of the cost while
being the arm *least* under study — diffusion NFE is fixed at 100 and is a
reference point, not a variable.

### Selected configuration: 16.22 GPU-h

Spend the budget on the arms that carry the hypotheses:

| Arms | Episodes | Seeds | GPU-h |
|---|--:|--:|--:|
| Flow @1/2/4/8/16 | 96 | 5 (101,202,303,404,505) | 8.27 |
| Diffusion @100 | 48 | 3 (101,202,303) | 7.96 |
| **Total** | | | **16.22** |

Rationale: H1–H3 are all statements about the **Flow NFE curve**, so the five
Flow arms keep full power (96 ep × 5 paired seeds, per-seed SE ~2.2%).
Diffusion needs only a stable reference level; 48 ep × 3 seeds gives a
per-seed SE of ~4.3% and a 3-seed mean SE of ~2.5%, adequate for a level
comparison.

Consequence for H1: the Flow@8-vs-Flow@4 comparison is fully paired across 5
seeds. The Flow@8-vs-Diffusion comparison is paired only on seeds 101/202/303
and on the first 48 episodes of each, and must be reported as such.

Episode and seed counts are fixed here **before launch** and must not be
adjusted after seeing results.
