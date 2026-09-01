# E1 analysis plan — FROZEN BEFORE ANY RESULT EXISTS

Committed prior to E1 GPU execution. No metric, threshold or rule below may be
changed after results.

## 1. Design (reconciled; both source docs agree)

Flow **seed 42**, **NFE 4**, frozen `replicate0_n96` (sha `35144910…`), H=100,
**num_envs = 16 (unchanged)**, 3 mappings: **identity + perm1 + perm2**
(perm seed 20260915). Only intended change: which GPU slot executes scenario *i*.

## 2. Endpoints

**PRIMARY:** full-episode success, per scenario.
**SECONDARY (established only):** per-object success (`goal_success_frac`),
`cubes_placed`, `avg_obj_dist`, `max_obj_dist`, `n_contacted`.
**No metric may be added after observing results.**

## 3. Predeclared classification

| level | rule | interpretation |
|---|---|---|
| **A NUMERICAL ONLY** | ≤1 discordant scenario per permutation **and** \|Δsuccess\| < **2.0 pp** | **E1 NULL** for the paper hypothesis |
| **B BEHAVIOURALLY DETECTABLE** | ≥2 discordant scenarios **or** \|Δsuccess\| ≥ **2.0 pp** | slot assignment affects policy-evaluation outcomes |
| **C COMPARISON-RELEVANT** | success **range across the three mappings ≥ 5.0 pp** | E2 becomes plausible |

**Threshold justification (frozen now).** 5.0 pp is the practical policy-effect
scale used throughout this project (the predeclared equivalence margin). 2.0 pp
is below the smallest calibrated NFE effect we care about (NFE4−NFE2 ≈ 0-1.7 pp
on 3-cube; NFE1 contrasts 2.8-9.0 pp), so anything at or above it is
behaviourally non-trivial. 1 discordant scenario ≈ 1.04 pp, i.e. single-episode
noise; ≥2 is the smallest count that cannot be one flipped episode.

## 4. Statistical treatment

96 scenarios × 3 mappings = 288 evaluations of **the same policy**. These are
**NOT 288 independent observations** and no algorithm-level p-value will be
manufactured. Analysis is **scenario-paired**: for each permutation vs identity
report Δsuccess, discordant counts with direction, the changed-scenario list, and
continuous-metric deltas; plus the success range across all three mappings and
the count of scenarios changing under any mapping.

## 5. Claim gate (frozen)

**E1 POSITIVE supports:** *Under the tested Isaac Gym configuration, reassigning
fixed task scenarios across GPU-parallel environment slots can change downstream
outcomes of an otherwise identical learned-policy evaluation.*

**E1 POSITIVE does NOT support:** any statement about policy **rankings** or
their reversal — that requires E2. Nor "GPU slots universally change rankings."
Nor mechanism attribution beyond slot-correlated execution.

**E1 NULL supports:** *Under fixed parallelism, static scenario-to-slot
reassignment was not a material source of the previously observed evaluation
variability.* A null does **not** invalidate the run-to-run physics
nondeterminism result, which was measured independently.

## 6. E2 gate (frozen)

E2 is proposed **only if E1 reaches classification C** (range ≥ 5.0 pp). Rationale:
our real policy contrasts are ~3-6 pp, so a slot effect must be of comparable
magnitude to plausibly perturb a policy comparison. E2 will **not** be proposed
for a single flipped episode, floating-point state differences, or a
classification-B result alone.

## 7. §10 red-team — traced in code, not memory

| # | question | finding |
|--:|---|---|
| 1 | Could permutation alter anything besides slot? | Batch **composition** is unchanged (permute within each 16-slot batch); only occupancy changes. |
| 2 | Is scenario policy noise invariant? | **Yes, by construction and verified**: noise drawn in scenario order under the same CRN seed, scattered to slots. Identity → **0.000e+00** vs canonical; multiset preserved. |
| 3 | Simulator RNG indexed by slot? | `reset_idx` draws `reset_noise = torch.rand((len(env_ids), 9))` but `frankaDofNoise = 0.0`, so it is **multiplied by zero**. Colour `torch.randperm` at line 662 is gated by `goal_reset`, which is **`False`** — never executes. |
| 4 | Init depends on creation/index order? | `_set_init_states` does `self._init_obj_state_dict[key][env_ids, :2] = init_states` — a pure scatter, order-correct under permutation. |
| 5 | Could collation remap results? | **No** — every record is keyed via `slot_to_scn`; asserted `scn not in recs` and `sorted(ids) == range(96)`. |
| 6 | Does policy batching change model output? | The model is a per-token transformer over a batch; rows are independent. Slot order changes batch **row order**, which can alter floating-point reduction order — this is *part of* the effect under test, not a confound. |
| 7 | Observations reordered correctly? | Obs return in **slot** order and are consumed in slot order; noise is scattered to slot order. Consistent end to end. |
| 8 | DLP batching semantics? | DLP encodes per-env rows independently; `deterministic=True`. |
| 9 | Results keyed by scenario ID? | **Yes** (see 5). |
| 10 | Identity exactly canonical? | Noise: **0.000e+00**. Mapping: `base[arange] == base`. |

**No uncontrolled variable found.** One dead-code path (partial-batch padding)
was replaced with an assertion since 96 = 6×16 exactly.

## 8. Provenance assertions (abort the run on failure)

checkpoint sha256 = `861dc344…` · scenario-set sha256 = `35144910…` ·
`num_objects == 3` · `horizon == 100` · `num_envs == 16` · `episodes == 96` ·
**calls/plan == NFE == 4** (forward-hook counted) · no duplicate/missing scenario ·
permutation equals the frozen map (hash recorded per run).

## 9. GPU-cleanliness gate

Immediately before each run, record `nvidia-smi` compute processes, memory,
utilization and timestamp. **No other GPU-heavy job may run concurrently.** If a
process appears mid-run, flag that run invalid and stop — do not reinterpret.
Unrelated jobs are never killed.
