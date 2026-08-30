# Common-Random-Numbers evaluator: built, validated, and BLOCKED at the §9 gate

**No training. No loss change. No new method.** Compute: ~0.05 GPU-h (validation
and diagnostics only). **The main 96×3 NFE4/NFE32 CRN run was NOT launched.**

Scripts: `crn.py`, `validate_crn.py`, `crn_control.py`
Data: `crn_validation.json`

**HEADLINE: CRN over policy noise is implemented and passes every sampler-level
gate bit-exactly — but the §9 reproducibility gate FAILS, because the dominant
stochastic source is not the policy. Isaac Gym itself is nondeterministic: two
rollouts with an identical reset and an identical fixed action sequence diverge
by 2.9e-2 within 30 steps and 1.03 by step 100, against a 0.04 success
threshold. CRN cannot fix this, so the paired experiment is not runnable as
specified.**

---

## 1. Stochasticity ledger

Every stochastic source affecting closed-loop evaluation, traced in code:

| # | source | location | status |
|---|---|---|---|
| 1 | **Flow initial noise** `torch.randn` | `flow_matching.py:358`, once per plan | **controllable** — the only draw in the Flow sampler; no per-step noise |
| 2 | Gaussian per-step noise `randn_like` | `diffusion.py:25,37,198` | not used in this experiment (Flow only) |
| 3 | Franka DOF reset noise `torch.rand(len,9)` | `isaac_panda_push_env.py:698` | **inert but consumes RNG** — `frankaDofNoise: 0.0` in config, so multiplied by zero; still advances the global stream |
| 4 | start position/rotation noise | `isaac_panda_push_env.py:448-454` | inactive — all four config values are `0.0` |
| 5 | DLP encoding | `dlp_utils.py:286,342` | **deterministic** — called with `deterministic=True` |
| 6 | policy sampling path | `policies.py` | no additional RNG use |
| 7 | **Isaac Gym GPU physics** | `env.step` | **NOT controllable — see §2** |

Config verification: `startPositionNoise`, `startRotationNoise`,
`frankaPositionNoise`, `frankaRotationNoise`, `frankaDofNoise` are **all 0.0**
in `env_config/generalization_num_cubes/IsaacPandaPushConfig.yaml`.

## 2. Exact source of prior run-to-run variation — **it is the simulator**

I did not assume the Flow noise was responsible. Direct test: fix the reset
states, fix a pre-generated action sequence, seed all RNGs identically, and run
the environment twice. **No policy is involved.**

| step | max abs difference between two identical-action rollouts |
|--:|--:|
| 0 | 4.005e-05 |
| 1 | 4.196e-05 |
| 5 | 1.942e-02 |
| 10 | 3.413e-02 |
| 50 | 1.219e-01 |
| **100** | **1.025e+00** |

**Divergence is present at step 0 — before any action is applied — and grows to
25× the 0.04 success threshold by the episode horizon.** This is characteristic
of GPU physics (non-associative parallel reductions), not of any RNG.

This reframes the previous report's finding. The 8.3-point run-to-run gap
(canonical `r0_flow_nfe4` = 0.8854 vs my rerun = 0.8021, disagreeing on 24/96
episodes) was attributed to unpaired policy sampling noise. **That attribution
was incomplete: simulator nondeterminism is present and sufficient on its own.**
The two sources cannot be separated by the experiment already run, and the
policy contribution has not been isolated.

## 3. Injectable-noise implementation

Canonical training/sampling code is **untouched**. `conditional_sample` draws
its noise from the global torch RNG with no injection point, so rather than edit
shared code (§4's risk clause) the wrapper **seeds the global RNG immediately
before each policy invocation**:

```python
s = derive_seed(base, batch_start, decision)   # sha256("base|batch|decision")
torch.manual_seed(s); torch.cuda.manual_seed_all(s)
return self.policy(*args, **kwargs)            # sampler draws the identical z
```

Keyed by **(episode-batch, decision index)**, never by physical state, so both
arms receive the same exogenous random sequence even after trajectories diverge
(§7). Disabled by default (`enabled=False` reproduces canonical behaviour
exactly). Model weights, Euler equations, conditioning, normalization, horizon,
timestep schedule and action extraction are all unchanged.

## 4. Bit-identity validation — **PASSED**

| gate | result |
|---|--:|
| **A** same seed, sampler run twice | **max abs diff = 0.000e+00** |
| **B** canonical sampler vs manual Euler replay with the same seeded z | **max abs diff = 0.000e+00** |
| **C** different seed (sanity: must differ) | max abs diff = 1.683 |

Gate B is the key one: it proves the seeded RNG reproduces exactly the tensor
the canonical sampler would have drawn, so CRN pairing is exact rather than
approximate.

## 5. Marginal-distribution validation — **PASSED**

40 canonical draws (natural RNG stream) vs 40 CRN-seeded draws on a fixed batch:

| | mean | sd |
|---|--:|--:|
| canonical | −0.21625 | 0.50800 |
| CRN | −0.21955 | 0.50640 |

Two-sample KS per action dimension: **KS = 0.031 / 0.020 / 0.025, p = 0.914 /
0.999 / 0.988.** The evaluator has **not created a new policy**; only the
coupling between arms changed.

## 6. Small-set repeatability — **FAILED (the §9 gate)**

One checkpoint (seed 42), NFE4, 16 episodes, identical CRN bank, run twice:

| run | success |
|---|--:|
| rep1 | 0.7500 |
| rep2 | 0.8750 |

**Outcomes did not reproduce**, so per §9 I stopped rather than spend on the
main experiment. §2 identifies why: the simulator is chaotic from reset onward.

**A second, self-inflicted bug this gate caught:** I invoked the harness with
`--episodes 16`, and `get_episode_set` silently **recorded a brand-new 16-episode
set** (`f3e22980`) instead of using the frozen `replicate0_n96`. That artifact was
deleted; the three canonical replicate sets are untouched. Had the gate not been
run, the main experiment would have used a non-frozen episode set.

## 7. Independent-noise vs CRN disagreement

Not measurable as intended. Under CRN the residual disagreement is dominated by
simulator divergence, so the comparison cannot isolate the policy-noise
contribution. Reporting this rather than presenting a number that would be
attributed to the wrong cause.

For reference, the independent-noise repeat measured previously: **0.8854 vs
0.8021, 24/96 episodes disagreeing.** Whether CRN would reduce that is
**unresolved** — the simulator floor is not yet quantified at the episode-outcome
level.

## 8. Frozen CRN bank definition

Defined and validated, though not yet used for a scientific run:

| property | value |
|---|---|
| base seed | **20260905** |
| derivation | `sha256("base|batch_start|decision")[:8]`, little-endian, mod 2⁶³−1 |
| keying | (episode-batch start, decision index) — not physical state |
| tensor | drawn by the canonical sampler: `(batch, horizon=5, transition_dim=483)`, float32, cuda:0 |
| shared across training seeds | yes — predeclared, so 42/43/44 face identical disturbances |
| validation record | `crn_validation.json` |

## 9-14. Main NFE4/NFE32 CRN run — **NOT RUN**

Blocked at the §9 gate. Per-seed success, discordant pairs, secondary
diagnostics, paired CIs, the ±5pp equivalence assessment and the A/B/C/D
classification are all **not available**, and I decline to report the
independent-noise numbers in their place (§17 forbids mixing the datasets).

## 15. Corrected interpretation of the old NFE control curve

The canonical NFE1/2/4/8/16 curve drew policy noise **independently per arm**
(each arm is a separate `build_policy` + `evaluate` call with no seeding), and
ran under a nondeterministic simulator. Therefore, per §1:

**Withdrawn as mechanistic conclusions:** "control saturates at NFE2",
"control peaks at NFE8", "NFE16 declines".

**Replacement wording:**

> In the existing independently sampled evaluations, control was broadly strong
> from NFE2–16, but the fine NFE ordering is unresolved because policy sampling
> noise and simulator nondeterminism are both substantial and uncontrolled.

The individual success rates remain valid **marginal** evaluations. Raw results
are unaltered.

The one contrast that survives is **NFE1 vs the rest** (0.8056 vs 0.868–0.899),
which was the study's only significant comparison (McNemar p = 0.0356) and whose
magnitude exceeds the ~8-point reproducibility spread.

## 16. Is a behavioural-equivalence region supported? **NO — not yet**

Per instruction, this claim is **not** frozen. The honest statement remains:

> Under the current independently sampled stochastic evaluator, NFE4→NFE32
> produces no reproducible three-checkpoint control difference distinguishable
> from evaluation variability at n=96/seed.

The §19 equivalence test cannot be applied: with a ±5pp predeclared margin and a
measured same-configuration spread of ~8.3pp, the evaluator **cannot resolve the
margin**, regardless of how many episodes are run under this protocol.

## 17. Loss-balancing direction: **NOT MOTIVATED** (frozen)

Unchanged and closed, per §20. Nothing in this work reopens it.

## 18. ONE next research direction

**Quantify and then reduce the simulator's contribution to evaluation variance,
before any further control comparison.**

Concretely and cheaply: run the *same* arm N times under CRN on the frozen
96-episode set and measure the episode-outcome disagreement attributable to
physics alone. That yields the evaluator's true noise floor, which determines
whether *any* control difference of the size in question is measurable at all.
Then test the two available levers — Isaac Gym's CPU-pipeline / deterministic
mode, and per-episode-independent batching — for whether either restores
reproducibility.

Rationale: every closed-loop claim in this project rests on an evaluator whose
noise floor is now known to be substantial but is **not yet quantified**. The
NFE4-vs-NFE32 question is not answerable until it is, and neither is any future
sampler comparison. This is the blocking dependency, and it is cheap to measure.

**Not launched.** It needs approval, and its cost depends on how many repeats
are wanted.

---

## HARD STOP OBSERVED

No training. No loss changes. No MeanFlow. No VP. No new method.
No main CRN control run (blocked at the mandatory §9 gate).
