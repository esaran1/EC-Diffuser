# NFE2 (n=500) vs published NFE100 (n=500) — confirmatory gate

**Cost: 711 s = 0.198 GPU-h** (approved 0.25; projected 0.239). No NFE50, no
NFE100 rerun, no additional budgets.

---

## 0. A CORRECTION TO THE STAGE-1 SCREEN (found by the §3 provenance audit)

My Stage-1 success definition was **wrong**. I used `max_reward > 0.95`. The
environment defines:

```python
reward = np.clip(coverage / self.success_threshold, 0.0, 1.0)
terminated = is_success = coverage > self.success_threshold
```

so success requires the reward to **saturate at 1.0**. Verified against the
published per-episode records:

| definition | agrees with published | pc_success |
|---|--:|--:|
| `max_reward > 0.95` (mine, wrong) | 390/500 | 87.4 |
| **`max_reward >= 1.0`** | **500/500** | **65.4** ✓ exact |

### Stage-1 screen, recomputed (no GPU, from cached per-episode rewards)

| NFE | 1 | 2 | 4 | 5 | 10 | 20 | 50 | 100 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| success (reported, WRONG) | 0 | 88 | 84 | 88 | 86 | 90 | 92 | 86 |
| **success (CORRECT)** | **0** | **48** | **70** | **56** | **74** | **58** | **68** | **72** |
| avg max reward (unaffected) | 0.088 | 0.967 | 0.921 | 0.963 | 0.937 | 0.960 | 0.979 | 0.928 |

**My earlier "behavioural saturation at NFE2" reading was an artifact of a
lenient threshold counting near-misses as successes.** The continuous
`avg_max_reward` was unaffected — and it still ranks NFE2 (0.967) above NFE100
(0.928). The two endpoints genuinely disagree, which is why §9 required both.

## 1. Published NFE100 provenance — MATCHES

| field | published | ours |
|---|---|---|
| policy type | diffusion | same checkpoint |
| noise_scheduler_type | DDPM | DDPM |
| num_train_timesteps | 100 | 100 |
| num_inference_steps | `None` → 100 | 2 (the only variable) |
| beta_schedule | squaredcos_cap_v2 [1e-4, 0.02] | identical |
| prediction_type | epsilon | identical |
| clip_sample | True, range 1.0 | identical |
| horizon / n_obs / n_action | 16 / 2 / 8 | identical |
| env / task | pusht / PushT-v0 | identical |
| episode_length | 300 | 300 |
| obs_type | pixels_agent_pos | identical |
| seeds | 1000-1499 (contiguous) | 1000-1499 |
| success | `max_reward >= 1.0` | now identical |

No load-bearing mismatch. The published run is usable as the control.

## 2. Per-seed data — AVAILABLE

`eval_info.json` carries `seed`, `max_reward`, `success`, `sum_reward` for all
500 episodes → the **scenario-matched paired design (§13)** applies, not an
unpaired aggregate comparison.

## 3. Runtime

711 s wall = **0.198 GPU-h**. `calls_per_plan = 2.00` verified; planner latency
18.9 ms.

## 4-11. Results

| | NFE2 (ours) | NFE100 (published) | Δ (NFE2 − NFE100) | 95% CI |
|---|--:|--:|--:|---|
| **success** | **60.8%** | **65.4%** | **−4.6 pp** | **[−10.6, +1.6]** |
| **avg max reward** | **0.9315** | **0.9551** | **−0.0236** | **[−0.0435, −0.0046]** |

Scenario-paired bootstrap over the 500 shared environment seeds (20,000
resamples). Discordant pairs: NFE2 fail/NFE100 ok = 133; NFE2 ok/NFE100 fail =
110; **McNemar exact p = 0.158** (secondary, descriptive only).

### 8. The ±5 pp margin — NOT satisfied

The CI on Δ_success spans **[−10.6, +1.6] pp**, extending well beyond −5 pp.
Per §11 I do **not** use `p > 0.05` as evidence of non-inferiority, and per §12
I do not promise equivalence. **Formal ±5 pp non-inferiority is unresolved.**

Worse, it is unresolvable in practice: the per-seed paired difference has
sd = 0.696, so the CI half-width is 6.10 pp at n=500 and 4.32 pp at n=1000. With
the observed Δ sitting essentially *on* the margin (−4.6 vs −5), excluding −5 pp
would require **n ≈ 116,000 episodes**. No feasible experiment settles this.

## 12-14. The contrast

| | NFE2 | NFE100 | ratio |
|---|--:|--:|--:|
| offline exec-action L2 | 5.9205 | 4.6500 | NFE100 **21% better** |
| UNet calls / plan | 2 | 100 | **50×** |
| planner latency | 18.9 ms | 702.8 ms | **37×** |
| success | 60.8% | 65.4% | −4.6 pp |
| avg max reward | 0.9315 | 0.9551 | −0.024 |

## 15. Is external behavioural saturation confirmed? **Partially — weaker than the screen suggested**

**Supported:** a **50× increase in denoiser calls and a 21% offline-error
improvement buys ~4.6 pp of success and 0.024 of max reward.** The return on
inference compute is heavily sublinear, and the offline metric's 21% improvement
does not translate proportionally.

**Not supported:** that NFE2 is behaviourally equivalent, or that behaviour
"saturates at 2 steps." NFE100 is better on **both** endpoints, and the reward
difference CI **excludes zero** ([−0.0435, −0.0046]). Under the corrected success
definition the screen's apparent NFE2 advantage disappears.

## 16. Corrected strongest cross-policy claim

> Across an entity-centric joint state/action Flow policy in a GPU-simulated
> pushing task and an independently trained action-only DDPM policy in a
> deterministic 2D pushing task, offline demonstrated-action error does not
> quantify the closed-loop value of inference compute. In the external policy a
> 50× increase in denoiser calls, which improves offline action error by 21%,
> yields 4.6 pp of task success (95% CI [−10.6, +1.6]) at 37× the planner latency.

Explicitly **not** claimed: that all generative policies saturate at two steps;
that offline error *inverts* the behavioural ranking in PushT; that NFE2 is
non-inferior within ±5 pp.

## 17. Decision: **D-CONDITIONAL**

Not **D-GO**: the direction is suggestive and the compute/benefit ratio is
striking, but NFE100 is better on both endpoints, the reward CI excludes zero,
and the predeclared margin is not met.

Not **D-WEAKENED**: NFE100's advantage (4.6 pp, 0.024 reward) is *not*
commensurate with its 21% offline-error advantage and 50× compute — a 37×
latency reduction for ~4.6 pp is a real efficiency finding, and the qualitative
dissociation between offline metric scale and behavioural payoff survives.

Not **D-NO-GO**: the published reference was provenance-clean and used correctly.

## 18. ONE next action

**Do not run another external-policy experiment.** The external axis has given
what it can: a clean, provenance-verified, n=500 measurement showing sublinear
return on inference compute, with a margin that no feasible sample size resolves.

The single next action is **a paper-level roadmap decision (§23)** on how to
spend remaining compute between (a) EC-Diffuser compositional breadth (4/5-cube),
and (b) evaluator-methodology validation — **not** additional policies. That
requires no GPU and should precede any further spending.

**HARD STOP.**
