# Closed-loop control: Flow NFE 4 vs NFE 32

**No training. No new checkpoints. No new solver. No loss change.**
Canonical `isaacgym_nfe_study` harness driven unmodified; only `PLAN` and the
flow checkpoint path were set per arm. **NFE was the sole experimental variable.**

Scripts: `experiments/scripts/nfe4_vs_nfe32_control.py`, `analyze_control.py`
Data: `experiments/isaacgym_control/nfe4_vs_nfe32/`, `control_analysis.json`

**HEADLINE: Classification C — no reliable control difference. The three-seed
mean favours NFE32 by +3.5 points, but the sign is not consistent (2 of 3
seeds), the SD across seeds (6.1 pts) is larger than the effect, and an
identical-configuration re-run of NFE4 differs from the canonical value by 8.3
points. Offline action-imitation error does NOT predict control.**

---

## 1. Frozen episode-set provenance

| property | value |
|---|---|
| artifact | `experiments/isaacgym_episode_sets/replicate0_n96.pkl` |
| stored sha256 | `35144910b1471b7b0d50d17da18b01db0b5e61e21d7e16e4ef1aa266ee80d511` |
| file sha256 | `c760dd58d4cde41c…` |
| episodes | 96, 3 cubes |
| generation seed | 20260820 |
| frozen | 2026-08-20, **long before this NFE32 question arose** |
| environment horizon | 100 (`max_episode_length`) |

This is the exact set used by the canonical 3-cube Flow NFE evaluations
(`3cube_r0_flow`, hash `35144910…`). **NFE4 and NFE32 saw byte-identical
episodes, and all three training seeds used the same set.**

**Deviation from §6, stated plainly:** the canonical NFE curve used **three**
replicate sets (n=288). Using all three here would cost 1.52 GPU-h, triple the
approved 0.508. I ran **replicate0 only (n=96 per seed)**. This is a real
reduction in power and is the main limitation of this experiment.

## 2. Checkpoint hashes (verified before launch)

| seed | file | sha256 (first 40) | internal step |
|---|---|---|--:|
| 42 | `data/panda_push/flow/…seed42/state_400000.pt` | `861dc34434474455a25dc3a15ea4e1754066202d` | **499000** |
| 43 | `…seed43/state_400000.pt` | `c8e00eadfed9b8a0b54c5423864457e5330434b7` | **499000** |
| 44 | `…seed44/state_400000.pt` | `c2c13f557aca7cf0eeb29b7baa572cf62a002702` | **499000** |

(The `400000` filename is the floored `label_freq` label; the internal step is
499000, confirmed by loading each state dict. Hashes match those recorded in the
offline NFE sweep.) No seed was selected or excluded on behaviour.

## 3. Stochastic / noise protocol — **episodes paired, policy noise NOT paired**

`conditional_sample` draws its own noise internally with no injection point.
Per §7 I did **not** rewrite the canonical evaluator to force x0 pairing, since
that would change policy behaviour relative to all prior canonical evaluations.

So the pairing structure is: **initial states and goals are byte-identical
across arms and seeds; policy sampling noise is independent per rollout.**

**This matters more than expected.** Re-running the *identical* configuration —
same episode set, same seed-42 checkpoint, same NFE 4 — gives:

| run | success | per-episode agreement |
|---|--:|---|
| canonical `r0_flow_nfe4` | **0.8854** | — |
| this run `r0_s42_nfe4` | **0.8021** | **identical on only 72/96** |

**An 8.3-point spread from policy sampling noise alone, disagreeing on 24 of 96
episodes.** That is larger than the +3.5-point effect this experiment set out to
measure, and it is the single most important caveat in this report.

## 4. Compute: estimate vs actual

| | value |
|---|--:|
| approved estimate | 0.508 GPU-h |
| timing sanity check (1 arm, seed 42 NFE32) | 471 s |
| main run (remaining 5 arms) | 1348 s |
| **total actual** | **1819 s = 0.506 GPU-h** |

Essentially exact. NFE accounting was verified by the harness's hard check on
every arm: **4.0 and 32.0 calls/plan, no mismatches.** Latency 77.0 ms (NFE4) vs
625–626 ms (NFE32) per plan — an **8.1× inference cost increase**.

## 5-6. Per-seed full-episode success (n=96, paired episodes)

| seed | NFE4 | NFE32 | Δ (32−4) | 95% CI |
|---|--:|--:|--:|---|
| 42 | 77/96 = **0.8021** | 87/96 = **0.9062** | **+0.1042** | [+0.0208, +0.1875] |
| 43 | 82/96 = **0.8542** | 83/96 = **0.8646** | **+0.0104** | [−0.0729, +0.0938] |
| 44 | 87/96 = **0.9062** | 86/96 = **0.8958** | **−0.0104** | [−0.0833, +0.0625] |

## 7-10. Secondary canonical diagnostics (3-seed means)

| metric | NFE4 | NFE32 | Δ |
|---|--:|--:|--:|
| full success | 0.8542 | 0.8889 | +0.0347 |
| goal_success_frac | 0.9340 | 0.9479 | +0.0139 |
| **cubes_placed** | 2.8021 | 2.8438 | +0.0417 |
| cubes_moved | 2.9583 | 2.9688 | +0.0104 |
| **avg_obj_dist** | 0.0287 | 0.0251 | **−0.0036** |
| **max_obj_dist** | 0.0593 | 0.0500 | **−0.0094** |
| mean_progress | 0.1693 | 0.1731 | +0.0038 |
| cubes_closer | 2.8472 | 2.8819 | +0.0347 |
| **cubes_farther (wrong-direction)** | 0.1111 | 0.0868 | **−0.0243** |
| n_contacted | 2.9444 | 2.9722 | +0.0278 |
| first_contact_step | 0.6806 | 0.7118 | +0.0313 |
| min_ee_to_cube | 0.0261 | 0.0259 | −0.0002 |
| action_abs_mean | 0.1216 | 0.1202 | −0.0015 |
| clip_fraction | 0.0036 | 0.0051 | +0.0015 |
| eef_path_length | 2.4774 | 2.4766 | −0.0008 |

Every continuous diagnostic points the same (weak) way: NFE32 leaves cubes
slightly closer to goal, pushes fewer cubes the wrong way, and contacts slightly
more often. **Effector kinematics are essentially unchanged** (path length,
z-height, action magnitude all within 1%), so NFE32 is not producing
qualitatively different motion — it is marginally more accurate placement.

## 11. Paired episode transitions

| seed | A both succeed | B 4✗→32✓ | C 4✓→32✗ | D both fail | McNemar exact p |
|---|--:|--:|--:|--:|--:|
| 42 | 70 | **17** | 7 | 2 | 0.0639 |
| 43 | 72 | 11 | 10 | 3 | 1.0000 |
| 44 | 78 | 8 | **9** | 1 | 1.0000 |

**Discordant pairs are numerous in both directions on every seed** (B+C = 24, 21,
17). Even on seed 42, where NFE32 wins by 10 net episodes, 7 episodes flip the
other way. This is the signature of a noisy comparison rather than a systematic
policy improvement — consistent with §3's finding that unpaired policy noise
alone flips ~24 of 96 episodes.

## 12. Paired statistical intervals

Per-seed 95% bootstrap CIs on Δ are in §5. Only seed 42's excludes zero
(+0.0208 to +0.1875), and its McNemar p = 0.064 does not reach 0.05. Seeds 43
and 44 are squarely null (p = 1.000 both).

**No pooling across seeds is reported as algorithm-level evidence.** 288
episodes are not 288 independent algorithm replications.

## 13. Extended seed-42 control curve

| NFE | success | source |
|--:|--:|---|
| 1 | 0.8056 | canonical, n=288 (3 replicates) |
| 2 | 0.8681 | canonical, n=288 |
| 4 | 0.8889 | canonical, n=288 |
| 8 | 0.8993 | canonical, n=288 |
| 16 | 0.8854 | canonical, n=288 |
| **32** | **0.9062** | this experiment, **n=96 (replicate0 only)** |
| *(4)* | *(0.8021)* | *this experiment, n=96 — vs 0.8889 canonical* |

**The NFE32 point is not directly comparable to the canonical curve**: different
n, single replicate, and — per §3 — the same configuration re-run moves by 8.3
points. Read literally the curve stays in a **0.81–0.91 band from NFE 2 to 32**,
which is inside the previously established 3–11 point evaluation noise floor.
**Control is saturated from NFE 2 onward; it does not decline at NFE 32.**

## 14. Offline vs online contrast (NFE4 → NFE32)

| quantity | NFE4 | NFE32 | change |
|---|--:|--:|--:|
| offline **action** imitation error | 0.02212 | 0.04136 | **+87.0% (worse)** |
| offline **state** prediction error | 0.04881 | 0.02493 | **−48.9% (better)** |
| closed-loop success (3-seed mean) | 0.8542 | 0.8889 | **+0.0347 (n.s.)** |

## 15. Does closed-loop track action imitation, state prediction, or neither?

**Not action imitation.** Action error nearly doubles from NFE4 to NFE32 while
control does not degrade at all — if anything it nominally improves. Combined
with the earlier NFE1→8 anti-alignment, **demonstrated-action imitation error is
refuted as a control proxy in this regime**, in both directions.

**Weakly consistent with state prediction, but not established.** State error
halves and control nominally improves, and the secondary diagnostics
(avg/max object distance, wrong-direction pushes) all shift the same way. But
the success effect is not significant on 2 of 3 seeds and is smaller than the
re-run noise, so this is at most **suggestive**.

**The honest summary is "neither is demonstrated."** What *is* demonstrated is a
negative: a large, replicated, 87% offline action-imitation degradation carries
**no detectable closed-loop cost**.

## 16. Three-seed replication classification: **C — effectively similar control**

- per-seed Δ: **+0.1042, +0.0104, −0.0104**
- signs: **+, +, −** — not consistent
- mean seed-level Δ = **+0.0347**, **SD = 0.0610** (N=3)
- No t-test reported: N=3 has no meaningful power, per §12.

The mean is positive but smaller than its own across-seed SD, and smaller than
the 8.3-point same-configuration re-run spread measured in §3.

## 17. Strongest mechanistic conclusion supported

> Moving from NFE 4 to NFE 32 changes the Flow sampler's offline predictions
> substantially — action imitation error +87%, state prediction error −49% — yet
> produces **no reliable change in closed-loop task success** across three
> independently trained checkpoints, at 8.1× the inference cost.

This is §15's third case: **a large behavioural-equivalence region in the Flow
sampler.** Integration resolution above the practical regime moves the generated
distribution a lot and the task outcome very little.

It also settles the practical question: **NFE 4 remains the right operating
point.** There is no measured control benefit to 8× more compute.

**Not claimed:** that NFE32 is genuinely better (2 of 3 seeds say no); that
coarse integration is beneficial for control (the earlier hypothesis — this
experiment does not support it either); or any causal link from either offline
metric to control.

## 18. What remains unresolved

- **Whether a real small effect exists.** n=96 × 3 seeds cannot resolve effects
  below ~8-10 points given the measured re-run spread. A +3.5-point true effect
  would need far more episodes and seed-paired policy noise.
- **Policy-noise pairing.** The canonical evaluator cannot pair x0 across arms;
  §3 shows this injects more variance than the effect. Any future control
  comparison at this effect size needs a seeded-noise evaluator — which would
  itself need validation against the canonical protocol.
- Whether the state-prediction/control association is real or coincidental.
- Everything offline remains an **in-distribution replay diagnostic**; no
  held-out generalization was measured anywhere in this line of work.
- 4-cube / 5-cube behaviour: deliberately not run (§5).

## 19. Is loss modification still worth pursuing? **NO — NOT MOTIVATED**

This is a downgrade from LOWER PRIORITY. The chain is now complete: the original
concern was that the 3 action dimensions were under-served by the loss. We have
since shown (a) the action block is *most* accurate at minimal integration and
degrades with convergence — the opposite of under-serving; (b) the state block
behaves as a well-posed convergence problem; and (c) **an 87% swing in action
imitation error produces no measurable control consequence at all.** There is no
longer a symptom for a loss change to target.

## 20. ONE recommended next research direction

**Characterise the behavioural-equivalence region directly, as the finding
itself — not as a bug to fix.**

The scientifically interesting result of this whole investigation is that the
deployed policy is `(learned field + discrete integrator + NFE)`, and that this
composite is remarkably insensitive to the integrator across an 8× compute range
even while its offline predictions change by 50-90%. The natural next step is a
**tolerance/robustness characterisation**: how far can the sampler be perturbed
(NFE, solver order, step schedule) before closed-loop behaviour actually moves?

That reframes the low-NFE result — already the project's strongest practical
claim, 2 NFE matching Gaussian at 100 for a 53× latency reduction — from "Flow
is cheap" to "there is a wide sampler-equivalence region, and the cheapest point
in it is the right one to deploy." That is a publishable, mechanism-level claim
that the current evidence base directly supports.

**No experiment is launched.** Any such study needs a seeded-noise evaluator
(§18) and explicit approval first.

---

## HARD STOP OBSERVED

No additional simulator experiments. No training. No new loss. No new solver.
No new policy method.
