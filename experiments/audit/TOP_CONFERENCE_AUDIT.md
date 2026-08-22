# Pre-submission scientific audit

Date: 2026-08-22. HEAD `6d75e92bcce4bdb6f7cd6ef0f4a650da532c5055`, tree clean.
Environment, artifact hashes: `experiments/audit/repository_snapshot.md`.

**Status of the 5-cube probe.** It was still running during the first pass of
this audit and was not touched. It has since **exited cleanly**; all three arms
completed (96 raw episode records each, one shared episode-set hash
`f8dff00dfd7b1752`, measured calls 1.00 / 4.00 / 100.00, no call-count
mismatches). Its results were then recomputed from raw records and are included
below. Claim C22 is now **VERIFIED**, and it **changes a headline conclusion**
(§3.2).

---

## 1. Audit verdict

> ### STRONG CHECKPOINT-LEVEL EVIDENCE

Not "algorithm-level", and not "submission-grade".

**Why this and not lower.** Every headline number recomputes exactly from raw
per-episode records (21 arms, agreement to 1e-9). Every episode-set hash
recomputes from the serialized state arrays and matches. Every arm's measured
denoiser calls equal its requested NFE. Latency is planner-only, CUDA-synced,
uncontended, and proportional to NFE. Checkpoint selection is the *final*
checkpoint everywhere, chosen before the sweep that would have favoured a
different one. Pairing is real, not filename-deep.

**Why not higher.** There is exactly **one independently trained model per
method**. Evaluation replicates are not training seeds. No comparative claim
between Flow and Gaussian can be attributed to the algorithms rather than to
these two particular checkpoints.

---

## 2. Verified headline results

Recomputed from raw episode records by `experiments/audit/recompute_results.py`;
canonical values in `experiments/audit/canonical_results.csv`.

### 3 cubes (3 paired evaluation replicates x 96 = 288 episodes, H=100)

| Arm | Calls/decision | Full success | 95% CI | Per-object |
|---|--:|--:|--:|--:|
| Flow @1 | 1.00 | 232/288 = 0.806 | [0.755, 0.850] | 0.905 |
| Flow @2 | 2.00 | 250/288 = 0.868 | [0.823, 0.905] | 0.948 |
| Flow @4 | 4.00 | 256/288 = 0.889 | [0.847, 0.923] | 0.948 |
| Flow @8 | 8.00 | 259/288 = 0.899 | [0.859, 0.932] | 0.960 |
| Flow @16 | 16.00 | 255/288 = 0.885 | [0.843, 0.920] | 0.932 |
| Gaussian @100 | 100.00 | 250/288 = 0.868 | [0.823, 0.905] | 0.928 |

### 5 cubes (1 paired set of 96, H=200)

| Arm | Calls/decision | Full success | 95% CI | Per-object |
|---|--:|--:|--:|--:|
| Flow @1 | 1.00 | 29/96 = 0.302 | [0.213, 0.404] | 0.721 |
| Flow @4 | 4.00 | 45/96 = 0.469 | [0.366, 0.573] | 0.800 |
| Gaussian @100 | 100.00 | 40/96 = 0.417 | [0.317, 0.522] | 0.760 |

Cubes-completed distribution (of 5):

| Arm | 0/5 | 1/5 | 2/5 | 3/5 | 4/5 | 5/5 |
|---|--:|--:|--:|--:|--:|--:|
| Gaussian | 3 | 5 | 7 | 18 | 23 | 40 |
| Flow @4 | 2 | 4 | 5 | 15 | 25 | 45 |
| Flow @1 | 1 | 6 | 11 | 23 | 26 | 29 |

Contact rate remains **1.0000** for all three arms; 4.89-4.94 of 5 cubes
contacted. Paired: Flow@4 vs Gaussian p=0.473 (not distinguishable); **Flow@1 vs
Flow@4 p=0.0166 full / p=0.0213 per-object, bootstrap [-0.140, -0.017]**.

### 4 cubes (1 paired set of 96, H=150)

| Arm | Calls/decision | Full success | 95% CI | Per-object |
|---|--:|--:|--:|--:|
| Flow @1 | 1.00 | 55/96 = 0.573 | [0.468, 0.673] | 0.792 |
| Flow @4 | 4.00 | 70/96 = 0.729 | [0.629, 0.815] | 0.904 |
| Gaussian @100 | 100.00 | 65/96 = 0.677 | [0.574, 0.769] | 0.854 |

### Verified structural facts

- **Model calls per planning decision == requested NFE** for all 21 arms.
- **Latency is proportional to NFE**: per-call 19.1-19.9 ms across 1->100 calls,
  planner-only, CUDA-synchronised, on a verified-uncontended GPU.
- **Contact rate = 1.0000** for every arm at 3 and 4 cubes.
- **Policy training data contains exactly 3 cubes in 200,000/200,000 frames**
  (direct tensor inspection; no padded or variable-count slots).
- **DLP encoder was trained on 6-cube scenes** — EC-Diffuser arXiv:2412.18907v2,
  Appendix B: *"we train a single DLP model on a total of 600,000 images ... on
  an environment with 6 cubes in distinct colors."*
- **Action normalization round-trips to 1.49e-08.**

---

## 3. Invalidated / corrected results

| # | Prior claim | Audit finding |
|---|---|---|
| **1** | Our Gaussian control "brackets" the EC-Diffuser paper | **INCORRECT.** Our checkpoint is the *generalization* variant (100 diffusion steps, 12 layers, hidden 512, H=5 per its `args.json`); the paper's Table 5 standard config is 5 steps, 6 layers, hidden 256, horizon 3. Episode draws are ours, not the paper's. Correct wording: **"broadly consistent with the published 0.894 ± 0.025"**. Not a reproduction. |
| **2** | The Flow@4−Flow@1 gap *grows* with object count | **The current checkpoint-level evidence does not support a monotonic increase in the NFE penalty with object count.** Gap: +0.043 [0.013, 0.073] at 3, +0.112 [0.044, 0.180] at 4, +0.079 [0.017, 0.140] at 5 — the point estimate falls from 4 to 5. Bootstrap on the difference of gaps: gap(4)−gap(3) = +0.069 [−0.005, +0.145] p=0.069; gap(5)−gap(3) = +0.036 [−0.033, +0.105] p=0.31; gap(5)−gap(4) = −0.033 [−0.126, +0.057] p=0.49. Every interval includes zero. What survives: the gap is **positive at all three object counts**. Note the native-horizon confound (§6.2) is unresolved here, so this evidence cannot yet separate object count from execution time. |
| **3** | "Flow @4 beats Gaussian" at 4 cubes | **UNVERIFIED.** Paired McNemar p=0.542, Wilcoxon p=0.125, bootstrap CI [−0.008, +0.109] includes zero. Nominal lead only. |
| **4** | Flow imagination is "worse" (dispersion 0.417 vs 0.566/0.716) | **UNVERIFIED as a quality claim.** Ad hoc statistic, n=48, no CI, transparency threshold 0.5 unjustified, never validated against any independent notion of imagination quality. |
| **5** | DLP reconstruction "trustworthy" at 1.8/255 | **SUPPORTED BUT LIMITED.** n=6 frames, front view, one episode, under *random* actions rather than the policy's own state distribution. Per-pixel MAE over a mostly-static white table under-weights cube-level error. |
| **6** | 4-cube latency percentiles | **REPORTING GAP.** `planning_stats()` computes p50/p90/p95/p99, but `isaacgym_fourcube_probe.py:256` persisted only `mean_ms`. The 0.00 values in earlier tables are **absent data, not measurements**. Mean latency is valid. |
| **7** | Flow "is not undertrained; final checkpoint is best" | **SUPPORTED BUT LIMITED.** The supporting 32-episode sweep was **exploratory** and its ordering (399k > 499k) **reversed** at n=96. The loss plateau is real; the checkpoint ranking is not. |
| **8** | OGBench BC-floor findings | **OBSOLETE** for the current story. One training seed; superseded direction. |

**New finding not previously tested:** Flow@1 vs Flow@4 at 4 cubes **is**
statistically distinguishable — McNemar p=0.0275, Wilcoxon p=0.0041, bootstrap
[−0.180, −0.044] excluding zero. This is the strongest single comparison in the
project. It is a *within-method* NFE contrast: **the inference-budget effect is
established within this trained Flow checkpoint; independent Flow training seeds
are still required to establish that it is an algorithm-level effect.**

---

## 4. Reproducibility status

**Every central number is fully traceable.** `experiments/audit/result_manifest.json`
maps each of 21 arms to: raw result file + SHA256, checkpoint + SHA256, episode
set file + SHA256, env config, and HEAD commit.

| Component | Status |
|---|---|
| Commit | `6d75e92`, tree clean |
| Configs | committed (`env_config/generalization_num_cubes`, checkpoint `*_config.pkl`) |
| Checkpoints | present, hashed |
| Raw results | present, hashed, per-episode records retained |
| Episode sets | present; hashes **recomputed from serialized state arrays** and matched |
| Analysis | `recompute_results.py` regenerates every number from raw |

**One gap:** the Gaussian checkpoint was downloaded, not trained here, so its
training run is not reproducible from this repository.

---

## 5. Statistical validity

**What the sample sizes allow.** 288 paired episodes (3 cubes) resolve roughly
7-point differences at 80% power; 96 episodes (4 cubes) resolve roughly 10-point
differences. Measured between-replicate spread of one fixed checkpoint on
96-episode sets was **3-11 points**.

**What the seed count allows — the binding constraint.** There is **1
independently trained Gaussian model** (seed 719594, downloaded) and **1
independently trained Flow model** (seed 42). Episode-level significance
establishes that *these two checkpoints* differ on *these episodes*. It does
**not** establish that flow matching beats Gaussian diffusion.

**Multiple comparisons.** The 3-cube study ran 5 paired contrasts per replicate;
none were corrected. All were non-significant, so no conclusion depends on it —
but a paper must state it.

**Test appropriateness (re-derived, not inherited).** Full success is binary per
episode -> exact McNemar on discordant pairs (correct; counts are small so the
normal approximation is inappropriate). Per-object success is a bounded
per-episode mean over *correlated* cubes -> a proportion test would be invalid;
Wilcoxon signed-rank is used, with skew reported and a paired bootstrap as an
assumption-light cross-check.

---

## 6. Confounds, ranked

| # | Confound | Severity | Detail |
|---|---|---|---|
| **1** | **One training seed per method** | **FATAL to algorithm claims** | Blocks every "Flow vs diffusion" statement. |
| **2** | **Episode horizon changes with object count** | **MAJOR** | H = 100/150/200 at 3/4/5 cubes (`entity_to_steps`, confirmed at runtime). Steps-per-cube *rises* (33.3 -> 37.5 -> 40.0), so the harder setting gets **more** time per object. Direction matters: measured degradation is a **lower bound** on pure-difficulty degradation. But the x-axis cannot be called "task difficulty" unqualified. |
| **3** | **DLP saw up to 6 cubes** | **MAJOR** | Only *policy* generalization is zero-shot. Any "compositional generalization" framing must say so. |
| **4** | **Full-success criterion tightens with N** | **MODERATE** | N-of-N. Mitigated by reporting per-object success as primary. |
| **5** | Single episode set at 4 (and 5) cubes | **MODERATE** | 3-cube used 3 replicates; 4-cube used 1. Between-set spread was 3-11 points at 3 cubes. |
| **6** | Gaussian checkpoint is the generalization variant | **MINOR** | Blocks reproduction claims (see §3.1); does not affect internal comparisons. |

---

## 7. Novelty status

Primary-source search, current through August 2026.

**Already established — cannot be claimed as new:**

- **Few-step / one-step flow policies for robot control.** [FlowPolicy (AAAI 2025)](https://github.com/zql-kk/FlowPolicy) — consistency flow matching for fast 3D manipulation policies; MP1 (MeanFlow, 1-step); [One-Step Flow Policy self-distillation](https://arxiv.org/pdf/2603.12480); [Invertible adapter for one-step FM](https://arxiv.org/pdf/2606.19194); [VITA](https://arxiv.org/pdf/2507.13231).
- **Equivariant/structured flow policies.** [EfficientFlow](https://arxiv.org/pdf/2512.02020).
- **Entity-centric multi-object manipulation with compositional generalization to more objects.** [EC-Diffuser (ICLR 2025)](https://arxiv.org/pdf/2412.18907) already trains on 3 cubes and evaluates zero-shot up to 6 — *this is the exact protocol we are running*.
- **Inference-compute scaling for generative policies.** [ELASTIC](https://arxiv.org/html/2606.31132) learns a meta-policy allocating denoising steps and samples.

**What we did not find prior work doing** (phrased as a search result, not a
claim of absence): a controlled study of **how the required inference budget
(NFE) interacts with the number of objects** in an entity-centric policy, with
paired episodes and verified call counts. ELASTIC — the closest work on adaptive
compute — was checked directly: it uses **single-object tasks** (PushT,
Robomimic Can/Square), **no object-centric representation**, and **does not vary
object count**.

**Honest assessment:** using vanilla Flow Matching in place of Gaussian
diffusion inside an existing architecture is **not a methodological
contribution**. The candidate contribution is the *measurement* — an NFE x
object-count interaction study — and that measurement is currently
underpowered (§3.2) and single-seed.

---

## 8. Hostile-review findings (top five)

1. **FATAL — "One training seed."** Every cross-method claim is checkpoint-level. *Resolution: 3 seeds per method, or restrict all claims to within-method NFE.*
2. **MAJOR — "Where is the method?"** Swapping the generative objective in EC-Diffuser is a known move. *Resolution: the contribution must be the measurement, or a method must emerge from a measured limitation.*
3. **MAJOR — "Your x-axis is confounded."** Horizon changes with object count. *Partially mitigated: the confound favours the harder setting, so degradation is a lower bound. Needs a fixed-horizon control to claim an effect size.*
4. **MAJOR — "Your headline scaling effect isn't significant."** The gap increase has overlapping CIs (§3.2). *Resolution: more episodes at 4/5 cubes, or drop to "positive at both counts".*
5. **MAJOR — "You call it zero-shot but the encoder saw 6 cubes."** *Already resolved by wording discipline — must survive into the paper.*

**Already resolved:** episode pairing (hashes recomputed), NFE counts (forward
hook), latency fairness (planner-only, uncontended), checkpoint-selection
leakage (final checkpoint used despite a sweep favouring another).

---

## 9. Claim boundary

| Finding | Safe claim | Too-strong claim |
|---|---|---|
| Low NFE suffices | "For the evaluated Flow checkpoint on 3-cube PushCube, 2-4 network calls per decision were not measurably worse than the 100-call Gaussian checkpoint (288 paired episodes; powered for ~7-point differences)." | "Flow matching matches diffusion with 25x less compute." |
| Flow vs Gaussian | "In this checkpoint-level comparison, Flow@4 was nominally ahead at 3 and 4 cubes; differences were not statistically distinguishable." | "Flow beats diffusion" / "Flow generalizes better." |
| Low-NFE penalty | "Flow@1 placed materially fewer cubes than Flow@4 at 4 cubes (paired, p<0.01). The penalty was positive at both 3 and 4 cubes; the CIs for its magnitude overlap, so an increase with object count is not established." | "The NFE requirement grows with compositional complexity." |
| Generalization | "Zero-shot **policy** generalization from 3 training objects to 4 test objects; the DLP encoder had prior exposure to 6-object scenes." | "Zero-shot compositional generalization." |
| Difficulty axis | "Increasing object count under the benchmark's native object-dependent episode budget." | "Task difficulty causes X." |
| Imagination | "Decoded Flow futures appear qualitatively more smeared, and a descriptive particle-dispersion statistic is lower for Flow, despite stronger control." | "Imagination quality is unnecessary" / "Flow imagination is worse." |
| Variance collapse | "The [-1,1] representation has E[x^2]=0.121 and the linear path loses ~89% of input scale mid-trajectory." | "Variance collapse is harmless" / "...is the bottleneck." |

---

## 10. Canonical artifacts

| Artifact | Path |
|---|---|
| Canonical results CSV | `experiments/audit/canonical_results.csv` |
| Recomputation script | `experiments/audit/recompute_results.py` |
| Recomputed details + paired tests | `experiments/audit/recomputed_details.json` |
| Claim ledger | `experiments/audit/claim_ledger.{json,csv}` |
| Result manifest (provenance) | `experiments/audit/result_manifest.json` |
| Canonical figures | `experiments/audit/figures/audit_canonical.png` |
| Figure script | `experiments/audit/make_canonical_figures.py` |
| Repository snapshot | `experiments/audit/repository_snapshot.md` |
| NFE gap by cube count | `experiments/audit/nfe_gap_by_cubes.json` |

Test suite at audit time: `PYTHONPATH="$PWD:$PWD/diffuser" python -m pytest
tests/ -q` -> **212 passed, 1 skipped, 1 warning in 3.46s**, 2026-08-22 01:22:06.

---

## 11. Minimum experiments remaining

Compute estimated from **measured** throughput: 0.2343 s/step steady-state
(3-cube Flow run `l1vkhnp9`) -> ~32.5 GPU-h per 500k-step training arm.
Evaluation: 96 episodes costs ~130 s (Flow@4) to ~2,000 s (Gaussian@100).

| # | Experiment | Purpose | GPU-h | Priority |
|---|---|---|--:|---|
| 1 | **2 additional Flow training seeds** | Turns within-method NFE findings from checkpoint-level to algorithm-level. Each seed is evaluable at *all* NFEs, so no per-NFE training is needed. | ~65 | **MUST HAVE** |
| 2 | **2 additional Gaussian training seeds** | Required only for cross-method claims. If the paper restricts itself to within-method NFE scaling, this is not needed. | ~65 | **STRONGLY RECOMMENDED** (MUST if claiming Flow vs Gaussian) |
| 3 | **Fixed-horizon control** at 4 cubes (H=100, matching 3-cube) | Deconfounds object count from time budget. Cheapest decisive control: Flow@4 + Flow@1 + Gaussian, 96 episodes. | ~0.6 | **MUST HAVE** |
| 4 | **More episodes at 4/5 cubes** (2 more sets each) | The gap-increase CIs currently overlap; this is what would resolve §3.2. | ~1.3 | **MUST HAVE** |
| 5 | Non-entity backbone control | Tests whether the NFE-vs-object-count interaction is *entity-specific* — the only route to a mechanism claim. | ~32.5 | STRONGLY RECOMMENDED |
| 6 | State-generation objective ablation | Reproduces the paper's ablation in our pipeline where imagination is measurable. | ~65 | OPTIONAL |
| 7 | Modern low-NFE baseline (e.g. consistency/MeanFlow-style) | Reviewer 2 will ask why not compare against current few-step methods. | ~32.5+ | STRONGLY RECOMMENDED |
| 8 | Isaac Lab replication / real robot | Reviewer 1 (Isaac Gym is deprecated). | large | OPTIONAL |

**Cheapest scientifically valid path to a defensible claim: items 3 + 4 (~1.9
GPU-h) fix the confound and the power problem for the *within-method* NFE
result, which item 1 (~65 GPU-h) then elevates to algorithm level.**

---

## 11b. 5-cube regime classification

**Regime B — ideal scaling regime.** Both methods retain substantial but clearly
reduced performance, with useful separation among the three arms.

| Regime | Verdict | Evidence |
|---|---|---|
| A. Still too easy | **No** | Best arm 0.469 full success; 51 of 96 episodes fail. |
| **B. Ideal scaling** | **YES** | Full success spans 0.302-0.469; per-object 0.721-0.800; all three arms separated; no arm at ceiling or floor. |
| C. Flow low-NFE weakness emerges | **No** | Flow@4 (0.469) is nominally *above* Gaussian (0.417), not below. |
| D. Flow advantage strengthens | **Not claimable** | Flow@4 leads but p=0.473; and with one training seed this could not be an algorithm claim regardless. |
| E. Joint collapse | **No** | 0-of-5 occurs in only 1-3 of 96 episodes per arm; contact rate 1.0000; 4.89-4.94 of 5 cubes contacted. |

Per the standing instruction, 5 cubes being Regime B means **stop; do not run
6 cubes.**

**Which benchmark should be primary?** **5 cubes**, with 4 cubes retained as a
mid-point. At 5 cubes the arms are furthest from ceiling (best 0.469) while
still well clear of the floor, giving the widest dynamic range for any future
comparison. 3 cubes is saturated and should be reported only as the in-
distribution reference.

---

## 12. Single next move

**Run the fixed-horizon control: 4 and 5 cubes at H=100 (matching the 3-cube
budget) — Flow@4, Flow@1, Gaussian, 96 episodes each, ~1.5 GPU-h.**

The 5-cube probe is complete and verified. The current checkpoint-level evidence
**does not support a monotonic increase in the NFE penalty with object count**
(§3.2): +0.043 / +0.112 / +0.079 at 3/4/5, with all gap-difference intervals
including zero. Before that pattern can be interpreted at all, the horizon
confound must be removed, because episode horizon rose 100 -> 150 -> 200 across
exactly those three points and is a live candidate explanation for the 4 -> 5
dip.

At ~1.5 GPU-h this costs roughly 2% of one training seed and is a prerequisite
for any scaling statement. It must precede training work: additional seeds would
otherwise only add precision to a confounded axis.
