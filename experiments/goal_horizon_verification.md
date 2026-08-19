# Verification of the 99.2% goal-horizon claim

Date: 2026-08-17. CPU-only. No GPU work, no training, no retraining.

## 1. Claim under audit

`imf_auxiliary_puzzle_progress_diagnostic_results_v3.json` reports, for the
OGBench puzzle-4x4 training adapter:

```
windows                            997000
policy_horizon                     5
fraction_goal_beyond_endpoint      0.9923279839518556
goal_offset_steps  median 190.0  mean 252.99  p99 862  max 1000
```

The statistic was produced ad hoc; **no code in the repository computes it**,
so it was independently recomputed here from the raw source.

## 2. Exact mechanism, traced to source

`diffuser/diffuser/datasets/benchmark_sequence.py`,
`OGBenchPuzzleWindowDataset`:

```python
def _goal_index(self, index, window_start, episode_end):
    lower = window_start + self.horizon - 1     # = window_start + 4
    span  = episode_end - lower
    return lower + _stable_integer(self.goal_seed, index, window_start) % span
```

and in `_batch`:

```python
observations[-1] = normalized_goal          # last obs slot REPLACED by goal
conditions[self.horizon - 1] = normalized_goal.copy()
```

So each training example is a 5-step window whose **final observation slot is
overwritten by a state sampled uniformly from `[t+4, episode_end]`**, and that
slot is also imposed as a hard conditioning constraint at sampling time.

Episodes are 1001 steps (manifest `episode_offsets`), giving 997 windows per
episode and 997,000 windows over 1,000 train episodes.

## 3. Independent reproduction — claim CONFIRMED

Recomputed from `train.npz` + manifest, replaying `_stable_integer` exactly:

| Quantity | Reported | Recomputed |
|---|---:|---:|
| windows | 997000 | 997000 |
| min offset | 4 | 4 |
| mean offset | 252.98824172517553 | 252.98824172517553 |
| median offset | 190.0 | 190.0 |
| p90 / p95 / p99 | 590 / 703 / 862 | 590 / 703 / 862 |
| max offset | 1000 | 1000 |
| fraction beyond endpoint | 0.9923279839518556 | 0.9923279839518556 |
| fraction >100 ahead | 0.6768525576730191 | 0.6768525576730191 |

Every digit matches. **The statistic is correct.**

## 4. But it is not a bug — it is arithmetic

For a window at position `p` in a 1001-step episode with H=5, the goal is
uniform over `span = 997 - p` candidates, so
`P(goal exactly at window endpoint) = 1/span`. Averaging over all windows:

```
analytic  E[P(goal at endpoint)] = 0.0075050
analytic  fraction beyond endpoint = 0.9924950
empirical fraction beyond endpoint = 0.9923280
```

The 99.2% is the **mechanical consequence of uniform within-episode goal
sampling over long episodes**, matching analytically to four decimals. It is
not evidence of a corrupted adapter, a leak, or a mislabeled dataset. Any
uniform hindsight-relabeling scheme on 1001-step episodes with a short window
produces ~99%.

**This corrects the framing in the previous report**, where the number was
listed as a data/temporal pathology. It is expected behavior of the declared
goal policy (`"sample goal time uniformly from [current_time, episode_end]"`).

## 5. What it means mechanically — HYPOTHESIS, not conclusion

**This section was previously overstated and is corrected here.** A goal lying
outside the H=5 window is *expected and normal* in goal-conditioned control:
hindsight relabeling deliberately trains on distant goals, and the policy is
re-planned every step. Distance-beyond-horizon is therefore **not by itself
evidence of a flaw**. What follows is a hypothesis to be tested by D1/D3/D4,
not an established bottleneck.

### 5.1 The original metric was not valid

The first pass reported "OGBench 2.6x vs 3-cube 1.1x" from a raw unscaled
Euclidean norm. That comparison does not survive scrutiny:

- OGBench's 83-D observation mixes continuous arm state with **32 binary
  dims** (16 buttons x 2 one-hot). Per-dimension std spans 0.009-0.96, a
  ~100x range, so an unweighted L2 is dominated by whichever dims happen to
  have large scale.
- The 3-cube figure was computed on **DLP latent particle features**, which
  are an unnormalized learned representation. Distances there have no
  physical meaning and are not comparable to OGBench state units.

Comparing those two numbers was not meaningful.

### 5.2 Scale-aware and task-relevant metrics

Recomputed on 6,000 sampled windows per task:

**OGBench puzzle-4x4**

| Metric | in-window (4 steps) | to goal | ratio |
|---|---:|---:|---:|
| raw L2, all 83 dims | 1.805 | 4.669 | 2.59x |
| standardized L2 (z-scored) | 6.910 | 11.641 | **1.68x** |
| **button Hamming (task metric)** | 0.475 | 6.635 | **13.96x** |

**3-cube PushCube**, physical metric (mean per-cube Euclidean displacement,
metres, from `state_observations`/`state_goals` rather than DLP latents):

| Metric | in-window | to goal | ratio |
|---|---:|---:|---:|
| per-cube displacement (m) | 0.0096 | 0.0352 | **3.66x** |

So the corrected picture is the **opposite** of the first pass on the
continuous metric: 3-cube's ratio (3.66x) is *larger* than OGBench's
standardized ratio (1.68x). The earlier "3-cube 1.1x" claim is withdrawn.

### 5.3 What survives, and why it is still worth testing

The one measurement that remains striking is **task-relevant** and does not
depend on scaling choices:

- **87.97%** of training windows contain **zero button state changes**.
- Goals differ from the current state by **6.64 of 16 buttons** on average.
- Only 7.5% of goals have zero button difference.

So the median training example asks: given a state, and a goal 6-7 button
flips away, produce 5 actions -- during which, 88% of the time, **no button
changes at all**. The supervision signal for the combinatorial part of the
task is present in ~12% of windows.

This is a **plausible mechanism** for weak puzzle learning, and it is
specific to puzzle's discrete subgoal structure. It is *not* established as
the bottleneck: a policy can legitimately learn to make progress toward a
distant goal without changing a button in every window. That is exactly what
D1/D3/D4 exist to test.

## 6. Status: OPEN HYPOTHESIS, not a finding

Summary of what is and is not established:

| Statement | Status |
|---|---|
| The 99.2% figure is numerically correct | **Verified** |
| It is the arithmetic of uniform relabeling, not a data bug | **Verified** |
| Goals lie far outside H=5 | **Verified, and expected** in goal-conditioned control |
| 88% of windows contain no button change | **Verified** |
| The original "OGBench 2.6x vs 3-cube 1.1x" contrast | **Withdrawn** — invalid metrics |
| Sparse button supervision limits puzzle learning | **Hypothesis, untested** |
| Goal-horizon mismatch is *the* bottleneck | **Not established** |

Nothing here licenses a claim that goal relabeling is the bottleneck. It
licenses one specific, cheap test: **does prediction/control quality actually
degrade as goal offset grows (D1), and does changing the relabeling or the
horizon actually improve performance (D3/D4)?** Only affirmative answers to
both would justify calling it a bottleneck.

**Scope limit — this analysis is puzzle-specific.** The 88% figure is about
discrete button toggles. cube-triple has no analogous discrete subgoal
structure; its goals are continuous cube poses. Neither the mechanism nor the
D1/D3/D4 ablations transfer automatically. See
`experiments/protocols/hard_task_diagnosis_v1.md` §7 for the transfer analysis.

## 7. Reproduction

```bash
source /home/jren313/miniconda3/etc/profile.d/conda.sh
conda activate ecdiffuser-linux
# recompute section 3 and 5 from raw sources; CPU only, ~1 min
python experiments/scripts/verify_goal_horizon.py
```

## 8. Cross-task invariant confirmed on real data

The unit test `test_ogbench_tasks_share_one_goal_relabeling_rule` pins the
shared goal rule on synthetic fixtures. Verified on the actual datasets
(2026-08-18): with the same `goal_seed`, the puzzle-4x4 and cube-triple
validation adapters produce **identical goal offsets** for the first 5,000
windows (mean offset 252.4 for both), confirming they share one code path.

This is what licenses comparing goal-offset structure across the two tasks.
