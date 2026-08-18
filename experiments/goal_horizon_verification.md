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

## 5. What it means mechanically — the part that does matter

The number itself is benign; its *consequence* is not. Measured on 6,000
sampled windows:

| Task | ‖o[t] − o[t+4]‖ (achievable in-window) | ‖o[t] − goal‖ (conditioning demands) | ratio |
|---|---:|---:|---:|
| OGBench puzzle-4x4 | 1.81 | 4.67 | **2.6×** |
| 3-cube PushCube | 16.40 | 17.91 | **1.1×** |

Mechanically, the OGBench model is trained to emit a 5-step action chunk whose
terminal state is pinned to a goal a **median of 187–190 steps away**, i.e. a
displacement ~2.6× larger than anything reachable in 4 transitions. The
regression target is therefore *systematically unreachable within the horizon*.
The model can only learn "move in the direction of the goal", never "reach the
goal", and the conditioning slot is a constraint it structurally cannot satisfy.

This is consistent with the v3 taxonomy's observed behavior — transient
progress that does not compound and often regresses.

### Why 3-cube does not show this

`GoalDataset.__getitem__` uses `normed_goals[path_ind, path_length-1]`: the
**fixed task goal**, constant within the episode, over 100-step episodes. Its
ratio is 1.1×, i.e. the conditioning target is roughly as far as the window
already travels. The two tasks pose structurally different problems despite
sharing H=5. Results are not transferable between them without controlling for
this.

## 6. Consequence for diagnosis

The goal-horizon mismatch is a **property of the OGBench adapter's goal
policy interacting with H=5**, not of the generative objective. It is a
confound that must be controlled before any statement of the form "low-NFE
generative modeling fails on this task" is defensible.

It is cheap to manipulate directly (goal-distance-conditioned subsets, horizon
changes, capped relabeling) without retraining for the subset analyses. Those
ablations are specified in `experiments/protocols/hard_task_diagnosis_v1.md`.

## 7. Reproduction

```bash
source /home/jren313/miniconda3/etc/profile.d/conda.sh
conda activate ecdiffuser-linux
# recompute section 3 and 5 from raw sources; CPU only, ~1 min
python experiments/scripts/verify_goal_horizon.py
```
