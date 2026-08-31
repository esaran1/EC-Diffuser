# Offline alignment — verified from source AND empirically

Given this project's earlier target-provenance failures, the alignment was
established two independent ways before any bulk evaluation.

## 1. From source

`DiffusionPolicy.select_action` docstring (schematic, verbatim):

```
(legend: o = n_obs_steps, h = horizon, a = n_action_steps)
|timestep            | n-o+1 | n-o+2 | ..... | n     | ..... | n+a-1 | n+a   | ..... | n-o+h |
|observation is used | YES   | YES   | YES   | YES   | NO    | NO    | NO    | NO    | NO    |
|action is generated | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   |
|action is used      | NO    | NO    | NO    | YES   | YES   | YES   | NO    | NO    | NO    |
```

`DiffusionModel.generate_actions` (verbatim):

```python
start = n_obs_steps - 1
end = start + self.config.n_action_steps
actions = actions[:, start:end]
```

With `n_obs_steps=2`, `horizon=16`, `n_action_steps=8`:
**executed slice = `actions[:, 1:9]`**, i.e. the 8 actions beginning at the
current frame.

## 2. Indexing diagram

```
dataset frame index:     t-1        t        t+1  ...  t+7      t+8 ... t+14
observation used:        YES       YES        no        no       no       no      (delta_timestamps -1/fps, 0)
demo action fetched:      -      A[0]       A[1]      A[7]     A[8]    A[15]      (delta_timestamps 0..15/fps)
model output index:     P[0]      P[1]       P[2]      P[8]     P[9]    P[15]
EXECUTED (generate_actions): -----> P[1:9] <-----
COMPARED AGAINST:            -----> A[0:8] <-----
```

`delta_timestamps` used:
`observation.image`/`observation.state` = `[-1/fps, 0.0]`;
`action` = `[0/fps, 1/fps, ..., 15/fps]`. Padding is handled by
`LeRobotDataset`'s own delta-timestamp machinery; conditions are drawn uniformly
from all 25,650 frames with seed 20260920.

## 3. Empirical confirmation — the offset scan

If the alignment were wrong, some other offset would fit better. It does not.
60 conditions, NFE 100:

| model slice start | L2 vs demo `A[0:8]` |
|--:|--:|
| 0 | 10.3283 |
| **1** | **6.1355** ← `n_obs_steps−1 = 1` |
| 2 | 12.6334 |
| 3 | 21.2314 |
| 4 | 29.8535 |
| 8 | 58.5780 |

Demo-shift control (holding the model slice at `P[1:9]`):

| demo offset | L2 |
|--:|--:|
| **+0** | **6.1355** ← ours |
| +1 | 10.6172 |
| +2 | 17.6859 |

**`start=1` is the unique minimum on both axes**, matching the source-derived
value exactly. Alignment is confirmed.

## 4. Metric definitions

- **PRIMARY:** mean L2 over the **8 actions the policy would actually execute**
  (`P[1:9]` vs `A[0:8]`), in pixel units, averaged over conditions and K samples.
- **SECONDARY:** full 16-step horizon L2 (`P` vs `A`).
- **Stochasticity:** K = 4 diffusion samples per condition, seeds
  `20260921 + k`, the **same seed set for every inference budget**.
  Reported as mean (primary), median, and SD across samples.
  **No best-of-K.**
- 200 replay conditions, predeclared by seed, chosen **independently of any
  closed-loop result**.
