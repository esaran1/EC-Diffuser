# Stage F — multi-object task audit (§32)

**Verified from `env_config/`, `diffuser/config/`, and the checkpoint tree — not
inferred from names.**

## 1. What the "nC" names actually mean

| mode | `env_config_dir` | `numObjects` | RandColor | numColors | `max_path_length` |
|---|---|--:|:--:|--:|--:|
| `1C_dlp` | `env_config/n_cubes` | — (see note) | False | — | 30 |
| `2C_dlp` | `env_config/n_cubes` | **2** | False | — | 50 |
| `3C_dlp` | `env_config/n_cubes` | **2** ← note | False | — | 100 |
| **`3C_dlp_randcolor`** | `env_config/generalization_num_cubes` | **3** | **True** | **6** | 100 |
| `1C_dlp_pusht` | `env_config/push_one_t` | **1** | False | — | 50 |
| `2C/3C_dlp_pusht` | `env_config/push_n_t` | **3** | False | — | 100/150 |

**Important caveat the directive was right to demand.** `numObjects` is a value
in a *shared* YAML file, not a per-mode value: `1C_dlp`, `2C_dlp` and `3C_dlp`
all point at the **same** `env_config/n_cubes/IsaacPandaPushConfig.yaml`, which
currently reads `numObjects: 2`. The distinguishing configured quantity between
those three modes in `pandapush_flow.py` is **`max_path_length` (30/50/100)**,
not the object count. The object count for those modes would come from the CLI
(`--num_entity`) or from editing that shared YAML.

So: **"3C" does not reliably denote three cubes via the config file alone.** For
our canonical policy it does, because `3C_dlp_randcolor` uses a *separate* config
directory (`generalization_num_cubes`) that genuinely sets `numObjects: 3`, and
because the production invocation passes `--num_entity 3 --rand_color`.

## 2. Our canonical task is already the hardest configured one

The canonical policy is `3C_dlp_randcolor`:
- **3 objects**, random colors drawn from 6, 2 camera views,
- success threshold `max_obj_dist <= 0.04` (matches the evaluator),
- ~79% macro success.

This directly addresses H5 (task saturation): the easy-task worry does **not**
apply in the form "we are testing on one cube." We are already on the
three-cube, random-color generalization setting. At ~79% it is neither saturated
(100%) nor floored (0%).

## 3. The blocking constraint for Stage F

**Every trained checkpoint in this repository is `3C_dlp_randcolor`.** Verified
by enumerating every run directory containing a `state_*.pt`: all 30+ are
`3C_dlp_adalnpint_randcolor_H5_T4_seed{42,43,44,45}`.

Datasets exist for `1C`, `2C`, `3C`, `3C_randcolor` (375M–774M each), but there
is **no pretrained policy for any other task**, and the canonical model required
**500,000 training steps**.

### Cost of creating one

Measured: our 5,000-step fine-tune costs ~0.33 GPU-h ⇒ ~6.5e-5 GPU-h/step.
A from-scratch policy at 500k steps ≈ **33 GPU-h** — **3.3× the entire 10.0
GPU-h program cap**, for one baseline on one task.

Phase 3 already established that short from-scratch PandaPush training is
insufficient: Standard Flow at 20k steps gave **0/96 on all five evaluation
sets**. A 20k screen would cost ~1.3 GPU-h and is known in advance to produce a
floored, uninformative baseline — precisely the "candidate task baseline ~0%"
case §33 says to avoid.

## 4. Verdict — Stage F is BLOCKED, and the reason is structural

> A harder-task comparison cannot be run inside this program's compute cap,
> because no pretrained policy exists for any task other than the one we are
> already on, and creating one costs ~33 GPU-h against a 10.0 GPU-h cap.

This is **not** a scheduling decision to be worked around by shortening arms
(§ "Do not shorten scientific arms to compensate"). It is reported as a hard
constraint.

### What partially substitutes

H5's concern is that the task may be too easy to reveal modest architectural
effects. Two things address it without a new task:

1. The canonical task is **already** the 3-object random-color setting at ~79%,
   not a one-cube toy.
2. The binding limit on detecting "modest architectural improvements" here is
   **not** task saturation but **evaluator resolution**: conservative RC95
   ±4.30 pp, with R=3 giving a ±2.48 pp half-width. An architectural effect of
   1–2 pp is unresolvable on *any* task at this evaluation budget.

That reframing is the honest response to H5 given what we can afford.

## 5. If Stage F were funded later

Cheapest informative version, for the record:
- pick `3C_dlp` (no random color) or `2C_dlp` as a *related* task reusing the
  same DLP latent checkpoints;
- train **one** baseline to convergence (~33 GPU-h) and freeze it as a permanent
  asset for all future comparisons, amortizing the cost;
- then fine-tune candidates from it at ~0.33 GPU-h each.

The one-off baseline is the expensive part; everything after is cheap.
