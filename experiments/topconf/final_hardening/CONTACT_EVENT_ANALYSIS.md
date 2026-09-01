# Contact mechanism — what the cache supports (§20-§22)

## What is available, and what is not

The open-loop divergence run (10 repeats, fixed actions, no policy) logged
**population-level divergence per timestep** but **not per-episode first-contact
time**. Exact event-alignment (τ = 0 at first contact) is therefore **not possible
from cache**; it would need a re-log. Stated as a limitation rather than
approximated.

## Population-level divergence (open loop, no policy, identical actions)

| t | cube_max | eef_max |
|--:|--:|--:|
| 0 | 5.878e-04 | 8.58e-07 |
| 1 | 5.739e-04 | 2.86e-06 |
| 2-5 | 4.593e-03 (flat) | 1.7-2.8e-03 |
| 6 | 6.303e-03 | 2.91e-03 |
| **7** | **2.513e-02** | 1.31e-02 |
| **8** | **2.259e-01** | 1.03e-02 |
| **9** | **9.225e-01** | 1.02e-02 |
| 20 | 7.175e-01 | 3.80e-02 |
| 100 | 9.225e-01 | 1.57e-01 |

Divergence is nonzero at **t=0, before any action** (5.9e-4), sits on a plateau
through t≈6, then rises by **three orders of magnitude between t=7 and t=9**
(largest single-step amplification **9.0× at t=7→8**), and then **saturates**
rather than compounding. Terminal cube divergence is **23× the 0.04 success
threshold**.

The shape is a **discontinuity followed by a plateau**, not exponential chaos —
consistent with a near-tie contact resolving differently and the object taking a
macroscopically different path.

## NEW: contact *timing* variability distinguishes sensitive episodes

From the closed-loop R=8 same-arm bank (seed 42, NFE4), per-episode
`first_contact_step` across the 8 realizations:

| | physics-sensitive (n=39) | robust-success (n=57) |
|---|--:|--:|
| mean first_contact_step | 0.660 | 0.651 |
| **within-episode SD of first_contact_step** | **0.0809** | **0.0296** |

**Mean contact time is essentially identical (0.660 vs 0.651), but its
realization-to-realization variability is 2.7× larger on sensitive episodes.**

This is a sharper mechanistic statement than the earlier "200× amplification
between steps 2-10": what distinguishes an unstable scenario is not *when* the
robot contacts an object, but **how reproducibly** it does so. Scenarios whose
contact timing jitters are the ones whose outcomes bifurcate.

## §21 — which contacts?

`n_contacted` counts objects contacted; the evaluator does **not** distinguish
robot-object, object-object and object-ground contacts, and PhysX contact
collection is disabled in the canonical config (`contact_collection: 0`,
CC_NEVER). **We cannot attribute the bifurcation to a specific contact type from
existing logs.** Stated as a limitation.

## §22 — outcomes bifurcate, they do not jitter around the threshold

On physics-sensitive episodes, split by realization outcome:

| realization outcome | mean max_obj_dist |
|---|--:|
| succeeded | **0.0184** |
| failed | **0.3036** |

A **16.5× separation**, and the failure value is **7.6× the 0.04 threshold**.
Reproduced at 4 cubes: sensitive episodes 0.1041 vs robust 0.0214 (4.9×).

Failures are not millimetre jitter across a decision boundary — the object ends
up somewhere qualitatively different. This is the single most important
mechanistic fact for the paper, because it rules out "just move the threshold" as
a fix.
