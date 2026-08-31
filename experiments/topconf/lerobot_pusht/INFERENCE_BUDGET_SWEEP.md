# Inference-budget sweep — lerobot/diffusion_pusht

Eight predeclared budgets, frozen 50-episode set (seeds 1000-1049, spec sha
`86867a8e…`), identical for every arm. Offline: 200 predeclared replay
conditions × K=4 diffusion samples with a shared seed set.

## Full results

| NFE | UNet calls/plan | len(timesteps) | first t | offline exec L2 | (sd over K) | full-horizon L2 | success % | avg max reward | planner latency (ms) |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 1.00 | 1 | 0 | **254.99** | 34.17 | 254.13 | **0.0** | 0.0879 | 12.7 |
| 2 | 2.00 | 2 | 50 | 5.9205 | 1.69 | 11.004 | 88.0 | **0.9674** | 19.9 |
| 4 | 4.00 | 4 | 75 | 4.7717 | 0.88 | 10.198 | 84.0 | 0.9206 | 34.3 |
| 5 | 5.00 | 5 | 80 | 4.7796 | 1.00 | 10.239 | 88.0 | 0.9633 | 40.2 |
| 10 | 10.00 | 10 | 90 | 4.7310 | 1.04 | 10.320 | 86.0 | 0.9370 | 74.1 |
| 20 | 20.00 | 20 | 95 | 4.7521 | 1.13 | 10.190 | 90.0 | 0.9600 | 139.8 |
| 50 | 50.00 | 50 | 98 | 4.6735 | 1.08 | 10.286 | **92.0** | **0.9789** | 340.6 |
| 100 | 100.00 | 100 | 99 | **4.6500** | 1.04 | 10.296 | 86.0 | 0.9277 | 702.8 |

`UNet calls/plan == len(timesteps) == requested num_inference_steps` for all
eight arms, verified by forward hook. Each arm's full scheduler timestep sequence
is stored in `results/nfe*_sweep.json`.

Note the schedule geometry is **not** uniform across budgets: the first timestep
moves from 99 (N=100) to 50 (N=2) to **0** (N=1). At N=1 the single step is taken
at t=0, which is why NFE1 degenerates — a scheduler-geometry fact, recorded per
§9, not a claim about one-step diffusion in general.

## Rankings (NFE1 excluded — it collapses on both axes, the trivial case §22)

| NFE | offline rank | success rank | max-reward rank |
|--:|--:|--:|--:|
| 2 | 7 (worst) | 3 | 2 |
| 4 | 5 | 7 (worst) | 7 (worst) |
| 5 | 6 | 4 | 3 |
| 10 | 3 | 6 | 5 |
| 20 | 4 | 2 | 4 |
| 50 | 2 | **1** | **1** |
| 100 | **1** (best) | 5 | 6 |

- **Spearman(offline L2, success) = −0.018** (perfect alignment would be −1)
- **Spearman(offline L2, max_reward) = +0.214** (wrong sign)
- offline optimum: **NFE 100**; success and max-reward optimum: **NFE 50**
- offline error varies **27.3%** across NFE 2-100 (5.92 → 4.65); success varies
  84-92%

## Resolvability — the essential caveat

At n=50 the closed-loop spread is **not** individually resolvable:

| NFE | successes | 95% CI |
|--:|--:|---|
| 2 | 44/50 = 88% | [76, 95] |
| 4 | 42/50 = 84% | [71, 93] |
| 5 | 44/50 = 88% | [76, 95] |
| 10 | 43/50 = 86% | [73, 94] |
| 20 | 45/50 = 90% | [78, 97] |
| 50 | 46/50 = 92% | [81, 98] |
| 100 | 43/50 = 86% | [73, 94] |

**All seven intervals overlap heavily.** Per §23 no claim is made about the
individual 2-8 pp orderings, and per §19 the near-zero Spearman must not be
over-read from seven noisy points.

Direct extreme comparison, NFE2 vs NFE100 (paired on the same 50 seeds):
discordant pairs 3 vs 4, **McNemar exact p = 1.000**; avg max reward 0.9674 vs
0.9277 (Δ +0.0397 favouring **NFE2**).

## What IS supported

Not "reordering", but **saturation with a large efficiency gap**:

> Offline demonstrated-action error keeps improving monotonically from NFE 2 to
> NFE 100 (5.92 → 4.65, a 21% reduction between the extremes), while closed-loop
> utility is statistically indistinguishable across that entire range
> (McNemar p = 1.000 between the extremes; all CIs overlapping). NFE 2 attains
> the *worst* offline error of any non-degenerate budget yet the *second best*
> average max reward, at **35.3× lower planner latency** than NFE 100.

The offline metric therefore does **not** identify the efficient operating point:
it prefers NFE 100, while behaviour is already saturated at NFE 2.
