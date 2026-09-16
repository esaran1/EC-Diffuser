# Closed-loop evaluation plan (§29, §10) — registered before running

## The §29 gate, checked against the offline diagnostics

| gate requirement | status |
|---|---|
| no NaNs | **PASS** — every path finite in `replay_*.json` |
| no gross imagination collapse | **PASS** — state error 0.16–0.34 across paths, no divergence |
| coherent state prediction | **PASS** — UWM state off/diag ratio 1.145 |
| action output within sane normalized range | **PASS** — checked in replay eval |
| off-diagonal model clearly adapted beyond the warm-start OOD artifact | **PASS** — corr(dist,err) +0.356 vs control +0.692; corner 6.4× better |
| sampler tests pass | **PASS** — oracle-field test exact for all 5 paths |

## Which models go to closed loop, and why only these

§29 forbids evaluating ten paths. Offline diagnostics narrow the set:

| candidate | decision | reason |
|---|---|---|
| **CTL, diagonal** | **EVALUATE** | the matched control — mandatory reference (§22) |
| **UWM, diagonal (P0)** | **EVALUATE** | the UWM arm at its own best operating point |
| UWM, P2 action-leads | **hold** | best non-diagonal UWM path offline (0.0550), but still 3.4× worse than UWM's own diagonal |
| UWM, P1/P3/P4 | **drop** | 0.117–0.251, far worse offline |

**Rationale for the pair chosen.** The offline evidence says the diagonal is the
best operating point on *both* arms. The scientifically load-bearing comparison
is therefore **"does independent-time training change closed-loop control at the
operating point anyone would actually use?"** — i.e. UWM-diagonal vs
CTL-diagonal. Evaluating a non-diagonal path that is already 3–16× worse offline
would spend 0.66 GPU-h to confirm a prediction the offline data makes
confidently.

This is the §29 instruction ("use offline diagnostics to narrow to at most 2 UWM
paths") applied honestly: the data narrowed it to one path, the diagonal.

## Protocol (§10, §51)

- **R = 3** independent evaluator realizations per frozen checkpoint
- the same five **fixed** scenario sets: E0 20260820, E1s 20261820, E2s 20262820,
  E3s 20263820, E4s 20264820 — never regenerated per policy
- 96 episodes per set, NFE=4, EMA weights, `max_obj_dist <= 0.04`
- report **every realization** and **every E set**, plus the R=3 mean
- terminology kept distinct: **training seed** (42) vs **environment/scenario
  seed** (the five E sets) vs **evaluator realization** (R=1..3)

## Predeclared interpretation (§49)

| observed |UWM − CTL| | reading |
|---|---|
| < ~3 pp | **UNRESOLVED / below practical routine resolution** — the measured conservative RC95 is ±4.30 pp and the R=3 half-width is ±2.48 pp |
| > ~3 pp with consistent sign across E sets | may justify stronger follow-up (3 training seeds) |

**3 pp is a practical resolution guide, not a significance cutoff.**

## Cost

2 models × R=3 × 0.2212 = **1.327 GPU-h**. Checked against the cap before launch
by `run_closed_loop.py`, which sums training and evaluation together.
