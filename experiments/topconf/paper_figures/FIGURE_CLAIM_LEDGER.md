# Figure Claim Ledger

One row per claim a figure makes. "Supported by" names the frozen artifact.
No claim here requires E1. E1 was not touched to build this package.

## Fig 1 — Nested realizations
| # | Claim | Supported by | Status |
|---|---|---|---|
| 1.1 | Fixing policy, scenario, and policy noise still yields different outcomes across physics realizations | `evaluation_noise/results/r0_s42_nfe4_crnrep{1..8}.json` | **WRITE** |
| 1.2 | Some scenarios are deterministic (p=1), others split near p≈0.5 | R=8 bank, 96 scenarios | **WRITE** |
| 1.3 | The three displayed scenarios were chosen by a stated rule (lowest-id robust / lowest-id mixed / closest to p=0.5), not by inspection | `make_figures.py` selection code | **WRITE** |

## Fig 2 — Contact-associated bifurcation
| # | Claim | Supported by | Status |
|---|---|---|---|
| 2.1 | Failures bifurcate rather than jitter around the threshold: successes 0.0177 m, failures 0.3036 m mean final distance, threshold 0.04 m | R=8 bank `max_obj_dist` | **WRITE** |
| 2.2 | Sensitive scenarios show ~2.7× the first-contact-step variability of robust ones (SD 0.0809 vs 0.0296) | R=8 bank `first_contact_step` | **WRITE** — associational only |
| 2.3 | Contact timing variability *causes* the bifurcation | — | **MISSING-EVIDENCE**. Association only; state as "consistent with", never causal. |
| 2.4 | This is a property of Isaac Gym GPU physics generally | — | **MISSING-EVIDENCE**. One simulator, one task family. Vendor-documented nondeterminism is not ours to claim as new. |

## Fig 3 — Conclusion instability
| # | Claim | Supported by | Status |
|---|---|---|---|
| 3.1 | 14.8% of nested single-realization views are **strict sign reversals** vs the R=3 reference (16/108) | `calibrated_contrasts.csv` | **WRITE** — 14.8%. Supersedes both 24% and 21.3% |
| 3.1b | 7/108 views land at exactly Δ=0. A tie is **not** a reversal and must not be counted as one | recomputation §10 | **WRITE** — excluding ties from the denominator gives 15.8%; report 14.8% |
| 3.2 | 25.0% disagree in **practical category** (27/108) | same | **WRITE** |
| 3.3 | Pooling is not driving the number: pooled and unweighted-per-contrast means agree to 0.0 pp (equal 9 views per contrast) | recomputation §10 | **WRITE** |
| 3.4 | Instability concentrates on small effects — per-contrast sign disagreement ranges 0%–56% | `calibrated_contrasts.csv` | **WRITE** — a single pooled number understates this; report the range alongside |
| 3.5 | The 108 views are 108 independent experiments | — | **FORBIDDEN**. 12 unique contrasts × 9 *nested* views. Descriptive rates only; no interval over the 108. |

## Fig 4 — Resolution calibration
| # | Claim | Supported by | Status |
|---|---|---|---|
| 4.1 | Resolution at R=1 is ~7.6–8.1 pp half-width, exceeding the practical band | frozen audit + analytic `2w/(NR)` | **WRITE** |
| 4.2 | R=3 brings it to ~4.4–4.7 pp, inside the band | same | **WRITE** |
| 4.3 | Analytic and bootstrap estimates agree in shape; analytic runs ~7% conservative at every R | this package vs frozen audit | **WRITE** — report both, do not pick one |
| 4.4 | The 5 pp threshold is an external or community standard | — | **FORBIDDEN**. Internally chosen. Must read "predeclared project convention". |

## Fig 5 — Held-out calibration
| # | Claim | Supported by | Status |
|---|---|---|---|
| 5.1 | Sign reliability is monotone in held-out signal-to-resolution ratio: 66.7% → 97.2% | frozen audit §5, reproduced here | **WRITE** |
| 5.2 | Practical-category reliability is **flat** (72–78%) and does not improve with ratio — a negative result | frozen audit §6, reproduced here | **WRITE** — must be reported, not buried |
| 5.3 | Calibration is genuinely held out (σ from other seeds only) | `heldout_SE_R1_pp` column | **WRITE** |
| 5.4 | Bins contain 1–6 contrasts; the [0,0.25) bin is a single contrast | CSV | **WRITE** — state bin counts on the figure |

## Claims NOT supported by any figure
| Claim | Status |
|---|---|
| Low-NFE flow policies are behaviorally equivalent to many-step ones | **NEED-E1** — 12/12 CIs include zero; absence of resolution is not equivalence |
| The NFE2 operating point transfers across task complexity | **MISSING-EVIDENCE** — 4-cube shows +4.53 pp, with a stated ceiling confound |
| Slot/environment-index assignment affects outcomes | **NEED-E1** |
| We corrected published results | **FORBIDDEN** — the corrected results were internal |
