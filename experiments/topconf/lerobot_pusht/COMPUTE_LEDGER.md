# Compute ledger — Stage 1 external policy (lerobot/diffusion_pusht)

Hard cap: **0.80 GPU-h**. GPU: single RTX 4080. No training.

| # | purpose | commit | detail | wall (s) | GPU-h | changed decision? |
|---|---|---|---|--:|--:|---|
| 1 | harness smoke (broken: lerobot 0.4.4) | d5863bb | 3 eps, nfe100 | 66 | 0.018 | **yes** — exposed the normalization-buffer bug |
| 2 | harness smoke #2 (stats-injection attempt) | d5863bb | 3 eps, nfe100 | 81 | 0.023 | yes — proved the 0.4.4 API refactor, forced version pin |
| 3 | harness validation (lerobot 0.3.2) | d5863bb | 10 eps, seeds 1000-1009 | 223 | 0.062 | **yes** — GATE PASSED |
| 4 | alignment offset scan | d5863bb | 60 cond, nfe100 | ~15 | 0.004 | **yes** — confirmed executed slice `P[1:9]` |
| 5 | offline action-error sweep | d5863bb | 200 cond x K=4 x 8 budgets | 67 | 0.019 | pending analysis |
| 6 | closed-loop screen | d5863bb | 8 budgets x 50 frozen eps | pending | ~0.66 | pending |
| | **total** | | | | **~0.79** | vs cap 0.80 |

Downloads: model 1,050,862,408 B + `lerobot/pusht` dataset; **1021 MB total on
disk**, cap 1.20 GB.

Earlier project stages (for context, not counted against this cap): NFE1-vs-NFE4
R=3 0.693 GPU-h; NFE2-vs-NFE4 R=3 0.761; noise-floor calibration 0.51;
self-state diagnostic 0.11.
