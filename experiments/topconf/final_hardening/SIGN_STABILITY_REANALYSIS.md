# Sign stability, rebuilt rigorously (§16-§18)

## The hierarchy — stated explicitly

```
task regime (3-cube, 4-cube)
└── policy contrast (NFE4-NFE1, NFE4-NFE2, NFE2-NFE1)
    └── training checkpoint (42, 43, 44)
        └── reconstructed single-realization view (3×3 = 9 per comparison)
```

**12 calibrated comparisons; 108 reconstructed views.** The 108 are **NOT
independent observations** — they are nested reconstructions inside 12
experiments, and only 12 (really 3 checkpoints × 4 contrasts) are independent at
the level that matters. Every aggregate below is descriptive.

## Per-comparison results

| task | contrast | seed | \|Δ_cal\| pp | ratio to noise | correct sign |
|---|---|--:|--:|--:|--:|
| 3-cube | NFE4−NFE1 | 42 | 9.03 | 1.51 | 9/9 |
| 3-cube | NFE4−NFE1 | 43 | 2.78 | 0.46 | 7/9 |
| 3-cube | NFE4−NFE1 | 44 | 6.25 | 1.04 | 9/9 |
| 3-cube | NFE4−NFE2 | 42 | 0.69 | 0.12 | 4/9 |
| 3-cube | NFE4−NFE2 | 43 | 1.04 | 0.17 | 4/9 |
| 3-cube | NFE4−NFE2 | 44 | 1.74 | 0.29 | 6/9 |
| 3-cube | NFE2−NFE1 | 42 | 7.29 | 1.22 | 9/9 |
| 3-cube | NFE2−NFE1 | 43 | 2.08 | 0.35 | 6/9 |
| 3-cube | NFE2−NFE1 | 44 | 3.47 | 0.58 | 7/9 |
| 4-cube | NFE4−NFE2 | 42 | 7.29 | 1.22 | 8/9 |
| 4-cube | NFE4−NFE2 | 43 | 3.47 | 0.58 | 7/9 |
| 4-cube | NFE4−NFE2 | 44 | 2.78 | 0.46 | 6/9 |

Overall **82/108 correct = 75.9%**, i.e. **24.1% sign disagreement**.

## §17 — the reliability curve (this is the key upgrade)

The obvious objection is *"of course near-zero effects flip sign."* That is true,
and it is why the raw 24% is weak on its own. The defensible result is the
**relationship** between effect size and sign reliability.

Noise scale: SD of a single-realization arm mean = **4.23 pp** (measured from the
R=8 same-arm bank), so SD of a single-realization Δ ≈ **5.98 pp**.

| \|Δ_cal\| / σ_Δ | comparisons | correct sign |
|---|--:|--:|
| [0.0, 0.5) | 6 | **61.1%** (33/54) |
| [0.5, 1.0) | 2 | 77.8% (14/18) |
| [1.0, 1.5) | 3 | **96.3%** (26/27) |
| ≥ 1.5 | 1 | **100%** (9/9) |

**Sign reliability is a clean monotone function of the effect/noise ratio, and it
collapses toward coin-flipping below ratio ≈ 0.5.** Both task regimes contribute
points across the range, so the curve is not an artifact of one setting.

This converts an anecdote into an empirical rule a practitioner can use: *estimate
your evaluator's σ, then know that effects below ~0.5σ are not sign-reliable at
R=1.*

## §18 — practical-conclusion flips (more consequential than sign)

Predeclared categories using the established ±5 pp practical threshold:
"A better" (< −5 pp), "indistinguishable", "B better" (> +5 pp).

**27/108 = 25.0%** of reconstructed single-realization views land in a different
practical category than the calibrated analysis. Per-comparison disagreement
ranges 1/9 to 4/9.

This matters more than sign: a category flip changes *what you would deploy*, not
merely the direction of an arrow.

## Honest limitations

- The 108 views are nested, not independent; no inferential interval is computed
  over them.
- Only 12 calibrated comparisons exist, so the reliability curve has few points
  per bin (1-6 comparisons).
- σ_Δ is estimated from one arm (seed 42, NFE4, 3-cube) and assumed to transfer;
  4-cube within-arm spread (6.6 pp) is broadly consistent but was measured at R=3.
