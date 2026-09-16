# Presentation figures

Three slide-ready figures for the evaluator-resolution study.

| file | shows |
|---|---|
| `pres1_same_policy_repeatability.png` | six macro measurements per frozen anchor, with mean lines; canonical SD 1.23 pp, clean-A42 SD 1.55 pp, pooled 1.40 pp |
| `pres2_pseudo_treatment_null.png` | apparent B−A effects from 3-vs-3 splits of one checkpoint's own runs, where the true effect is exactly zero; up to 2.64 pp |
| `pres3_precision_vs_repeats.png` | 95% B−A half-width for R=1..6 under both the pooled (1.399 pp) and conservative (1.552 pp) SDs, with R=3 marked |

Regenerate with:

```
python experiments/policy_improvement/evaluator_resolution/presentation_figures/make_presentation_figures.py
```

**No new science.** Every plotted value is read back from the committed
`ANCHOR_ANALYSIS.json`. The only arithmetic performed is the half-width formula
already documented in `RESULT.md` section 16 (`1.96 · s · √(2/R)`), applied to
the committed `s_w` values. Nothing is recomputed from raw diagnostics and no
GPU work is involved.

Figure 2 plots the stored pseudo-treatment array, which holds all 10 unique
3-vs-3 splits per anchor under both sign labelings. This is a complete
enumeration of a single six-run panel, **not** a set of independent statistical
samples, and the figure says so on its face.

Colors are categorical slots 1 and 2 of the validated reference palette
(`#2a78d6`, `#eb6834`). Validated against the white surface: worst CVD ΔE 24.7
(target ≥8), normal-vision ΔE 33.6 (floor 15), contrast 4.42:1 and 3.20:1 (≥3:1).

The `.png` files are gitignored repo-wide; the generator is committed.
