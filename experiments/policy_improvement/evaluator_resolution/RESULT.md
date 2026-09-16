# PandaPush evaluator resolution — result

## 1. Executive summary

**Measured: the PandaPush evaluator's within-checkpoint SD is ~1.4 pp. A single
evaluation of each of two policies resolves differences of roughly 4 pp, not 1–2
pp.** Every policy effect this project has ever reported was at or below that
resolution.

The headline numbers, which answer three *different* questions and must not be
conflated:

| | value | what it answers |
|---|--:|---|
| **A. largest observed repeat shift** | **3.75 pp** | an extreme of a finite sample — **not** an SD |
| **B. within-checkpoint SD (s_w)** | **1.399 pp** pooled; 1.227 canonical, 1.552 cleanA42 | typical realization-to-realization spread |
| **C. 95% repeatability coefficient** | **±3.88 pp** pooled / **±4.30 pp** conservative | how far two *single* evaluations of one unchanged policy can differ |

**Which SD generates which number.** Two variance estimates are in play and must
not be mixed:

- **pooled s_w = 1.399 pp** — the descriptive estimate, pooled over both anchors
  (dof 10). It generates the **pooled RC95 = 1.96·√2·1.399 = ±3.88 pp**.
- **conservative s_w = 1.552 pp** — the *noisier anchor* (cleanA42) alone. It
  generates the **conservative RC95 = 1.96·√2·1.552 = ±4.30 pp** and **every
  planning table in §16–§22**.

All planning deliberately uses the noisier anchor, so recommendations stay valid
if a future checkpoint behaves like the worse-behaved of the two rather than like
the average. ±4.30 pp is **not** derived from pooled s_w = 1.399.

The prior program's shorthand — "the evaluator noise floor is 3.75 pp" — was
wrong in kind, not just degree: it reported a maximum as a standard deviation.
That phrasing is **withdrawn**. The corrected statement is B, and C is the figure
that actually governs single-evaluation comparisons.

**Operationally:**

- 40% of same-policy evaluation *pairs* differ by ≥2 pp; 13% by ≥3 pp.
- Splitting one checkpoint's own six realizations 3-vs-3 — where the true effect
  is exactly **zero** — produces apparent "treatment effects" up to **2.64 pp**,
  and ≥1 pp half the time.
- Resolving a 1 pp true difference needs **R ≈ 38 per policy (~16.81 GPU-h per
  A/B pair)**. Resolving 2 pp needs R ≈ 10 (~4.42 GPU-h).

### Classifications

| axis | classification |
|---|---|
| evaluator resolution | **EVALUATOR-RESOLUTION-CHARACTERIZED** |
| R=1 evaluation for 1–4 pp effects | **INADEQUATE** |

---

## 2. Motivation from the loss study

The completed loss program reported a terminal-mask improvement of +2.24 pp over
four fine-tuning seeds, which fell to +0.39 pp when a single extra simulator
realization was averaged in. A frozen checkpoint had been observed to shift by
3.75 pp on re-evaluation. That warning was real but imprecisely stated, and it
made every conclusion in that program provisional.

This study converts the warning into a measurement model.

---

## 3. Transparency note on the predeclared clean rule

The clean-first-step experiment **mechanically satisfied** the predeclared
directional conditions that would have yielded `CLEAN-TERMINAL-MASK-SUPPORTED`
and `TRAINER-CONFOUND-UNLIKELY` (clean mean > 0, ≥3/4 seeds non-negative,
leave-one-out means all positive). The final interpretation was nevertheless
downgraded to UNRESOLVED.

To state this precisely, and without rewriting history:

> The predeclared directional rule was mechanically satisfied. We subsequently
> measured same-checkpoint evaluator variation of comparable or larger magnitude
> than the treatment signal, invalidating an implicit resolution assumption
> behind that rule. We therefore conservatively downgraded the scientific
> interpretation while preserving the registered outcome in the record.

This was a post-hoc conservatism adjustment prompted by a newly measured
assumption violation — **not** licence to redefine rules when an outcome is
inconvenient. The present study supplies the measurement that justifies it: the
clean effect (+1.46 pp) is 0.67× the standard error of a single-realization A/B
difference.

---

## 4. The frozen evaluator

Unchanged from the loss program; this study measures it, does not improve it.

| parameter | value |
|---|---|
| scenario sets | E0 20260820, E1s 20261820, E2s 20262820, E3s 20263820, E4s 20264820 |
| episodes per set | 96 (480 total per evaluation) |
| NFE | 4 |
| solver | fixed-step forward Euler (`ConditionalFlowMatching`) |
| weights | EMA |
| success | `max_obj_dist <= 0.04` |
| policy noise | `torch.manual_seed(NOISE_SEED + bs)`, NOISE_SEED = 20260902 |
| evaluator commit (as run) | `08d77ab` |
| analysis-tooling commit | `72942a4` (evaluator and `diffuser/` byte-identical to `08d77ab`) |

---

## 5. Sources of evaluator randomness

Full table in `RANDOMNESS_AUDIT.md`. Summary: scenario states load from frozen
`.pkl` files; scenario ordering and batching are deterministic functions of the
episode count; policy diffusion noise is re-seeded before **every** batch; NFE,
solver, horizon, env count and success threshold are constants; checkpoint bytes
are SHA-verified identical across repeats.

> All audited Python-level randomness relevant to scenario selection and policy
> sampling is fixed across repeats. The remaining variation is therefore
> consistent with lower-level GPU/simulator nondeterminism (e.g. Isaac Gym
> physics and/or CUDA execution), though the current study does not isolate
> those components individually.

The measured quantity is therefore called **end-to-end evaluator realization
variability** throughout. It is *not* attributed to "physics noise".

---

## 6. Existing same-checkpoint repeat data — PRELIMINARY BASELINE

Mined before any new GPU work, from 40 catalogued artifacts
(`evaluation_registry.json`): **8 distinct checkpoints, 19 valid evaluations** —
one R=5 group and seven R=2 pairs.

| statistic | value |
|---|--:|
| pooled within-checkpoint SD | **1.313 pp** (dof 11) |
| mean \|repeat difference\| | 1.348 pp |
| median \|difference\| | 1.250 pp |
| RMS difference | 1.677 pp |
| max \|difference\| *(extreme)* | 3.750 pp |
| per-set pooled SD | E0 2.61, E1s 3.51, E2s 1.74, E3s 2.66, E4s 3.04 pp |

**Frozen as PRELIMINARY EXISTING-DATA BASELINE**, not the final estimate: the
structure (one R=5 group, seven R=2 pairs) is thin. The six-repeat anchor panel
is the primary estimator. The two agree closely (1.313 vs 1.399 pp), which is
itself reassuring.

### Provenance of the R=5 group

Three of those five evaluations came from the kwargs-defect B/C/D runs.

> These runs are invalid as distinct loss-intervention arms but valid as repeated
> evaluations of one byte-identical frozen policy.

Their preserved checkpoints were verified byte-identical to legacy A42
(`965e7e32fe19cf01`), and the evaluations themselves were uncontended with all
five sets VALID. The original failed experiment is **not** relabelled as
scientifically valid; only its evaluator realizations are reused, for a question
they legitimately answer.

Two records were excluded from repeatability estimation: a degenerate policy
(macro 0.0 — variance at the range boundary says nothing about the ~78% regime)
and a SUSPECT-flagged file.

---

## 7–8. Anchor selection and provenance

Chosen **before any new repeat outcome was observed**, on role rather than
result:

| anchor | role | SHA256 (first 16) | step |
|---|---|---|--:|
| canonical | stable pre-fine-tuning reference | `861dc34434474455` | 499000 |
| cleanA42 | predeclared clean baseline, corrected protocol | `49bbc3f5a041d32f` | 2000 |

Neither was selected for an unusual treatment effect. Full record in
`ANCHOR_MANIFEST.json`.

---

## 9–10. Six-repeat results and macro repeatability

Raw values first, before any pooled statistic.

**canonical** (SHA `861dc344…`)

| R | macro | E0 | E1s | E2s | E3s | E4s |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 0.7917 | 0.7604 | 0.7812 | 0.8438 | 0.7500 | 0.8229 |
| 2 | 0.7896 | 0.7292 | 0.8125 | 0.7917 | 0.8229 | 0.7917 |
| 3 | 0.8125 | 0.8229 | 0.8021 | 0.8125 | 0.8021 | 0.8229 |
| 4 | 0.7917 | 0.7708 | 0.7812 | 0.8125 | 0.7812 | 0.8125 |
| 5 | 0.7854 | 0.7396 | 0.8021 | 0.7917 | 0.7812 | 0.8125 |
| 6 | 0.7750 | 0.7708 | 0.8021 | 0.7708 | 0.7812 | 0.7500 |

mean 0.7910, **SD 1.227 pp**, median 0.7906, range **3.750 pp**.

**cleanA42** (SHA `49bbc3f5…`)

| R | macro | E0 | E1s | E2s | E3s | E4s |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 0.8021 | 0.7812 | 0.7812 | 0.8229 | 0.8229 | 0.8021 |
| 2 | 0.7750 | 0.8125 | 0.7292 | 0.8021 | 0.7708 | 0.7604 |
| 3 | 0.7833 | 0.7500 | 0.8021 | 0.7812 | 0.8021 | 0.7812 |
| 4 | 0.7979 | 0.7708 | 0.8021 | 0.8646 | 0.7708 | 0.7812 |
| 5 | 0.8042 | 0.8021 | 0.8021 | 0.8125 | 0.8021 | 0.8021 |
| 6 | 0.7667 | 0.7500 | 0.8021 | 0.7604 | 0.7708 | 0.7500 |

mean 0.7882, **SD 1.552 pp**, median 0.7906, range **3.750 pp**.

**Both anchors independently reproduce a 3.75 pp range.** The prior maximum was
a typical extreme for six draws, not an anomaly.

A useful incidental control: these are two genuinely *different* policies, and
their means differ by **−0.28 pp**, 95% CI [−2.03, +1.48] even at R=6 — i.e. this
benchmark cannot distinguish the canonical policy from a 2,000-step fine-tune of
it.

---

## 11. Per-E-set repeatability

| set | canonical SD | cleanA42 SD | pooled SD | max \|diff\| |
|---|--:|--:|--:|--:|
| E0 | 3.278 | 2.608 | 2.962 | 9.375 |
| E1s | 1.276 | 2.928 | 2.258 | 7.292 |
| E2s | 2.501 | 3.593 | **3.096** | 10.417 |
| E3s | 2.443 | 2.226 | 2.337 | 7.292 |
| E4s | 2.795 | 2.126 | 2.483 | 7.292 |

Per-set SDs (2.3–3.1 pp) exceed the macro SD (1.4 pp) as expected: each set
averages only 96 episodes, while the macro averages five sets. **No set is
dropped or reweighted** — this is diagnostic only. Nor is any set consistently
noisiest across both anchors: E0 is worst for canonical, E2s for cleanA42, and
E1s ranges from best (1.28) to second-worst (2.93). The per-set ranking does not
replicate, which argues against a per-set defect.

---

## 12. Episode-level repeatability — NOT AVAILABLE

No historical artifact stores episode identities or per-episode success vectors
(`evaluation_registry.json`: `episode_level_data_available: false`). Paired
same-scenario flip analysis is therefore impossible from existing data, and
passive logging was **not** added — under §15 the macro study took priority over
instrumentation that would have needed its own validation.

Consequence: the hypothesis that macro variance is driven by a small number of
borderline episodes flipping is **plausible but undemonstrated**. It is recorded
as unresolved, not asserted.

---

## 13. Within-checkpoint variance

Estimator verified explicitly (§2 of the directive): residuals about **each
checkpoint's own mean**,

```
s_w² = Σ_g Σ_r (y_gr − ȳ_g)² / Σ_g (n_g − 1)
```

| estimate | value |
|---|--:|
| canonical s_w | 1.227 pp |
| cleanA42 s_w | 1.552 pp |
| **pooled s_w** | **1.399 pp** (dof 10) |
| pooled RC95 = 1.96·√2·1.399 | **±3.88 pp** |
| conservative RC95 = 1.96·√2·1.552 | **±4.30 pp** |
| variance ratio | 1.60 |
| heteroscedastic (ratio > 3)? | no |
| **conservative s_w for planning** | **1.552 pp** (= cleanA42, the noisier anchor) |

**The four estimates, and what each drives:**

| estimate | value | used for |
|---|--:|---|
| canonical anchor SD | 1.227 pp | descriptive only |
| cleanA42 anchor SD | 1.552 pp | **= the conservative planning SD** |
| **pooled within-checkpoint SD** | **1.399 pp** | descriptive headline; pooled RC95 ±3.88 pp |
| **conservative planning SD** | **1.552 pp** | conservative RC95 ±4.30 pp; §16–§22 tables |

Cross-checks:
- Pooling around a *grand* mean would give 1.341 pp — inflated by the
  between-anchor policy difference. **Not used.**
- The difference-based estimator, SD(Y₁−Y₂)/√2 = 1.346 pp, agrees within 4%.
- The preliminary existing-data baseline, 1.313 pp, agrees within 7%.

Variances differ by 1.6× — not enough to call heteroscedastic, but the **noisier
anchor (1.552 pp) drives every planning recommendation** regardless, for safety.

---

## 14. Same-policy null difference distribution

| anchor | pairs | mean \|d\| | max \|d\| |
|---|--:|--:|--:|
| canonical | 15 | 1.389 pp | 3.750 pp |
| cleanA42 | 15 | 1.889 pp | 3.750 pp |

Descriptive tail probabilities, pooled over both anchors (30 pairs):

| threshold | P(\|difference\| ≥ threshold) |
|---|--:|
| ≥1 pp | 0.633 (19/30) |
| ≥2 pp | **0.400** (12/30) |
| ≥3 pp | 0.133 (4/30) |
| ≥4 pp | 0.000 (0/30) |

**These 15 pairs per anchor are not 15 independent observations** — they share
realizations. Bootstrapping at the **realization** level gives a 95% interval for
P(\|d\| ≥ 2 pp) of **[0.067, 0.567]**: the point estimate of 0.40 is itself very
uncertain at n=6. Treat these as descriptive, not precise tail probabilities.

---

## 15. Pseudo-treatment experiment

Split each anchor's own six realizations 3-vs-3. The true effect is **exactly
zero** by construction. C(6,3)=20 directed allocations reduce to **10 unique
undirected splits**; both sign labelings are reported since which half is called
"B" is arbitrary. These are a finite enumeration, not independent experiments.

| anchor | unique splits | SD of pseudo-effect | max \|pseudo-effect\| | ≥1 pp | ≥2 pp |
|---|--:|--:|--:|--:|--:|
| canonical | 10 | 1.028 pp | 1.528 pp | 0.500 | 0.000 |
| cleanA42 | 10 | 1.300 pp | **2.639 pp** | 0.500 | 0.100 |

**Half of all zero-effect splits produce an apparent effect ≥1 pp, and the
largest reaches 2.64 pp** — comfortably inside the range of every effect the loss
program reported.

---

## 16. Uncertainty vs R

**All rows below use the conservative s_w = 1.552 pp (the noisier anchor,
cleanA42), not the pooled 1.399 pp.** Assumes independent evaluator realizations
between the two policies (see §17 caveat). For reference, the pooled-SD
equivalents are ~10% tighter: at R=3 the half-width is ±2.24 pp pooled versus
±2.48 pp conservative.

| R | SE(mean) | SE(B−A) | 95% half-width | GPU-h per A/B |
|--:|--:|--:|--:|--:|
| 1 | 1.552 | 2.194 | **4.301** | 0.445 |
| 2 | 1.097 | 1.552 | 3.041 | 0.890 |
| 3 | 0.896 | 1.267 | 2.483 | 1.335 |
| 4 | 0.776 | 1.097 | 2.150 | 1.780 |
| 5 | 0.694 | 0.981 | 1.923 | 2.225 |
| 6 | 0.633 | 0.896 | 1.756 | 2.670 |

This quantifies exactly how much the loss program's R=2 averaging helped: it cut
the half-width from 4.30 to 3.04 pp (conservative s_w throughout) — still far wider than the +0.39 pp effect it
was used to evaluate.

---

## 17–18. Planning

**Precision-based** (conservative s_w = 1.552 pp):

| target 95% half-width on B−A | R per policy | GPU-h per A/B |
|---|--:|--:|
| ≤0.5 pp | 74 | 32.74 |
| ≤1.0 pp | 19 | 8.41 |
| ≤1.5 pp | 9 | 3.98 |
| ≤2.0 pp | 5 | 2.21 |

**Power-based** (α=0.05, power=0.80, approximation
R ≥ 2·((1.96+0.84)·s_w/δ)²):

Conservative s_w = 1.552 pp; R rounded **up** to whole realizations.

| true effect δ | R per policy | GPU-h per A/B |
|---|--:|--:|
| 1 pp | 38 | 16.81 |
| 2 pp | 10 | 4.42 |
| 2.24 pp *(= the R=3 half-width)* | 8 | 3.54 |
| 3 pp | 5 | 2.21 |
| 4 pp | 3 | 1.33 |

**Precision and power are different questions.** R=3 gives a ±2.48 pp *confidence
interval*; it does **not** give 80% power to detect a 2.48 pp effect. Detecting a
3 pp effect at 80% power needs R=5, and a 2 pp effect needs R=10.

Phrasing matters here. Not "R=4 detects a 2 pp effect", but:

> Under the measured variance and the stated independence/normal approximation,
> approximately R=10 realizations per checkpoint would provide 80% planning power
> for a true 2 pp difference, and R=19 would give a 95% CI half-width of 1 pp.

The variance estimate is itself uncertain (n=6 per anchor), so these R values
carry real uncertainty.

**Independence caveat:** these formulas assume evaluator realizations for the two
policies are independent. Paired *training* seeds do not make *evaluator*
realizations paired — they are different things, and pairing training seeds buys
no variance reduction here.

---

## 19. GPU cost vs precision

Measured cost this study: **2.6544 GPU-h / 12 evaluations = 0.2212 GPU-h per
five-set evaluation.** The cost column above uses that measured figure, not an
estimate.

Arithmetic, shown explicitly:

```
GPU-h per A/B pair = R x 2 policies x 0.2212 GPU-h per five-set evaluation

  R=3  (routine default)   3 x 2 x 0.2212 =  1.33 GPU-h
  R=5  (80% power, 3 pp)   5 x 2 x 0.2212 =  2.21 GPU-h
  R=10 (80% power, 2 pp)  10 x 2 x 0.2212 =  4.42 GPU-h
  R=19 (1 pp precision)   19 x 2 x 0.2212 =  8.41 GPU-h
  R=38 (80% power, 1 pp)  38 x 2 x 0.2212 = 16.81 GPU-h
```

Note: an earlier draft quoted **16.9** GPU-h for the R=38 row. That used the
*pre-registered* cost estimate of 0.2225 GPU-h/eval (38x2x0.2225 = 16.91). All
cost figures now use the **measured** 0.2212 GPU-h/eval, giving 16.81.

For context: one A/B comparison at R=19 (1 pp precision) costs 8.41 GPU-h — more
than the entire loss program's corrected rerun plus replication combined.

---

## 20. Interpretation of prior loss results

Re-expressed against measured resolution. SE(B−A) at R=1 is **2.194 pp**.

| effect | size | \|d\|/SE(R=1) | R for 80% power |
|---|--:|--:|--:|
| terminal mask, legacy R=1, 4-seed | +2.24 | 1.02 | 8 |
| terminal mask, legacy R=2 | +0.39 | 0.18 | 249 |
| terminal mask, clean first-step | +1.46 | 0.67 | 18 |
| denominator B′−A | +1.82 | 0.83 | 12 |
| B′−B | −0.42 | 0.19 | 215 |
| λ=2 vs masked λ=1 | −1.88 | 0.86 | 11 |
| λ=5 vs masked λ=1 | −1.67 | 0.76 | 14 |

**Every effect the loss program reported sits at or below the resolution of a
single evaluation per policy** (|d|/SE ≤ 1.02). The largest, +2.24 pp, would have
needed R≈8 per policy to measure with useful precision; it was measured at R=1.

No hypothesis test is re-run and no effect is reclassified as positive. The
operational reading is simply: those experiments were under-powered by roughly an
order of magnitude in evaluation budget.

---

## 21. Recommended future PandaPush evaluation standard

**Adopt R = 3 as the default, and treat effects below ~2.5 pp as below practical
resolution.**

| element | specification |
|---|---|
| scenario sets | the same five fixed E sets, 96 episodes each |
| realizations per checkpoint | **R = 3** independent evaluator realizations |
| reporting | preserve all R per-realization results; average within checkpoint |
| per-set | report all five sets separately; never drop or reweight |
| training comparisons | paired fine-tune seeds, but do **not** assume evaluator pairing |
| uncertainty | report SE(B−A) ≈ s_w·√(2/R) with the conservative s_w = 1.552 pp, and the 95% half-width |
| precision at R=3 | half-width **±2.48 pp** conservative (**±2.24 pp** pooled), cost **1.33 GPU-h** per A/B pair |

> Using the noisier anchor for conservative planning, R=3 gives an approximate
> 95% B−A half-width of **±2.48 pp**. Using the pooled variance instead it is
> **±2.24 pp**. The conservative figure is the one recommended for planning.

Why R=3 rather than the R=19 that 1 pp precision demands: this is a cost-aware
choice, as directed. R=3 narrows the interval from ±4.30 to ±2.48 pp for 1.33 GPU-h, which is
affordable for routine screening. R=19 costs 8.41 GPU-h per comparison — justified
only for a final confirmatory result, never for exploration.

**The honest corollary, stated carefully.** R=3 is a practical default that
substantially improves precision at manageable cost, but it is **not** sufficient
to make 1–3 pp effects automatically confirmatory. Effects below roughly 3 pp
should generally be treated as below practical routine resolution unless
additional repeats are budgeted.

Concretely, at R=3 a 3 pp observed difference has a 95% interval of roughly
[0.5, 5.5] pp — consistent with a real effect, but not confirmation of one, and
80% power at 3 pp requires R=5. R=3 is a **screening** tier: it is for deciding
what deserves a confirmatory run, not for declaring results.

---

## 22. Is the benchmark fit for small-effect research?

**No, not at practical cost.** Resolving 1–2 pp requires R=19–38 per policy
(8.41–16.81 GPU-h per single A/B comparison) on the RTX 4080. Multiply by four
training seeds and a realistic experiment costs 34–67 GPU-h of *evaluation alone*.

The benchmark is well suited to effects ≥3 pp. For smaller effects it is
expensive and poorly resolved under the current protocol. No redesign is proposed
here — that is the recommended next action.

---

## 23. More episodes vs more realizations — UNRESOLVED

Cannot be answered from available data. Deciding between more episodes per set
and more realizations requires knowing whether macro variance is dominated by
finite-sample binomial noise over 480 episodes or by realization-level effects
that resample equally at any episode count. **That decomposition needs
episode-level success vectors, which do not exist in any artifact** (§12).

A binomial floor at p≈0.79, n=480 would be √(p(1−p)/480) = 1.86 pp per
evaluation — *larger* than the measured 1.40 pp macro SD. That is only an
apparent contradiction: the five scenario sets are fixed and reused, so
realizations are not independent draws of new scenarios, and the simple
independent-binomial model does not apply. The comparison is recorded but
**not** used to claim the variance is or is not finite-sample dominated.

---

## 23b. Figures

Six figures in `figures/`. They are **not committed** — the repo `.gitignore`
excludes `*.png` — but regenerate exactly from the committed
`ANCHOR_ANALYSIS.json` with:

```
python experiments/policy_improvement/evaluator_resolution/make_figures.py
```

fig1 same-checkpoint realizations - fig2 per-set vs macro repeatability -
fig3 same-policy pairwise differences - fig4 pseudo-treatment null -
fig5 uncertainty vs R - fig6 cost vs precision.

---

## 24. Red-team

| check | result |
|---|---|
| both anchors byte-identical across repeats? | **yes** — one SHA each, recorded in manifest |
| EMA vs model correct? | **yes** — `control()` evaluates `exp.ema` |
| NFE=4, exact E seeds, n=96, same success metric? | **yes**, all 12 |
| same evaluator commit? | **yes** — all 12 ran under `08d77ab`; later analysis commits left `phase2_diagnostics.py` and `diffuser/` untouched (verified by `git diff`) |
| foreign GPU contention in any accepted run? | **none** — 0 MiB mid-set, all 12 |
| invalid artifacts included? | **no** — quarantine rule enforced |
| historical protocol accidentally pooled? | **no** — n≠96 records bucketed separately |
| training-seed variance called evaluator variance? | **no** — only byte-identical checkpoints used |
| dependent pairwise samples treated as independent? | **no** — flagged, bootstrap at realization level |
| maximum range called SD? | **no** — A/B/C distinguished explicitly |
| bootstrap unit correct? | **yes** — resamples realizations, not pairs |
| outcome-dependent anchor selection? | **no** — chosen on role, before any new repeat |
| temporal drift? | **not distinguishable from chance** — permutation p=0.31 (canonical), 0.61 (cleanA42) |
| anchor heteroscedasticity? | ratio 1.60, below the 3× flag; noisier anchor used anyway |
| recommended R robust to the noisier anchor? | **yes** — all planning uses s_w = 1.552 pp |
| compute estimate measured, not guessed? | **yes** — 0.2212 GPU-h/eval from this study's own log |

---

## 25. Compute ledger — EVALUATOR-RESOLUTION program

| category | GPU-h |
|---|--:|
| valid canonical repeats (6) | ~1.327 |
| valid cleanA42 repeats (6) | ~1.327 |
| invalid / contended | **0.000** |
| invalid / other | **0.000** |
| **total** | **2.6544 of 3.00 cap** |

Fresh ledger; no borrowing from any prior allocation. CPU analysis uncharged.
Zero wasted GPU time in this program.

---

## 26. Claim ledger

**ESTABLISHED**
- Within-checkpoint evaluator SD: **1.399 pp pooled** (1.227 canonical, 1.552
  cleanA42), from 6 realizations of each of two byte-identical checkpoints.
- 95% repeatability coefficient for two single evaluations: **±3.88 pp** from
  pooled s_w, **±4.30 pp** from the conservative (noisier-anchor) s_w. Planning
  uses the conservative figure.
- Same-policy pairs differ by ≥2 pp **40%** of the time (descriptive; bootstrap
  interval [0.067, 0.567]).
- Zero-effect 3-vs-3 splits produce apparent effects up to **2.64 pp**.
- All Python-level randomness in the evaluator is fixed across repeats.
- Measured evaluation cost: **0.2212 GPU-h** per five-set evaluation.
- The kwargs-defect B/C/D evaluations are valid repeat realizations of one
  byte-identical policy (invalid as loss arms).

**PRELIMINARY**
- Existing-data baseline s_w = 1.313 pp (thin structure: one R=5, seven R=2).
- Recommended R=3 default — derived from n=6 per anchor, so the variance estimate
  underlying it is itself uncertain.

**NULL / NEGATIVE**
- No temporal drift distinguishable from chance.
- No E-set consistently noisiest across both anchors.
- canonical vs cleanA42 macro difference −0.28 pp, CI [−2.03, +1.48] at R=6.

**WITHDRAWN**
- **"3.75 pp is the evaluator SD / noise floor."** It was the largest observed
  same-checkpoint macro shift in a finite sample. The SD is ~1.4 pp pooled; the 95%
  repeatability coefficient is ±3.88 pp pooled / ±4.30 pp conservative. Notably, both new anchors independently
  produced a 3.75 pp range, confirming it as a typical *extreme* for six draws —
  which is exactly why it must not be reported as an SD.

**UNRESOLVED**
- Whether Isaac Gym physics or CUDA nondeterminism dominates (not isolated).
- Whether macro variance is driven by a few borderline episodes (no episode-level
  data).
- More episodes vs more realizations (same reason).

---

## 27. Limitations

n=6 per anchor and only two anchors. Both anchors are competent policies near
78% success; variance may differ elsewhere in the performance range (the
excluded macro-0.0 policy is direct evidence that it does at the boundary). The
normal/independence approximations behind RC and the planning tables are
untested at this sample size. No episode-level decomposition was possible.

---

## 28. Final classifications

**EVALUATOR-RESOLUTION-CHARACTERIZED** — the primary question is answered with a
measured variance model, verified estimators, and cost-aware planning tables.

**R1-SMALL-EFFECT-EVALUATION: INADEQUATE** — for policy differences in the 1–4 pp
regime, a single evaluation per policy has SE(B−A) = 2.19 pp and a 95%
half-width of 4.30 pp (conservative s_w; ±3.88 pp on the pooled estimate). This
is a statement about measurement resolution, not
about policy quality.

---

## 29. One next experiment (not run)

**Add episode-level success logging to the evaluator and re-run the two anchors
at R=3 each, to decompose macro variance into per-episode instability versus
realization-level effects.**

**Purpose, stated precisely:** determine whether the ~1.4 pp realization
variability is driven mainly by a small set of nominally identical episodes
flipping outcomes, which would inform whether increasing episode count can
improve resolution more cheaply than repeated full evaluator realizations.

**Precondition — the logger must be demonstrated passive before it is used.**
Prior to any scientific run it must be shown not to alter:

- RNG consumption (identical draw counts and ordering)
- simulator stepping
- action calls
- episode ordering
- aggregate evaluator behaviour (macro and per-set results byte-identical to an
  unlogged run of the same checkpoint)

The last item is the decisive test: a logged and an unlogged evaluation of one
frozen checkpoint must agree exactly, not merely closely.

Rationale: the single highest-value unknown is *where* the 1.4 pp comes from. If
a small set of borderline episodes flips, the fix is cheap — more episodes, or
reporting a stability-weighted metric — and 1 pp resolution becomes affordable.
If variance is spread across all episodes, no episode-count increase will help
and only repeated realizations (or a different benchmark) will. That fork
determines whether small-effect policy research on PandaPush is practical at all,
and it costs ~1.3 GPU-h plus a logger that must be validated as passive (no RNG
calls, no ordering or stepping changes).

This is deliberately *not* "go test another policy idea": doing so before knowing
the answer is what produced the last program's vanishing effects.

---

## 30. Final commit

This result is recorded at commit **`4596515`** (numerical-consistency audit;
scientific content first recorded at `435d3b1`) ("evaluator-resolution: measure
PandaPush repeatability, complete program"). The evaluations themselves ran
under `08d77ab`.

**PROGRAM CLOSED.**
