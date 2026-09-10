# MeanFlow Failure Diagnostics — saved 20k checkpoint, no retraining

GPU cost **~0.06 GPU-h** (cap 0.5). Checkpoint `imf_viability/…` at internal step
18981 (EMA), Standard Flow control = Arm A `state_final.pt` at step 20000 (EMA),
evaluated on **identical replay batches** with matched RNG seeds.

**Verdict up front: of the three candidate mechanisms, only the boundary /
interval condition shows a clear failure. Magnitude-starvation and
representation-degeneration are NOT supported.**

---

## A. IVC / boundary underdetermination — **SUPPORTED**

**Exact quantity.** Our implementation is *boundary* iMF: it parameterizes the
instantaneous velocity as `v(z,t) := u(z,t,t)`, i.e. the average velocity over a
zero-length interval. The mathematically required relation, and the one MVP/IVC
(arXiv:2602.13810 Eq. 15) enforces explicitly, is therefore

```
u_θ(z, t, t)  ==  v*(z, t)  =  (noise − data)      [our repo's target sign]
```

Measured by calling `_call_interval_model(zt, cond, t·time_scale, interval=0)` and
comparing to `z − x1`, over free (unconditioned) coordinates only.

| quantity | value | n |
|---|--:|--:|
| residual norm (mean) | **25.74** | 192 |
| cosine(pred, target) | **0.738** | 192 |
| **norm ratio pred/target** | **0.481** | 192 |

**Interpretation.** At the boundary the model outputs **less than half** the
required magnitude (0.481) and is **22° further off in direction** than the same
model's ordinary (r<t) predictions (cos 0.738 vs 0.914 pooled). The boundary
condition that our parameterization *depends on* is not satisfied.

**Status: SUPPORTS the IVC/boundary mechanism.** This is a measured property of
the exact relation our code requires, not a generic endpoint metric.

**Caveat:** this is an *association*, not a demonstrated cause of the 0/96 control
result. MVP/IVC itself never measured its failure mode empirically (theory only),
so we are the first to test it here — and a fix would have to be shown to help,
not assumed.

## B. OMP velocity-magnitude / directional starvation — **NOT SUPPORTED**

**Exact quantity.** OMP (arXiv:2512.19347 Eq. 9) predicts
`∂L/∂α = 2ρρ* sin α`, so as the **target** magnitude `ρ*` → 0 the directional
gradient vanishes and the model should shrink its output instead of rotating it.
Signature: at small target norm, **norm ratio ≪ 1 together with low cosine**.

Binned by target-norm quartile, **per semantic block** (48 samples per cell):

**MeanFlow — action block**

| target-norm bin | target | pred | ratio | cos | ang err | abs err |
|---|--:|--:|--:|--:|--:|--:|
| smallest | 3.147 | 3.126 | **0.994** | **0.943** | 11.7° | 0.1218 |
| 2nd | 3.807 | 3.678 | 0.967 | 0.964 | 7.4° | 0.0967 |
| 3rd | 4.309 | 4.187 | 0.972 | 0.957 | 9.4° | 0.1225 |
| largest | 5.168 | 4.892 | 0.948 | 0.958 | 9.7° | 0.1562 |

**MeanFlow — state block**: ratio 0.932–0.953, cos 0.897–0.913, flat across bins.

**Standard Flow control — action block**: ratio 0.960–0.988, cos 0.968–0.977.

**Interpretation.** The predicted signature is **absent, and in fact reversed**:
the *smallest*-target action bin has the **highest** norm ratio (0.994) and a high
cosine (0.943). There is no regime where magnitude collapses while direction goes
unlearned. Standard Flow is slightly better in direction (cos ~0.97 vs ~0.95) but
the two are qualitatively the same shape.

**Status: DOES NOT SUPPORT the OMP mechanism. Per §3 of the brief, do not import
`L_DA`.**

## C. DM1 representation degeneration — **NOT SUPPORTED** (and one artifact retracted)

**Predeclared locations** (not fished): time embedding, early block, middle block,
late block. **Battery**: effective rank (entropy of singular spectrum), pairwise
cosine, per-feature variance, feature norm, top-singular-value fraction.

**Transformer blocks — MeanFlow vs Standard Flow:**

| location | eff rank MF | eff rank SF | pairwise cos MF | pairwise cos SF |
|---|--:|--:|--:|--:|
| early block | 93.12 | 89.64 | 0.493 | 0.440 |
| middle block | 101.85 | 90.62 | 0.472 | 0.495 |
| late block | 105.98 | 85.86 | 0.805 | 0.807 |

MeanFlow's block representations are **not** degenerate — effective rank is in fact
*higher* than Standard Flow's at every block, and pairwise cosine is comparable.

**Model outputs:** effective rank **187.56** (MF) vs **187.42** (SF) on identical
inputs — indistinguishable.

### Retracted measurement — my probe was wrong

My hooked battery initially reported the MeanFlow **time embedding** at effective
rank **1.06** with top singular value carrying 98.9% of the spectrum, versus 28.20
for Standard Flow. That looked like a dramatic, precisely-localized collapse.

**It is an artifact of my measurement, and I withdraw it.** Two controls:

1. **Controlled probe** — call `time_mlp` directly on an identical fixed `t` grid
   (192 points, no hooks, same architecture, same `time_scale=1000`):

   | model | eff rank | top-sv frac | pairwise cos |
   |---|--:|--:|--:|
   | MeanFlow | **138.56** | 0.047 | 0.870 |
   | Standard Flow | **62.93** | 0.245 | 0.671 |

   The ordering **reverses** — MeanFlow's time embedding is *richer*, not collapsed.

2. **Why the hook lied** — MeanFlow's loss calls `time_mlp` **several times per
   training step** (at `t`, at the perturbed times, and at the boundary `r=t`).
   The hook pooled all of those calls: 24 calls yielded 192 rows but only **96
   unique `t` values**, and the near-duplicate boundary/interval rows dominate.
   Pooling duplicated rows deflates effective rank mechanically.

**Status: DOES NOT SUPPORT the DM1 mechanism.** Do not import a dispersive loss.
Note DM1 itself never measured representation collapse either — it asserted it and
validated only downstream — so there was no published metric to reproduce.

## D. EMA — control only

Known and unchanged: LIVE ≈ EMA (objective 0.2606 vs 0.2613; replay 0.1275/0.2275
vs 0.1164/0.2283). Not re-investigated.

## E. Action branch — control only

Known and unchanged: action gradients alive (LIVE 0.0852, EMA 0.0720). **The
failure is not a dead action branch.**

---

## Summary table

| mechanism | exact quantity | MeanFlow | Standard Flow | verdict |
|---|---|--:|--:|---|
| **IVC / boundary** | `u(z,t,t)` vs instantaneous target | ratio **0.481**, cos **0.738** | n/a (no interval path) | **SUPPORTS** |
| **OMP / starvation** | norm ratio & cos binned by target norm | 0.994 ratio at smallest bin, cos 0.943 | 0.988 / 0.972 | **DOES NOT SUPPORT** |
| **DM1 / representation** | eff rank, pairwise cos at 4 predeclared sites | blocks 93–106, output 187.6 | blocks 86–91, output 187.4 | **DOES NOT SUPPORT** |

**Single most important finding:** the MeanFlow checkpoint's pooled velocity
predictions are **almost indistinguishable from Standard Flow's** (norm ratio 0.943
vs 0.940, cosine 0.914 vs 0.917, effective rank 187.6 vs 187.4) — yet it scored
0/96 on all five control sets while Standard Flow @500k is a competent policy.
So the failure is **not** visible in ordinary `r<t` velocity quality. It is
visible **only at the boundary `r=t`**, which is exactly the quantity our
one-step deployment path depends on.

**Claim discipline.** We do **not** claim "MeanFlow collapsed." Two of the three
literature mechanisms are affirmatively **not** present. The boundary result is a
measured inconsistency that is *consistent with* the IVC diagnosis; it is not a
demonstrated cause of the control failure, and testing that would require a
retrain we have not authorized and do not propose.
