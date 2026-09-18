# Tal-followup program — result

**Status: PARTIAL — blocked on GPU contention from a foreign user job.**
Sections marked ⏳ await GPU work that cannot currently run. Nothing below is
written ahead of its data.

**What is complete and matched:** the 5,000-step pilot pair (UWM vs matched
diagonal control) and its full Stage-E analysis. Every finding in §7 rests on
that intact pair.

**What is blocked:** the §23 extension to 10,000 steps completed for the
treatment arm but was stopped at step 2261/5000 for the control when a foreign
GPU job appeared. The matched 10k comparison is therefore unavailable —
reporting a 10k treatment against a 5k control would confound "trained
off-diagonal" with "trained longer". Details and resumption rule in
`CONTENTION_GATE.md`.

Central question (Tal): **what is action denoising actually buying us?**

---

## 1. Executive summary

**Tal's question — what is action denoising buying us? — is not yet answered at
the control level, but three registered stages resolved on evidence and one
produced a decisive mechanistic result.**

### The headline finding

Training EC-Diffuser on **independent state/action diffusion times** (faithful
UWM, `p_diag = 0.0`) genuinely adapts the model to the off-diagonal regime,
against a matched diagonal control that differs in **exactly one value**:

| metric (action block) | UWM | matched control |
|---|--:|--:|
| off-diagonal / diagonal error | **1.558** | 3.003 |
| corr(\|t_s − t_a\|, error) | **+0.356** | +0.692 |
| inverse-model corner (state clean, action noisy) | **0.2475** | 1.5941 (**6.4×**) |
| forward-dynamics corner | **0.2737** | 1.0833 (**4.0×**) |

The control reproduces the closed warm-start branch's non-adaptation signature;
the UWM arm roughly halves it. **The out-of-distribution-time confound that
invalidated that branch is removed by training, not assumed away.**

### The finding that cuts against the interesting hypothesis

**On both arms, the plain diagonal is the best inference path** for predicting
the executed action, at equal model calls. The UWM arm beats the control on
every non-diagonal path, but non-diagonal paths are 3–16× worse than the
diagonal on the same model.

> Offline, **"state-leading inference improves action prediction" is NOT
> supported.** Independent-time *training* buys off-diagonal *robustness*; it
> does not buy a better operating point.

### The methodological finding that may outlast both

`corr(dist, err)` sat at **+0.695 / +0.705 / +0.695** through step 2000 —
indistinguishable from the never-adapted control's **+0.692** — and only
collapsed to +0.356 between 2000 and 5000.

> At the 2,000-step budget the prior loss program used, this experiment would
> have concluded that off-diagonal training does not work. **Adaptation is late
> and nonlinear.**

### Stages closed on evidence, without spending their budget

| stage | outcome |
|---|---|
| F — harder multi-object task | **BLOCKED**: no pretrained policy exists for any other task; creating one costs ~33 GPU-h vs a 10.0 cap |
| G — register implementation | **NOT JUSTIFIED**: no artifact found (max/mean ≤ 2.53 vs Darcet's ~10×) |
| H — action-latent bottleneck | **NOT AUTHORIZED** by its own predeclared criterion (75.7% vs required ≥99%) |

### Not yet answered

The **closed-loop** comparison — the only thing that can support a control claim
(§48) — has not run. Nothing here is a performance claim.

---

## 2. What was audited, and what changed because of it

### UWM is arXiv:2504.02792 (RSS 2025)

Three findings changed the design before any GPU was spent:

| finding | consequence for us |
|---|---|
| UWM is **DDPM with t=0 CLEAN** — the opposite of our flow convention | corner semantics invert; a naive port would have inverted every endpoint |
| UWM trains on the **plain product measure with NO diagonal mass** | our inherited code defaulted to `p_diag=0.5`; **faithful UWM is 0.0**, and the mixture is demoted to the §20 rescue arm |
| UWM's time embedding is a **joint MLP over the concatenated pair** | our zero-parameter construction diverges here — declared, with the tradeoff stated below |
| UWM **trains the 2D interior but never traverses it** | interior-path traversal is **unclaimed prior art**, and is our Stage E |

### Registers (arXiv:2309.16588) — the prior is negative, and that is the finding

The phenomenon is a *conjunction*: ≥ViT-Large, >⅓ of a long schedule, and a long
sequence containing redundant background tokens. EC-Diffuser satisfies none of
them (60.6M params, 8 blocks, 49 tokens/step, every DLP particle a
meant-to-be-distinct object slot) — **and already has a global-aggregation slot,
the action token, which is functionally what a register is.**

### Action-to-Action (arXiv:2602.07322, RSS 2026) — verified, guess was correct

Its Fig. 10 supplies a falsifiable prior for H1: flow vs deterministic regression
reach comparable in-distribution success, but regression "fails to generalize to
unseen scenarios." It also argues **against** our Stage H idea: their latent is
512-d for an 8-step chunk; our entire action space is 15 numbers.

### MeanFlow (arXiv:2512.02012) — closure stands

Different diagnosed failure (JVP gradient variance, not a deployment boundary),
no failed-run rescue anywhere, image generation only. It did surface a
**correction to our own records**: "IVC-style" refers to an **r = t** boundary,
not the (r=0, t=1) deployment boundary. We had conflated two different
boundaries.

---

## 3. Architecture, traced from production code

Token index **0 is the action**; 1–48 are DLP particles (2 views × 24). Time
reaches the denoiser by **two** paths — additive per-token, and AdaLN via
`c = action_embed + t_embed`. `time_mlp` is a **single shared network**, which is
what makes a zero-added-parameter two-time construction possible.

Flow convention verified from source (`flow_matching.py:330-332`):
`xt = (1−t)·x0 + t·x1` with `x0 = randn` ⇒ **t=0 noise, t=1 clean**.

Full detail: `architecture/ARCHITECTURE_AUDIT.md`.

---

## 4. The direct-action finding could not be recovered — it had to be built

**No direct-action implementation, checkpoint, or result exists anywhere** in
this repository: not in the working tree, not in git history across all branches,
not on disk. Searched exhaustively (`direct_action/DIRECT_ACTION_RECOVERY.md`).

> Tal's observation has no corresponding artifact here. It is therefore **built
> as a new construction** and must never be reported as reproducing a prior
> finding.

Per §15 it is also kept strictly separate from any diffusion endpoint: a learned
query token carrying no action observation is not `t_action=0` (which feeds
noise) nor `t_action=1` (which feeds the clean action).

---

## 5. Pre-GPU gates — all passed

| gate | result |
|---|---|
| §18 diagonal compatibility | **bit-exact**, max\|d\| = 0.000e+00 over 3 batches × 5 t-values |
| zero added parameters | 213,325 = 213,325 |
| loss/mask/denominator identity on the diagonal | identical under unconditioned, t=0-conditioned and goal-conditioned coordinates |
| gate non-vacuous | off-diagonal output differs by 4.0e-2 |
| §25 sampler, oracle field | every path lands on the clean target for any step count, max err 2e-6 |
| §25 no clean-target leakage | zero field leaves `x` bit-unchanged |
| §53 production-path config | `p_diag` survives the `diffusion_wrapper_kwargs` whitelist; arms produce different kwargs |
| matched control exactness | `p_diag=1.0` reproduces the true baseline loss **bit-identically** (1.9936090708) |

### Two of my own unit tests were wrong

Recorded rather than quietly fixed (`uwm/TWO_TIME_SAMPLER.md` §6):

1. The "naive shared-du is wrong" test used a **constant** field, where both
   updates coincidentally agree because each block's increments sum to 1. It had
   no teeth. Fixed with a time-dependent field; the gap is now 1.17.
2. I asserted Euler on `v = c(x1−x)` converges to `x1`. It converges to
   `x1 + (x0−x1)e^{−c}` = 0.8647 for c=2. Error appeared to *grow* with steps
   because I measured against a limit the ODE never reaches.

**The sampler was correct both times.**

---

## 6. The experiment design ⏳ (results pending)

Two arms differing in **exactly one value**, verified programmatically:

| arm | `p_diag` | training times | runtime evidence |
|---|--:|---|---|
| UWM | 0.0 | `t_s, t_a ~ U(0,1)` independent | `frac_diag = 0.0000`, mean \|t_s−t_a\| = 0.376 (theory 0.333) |
| CTL | 1.0 | `t_s = t_a` | `frac_diag = 1.0000`, `abs_time_gap = 0.0000` |

Same class, same denoiser, same parameters, same code path, same canonical
warm-start bytes, same optimizer/LR/EMA/schedule/dataloader seed.

**§54 divergence is confirmed at the distribution level, not merely checkpoint
bytes** — the check whose absence let the historical kwargs defect pass silently.

**§9 clean-first-step verified from the runtime manifest**: gradient norm
**0.651 before the clear, exactly 0.0 after**, for both model and EMA.

---

## 7. Stage E — the 2D field and inference paths

### 7.1 The adaptation verdict: ADAPTED, on both predeclared criteria

Criterion registered in `uwm/ADAPTATION_CRITERION.md` **before** the data was
read. Measured on the identical 5×5 grid with identical replay states, noise and
conditioning for both arms:

| metric (action block) | UWM (`p_diag=0`) | CTL (`p_diag=1`) |
|---|--:|--:|
| off-diagonal / diagonal error ratio | **1.558** | 3.003 |
| corr(\|t_s − t_a\|, error), off-diagonal | **+0.356** | +0.692 |
| *(state block)* off/diag ratio | **1.145** | 1.673 |
| *(state block)* corr(dist, err) | **+0.268** | +0.654 |

The control — which by construction never saw an off-diagonal time — reproduces
the closed warm-start branch's **non-adaptation signature** (corr ≈ +0.69; that
branch measured |corr| ≈ 1.0 on a related quantity). The UWM arm roughly halves
it on both blocks.

**Semantic corners** (our convention, t=1 clean):

| regime | corner | CTL | UWM | improvement |
|---|---|--:|--:|--:|
| inverse model (state clean, action noisy) | (1.0, 0.0) | 1.5941 | **0.2475** | **6.4×** |
| forward dynamics (action clean, state noisy) | (0.0, 1.0) | 1.0833 | **0.2737** | **4.0×** |

> The off-diagonal OOD confound that invalidated the previous branch is removed
> **by training**, not assumed away. This is the precondition that makes a field
> study meaningful, and it is now satisfied.

### 7.2 Adaptation happens LATE — and this matters beyond this experiment

| step | off/diag | corr(dist, err) | corner (1,0) |
|--:|--:|--:|--:|
| 500 | 2.715 | +0.6985 | 1.4077 |
| 1000 | 2.502 | +0.7053 | 1.2990 |
| 2000 | 2.240 | +0.6947 | 0.9023 |
| **5000** | **1.558** | **+0.3564** | **0.2475** |
| *CTL* | *3.003* | *+0.6920* | *1.5941* |

Through step 2000 the correlation sits at **+0.695 / +0.705 / +0.695** —
statistically indistinguishable from the never-adapted control's **+0.692**. On
that metric the model had learned essentially **nothing** about off-diagonal
structure by 2000. It collapses to +0.356 only between 2000 and 5000.

> **Had this been run at the 2000-step budget the prior loss program used, the
> conclusion would have been "off-diagonal training does not help" — and it would
> have been wrong.** Adaptation is late and nonlinear.

§23 therefore authorized **one** extension to 10000 steps, applied to **both**
arms so the matched control keeps an identical optimizer budget
(`uwm/EXTENSION_DECISION.md`).

### 7.3 The result that cuts AGAINST the interesting hypothesis

**On both arms, the diagonal path P0 is the best path for predicting the executed
action**, at equal model calls (NFE=4 for every path):

| path | UWM act@0 | CTL act@0 | UWM − CTL |
|---|--:|--:|--:|
| **P0 diagonal** | **0.0160** | **0.0118** | +0.0042 |
| P1 state-leads | 0.1169 | 0.2118 | −0.0949 |
| P2 action-leads | 0.0550 | 0.1076 | −0.0527 |
| P3 state-first axis | 0.2508 | 0.4200 | −0.1692 |
| P4 action-first axis | 0.0734 | 0.1382 | −0.0647 |

Two readings, both required:

1. **The UWM arm beats the matched control on every non-diagonal path**
   (−0.053 to −0.169). That is the adaptation showing up as usable behaviour.
2. **But the non-diagonal paths are 3–16× worse than the diagonal on the same
   model**, and the UWM arm is marginally *worse* on the diagonal itself
   (+0.0042). The best absolute action prediction available anywhere in this
   experiment is the plain diagonal on the control.

> **Offline, "state-leading inference improves action prediction" is NOT
> supported.** Independent-time *training* buys a large improvement in
> off-diagonal *robustness*; it does not buy a better operating point than the
> diagonal.

Per §48 these are offline metrics. They narrow the closed-loop candidate set;
they do not support a control claim.

---

## 8. Stage C — direct action ⏳

---

## 9. Stage F — harder multi-object task: BLOCKED, structurally

Every trained checkpoint in this repository is `3C_randcolor`. No pretrained
policy exists for any other task, and the canonical model took 500,000 steps ≈
**33 GPU-h at our measured rate — 3.3× the entire 10.0 GPU-h program cap.** A
20k-step screen is already known from Phase 3 to give **0/96 on all five sets**,
i.e. exactly the floored baseline §33 says to avoid.

Reported as a hard constraint, **not** worked around by shortening arms.

The audit also vindicated the directive's caution about names: `1C_dlp`,
`2C_dlp` and `3C_dlp` all point at the **same** shared YAML reading
`numObjects: 2`, differing only in `max_path_length`. Our canonical mode is the
exception, using a separate config dir that genuinely sets `numObjects: 3`.

**Honest reframing of H5:** the canonical task is *already* the 3-object
random-color setting at ~79% — neither saturated nor floored. The binding limit
on detecting modest architectural effects is **not** task saturation but
**evaluator resolution** (conservative RC95 ±4.30 pp; R=3 half-width ±2.48 pp).
A 1–2 pp architectural effect is unresolvable on *any* task at this budget.

---

## 10. Stage H — action latent: bottleneck NOT AUTHORIZED by its own criterion

Predeclared §41 criterion: authorize latent dim 2 only if the top-2 PCA
dimensions explain **≥99%** of nonterminal action variance.

**Measured: 75.7%.** Spectrum nearly flat (0.445 / 0.312 / 0.243); dimensions
nearly uncorrelated (max |r| = 0.155). Projecting to 2-D would discard ~24% of
the action variance.

Per §41, no lossy bottleneck was invented. Combined with the A2A evidence, Stage
H is **deprioritized on evidence**, not skipped for convenience.

Incidentally confirmed from data: the terminal row is **identical across all
198,000 windows** with zero variance, mean `[0, 0, 0.706]` — a constructed
normalized-zero target, not padding.

---

## 11. Compute ledger ⏳

---

## 12. Claim ledger

See `CLAIM_LEDGER.md`, which is maintained live and records what is established,
what is preliminary, what is withdrawn, and — importantly — what is **explicitly
not claimed**.

---

## 13. Red-team ⏳

---

## 14. Recommended next action ⏳
