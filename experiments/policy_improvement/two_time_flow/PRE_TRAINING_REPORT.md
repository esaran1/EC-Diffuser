# Two-Time Flow — Pre-Training Report

Everything below is complete. **No two-time training has been run.** Total new GPU
spend for this phase: **~0.36 GPU-h** (Arm A control + rerun ~0.30, MeanFlow
diagnostics ~0.06), all within approved caps.

---

## 1. Current repo / experiment status

| item | state |
|---|---|
| git HEAD | `c61e764` (was `3035f16` at session start), branch `fast-generative-policies` |
| **E1** (scenario-slot experiment) | **UNRUN, untouched.** Not to be confused with eval seed **E1s** |
| Phase 2 (MeanFlow) | **CLOSED** — MF-NO-GO for the tested configuration |
| Phase 3 Arm A @20k | **CLOSED — gate returned STOP** |
| Phase 3 arms B/C/D | **DESIGNED, REGISTERED (`c75d701`), UNRUN** |
| two-time work | design + retrofit + unit tests + diagnostics only; **no training** |
| unrelated GPU jobs | RLinf `train_sac.py` seen earlier; **never touched**, since exited |

### Phase 3 frozen result

| E0 | E1s | E2s | E3s | E4s | macro mean | SD |
|--:|--:|--:|--:|--:|--:|--:|
| 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.0000** | **0.0000** |

All five accepted sets clean and uncontended (n=96 each).

> **Registered conclusion: a 20k Standard Flow training budget is too short for an
> informative loss-formulation screen.**

**Nothing is concluded about terminal masking or λ_action.** The loss-balancing
branch is **not** negative — it is **deferred until we choose a training budget at
which the baseline is meaningfully functional**. Running B/C/D at 20k anyway is
explicitly not proposed: every arm would be compared at the floor (0/96), where no
loss difference can be resolved.

### Infrastructure note — evaluator guard limitation

The contention guard sampled `nvidia-smi` **at set boundaries only**. A short-lived
foreign job that appears *and disappears within* a set is invisible to it. That is
exactly what happened: two diagnostics of mine ran briefly inside the E3s and E4s
windows and were gone before the post-snapshot, so the guard reported all five
VALID when two were suspect. Detected by comparing kill timestamps against per-set
windows.

Both sets were invalidated (originals preserved in
`armA20k_diagnostics_SUSPECT_E3sE4s.json`) and rerun clean. **The contaminated
values are not used, even though they happened to agree (0.0000 both times).**

**Recommendation (infrastructure, not research):** future Isaac Gym evaluations
should poll for contention *during* each set — e.g. a background sampler at
5–10 s intervals whose maximum foreign memory is recorded per set — rather than
boundary-only snapshots. **Not validated now; no GPU spent on it.**

## 2. Literature audit

Full detail in `LITERATURE_AUDIT.md`. Read from primary sources; prior summaries
not used.

**The three papers describe three different failures. Merging them under
"collapse" would be a category error.**

| | arXiv 2602.13810 (MVP/IVC) | OMP | DM1 |
|---|---|---|---|
| venue | ICLR 2026 (Oral) | ICML 2026 (poster) | preprint (page says AAMAS'26 submission) |
| failure | **boundary-condition underdetermination** | **gradient starvation → velocity-magnitude collapse** | **encoder representation collapse** |
| uses "collapse"? | **no — 0 occurrences** | only as "magnitude collapse" | yes, 20+ times |
| fix | `L_IVC = ‖u(a,t,t) − v‖²`, λ=1 | `L_DA = −log((cos α + 1)/2)` | InfoNCE/hinge/cov dispersive loss |
| measured its own failure? | **no — theory only** | synthetic toys only | **no — asserted, no metric** |
| code | none found | none (dead links) | yes, 4 paper/code discrepancies found |

**Key equations.**

IVC (Eq. 15): `L_IVC(θ) = E‖u_θ(a(t), t, t) − v‖²₂`, `v = a* − a(0)`; total
`L = L_MF + λ·L_IVC`. Theory: `Δu = C(a,r)/(r−t)` (Thm. 2), which diverges as
`r→t` unless `C ≡ 0` (Thm. 3).

OMP (Eq. 9, 12): `∂L_MSE/∂α = 2ρρ* sin α`; fix
`L_DA = −log((cos α + 1)/2)` with `cos α = (v₀·u)/(‖v₀‖‖u‖)`, `v₀ = z_T − z₀`.

DM1 (Eq. 16): `L = L_MF + α_disp·[L_disp(H^T) + L_disp(H^R) + L_disp(H^Cond)]`.

**Relevance to our MeanFlow failure:** IVC **DIRECTLY TESTABLE** (our code *is*
boundary iMF); OMP **DIRECTLY TESTABLE** (binned magnitude signature); DM1
**POSSIBLY RELEVANT** (representation battery). All three were tested — §3.

**Relevance to two-time flow: LOW for all three.** All keep one shared time on the
action block with clean, never-noised observation conditioning. Zero hits across
all three for per-block noise levels or inverse-vs-forward dynamics in a denoising
context.

## 3. MeanFlow failure diagnostics

Full detail in `MEANFLOW_FAILURE_DIAGNOSTICS.md`. Saved checkpoint only, EMA
weights, identical replay batches, matched RNG. **~0.06 GPU-h.**

### The central observation

| pooled metric (free coords, r<t) | MeanFlow | Standard Flow |
|---|--:|--:|
| norm ratio pred/target | 0.943 | 0.940 |
| cosine | 0.914 | 0.917 |
| output effective rank | 187.56 | 187.42 |

**On ordinary `r<t` samples the two are nearly indistinguishable — yet MeanFlow
scores 0/96 on all five control sets.**

At the deployment-relevant boundary `r=t`:

| boundary metric | MeanFlow |
|---|--:|
| norm ratio | **0.481** |
| cosine | **0.738** |
| residual norm | 25.74 |

**Frozen wording:**

> The diagnostics are consistent with the boundary-condition failure mode
> highlighted by IVC: MeanFlow appears relatively normal on generic `r<t` samples
> but degrades substantially at `r=t`, the regime used for one-step deployment.

We do **not** write "IVC explains our MeanFlow failure" or "boundary
underdetermination caused the 0% control." That would need a causal intervention
(retraining with `L_IVC`), which is not proposed.

### 3.1 Boundary test — exact, auditable specification (§4)

| item | value |
|---|---|
| `t` | flow time, `t ~ Uniform(0,1)` per sample, then scaled by `time_scale=1000` |
| `r` | interval endpoint; the model takes `(time, interval)` where `interval = t − r` |
| prediction at `r<t` | average velocity `u_θ(z_t, r, t)` over the interval |
| **what changes at `r=t`** | `interval = 0`; the average over a zero-length interval **is** the instantaneous velocity. Our code is *boundary* iMF: it **parameterizes** `v(z,t) := u(z,t,t)`, so this identity is a hard requirement, not an approximation |
| exact predicted quantity | `m._call_interval_model(z_t, cond, t·1000, interval=0)` |
| exact target quantity | `z − x₁` (**our repo's sign convention**, `target = noise − data`) |
| corruption | `z_t = (1−t)·z + t·x₁` |
| norm ratio | `‖u ⊙ mask‖₂ / ‖target ⊙ mask‖₂`, per sample, then averaged |
| cosine | `cos(u ⊙ mask, target ⊙ mask)` flattened per sample, then averaged |
| conditioning | **excluded** — `mask` zeroes the observation slice at rows 0 and 4; conditioned coords are also set equal in `x₁` and `z`, so they contribute nothing |
| n | **192** samples (24 batches × 8) |
| weights | **EMA** |
| state/action split | **N/A — pooled.** Not computed separately |

**Flagged as a future diagnostic:** the boundary test was pooled over all free
coordinates. A per-block split (action vs state boundary error) would be
informative — the state block has ~160× the coordinate count of the action block,
so the pooled number is dominated by state. **Not run; no GPU spent to obtain it.**

### 3.2 OMP — frozen as negative

Binned by target-norm quartile, per block, 48 samples per cell.

MeanFlow action block, smallest-target bin: **norm ratio 0.994, cosine 0.943**
(Standard Flow: 0.988 / 0.972). The pattern is **absent and reversed** — the
smallest-target bin has the *highest* ratio and a high cosine.

> Our saved MeanFlow checkpoint does not show the specific small-target
> velocity-magnitude starvation signature tested here.

⇒ no OMP directional loss, no OMP-inspired rescue, **and no claim that OMP is
irrelevant in general** — we tested one checkpoint against one signature.

### 3.3 DM1 — frozen as negative, with a retraction

| location | eff. rank MF | eff. rank SF |
|---|--:|--:|
| early block | 93.12 | 89.64 |
| middle block | 101.85 | 90.62 |
| late block | 105.98 | 85.86 |
| model output | 187.56 | 187.42 |

MeanFlow's effective rank is **higher** than Standard Flow's at every block.

**Retraction (red-teaming my own diagnostic).** My hooked battery first reported
the MeanFlow **time embedding** at effective rank **1.06**, top singular value
98.9% — which looked like a dramatic, precisely-localized collapse and fit the DM1
story neatly. Two controls killed it:

1. **Controlled probe** — `time_mlp` called directly on an identical fixed `t`
   grid (192 points, no hooks): MeanFlow **138.56** vs Standard Flow **62.93**.
   The ordering **reverses**.
2. **Root cause** — MeanFlow's loss calls `time_mlp` several times per step (at
   `t`, at perturbed times, at the boundary). The hook pooled them: 192 captured
   rows but only **96 unique `t` values**. Pooling near-duplicate rows deflates
   effective rank mechanically.

**The effective-rank = 1.06 observation is WITHDRAWN.** It is worth recording as a
case where the first measurement supported the expected story and the control
overturned it.

> The measured representations do not support a DM1-style representation-collapse
> explanation for this checkpoint.

⇒ no dispersive regularizer is motivated by current evidence.

### 3.4 Controls not re-investigated

**EMA:** LIVE ≈ EMA (objective 0.2606 vs 0.2613). **Action branch:** gradients
alive (LIVE 0.0852, EMA 0.0720) — the failure is **not** a dead action branch.

## 4. Control-frequency audit

Full detail in `CONTROL_FREQUENCY_AUDIT.md`. Traced from configs and source.

| | PandaPush 3-cube | OGBench manipspace |
|---|--:|--:|
| physics | 120 Hz (60 Hz × 2 substeps) | 500 Hz |
| **policy control** | **3 Hz** | **20 Hz** |
| sim steps per action | 19 | 25 |
| seconds per action | 0.333 | 0.050 |

PandaPush: `controlFrequency: 3` declared *and* consumed
(`isaac_panda_push_env.py:235`, loop at `:932`). OGBench:
`control_timestep=0.05` (`manipspace_env.py:26-27`), inherited by `CubeEnv`.
**Ratio ≈ 6.7×.**

**Frozen interpretation (hypothesis, not result):**

> OGBench provides a substantially higher-frequency control setting in which
> action-generation effects may be easier to expose.

We do **not** claim frequency will increase the effect until tested.

## 5. Current architecture trace

See `TWO_TIME_FLOW_DESIGN.md` §4 for the full trace with shapes. Two findings:

1. **State and action are literally separate transformer tokens** — action is
   token index 0 with its own projection/encoding/decoder; state particles are
   tokens 1..48 (`pint.py:170-187`). Per-block time conditioning is therefore
   surgical, not structural.
2. **The time embedding enters twice** — the token-wise add (`pint.py:191`) *and*
   the AdaLN conditioning `c = action_embed + t_embed`
   (`transformer_modules.py:493`), which modulates every block. Handling only the
   token add would leave the network silently single-time conditioned.

## 6. Two-time flow math

```
x^s(t_s) = (1 − t_s)·z^s + t_s·x^s_1        state block  (dims 3:483)
x^a(t_a) = (1 − t_a)·z^a + t_a·x^a_1        action block (dims 0:3)
v^s* = x^s_1 − z^s ,   v^a* = x^a_1 − z^a   (path-independent, linear interpolant)
v_θ( x^s(t_s), x^a(t_a), t_s, t_a, cond ) -> (v̂^a, v̂^s)
L = Σ_blocks masked_mean( |v̂ − v*| ⊙ W ⊙ mask )
```

Implementation follows the repo's `target = noise − data` sign convention; unit
tests assert the repo convention, not the sketch.

**Conditioned tokens (§9 of the brief).** The mask clamps `x[:, t, 3:]` for
`t ∈ {0,4}` at **every** solver step, so:

| token group | corrupted by t_state? | in loss? |
|---|---|---|
| current observation (row 0, obs) | **NO — clamped clean** | no |
| goal (row 4, obs) | **NO — clamped clean** | no |
| **predicted state trajectory (rows 1–3, obs)** | **YES — what t_state governs** | yes |
| action tokens (rows 0–4) | YES — governed by t_action | yes |

**The semantic statement is therefore about the *unconditioned predicted state
trajectory*, never about the observation or goal.** Saying "the goal becomes
noisy at low t_state" would be wrong, and every figure must reflect this.

## 7. Semantic interpretation

| regime | condition | meaning |
|---|---|---|
| joint generation | `t_s = t_a` | current EC-Diffuser; one clock |
| **inverse-model / plan-to-action** | `t_s > t_a` | predicted state plan cleaner than actions ⇒ infer actions that execute it |
| **world-model / forward-dynamics** | `t_a > t_s` | actions cleaner than predicted future state ⇒ predict the state that follows |

Wording: "inverse-model regime" / "plan-conditioned action inference" — **not**
classical inverse dynamics, since EC-Diffuser is a goal-conditioned
trajectory-level joint model, not a per-step `(s_t, s_{t+1}) → a_t` map.

## 8. Training-distribution options — computed, not described

400k samples per option.

| option | P(exact diagonal) | P(t_s>t_a) | P(t_a>t_s) | t_s marginal (mean/sd) | mean \|t_s−t_a\| | boundary mass |
|---|--:|--:|--:|---|--:|--:|
| **A** independent uniform | **0.0000** | 0.499 | 0.501 | 0.500 / 0.289 | 0.334 | — |
| **B** mixture ⅓/⅓/⅓ | 0.339 | 0.332 | 0.329 | 0.500 / 0.289 | 0.221 | — |
| **B′** mixture 0.50/0.25/0.25 | **0.500** | 0.251 | 0.250 | 0.500 / 0.289 | 0.167 | — |
| **B″** mixture 0.60/0.20/0.20 | 0.600 | 0.200 | 0.200 | 0.500 / 0.289 | 0.133 | — |
| **C** structured offset `t ± δ/2` | 0.0000 | 0.498 | 0.502 | 0.499 / **0.304** | 0.229 | **0.124** |

**Recommendation: Option B′ — `p_diag = 0.50`, `p_inverse = p_world = 0.25`.**

Reasons, each from the computed table:

- **A is disqualified for warm-start.** Exactly **zero** mass on the diagonal
  means fine-tuning would never train the configuration we deploy. Starting from a
  bit-exact competent policy and then never revisiting the diagonal is the
  catastrophic-forgetting risk in concrete form.
- **C distorts the problem.** Clipping puts **12.4%** of mass on the boundary and
  inflates the marginal SD from 0.289 → 0.304. It *looks* symmetric while changing
  marginal training difficulty — precisely the trap the brief warns about.
- **B/B′/B″ keep marginals identical to uniform** (mean 0.500, sd 0.289) *and*
  retain explicit diagonal mass, *and* are exactly symmetric between the inverse
  and world components by construction.
- **B′ specifically** matches two independent precedents: DM1 forces `r=t` for
  **ρ = 0.5** of samples, and MVP/IVC adds a dedicated diagonal loss term. Two
  groups independently found it necessary to over-train the degenerate
  sub-manifold of a two-time scheme.

`p_diag` is **predeclared, not tuned.** If Stage 1 shows the diagonal drifting,
the registered fallback is B″ (0.60) — not a search.

## 9. Architecture options and recommendation

| option | expressivity | params added | risk |
|---|---|--:|---|
| **A. per-semantic-block time embeddings (shared MLP)** | state tokens get `e_s`, action token gets `e_a` | **0** | **lowest — chosen** |
| B. token-type-conditioned embedding | equivalent here, since blocks *are* token groups | 0 | same as A |
| C. two embeddings + type embedding | marginally more expressive | >0 | breaks strict `state_dict` load |
| D. global `(t_s,t_a)` embedding | one vector for the pair | >0 | **cannot tell state vs action features which time is theirs**; also breaks bit-exactness |

**Chosen: A**, implemented in `twotime_denoiser.py`.

```
e_s = time_mlp(t_s) ;  e_a = time_mlp(t_a)     # SAME existing MLP
token 0     <- A   + e_a                        # action
tokens k>=1 <- S_k + e_s                        # state particles
c           = action_embed + [ w·e_s + (1−w)·e_a ]   # AdaLN
```

**Measured cost:** parameters **0 added** (8,101,389 both); CPU fwd+bwd ratio
**0.989×** at the canonical config (256-dim, 6 layers). GPU overhead to be
confirmed in Stage 0.

## 10. Warm-start / checkpoint compatibility

**Algebraic proof at `t_s = t_a = t`:**

1. `time_mlp` is deterministic ⇒ `e_s = e_a = e`.
2. token 0: `A + e_a = A + e` ✓; tokens k≥1: `S_k + e_s = S_k + e` ✓
3. AdaLN: `w·e + (1−w)·e = e·(w + 1 − w) = e` ⇒ `c = action_embed + e` ✓
   **for any `w`**
4. `F`, its weights, decoders and all encodings are untouched.

All inputs to `F` identical ⇒ output identical. **QED**

Consequences: **`w` is a pure off-diagonal knob** that provably cannot affect the
diagonal; and **zero new parameters**, so the canonical `state_dict` loads
strictly.

**Empirical:** diagonal `max|base − two| = 0.000e+00` (bit-exact); fallback path
(no `time_action`) also 0.000e+00; off-diagonal genuinely differs (3.78e-02);
parameter delta 0.

**Why this matters more than any other result here.** We do **not** need another
weak from-scratch model. We can start from the competent fully-trained 500k policy
and ask directly:

> Can the model learn off-diagonal state/action-time behavior while preserving its
> already-good diagonal policy?

This sidesteps the exact undertraining problem that made Phase 3 uninformative.

## 11. Inference schedules

One symmetric scalar:

```
t_state(s)  = s^(2^(−b))
t_action(s) = s^(2^(+b))
```

| b | γ_s | γ_a | mean(t_s−t_a) | max lead | regime |
|--:|--:|--:|--:|--:|---|
| −1.0 | 2.000 | 0.500 | −0.330 | 0.473 | strong action-leading (world) |
| −0.5 | 1.414 | 0.707 | −0.170 | 0.250 | mild action-leading |
| 0.0 | 1.000 | 1.000 | 0.000 | 0.000 | **diagonal** |
| +0.5 | 0.707 | 1.414 | +0.170 | 0.250 | mild state-leading |
| +1.0 | 0.500 | 2.000 | +0.330 | 0.473 | strong state-leading (inverse) |

Properties, exact for every `b`: endpoints `(0,0)→(1,1)`; monotone; within `[0,1]`;
**`b → −b` exactly swaps the two roles**; NFE unchanged (path reparameterization,
not extra network calls). Verified in unit test 10.

**Recommended set: `b ∈ {0, ±0.5, ±1}`.** The piecewise schedule
(`t_s=min(1,2s)`, `t_a=max(0,2s−1)`) is retained as an interpretable **extreme
control**, not part of the main family.

This gives a single-axis figure — *error / control vs schedule bias `b`* — rather
than five unrelated curves.

## 12. Unit-test plan and results — 22/22 PASS

CPU only, `test_two_time.py`.

| # | test | result |
|--:|---|---|
| 1 | diagonal corruption == ordinary CFM | PASS (0.000e+00) |
| 2 | extreme inverse `t_s=1,t_a=0`: state clean, action pure noise | PASS |
| 3 | extreme world `t_s=0,t_a=1`: action clean, state pure noise | PASS |
| 4 | conditioning mask exactly imposed for all `(t_s,t_a)` | PASS |
| 5 | velocity targets correct per block | PASS |
| 6 | shapes `(4,5,483)` | PASS |
| 7 | `t_state` and `t_action` each change the output | PASS |
| 8 | zero added params; diagonal **bit-exact** vs parent | PASS (0.000e+00) |
| 9 | finite nonzero grads in diagonal / inverse / world | PASS |
| 10 | all 5 schedules `(0,0)→(1,1)`, monotone, in `[0,1]` | PASS |

## 13. Experiment plan

**Two questions, kept separate (§12/§14 of the brief).**

**EXPERIMENT A — 2D conditional capability.** At *fixed* `(t_s,t_a)` on a 5×5
grid, measure action error and state error separately on neutral replay. Produces
the two heatmaps. **This can establish the inverse/world-model interpretation even
if no inference path improves control.**

**EXPERIMENT B — integration path.** From `(0,0)` to `(1,1)` along `b ∈ {0,±0.5,±1}`,
measure final generation and control. Produces the scheduler result.

A positive A with a null B is still a publishable mechanistic finding. They must
not be merged.

**Registered controls (§11 of the brief):**

| id | what | purpose |
|---|---|---|
| **CANONICAL** | original 500k Standard Flow | reference policy |
| **INIT-DIAGONAL** | two-time retrofit **before** fine-tuning, diagonal | must reproduce CANONICAL **bit-exactly** (already proven) |
| **POST-FT DIAGONAL** | fine-tuned ckpt, diagonal | did fine-tuning damage the ordinary policy? |
| **POST-FT INVERSE** | same ckpt, `b>0` | inverse-model regime |
| **POST-FT WORLD** | same ckpt, `b<0` | world-model regime |

The decisive comparison is **POST-FT DIAGONAL vs INVERSE vs WORLD — identical
weights**, so schedule effects cannot be confounded with the effect of
fine-tuning. **One checkpoint, multiple schedules.**

## 14. Compute budget

Estimated from the **actual two-time implementation** (measured 0.989× CPU
overhead on the canonical architecture), against the measured Standard Flow rate
of **0.230 s/step**.

| stage | what | GPU-h | stop gate |
|---|---|--:|---|
| **0** | throughput/memory probe, 600 steps | **0.04** | confirm overhead ≈1.0× and memory fits; **STOP if >1.3×** |
| **1** | smoke fine-tune, 1k steps | **0.06** | loss finite; diagonal retained; off-diagonal grads alive; **STOP on NaN or diagonal drift** |
| **2** | mechanistic fine-tune, 10k steps, ONE checkpoint | **0.64** | training healthy |
| **3** | offline 2D surfaces + partial-denoising figures | **0.15** | inference only |
| **4** | five-seed Panda control × 3 schedules | **0.58** | only if Stage 3 shows asymmetry |
| | **total through Stage 4** | **1.47** | |
| | **Stage 0+1 only (the gated ask)** | **0.10** | |

For comparison: Phase 3 spent **3.25 GPU-h** and returned a floor result; one
20k-from-scratch arm alone costs **1.28 GPU-h**. **No 20k-from-scratch screen is
proposed.**

## 15. Expected figures

| file | content |
|---|---|
| `fig1_two_time_plane.pdf` | the `(t_s,t_a)` square with diagonal / inverse / world regions labelled, and the `b` family drawn as paths |
| `fig2_action_error_surface.pdf` | action replay error heatmap over the 5×5 grid |
| `fig3_state_error_surface.pdf` | state replay error heatmap over the same grid |
| `fig4_partial_denoising.pdf` | diagonal vs inverse-biased vs world-biased on identical examples, columns = integration progress, **row 4 labelled GOAL CONDITION** |
| `fig5_control_vs_bias.pdf` | five-seed control vs schedule bias `b`, per-seed points + macro mean |

## 16. Claim ledger

**Established (measured):**
- Arm A @20k: 0/96 on all five sets; 20k is too short for a loss screen.
- MeanFlow pooled `r<t` velocity quality ≈ Standard Flow, yet 0/96 control.
- MeanFlow boundary `r=t`: norm ratio 0.481, cosine 0.738.
- OMP small-target starvation signature **absent and reversed**.
- DM1 representation degeneration **absent** at four predeclared sites.
- PandaPush 3 Hz; OGBench manipspace 20 Hz; ratio 6.7×.
- Two-time retrofit: 0 added params, strict load, **bit-exact diagonal**, 22/22 tests.

**Hypothesis (not established):**
- H1 inverse-model effect; H2 world-model effect; H3 tradeoff; H4 asymmetry;
  H5 control-frequency dependence.
- That the boundary inconsistency *caused* MeanFlow's control failure.

**Withdrawn:**
- MeanFlow time-embedding effective rank 1.06 — measurement artifact from pooling
  repeated `time_mlp` calls (96 unique `t` values across 192 rows).

**What the proposed experiment can and cannot conclude.** It **can** establish
whether relative state/action cleanliness changes what the model computes
(Experiment A) and whether the integration path changes generation and control
(Experiment B), with fine-tuning effects controlled. It **cannot** establish that
any advantage transfers to higher control frequency — that needs OGBench — and it
cannot revive the loss-balancing question, which is deferred separately.

## 17. Novelty search — **the (B) claim is already published**

Searched all twelve target categories. **Verdict: the specific claim is anticipated
in full, in robotics, by a well-known paper.** I verified this against the primary
source rather than a summary.

### DIRECT PRIOR ART (B) — the claim as stated

**Unified World Models: Coupling Video and Action Diffusion for Pretraining on
Large Robotic Datasets.** Zhu, Yu, Feng, Burchfiel, Shah, Gupta (UW + Toyota
Research Institute). **arXiv:2504.02792**, 3 Apr 2025, rev. 23 May 2025.
Project: `weirdlabuw.github.io/uwm`.

Verbatim from the abstract (fetched directly from arXiv):

> "a UWM integrates an action diffusion process and a video diffusion process
> within a unified transformer architecture, **where independent diffusion
> timesteps govern each modality**. **By controlling each diffusion timestep, UWM
> can flexibly represent a policy, a forward dynamics, an inverse dynamics, and a
> video generator.**"

One shared denoiser `s_θ(o, a_{t_a}, o'_{t_o'}, t_a, t_o')` with `t_a, t_o'`
sampled **independently** from `U(0,T)`. Their inference modes are our regimes:

| UWM mode | UWM setting | our framing |
|---|---|---|
| inverse dynamics | `t_o' = 0` (obs clean), denoise action | `t_state > t_action` — plan-to-action |
| forward dynamics | `t_a = 0` (action clean), denoise obs | `t_action > t_state` — world model |
| policy | `t_o' = T` | — |
| video prediction | `t_a = T` | — |

Sign conventions differ (DDPM `t=0` clean vs our flow `t=1` clean), but the
relation is identical: **the block held cleaner acts as the conditioner, and the
regime is set by relative cleanliness of state vs action.**

Adjacent: **Unified 4D World Action Modeling ... with Asynchronous Denoising**
(arXiv:2604.26694) uses per-modality schedules, but for *inference speed*, not
regime switching. **UniNav** (arXiv:2608.03244) is in the same family (timestep
mechanism not verified).

### GENERAL PRIOR ART (A) — heterogeneous noise levels

- **UniDiffuser** (Bao et al., ICML 2023, arXiv:2303.06555) — origin of the
  per-modality-timestep trick: *"the perturbation levels (i.e., timesteps) can be
  different for different modalities."* Text/image. UWM is essentially UniDiffuser
  applied to (video, action).
- **Diffusion Forcing** (Chen et al., NeurIPS 2024, arXiv:2407.01392) —
  independent per-**token** noise levels, but the axis is *position along the
  sequence*, not *variable type*. Does not noise state differently from action.
- **Mixture of Noise Levels for Audiovisual Generation** (arXiv:2405.13762) —
  explicitly enumerates "Per Modality" and "per time-segment and per-modality"
  noise assignment.
- **Hybrid Diffusion for Symbolic and Continuous Planning** (arXiv:2509.21983) —
  a "Separate Diffuser" whose two blocks *"differ in their level of noise
  corruption"*; symbolic vs motion plan, no regime-interpolation claim.
- **Neither Diffuser, Decision Diffuser, Diffusion Policy, UniPi, AVDC, nor
  EC-Diffuser** uses asymmetric state/action noise — all share one timestep.

### What is genuinely NOT anticipated

Three gaps, stated without inflation:

1. **Flow matching rather than DDPM.** A reformulation, defensible given this
   project's Flow-vs-Gaussian results, but not a novelty claim on its own.
2. **Object-centric DLP-particle state tokens rather than pixels.** Different
   substrate, same mechanism.
3. **The interior of the 2D domain is unstudied.** UWM trains on `U(0,T)²` but
   every reported inference mode **pins a timestep to a boundary** (0 or T).
   No work found sweeps the interior and asks what a *partially* clean state plan
   buys — whether relative cleanliness is a useful continuous dial rather than
   four discrete modes.

**Honest assessment:** only (3) survives, and it is an *empirical
characterization* of a mechanism someone else introduced — workshop-shaped, not a
method paper. Framed as "one model interpolates between inverse-model and
world-model behavior," the claim reads as UWM's abstract, and a reviewer who knows
UWM will say so immediately. **Given the MP1/OMP novelty constraint already
recorded for this project, a submission should not be built on the (B) claim as
stated.**

**Frozen wording:**

> Independent per-modality diffusion timesteps for action and observation blocks,
> used to switch between inverse-dynamics and forward-dynamics behavior, are
> established prior art (UWM, arXiv:2504.02792; UniDiffuser, arXiv:2303.06555).
> What remains open is the behavior of the *interior* of the two-time domain,
> which existing work does not characterize.

---

# Decision questions

**Q1. Is there measured evidence that our MeanFlow failure matches IVC, OMP, DM1,
or none?**

**IVC — consistent. OMP and DM1 — affirmatively not.** At `r=t` the model produces
norm ratio **0.481** and cosine **0.738**, versus 0.943/0.914 on ordinary `r<t`
samples, and our code *parameterizes* `v(z,t) := u(z,t,t)`, so that identity is a
hard requirement. OMP's small-target starvation signature is absent and reversed
(smallest bin has the *highest* ratio, 0.994). DM1-style degeneration is absent
(effective rank higher than Standard Flow at all three blocks). **Association, not
demonstrated cause** — a causal test needs a retrain we do not propose.

**Q2. Is the retrofit mathematically and empirically diagonal-equivalent?**

**Yes, both.** Algebraically: `e_s = e_a = e` at `t_s = t_a`, the token adds reduce
to the original broadcast, and the AdaLN blend gives `w·e + (1−w)·e = e` for **any**
`w`. Empirically: `max|base − two| = 0.000e+00`, zero added parameters, strict
`state_dict` load, 22/22 unit tests.

**Q3. Can we warm-start from canonical Flow without changing its initial diagonal
policy?**

**Yes — provably and bit-exactly at initialization.** Whether it *stays* preserved
after off-diagonal fine-tuning is the open question, which is why `p_diag` matters
and why POST-FT DIAGONAL is a registered control.

**Q4. What training distribution best teaches off-diagonal behavior while
preserving the diagonal?**

**Mixture B′: `p_diag = 0.50`, `p_inverse = p_world = 0.25`.** It is the only
family that keeps marginals identical to uniform (mean 0.500, sd 0.289) *and*
retains explicit diagonal mass *and* is exactly symmetric. Independent uniform has
**zero** diagonal mass (disqualifying); structured offset puts 12.4% of mass on the
boundary and inflates marginal SD to 0.304. Precedent: DM1 forces `r=t` for 50% of
samples; IVC adds a dedicated diagonal loss.

**Q5. What is the smallest schedule family spanning world ↔ joint ↔ inverse?**

One scalar: `t_state = s^(2^−b)`, `t_action = s^(2^+b)`, with `b ∈ {0, ±0.5, ±1}`.
Exact endpoints, monotone, `b → −b` exactly swaps roles, NFE unchanged. The
piecewise schedule is kept as an interpretable extreme control.

**Q6. What is the cheapest decisive PandaPush experiment?**

**Stage 0+1 = 0.10 GPU-h** to establish whether off-diagonal fine-tuning is viable
at all without destroying the diagonal. **Stage 2+3 = 0.79 GPU-h** produces the 2D
error surfaces — the mechanistic figure — from **one** checkpoint. Total through
five-seed control: **1.47 GPU-h**, against Phase 3's 3.25 for a floor result.

**Q7. What exact evidence would justify moving to OGBench?**

Both of: (a) a **clear asymmetry in the 2D surfaces** — action error falling as
`t_state` rises, and/or state error falling as `t_action` rises, beyond the
evaluator's resolution; and (b) preferably a **schedule-dependent effect** on
action error or control at matched NFE. Low control frequency alone is **not**
sufficient justification.

---

# Verdict: **GO WITH MODIFICATIONS**

**Go**, because the enabling result is unusually strong: a zero-parameter,
bit-exact warm start from the competent 500k policy makes the first informative
experiment cost **0.10 GPU-h** to gate and **1.47** in full — cheaper than the
Phase-3 screen that returned a floor. The mechanism is testable offline through
Experiment A regardless of whether any schedule improves control.

**The modifications are not optional:**

1. **Reframe the contribution.** The inverse/world regime-switching claim is
   **published** (UWM, arXiv:2504.02792). Do not present it as novel. The
   defensible open question is the **interior of the two-time domain** — is
   relative cleanliness a useful continuous dial, or only four discrete modes? —
   plus the flow-matching and object-centric-state substrate.
2. **Warm-start, never from scratch.** Phase 3 demonstrated what a 20k-from-scratch
   model buys: nothing measurable.
3. **Diagonal-preserving mixture B′**, `p_diag` predeclared, fallback B″ registered.
4. **One checkpoint, multiple schedules**, with POST-FT DIAGONAL as the control that
   separates fine-tuning effects from schedule effects.
5. **Experiment A before B.** A can succeed mechanistically even if B is null.
6. **Panda first; OGBench only on the Q7 trigger.**
7. **No collapse remedies imported** — OMP and DM1 mechanisms were measured absent.

**Honest caveat on ambition.** With UWM in the literature, this is now a
*mechanistic characterization* study rather than a new-method paper. That is worth
1.47 GPU-h and could support a workshop paper or a strong section of a larger
work. It should not be positioned as the centerpiece novelty of a top-conference
submission without a further differentiating result.
