# Loss Fine-Tune — Pre-Launch Audit (ZERO GPU)

**Verdict: GO WITH MODIFICATIONS.** One design-breaking defect found and fixed in
the plan; one required new measurement added. Nothing launched.

---

## 1. Warm-start semantics — model vs EMA

`Trainer.load()` (`training.py:387-388`) does:

```
self.model.load_state_dict(data['model'])       # trainable network
self.ema_model.load_state_dict(data['ema'])     # deployed weights
```

and `load_diffusion` returns `DiffusionExperiment(..., model, diffusion,
trainer.ema_model, ...)` — so `exp.diffusion` is the **trainable** path and
`exp.ema` is the **deployed** path. Gradients flow to `model`; control uses EMA.

**Recommended rule (matches original trainer semantics — continuation-like):**

```
trainable network  <-  canonical `model`
EMA                <-  canonical `ema`
```

This is what `Trainer.load` already does. **Do not copy EMA → model.** Applied
identically to A/B/C/D.

## 2. Canonical model vs EMA — measurably different

Over all **60,649,640** float parameters of `state_400000.pt` (internal step
499000):

| quantity | value |
|---|--:|
| max \|model − ema\| | 1.194e-02 |
| RMS \|model − ema\| | 1.739e-03 |
| **relative RMS** | **3.66e-02 (3.66%)** |
| identical? | **no** |

They are genuinely distinct, so the initialization rule is not cosmetic.

## 3. EMA update rule — exact

`EMA.update_average`: `new_ema = beta·old + (1−beta)·new`, `beta = 0.995`.
Called from `step_ema()` when `step % update_ema_every == 0`.

**Defaults, never overridden in the pandapush config or `train.py`:**

```
ema_decay        = 0.995
update_ema_every = 10
step_start_ema   = 2000        <-- the problem
```

```python
def step_ema(self):
    if self.step < self.step_start_ema:
        self.reset_parameters()     # EMA := HARD COPY of the live model
        return
    self.ema.update_model_average(self.ema_model, self.model)
```

## 4. **DESIGN-BREAKING DEFECT — `step_start_ema = 2000`**

A fresh fine-tune restarts `self.step` at 0. Therefore **for steps 0–1999 the EMA
is repeatedly overwritten by a hard copy of the live model**, and the canonical
EMA is **discarded at step 0**. The first true averaging update would occur at
step 2000 — the very end of the proposed budget.

**Consequence for the design as written:** the "EMA" checkpoint at 2,000 steps
would be ≈ the LIVE model, not an average. The primary deployment endpoint would
be measuring something different from what we intend, and would silently throw
away the competent canonical EMA.

### Retention arithmetic if averaging were enabled from step 0

`step_start_ema = 0`, so `n/10` updates in `n` steps:

| steps | EMA updates | retained canonical fraction `0.995^u` |
|--:|--:|--:|
| 500 | 50 | **77.83%** |
| 1000 | 100 | **60.58%** |
| 2000 | 200 | **36.70%** |

Effective timescale: `1/(1−β) = 200` updates × 10 steps = **~2,000 optimizer
steps** — comparable to the whole budget.

## 5. Is 2,000 steps enough for EMA to react? — **Yes, but only with the fix**

**With the fix (`step_start_ema = 0`):** after 2,000 steps only 36.7% of the
canonical EMA survives, so EMA tracks the fine-tuned model substantially. B/C/D
differences can realistically appear. At 500 steps (77.8% retained) EMA would be
sluggish — which is exactly why the 500/1000 checkpoints are diagnostic, not
endpoints.

**Without the fix:** EMA ≡ LIVE, and the canonical EMA is destroyed. Underpowered
in a different and worse way.

**Required modification:** pass **`step_start_ema = 0`** for all four arms, so EMA
performs true averaging from the canonical EMA throughout. Record this as an
explicit, identical deviation from the canonical training config.

**Both LIVE and EMA must be recorded at every checkpoint.** A null EMA result with
a large LIVE effect means "EMA has not caught up," **not** "loss weighting has no
effect."

## 6. Canonical five-seed anchor — **does NOT exist; must be added**

Searched all five-seed artifacts. Every existing one is for a *from-scratch* or
MeanFlow checkpoint:

| artifact | loadpath | n_valid |
|---|---|--:|
| `armA20k_diagnostics.json` | `phase3_armA/...` (20k from scratch) | 5 |
| `armA15k_diagnostics.json` | `phase3_armA/...` (15k) | 5 |
| `phase2_20k_diagnostics.json` | `imf_viability/...` (MeanFlow) | 5 |

**No `flow/3C_dlp_adalnpint_randcolor_H5_T4_seed42` five-seed result exists.**

The closest canonical evidence is **E0 only**: `r0_s42_nfe4_r3rep1.json` →
**0.8958** success at NFE4 — confirming the checkpoint is competent, but on one
of five sets and under the older harness.

⇒ **One canonical five-seed evaluation is required** (0.194 GPU-h). It is an
**anchor**, not a training arm.

## 7. Frozen inference settings

| setting | value | source |
|---|---|---|
| NFE | **4** | canonical `n_diffusion_steps: 4` |
| solver | forward Euler (flow wrapper) | `conditional_sample` |
| deployment weights | **EMA** | `DiffusionExperiment(..., trainer.ema_model, ...)` |
| action extraction | `samples[:, 0, :action_dim]`, unnormalized, replan every step | `exe_steps: 1` |
| scenario sets | E0/E1s/E2s/E3s/E4s, n=96 each | frozen manifest |
| policy noise | `torch.manual_seed(NOISE_SEED + batch_start)` | existing harness |
| success | `max_obj_dist <= 0.04` | verified 768/768 consistent |
| horizon | H = 100 env steps | fixed |

**Identical for all arms and the anchor. NFE is not tuned per arm.**

## 8. LR pilot — safety only, thresholds predeclared

**PILOT A1:** canonical checkpoint, unchanged baseline loss, **LR 1e-5**,
100-step warmup, **300 steps**. Records LIVE + EMA replay error, output drift from
canonical, loss stability, gradient norms.

**Catastrophic-damage thresholds — fixed BEFORE seeing results:**

| check | fail condition |
|---|---|
| numerical | any NaN/Inf in loss or grads |
| gradient | grad norm > 10× its step-0 value at any logged step |
| replay | LIVE replay action error > **2.0×** canonical, or state error > **2.0×** canonical |
| action collapse | >50% of executed action coords at the clipping bound |
| imagination | NFE10/15 grossly incoherent (qualitative, recorded before deciding) |

If **any** fires → run the single predeclared fallback **3e-6**. If that also
fires → **STOP and return**. No intermediate values; that would be fishing.

**The five evaluation seeds are NOT used for LR selection.** The pilot is never
continued into the experiment — Arm A restarts from canonical.

## 9. Arm A damage criterion — predeclared, includes control

Compare **CANONICAL** vs **A-LIVE** vs **A-EMA** at 2,000 steps. A need not be
bit-identical to canonical. **STOP before B/C/D if any of:**

| endpoint | fail condition |
|---|---|
| **five-seed control (primary)** | A-EMA macro mean < **0.60 ×** canonical macro mean |
| replay | A-EMA replay action error > **1.5×** canonical |
| imagination | NFE10/15 qualitatively collapsed vs canonical panel |
| numerical | any NaN/Inf |

If A collapses, the design is invalid at this optimizer/LR/budget. **Do not
rescue it post hoc** — return with a new proposal.

## 10. Attribution — frozen

| comparison | measures |
|---|---|
| CANONICAL → A | optimizer restart + warmup + 2k extra baseline training |
| **A vs B** | removing terminal zero-action supervision |
| **B vs C** | λ_action 1 → 2 |
| **B vs D** | λ_action 1 → 5 |

**CANONICAL vs B/C/D is NOT the clean causal comparison.** Canonical is
context/anchor; **A is the experimental control.**

Effective action shares unchanged: A 0.879%, B 0.816%, C 1.619%, D 3.952%.
**B removes supervision — it is not action emphasis.** λ_action multiplies the
entire action-side objective including the inherited t=0 weight of 10.

## 11. Revised compute

| item | GPU-h |
|---|--:|
| **canonical five-seed anchor (new, required)** | **0.194** |
| LR safety pilot (arm A, 300 steps) | 0.019 |
| training, 4 arms × 2,000 steps | 0.511 |
| offline diagnostics, 4 arms (LIVE + EMA) | 0.100 |
| five-seed control, 4 arms | 0.778 |
| **TOTAL** | **1.603** |

**Within the 1.65 STOP-for-approval threshold.** Arm-A-only gate (pilot +
training + diagnostics + control) = **0.366 GPU-h**.

## 12. Launch safety — atomic lock still TO DO

The single-run lock specified after the duplicate-launch race
(`PHASE3_INCIDENT_LOG.md`) was **never implemented** — it was deferred pending the
Arm-A gate, which then returned STOP.

**Required before A/B/C/D:** an `O_CREAT|O_EXCL` lock keyed to the run directory,
recording PID and start timestamp, released on clean exit, refusing to launch if
held. **Must be validated training-inert** the same way the final-save fix was
(no-lock control run; weights, EMA and batch order bit-identical). Plus one launch
path only, clean-GPU check, and continuous mid-set contention polling for control.

## 13. Verdict: **GO WITH MODIFICATIONS**

Two modifications are mandatory, not optional:

1. **`step_start_ema = 0`** for all arms — without it the EMA endpoint is
   meaningless and the canonical EMA is discarded at step 0.
2. **Add the canonical five-seed anchor** (0.194 GPU-h) — it does not exist, and
   without it Arm A's damage criterion has no reference.

Plus: implement and validate the atomic lock before launch.

With those, the design answers Tal's question with a valid control at
**1.603 GPU-h**, versus Phase 3's 3.25 that returned an uninterpretable floor.
