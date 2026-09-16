# Terminal-mask investigation — final result

*Final commit: `19c6809`. Figures: `mechanism/figures/` (regenerate with `mechanism/make_figures.py`).*

## 1. Executive summary

**The terminal-mask improvement did not survive its own controls. The honest
conclusion is that this evaluator cannot resolve effects of the size in
question.**

Four findings, in the order they matter:

1. **Action-side loss weighting did not help.** λ=2 and λ=5 gave no control
   improvement over masked λ=1 in the tested 2,000-step fine-tuning screen.
2. **The legacy terminal-mask signal (+2.24 pp over four seeds) collapsed to
   +0.39 pp when a second simulator realization was averaged in.** The original
   "replicated" result was substantially a single-realization artifact.
3. **The 0.207% denominator change is not the explanation.** The B′ control
   retained most of the legacy signal (+1.82 pp, positive in 4/4 seeds).
4. **The clean-first-step protocol gave +1.46 pp with SD 2.98 pp and one seed
   reversing (−2.29 pp).** The mechanical classification criteria pass, but the
   scatter is twice the mean and 59% of the positive mass comes from one seed.

**The decisive number:** re-evaluating the *same frozen checkpoint* on a new
physics realization moves macro success by **up to 3.75 pp** — as large as or
larger than every treatment effect measured. No conclusion about a ~1–2 pp effect
is supportable at this evaluation resolution.

### Classifications

| axis | classification |
|---|---|
| action weighting | **no improvement found** (plain-language null, tested regime only) |
| denominator | **DENOMINATOR-NEGLIGIBLE** (as primary explanation; not asserted as exact zero) |
| trainer confound | **TRAINER-CONFOUND-UNRESOLVED** |
| clean terminal mask | **CLEAN-TERMINAL-MASK-UNRESOLVED** |
| mechanism | **MECHANISM-UNRESOLVED** |

**Note on the two "UNRESOLVED" calls.** The predeclared arithmetic criteria for
TRAINER-CONFOUND-UNLIKELY and CLEAN-TERMINAL-MASK-SUPPORTED are formally met
(clean mean > 0, 3/4 seeds non-negative, leave-one-out means all positive). I am
**not** taking those labels, because the criteria were written assuming an effect
large relative to evaluator noise, and that premise fails here: clean SD (2.98 pp)
is twice the clean mean (1.46 pp), dropping seed 44 leaves +0.35 pp, and
same-checkpoint re-evaluation noise reaches 3.75 pp. Reporting SUPPORTED on this
evidence would be letting a threshold formula override what the data show.

---

## 2. The question

> When EC-Diffuser is fine-tuned from the competent canonical checkpoint WITHOUT
> the random-initialization sanity gradient contaminating optimizer step 1, does
> removing supervision on the constructed, non-executed terminal zero-action
> target reproducibly improve PandaPush closed-loop control?

---

## 3. Action-weighting screen

Single seed, 2,000-step fine-tune, five fixed scenario sets, n=96 each, NFE=4.

| arm | objective | macro | SD |
|---|---|--:|--:|
| canonical | *(context only)* | 0.7708 | 0.0338 |
| A | baseline | 0.7750 | 0.0597 |
| B | terminal masked, λ=1 | 0.8042 | 0.0324 |
| C | masked, λ=2 | 0.7854 | 0.0469 |
| D | masked, λ=5 | 0.7875 | 0.0524 |

λ demonstrably reached the objective (EMA action-gradient ratio vs B: C 1.81×,
D 4.36×) and did not damage state prediction or imagination — but produced no
control improvement.

> In the tested single-seed 2,000-step fine-tuning experiment, increasing the
> action-side loss weight to λ=2 or λ=5 did not improve closed-loop control
> relative to terminal masking alone.

This does **not** support "action weighting never helps", "λ=1 is globally
optimal", or claims about longer training or untested λ values.

---

## 4. Legacy terminal-mask result (single realization)

| seed | A | B | B−A |
|---|--:|--:|--:|
| 42 | 0.7750 | 0.8042 | **+2.92** |
| 43 | 0.7729 | 0.7729 | +0.00 |
| 44 | 0.7500 | 0.7896 | **+3.96** |
| 45 | 0.7833 | 0.8042 | **+2.08** |

mean **+2.24 pp**, median +2.50, SD 1.68, signs 3+/1 zero/0−.
Descriptive 95% interval −0.43 … +4.91 pp (already includes zero).

This is what was originally classified TERMINAL-MASK-REPLICATED.

---

## 5. Legacy R=2 — the result that changed the story

Each of the eight legacy policies was re-evaluated once more on an independent
simulator realization (Stage C, 8/8 VALID, zero contention).

**Same checkpoint, new realization:**

| seed | arm | original | repeat | shift |
|---|---|--:|--:|--:|
| 42 | A | 0.7750 | 0.7958 | +2.08 |
| 42 | B | 0.8042 | 0.7750 | **−2.92** |
| 43 | A | 0.7729 | 0.7854 | +1.25 |
| 43 | B | 0.7729 | 0.7729 | +0.00 |
| 44 | A | 0.7500 | 0.7875 | **+3.75** |
| 44 | B | 0.7896 | 0.7812 | −0.83 |
| 45 | A | 0.7833 | 0.7896 | +0.63 |
| 45 | B | 0.8042 | 0.7708 | **−3.33** |

**Physics-averaged (R=2) treatment effect:**

| seed | Jbar A | Jbar B | Delta |
|---|--:|--:|--:|
| 42 | 0.7854 | 0.7896 | +0.42 |
| 43 | 0.7792 | 0.7729 | **−0.62** |
| 44 | 0.7688 | 0.7854 | +1.67 |
| 45 | 0.7865 | 0.7875 | +0.10 |

mean **+0.39 pp**, median +0.26, SD 0.96, signs 3+/1−.

**The legacy effect fell by 83% (+2.24 → +0.39 pp) on adding one realization.**
That is the single most important measurement in this program.

---

## 6. Denominator control (B′) — legacy protocol

B′ = masked numerator, A's denominator (46560). Isolates the 0.207%
renormalization the mask inherits.

| seed | B−A | **B′−A** | B′−B |
|---|--:|--:|--:|
| 42 | +2.92 | **+1.87** | −1.04 |
| 43 | +0.00 | **+1.67** | +1.67 |
| 44 | +3.96 | **+2.92** | −1.04 |
| 45 | +2.08 | **+0.83** | −1.25 |
| **mean** | **+2.24** | **+1.82** | **−0.42** |

B′−A positive in **4/4** seeds.

> Restoring A's denominator preserves most of the masking-associated behavioral
> signal across all four seeds. The 0.207% renormalization is therefore not
> sufficient to explain the legacy A-vs-B result.

Exact zero contribution is **not** claimed. B′−B (−0.42 pp, mixed signs) is
within evaluator noise and is not interpreted.

All four B′ arms verified: step 2000, denominator 46560 with mask active, 20/20
sets VALID, zero contention, four distinct SHAs.

---

## 7. The random-init first-step confound

`train.py` (L153–160) runs a sanity forward/backward **before** the warm-start
load (L198), and `Trainer.train()` clears gradients only **after** the first
optimizer step (`training.py` L266). So every legacy fine-tune's step 1 applies:

```
g_sanity(random-init network, dataset[0], batch size 1) + g_batch1 + g_batch2
```

**Measured — and objective-dependent, not a shared constant:**

| arm | sanity loss | ‖g_test‖ | rel. vs A | cos vs A |
|---|--:|--:|--:|--:|
| A | 0.8488690853 | 5.2007e-01 | — | — |
| B | 0.8486053944 | 5.2224e-01 | 2.25e-02 | 0.99976 |
| C | 0.8727948070 | 5.5712e-01 | 2.50e-01 | 0.97329 |
| D | 0.9453629255 | 7.7480e-01 | 9.50e-01 | 0.77779 |
| B′ | 0.8468557000 | 5.2116e-01 | 2.22e-02 | 0.99976 |

‖g_test‖ ≈ 0.52 is **~28×** a typical training-gradient EMA (1.84e-2).

An earlier claim in this project that the residual was "identical across arms, a
shared constant" was **wrong and is withdrawn**.

---

## 8. Clean-first-step protocol

`--clean_first_step` (opt-in, default OFF — no historical arm altered): clears
optimizer/model/EMA gradients after the sanity backward and after warm-start,
immediately before training; aborts if any survive.

**Verified on all eight clean arms:**

| arm | grads before clear | norm before | after | EMA before/after |
|---|--:|--:|--:|---|
| A42 | 269 | 0.6689 | 0 / 0.0 | 0 / 0 |
| B42 | 269 | 0.6696 | 0 / 0.0 | 0 / 0 |
| A43 | 269 | 0.5935 | 0 / 0.0 | 0 / 0 |
| B43 | 269 | 0.5945 | 0 / 0.0 | 0 / 0 |
| A44 | 269 | 0.6179 | 0 / 0.0 | 0 / 0 |
| B44 | 269 | 0.6195 | 0 / 0.0 | 0 / 0 |
| A45 | 269 | 0.5380 | 0 / 0.0 | 0 / 0 |
| B45 | 269 | 0.5370 | 0 / 0.0 | 0 / 0 |

The contamination is real, large, fully removed, and never present on the EMA
side (verified empirically, not assumed). Independent confirmation: clean arms
draw **4000** batches (2×2000) versus legacy's 4002.

---

## 9–10. Clean four-seed result — the decisive experiment

| seed | A_clean | B_clean | Delta |
|---|--:|--:|--:|
| 42 | 0.8146 | 0.7917 | **−2.29** |
| 43 | 0.7958 | 0.8042 | +0.83 |
| 44 | 0.7562 | 0.8042 | **+4.79** |
| 45 | 0.7604 | 0.7854 | +2.50 |

mean **+1.46 pp**, median +1.67, **SD 2.98**, range −2.29 … +4.79, signs 3+/1−.
Descriptive 95% interval **−3.28 … +6.20 pp**.

Leave-one-seed-out means: drop 42 → +2.71; drop 43 → +1.67; **drop 44 → +0.35**;
drop 45 → +1.11.

**Why this is UNRESOLVED rather than SUPPORTED:**

- SD is **2.0×** the mean.
- Seed 42 **reverses sign** relative to its legacy result (+2.92 → −2.29).
- Seed 44 contributes **59%** of the positive mass; removing it leaves +0.35 pp.
- Same-checkpoint re-evaluation noise reaches **3.75 pp** — larger than the mean.
- The descriptive interval spans zero by a wide margin.

**Per-set effects (clean, mean over seeds):** E0 −0.26, E1s +2.34, E2s −0.26,
E3s +4.17, E4s +1.30 pp. No set favors B consistently (best is 3/4). E3s is
dominated by seed 44's +10.42 pp.

---

## 11. Legacy vs clean

| seed | legacy 1-realization | legacy R=2 | clean | clean − legacy R2 |
|---|--:|--:|--:|--:|
| 42 | +2.92 | +0.42 | **−2.29** | −2.71 |
| 43 | +0.00 | −0.62 | +0.83 | +1.46 |
| 44 | +3.96 | +1.67 | +4.79 | +3.12 |
| 45 | +2.08 | +0.10 | +2.50 | +2.40 |
| **mean** | **+2.24** | **+0.39** | **+1.46** | **+1.07** |

Question A — *does masking help under the clean trainer?* **Unresolved.**
Question B — *did the contaminated step distort the effect?* **Unresolved**: the
clean mean is *higher* than legacy R=2, the opposite of an inflation story, but
seed-by-seed shifts range −2.71 to +3.12 pp with no consistent direction, all
within the evaluator's own noise.

---

## 12. Offline diagnostics (legacy, four seeds, EMA)

Executed action t=0: A 0.01600–0.01634, B 0.01616–0.01643 — **unchanged**, if
anything marginally worse for B. Nonterminal t=1..3 flat. Terminal t=4: A ≈0.0006
(learns the constant placeholder), B 0.72214 (correctly stops emitting it).

State replay 0.20527–0.20541 across every arm. Imagination clean at NFE10/15 in
all arms including D at λ=5. No NaN.

**t=4 is reported separately everywhere and never pooled into an executed-action
metric** — pooling it manufactures a 6.4× artifact.

---

## 12b. First-step contamination persistence — NOT COMPLETED

Registered as a secondary analysis (section 27): compare `A_clean` vs `A_legacy`
and `B_clean` vs `B_legacy` at each shared checkpoint, to measure how long the
single contaminated Adam update leaves a measurable trace.

**Not completed.** The comparison loads ~40 checkpoints of ~1 GB each and runs
CPU forward passes; two attempts ran past 29 minutes without producing output and
were stopped rather than spend further effort on a secondary diagnostic (the
directive explicitly says not to burn material effort reconstructing this). The
script `clean_first_step/first_step_persistence.py` is committed and its
checkpoint-path resolution was verified; it can be re-run on a larger-memory
machine or adapted to stream parameters rather than hold them.

This affects no conclusion. The primary causal question is answered by the clean
four-seed control, and the contamination's existence and complete removal are
already established directly from the gradient manifests (section 8).

---

## 13. Mechanism

**MECHANISM-UNRESOLVED**, and with no behavioral effect to explain, the question
is moot for now.

Established negatives:
- **Gradient conflict NOT supported.** cos(g_term, g_exec) is positive in every
  predeclared group: full +0.46, trunk +0.35, action_head +0.53, time_cond +0.48.
  No search was made for a group showing conflict.
- Terminal loss component is **3.28e-5** — ~46× smaller than executed-action,
  ~7000× smaller than state — with gradient *aligned* (projection +0.08).
- A vs B outputs differ by only 2.8–4.5% of each arm's displacement from
  canonical; t=0 action cosine 0.999999.

---

## 14. What the terminal target actually is

Not padding. `GoalDataset.__getitem__` builds every trajectory's last row as the
**goal-conditioning row**: goal observation + an explicitly constructed
`normalize(zeros)` action. One unique value across 5,000 sampled windows, per-dim
std [0,0,0]; the terminal observation *is* `cond[H−1]`.

An intermediate analysis concluded "padding" and was wrong — 3.03% of windows do
extend past episode end, but a normal in-episode window showed the same constant
while the raw action there was [0.085, −1.0, 0.048]. Recorded because it would
have produced a confidently incorrect claim.

**Terminology:** constructed terminal zero-action target / non-executed terminal
action target / goal-row zero-action supervision. **Never "padding."**

---

## 15. Literature context

Related sequence-policy implementations commonly mask action positions that are
not semantically valid supervision (LeRobot's `action_is_pad`), and LeRobot issue
#3433 treats dividing by the full element count while masking as a bug, fixed in
PR #3377 by averaging over valid entries only — which matches our registered B
denominator. But **EC-Diffuser's goal-row target is a distinct constructed target
rather than literal sequence padding.** No novelty or equivalence is claimed.

---

## 16. Red-team

| check | result |
|---|---|
| all 8 clean arms `clean_first_step=True` | **yes** |
| exact historical dataloader-seed mapping | **yes** — verified against legacy `args.json` |
| same canonical source / EMA | **yes**, all 8 |
| residual model grads before real backward | **0**, norm 0.0, all 8 |
| EMA grads | **0** before and after, all 8 |
| mask / denominator / λ=1 / aw=10 / LR / warmup / sse=0 / 2000 steps | **all correct**, 14 checks × 8 arms |
| final internal step 2000 | **yes**, sentinels `step_matches` |
| interrupted-A42 state leaked? | **no** — all checkpoints post-restart, `batches_drawn=4000` |
| hidden resume | **no** — fresh Adam, `trainer.step=0`, atomic lock |
| contention in any accepted eval | **none** — 0 MiB foreign, all 40 clean+legacy sets |
| one seed driving the aggregate | **YES — seed 44 gives 59% of positive mass** |
| one E-set driving it | **partly** — E3s +4.17 pp, dominated by seed 44 |
| t=4 pooled into executed-action metric | **no** |
| claim stronger than n=4 supports | **avoided** — hence UNRESOLVED |
| simulator variability treated as significance test | **no** |

---

## 17. Compute ledger

**A. Historical loss-FT program** — screening + corrected B/C/D + replication:
valid 0.619 + 1.133 + 2.134; **invalid 1.005** (kwargs-defect B/C/D) + 0.060
(contention).

**B. Terminal-mask mechanism / legacy robustness** — hard cap **3.60**:

| item | GPU-h |
|---|--:|
| Stage B probes (VOID, see below) | 0.071 |
| Stage B′ four-seed (valid) | 1.421 |
| Stage C repeats (valid) | ~1.611 |
| contended B45 repeat (invalid) | 0.061 |
| **accidental B′ seed42 retrains ×2 (wasted)** | **0.266** |
| pre-lease charges | 0.120 |
| **total** | **~3.58 of 3.60** |

**C. Clean-first-step program** — hard cap **2.90**:

| item | GPU-h |
|---|--:|
| 8 clean arms (train + full diagnostics + control) | 2.859 |
| **premature-intervention duplicate-launch (wasted)** | **0.0053** |
| **total** | **2.8647 of 2.90** |

No borrowing between allocations; both caps code-enforced.

---

## 18. Engineering / validity ledger

Seven defects found, each of which would have produced a wrong scientific claim:

1. `diffusion_wrapper_kwargs` whitelist dropped both loss knobs → all four arms
   trained one objective (byte-identical checkpoints).
2. `loss_weight_matrix` is a registered buffer → checkpoint load silently
   clobbered the λ-scaled matrix while the scalar attribute still read correctly.
3. Direct-Trainer probe did not reproduce `train.py` → its
   `DENOMINATOR-NOT-NEGLIGIBLE` verdict (R_param 0.917) is **VOID**; probe B(500)
   sat 1.70 from scientific B(500) while the whole A–B difference is 0.0996.
4. Residual random-init sanity gradient entering optimizer step 1,
   objective-dependent (§7).
5. Result-file existence treated as evaluator validity → fixed to require
   `control_mean` non-null, `n_valid_sets==5`, all five VALID.
6. Two independent runners armed on the same clean-GPU check → single global
   lease + one orchestrator.
7. **Premature intervention:** I killed a healthy training run because its log
   was silent, then briefly duplicate-launched. `wait_stable_clean` is silent by
   design during a successful streak.

> **Lesson:** silent stable-clean waits must not be diagnosed as hangs from log
> inactivity alone; process tree, lease, and sentinels must be checked first.

Incidental: an accidental B′ seed42 retrain reproduced the original checkpoint
**byte-for-byte** (sha `f482ae9d219b4628`) — a determinism check on the training
path. The compute remains classified wasted.

---

## 19. Claim ledger

**ESTABLISHED**
- Seven production-path defects and their fixes (§18).
- Terminal target is a constructed goal-row constant, not padding (1 unique value
  / 5,000 windows, std [0,0,0]).
- `--clean_first_step` removes the residual gradient completely (269→0, EMA 0/0,
  all 8 arms).
- Same-checkpoint re-evaluation noise reaches **3.75 pp** on this evaluator.
- B′ semantics: masked numerator + A denominator, exact c = 46464/46560.

**PRELIMINARY / NOT SUPPORTED AT THIS RESOLUTION**
- Any terminal-mask control benefit. Clean mean +1.46 pp with SD 2.98 pp, one
  seed reversed, one seed carrying 59% of the mass.

**NULL / NEGATIVE**
- λ=2 / λ=5 action weighting: no control improvement in the tested regime.
- Simple terminal-vs-executed gradient conflict: not supported.
- Executed-action replay accuracy: unchanged by masking.

**WITHDRAWN**
- "Terminal target is padding."
- Stage-B probe denominator verdict (R_param 0.917).
- "Sanity gradient is identical across arms / a shared constant."
- TERMINAL-MASK-REPLICATED as a robust finding — it did not survive R=2.

**UNRESOLVED**
- Whether terminal masking helps at all; mechanism; generalization.

---

## 20. Tal's questions — direct answers

**1. Did increasing action-side loss weight help?**
No. λ=2 and λ=5 gave no control improvement in the tested 2,000-step screen.

**2. Did masking the terminal constructed zero-action target help under the
legacy trainer?**
It appeared to: +2.24 pp across four seeds, 3 positive, none negative. That was
the basis for calling it replicated.

**3. Did the benefit survive removing the random-init gradient from step 1?**
Unresolved — and the more important finding came earlier. Averaging in **one
extra simulator realization** dropped the legacy effect from +2.24 to **+0.39
pp**. The clean four-seed result is +1.46 pp with SD 2.98 pp and one seed
reversing. Neither protocol gives an effect this evaluator can resolve.

**4. Was the 0.207% denominator change responsible?**
No. B′ retained +1.82 pp, positive in 4/4 seeds.

**5. Did executed-action replay accuracy improve?**
No. t=0 error is unchanged (0.0160–0.0164), marginally worse for B if anything.

**6. Do we understand why behavior changed?**
No — and there is no longer a robust behavioral change to explain. Gradient
conflict is ruled out; the terminal term is ~46× smaller than executed-action
loss and *aligned* with it.

**Bottom line:** the loss branch is closed. Neither weighting nor terminal
masking produced an improvement that survives simulator resampling. The real
deliverable is a measured evaluator resolution limit (~3.75 pp per realization)
and seven fixed production-path defects.

---

## 21. One next experiment (not run)

**Raise evaluator resolution before testing any further policy intervention.**
Concretely: characterise how macro success converges as episodes per set and
realizations per policy increase — e.g. 3–5 realizations × the existing 5 sets on
two frozen checkpoints — and derive the minimum detectable effect.

Rationale: every intervention tested (λ=2, λ=5, terminal mask, denominator) lives
in the 0–4 pp range, and this program measured 3.75 pp of same-checkpoint
realization noise. Until the evaluator can resolve ~1 pp, further policy
variations cannot be distinguished from noise, and we will keep generating
effects that evaporate under resampling — which is exactly what happened here.

Everything else (more seeds, longer fine-tunes, new λ, other tasks) is premature
until the measuring instrument is characterised.
