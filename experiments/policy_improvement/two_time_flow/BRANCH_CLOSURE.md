# Two-Time Warm-Start Branch — CLOSED, NO-GO

Stages B and C not run. **~0.20 of the approved 1.40 GPU-h used; ~1.20 unused,
deliberately.** Stopping early is the intended behaviour of a failed
falsification gate, not an under-delivery.

---

## 1. Final claim ledger

### Withdrawn — do not cite

| withdrawn claim | why |
|---|---|
| "Partially denoised state context is optimal for action inference" | pre-FT the minimum is *on the diagonal* in every row; post-FT the interior advantage is +0.0% in half the rows |
| "The interior of the state/action-time plane contains an inverse-model sweet spot" | the structure is conditioning-geometry drift, not semantics |
| "Interior optimum in 5/5 rows" | count was arithmetically right, but the 5th row is degenerate (+1.0% margin on a fully-clean action block); the honest 4/4 is **also withdrawn** because the post-FT optimum coincides with the diagonal |
| **"0.0964 vs 0.4490 ≈ 4.7×"** | measured against `t_state=1`, **not** against the diagonal. Retained only as evidence that the clean-state endpoint is badly OOD under this retrofit — **never** as evidence that partial state context beats joint generation |
| matched-cleanliness asymmetry (9/9 positive, mean +0.205/+0.231) | already retracted at the gate: the contrast swaps the target block's own noise level, and own-block time dominates cross-block by 2.67×/9.09× |

### Frozen as correct

> The zero-parameter warm-start retrofit is exactly equivalent to canonical
> Standard Flow on the diagonal, but its off-diagonal conditioning is
> out-of-distribution. Action error increases almost perfectly with the geometric
> drift of the time-conditioning representation away from the pretrained
> diagonal.

**corr(action error, cosine-to-diagonal conditioning) = −0.995 / −0.999 / −1.000**
at `t_action = 0.25 / 0.50 / 0.75`.

### Pre/post decomposition — frozen

**PRE-FT:** every row's minimum lies exactly on `t_state = t_action`; no semantic
interior optimum; behaviour worsens monotonically with distance from the
pretrained diagonal.

**POST-FT:** off-diagonal fine-tuning reduces that penalty (1.5–7.4×) **and**
degrades diagonal behaviour (+82.7% / +44.8% / +31.7% / +3.1% at
`t = 0.00/0.25/0.50/0.75`). Against the correct diagonal baseline the apparent
gains are **+11.8% / +1.7% / 0.0% / 0.0%**.

⇒ The 1k-step fine-tune purchased **off-diagonal robustness**, not a new optimum.

### Boundary strip — useful negative

Degradation is **not** a singular exact-endpoint artifact: it begins near
`t_state ≈ 0.7` (+21%) and worsens smoothly to +460% at 1.0.

> Off-diagonal degradation extends through the interior rather than occurring
> only at the exact boundary, but it is strongly explained by representation
> drift away from the pretrained diagonal.

## 2. Why Stage B/C would be invalid

Any path farther from the diagonal necessarily traverses regions where the
pretrained conditioning is increasingly OOD. Schedule ordering would therefore be
confounded with conditioning-distance artifact, and closed-loop control would
compound it. **Stopping before B/C is the correct call.**

## 3. What this does NOT disprove

We do **not** claim "separate state/action flow times do not work" or "Tal's
scheduler hypothesis is false." The actual conclusion is narrower:

> A diagonal-pretrained Standard Flow model cannot be used as an unbiased
> warm-start probe of off-diagonal state/action times with this zero-parameter
> retrofit.

A properly trained dual-time model could still answer the scientific question.

## 4. From-scratch two-time: **SCIENTIFICALLY VALID BUT DEFERRED**

Not launched, for five recorded reasons:

1. UWM (arXiv:2504.02792, RSS 2025) already establishes independent modality
   timesteps, policy, inverse dynamics, forward dynamics, joint sampling, and
   training over timestep combinations.
2. Our remaining distinction was interior/path characterization — which this
   experiment cannot validly establish.
3. Phase 3 showed short from-scratch PandaPush training is insufficient:
   Standard Flow at 20k gave **0/96 on all five evaluation sets**.
4. A meaningful from-scratch dual-time EC-Diffuser therefore needs a
   substantially larger budget.
5. That compute is not justified by the current novelty/expected-information
   tradeoff unless Tal explicitly wants the characterization.

## 5. UWM positioning — unchanged

Do **not** position two-time EC-Diffuser as a novel method. UWM covers the
framework; our attempted differentiation was the interior, and the warm-start
probe cannot establish it.

## 6. The surviving methodological observation

> Exact agreement on the pretrained diagonal does not imply meaningful
> off-diagonal behaviour. A model can be bit-exact at `t_state = t_action` while
> becoming sharply out-of-distribution as semantic timesteps separate.

Useful for notes/discussion on multimodal or block-specific diffusion-time
retrofits. **Not** a central paper contribution.

## 7. Compute record

| | GPU-h |
|---|--:|
| approved Stage-A cap | 1.40 |
| **actually used** | **~0.20** |
| unused | ~1.20 |

Unused allocation is **not** to be spent because it exists.
