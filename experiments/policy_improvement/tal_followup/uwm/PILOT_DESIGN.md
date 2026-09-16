# UWM pilot — registered design (pre-GPU)

Registered **before** any GPU work. Zero-GPU gates already passed:
diagonal-compatibility (§18), sampler unit tests (§25), production-path config
validation (§53).

## 1. The two arms — differing ONLY in the time distribution

| | arm | `p_diag` | training times |
|---|---|--:|---|
| **UWM** | faithful independent-time | **0.0** | `t_s, t_a ~ U(0,1)` independently |
| **CTL** | matched diagonal control | **1.0** | `t_s = t_a = u ~ U(0,1)` |

Both use the **same class** (`UWMConditionalFlowMatching`), the **same denoiser**
(`TwoTimePINTDenoiser`), the same parameters, the same code path. This is the
strongest possible form of the §22 matched control: there is no architectural,
parameter-count, or code-path difference to confound the comparison.

**Verified exactly** (CPU): at `p_diag=1.0` the control reproduces the true
baseline `ConditionalFlowMatching` loss **bit-identically** (1.9936090708 both)
on the same data, noise and time, while sampling 100% on the diagonal with a
correct U(0,1) marginal. So CTL is simultaneously (a) a faithful reproduction of
canonical training and (b) architecturally identical to UWM.

## 2. Matched everything else (§22)

same canonical source checkpoint · same training seed · same dataloader stream
(`dataloader_seed`) · same optimizer · same LR · same steps · same EMA ·
**clean-first-step semantics mandatory** (§9).

## 3. Warm start

From canonical `861dc34434474455…` (`state_400000.pt`, internal step 499000).
Both arms warm-start from the identical bytes, so any divergence is caused by
the time distribution alone.

The `loss_weight_matrix` registered-buffer hazard (§55) is handled by train.py's
existing `rebuild_loss_weight_matrix()` call after warm-start load. **No new
config-dependent buffer is introduced by this program** — `p_diag` and
`adaln_state_weight` are plain float attributes, not buffers, so they cannot be
clobbered by `load_state_dict`. Verified: neither appears in any `state_dict`.

## 4. Checkpoints and adaptation criterion (§21, §23)

Checkpoints at **500 / 1000 / 2000 / 5000** steps. At each, evaluate OFFLINE
(no control, no GPU-expensive rollouts):

- diagonal replay state error and executed-action error;
- off-diagonal prediction across the 2D grid;
- distance of off-diagonal behaviour from diagonal behaviour;
- LIVE vs EMA;
- parameter drift from canonical;
- gradient health.

**Stop at the shortest budget that demonstrably adapts.** If off-diagonal quality
has plateaued by 2000, do not train to 5000 merely because it was registered. If
still improving strongly at 5000, one extension to 10000 is authorized if the
projection fits (§23).

## 5. The confound this pilot must avoid

The closed warm-start branch failed because it queried times the model had never
been trained on, and the apparent off-diagonal structure tracked *time-embedding
distance from the training diagonal*
(corr = −0.995/−0.999/−1.000). This pilot **trains on off-diagonal times**, which
removes that confound by construction.

**Predeclared adaptation check:** the UWM arm must show the off-diagonal penalty
*shrinking* relative to the canonical warm-start baseline. If off-diagonal error
still correlates near-perfectly with distance-from-diagonal after training, the
model has not adapted and the Stage-E field study is not yet meaningful.

## 6. Predeclared interpretation (§30)

| outcome | criterion |
|---|---|
| **PROMISING** | effect exceeds practical evaluator resolution (~3 pp) **or** a major qualitative capability gain (e.g. a working inverse-dynamics endpoint the baseline lacks) |
| **UNRESOLVED** | small effect, mixed sets, or difference at evaluator scale |
| **NEGATIVE** | clear degradation or instability |

A one-seed pilot is **not** evidence of general improvement. If the easy-task
effect is <~3 pp, prefer the harder-task stage over burning budget proving a tiny
effect (§30).

## 7. Compute (measured unit costs)

| item | GPU-h |
|---|--:|
| UWM + CTL, 1 seed, 5k steps | ~0.84 |
| Stage-E offline 2D grid + path diagnostics | ~0.15 |
| closed-loop gate: R=3 × 2 models | 1.327 |
| **pilot subtotal** | **~2.31** |

Against the 10.0 cap, with direct-action (~1.08) that is ~3.39 spent, leaving
~6.6 for confirmation / harder task / registers.

## 8. What would make this stage STOP

- diagonal compatibility regressing (it is re-checked at load);
- NaNs or gross imagination collapse;
- the two arms producing byte-identical checkpoints (the §54 divergence
  sentinel — this is exactly the historical kwargs-defect failure);
- projected total breaching 10.0 GPU-h.
