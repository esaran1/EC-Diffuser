# UWM Differentiation Table (§3) — verified before GPU work

Source: **Unified World Models**, arXiv:2504.02792 (RSS 2025). Read: abs page,
full text v1 + v2 + ar5iv **including all appendices (-A2, A-A1, A-A2, A-A3)**,
project page, and the official code
`github.com/WEIRDLabUW/unified-world-model` (`models/uwm/uwm.py` read in full).

| question | UWM | our proposed test |
|---|---|---|
| independent modality times | **YES** | yes (prior art — not claimed) |
| forward-dynamics endpoint | **YES** (Eq. 4, `t_a=0` frozen) | control only |
| inverse-dynamics endpoint | **YES** (Eq. 5, `t_o'=0` frozen) | control only |
| all combinations seen in training | **YES** (`t_a,t_o' ~ U(0,T)` independent) | yes (prior art) |
| **interior 2D error surface** | **NO** | **proposed** |
| **matched-cleanliness asymmetry** | **NO** | **proposed** |
| **continuous lead/lag parameter** | **NO** | **proposed** |
| **non-diagonal path through the plane** | **NO** | **proposed** |
| **matched-NFE path comparison** | **NO** | **proposed** |
| **non-monotonicity / interaction** | **NO** | **proposed** |
| flow-matching implementation | no (DDPM/DDIM) | yes (difference, not a contribution) |
| object-centric latent state | no (pixels, ResNet-18) | yes (difference, not a contribution) |

## Evidence for each "NO"

**Interior surface.** No heatmap, grid, sweep or table over interior
`(t_a, t_o')` anywhere in text, tables I–IX, or figures 1–14. The only
"all combinations" statements are about **training exposure**, never evaluation:
§III-B *"this training paradigm exposes the model to all combinations of noise
levels of the modalities, which enables flexible sampling at inference"*, and its
twin in §VI. Both are motivational; no measurement accompanies them.

**Every reported inference mode pins a timestep to a boundary** (§III-B, Eqs. 2–5):

| mode | frozen | swept |
|---|---|---|
| policy `p(a\|o)` | `t_o' = T` | `t_a: T→1` |
| video prediction `p(o'\|o)` | `t_a = T` | `t_o': T→1` |
| forward dynamics `p(o'\|o,a)` | `t_a = 0` | `t_o': T→1` |
| inverse dynamics `p(a\|o,o')` | `t_o' = 0` | `t_a: T→1` |

**Code confirms it.** In `models/uwm/uwm.py`, every sampler holds one modality's
timestep as a frozen scalar taken from a schedule endpoint —
`sample_marginal_action` uses `next_obs_t = timesteps[0]` (=T);
`sample_inverse_dynamics` uses `next_obs_t = timesteps[-1]` (=0); mirrored in
`sample_forward_dynamics` and `sample_marginal_next_obs`.

**One unreported code-only mode exists:** `sample_joint` moves both timesteps but
is **strictly diagonal** — the identical `t` is passed to both slots. It appears
only as a training-time visualization call (`experiments/uwm/train.py:111`).

**Training distribution: independent uniform, no diagonal mass.** Eq. 1 and
Appendix -A2: *"sample action and observation diffusion timesteps t_a, t_o'
independently from the uniform distribution U[0,T)"*. Code matches — two separate
`torch.randint` draws. **No ablation over the timestep-sampling distribution
exists** (their ablations cover objective, register count, AdaLN vs
cross-attention, video source, pretrain-vs-scratch).

**Matched-NFE.** Never appears. Inference steps are one fixed hyperparameter for
all modes (Table VI: 10 DDIM steps).

## Consequences for how we must write this

Three cautions, taken directly from the verification:

1. The *"exposes the model to all combinations of noise levels"* sentences
   (§III-B, §VI) could be misread as interior characterization. **State
   explicitly** that these are training-coverage claims with no accompanying
   evaluation.
2. `sample_joint` means **"moving both timesteps at once" is not itself novel.**
   Our differentiation must rest on **non-diagonal** paths, the lead/lag
   parameterization, and matched-NFE path comparison.
3. UWM's PAD comparison already occupies the *"independent beats shared
   timesteps"* claim at the coarse end-to-end level. **Do not frame our
   contribution there.**

## Net

Items 1–6 of the §3 table are **all open**. UWM establishes the 2D domain and the
boundary-mode interpretation, then treats the interior purely as a training-time
regularizer it asserts is beneficial **without ever measuring it**. That is
exactly the gate question, and it survives a careful search of paper, appendices
and code.
