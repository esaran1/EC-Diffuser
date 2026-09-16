# UWM audit — Unified World Models

**Identified:** *Unified World Models: Coupling Video and Action Diffusion for
Pretraining on Large Robotic Datasets* — Zhu, Yu, Feng, Burchfiel, Shah, Gupta
(UW + Toyota Research Institute), **RSS 2025**, **arXiv:2504.02792** (v3 audited).

- https://arxiv.org/abs/2504.02792 · https://arxiv.org/html/2504.02792v3
- Project: https://weirdlabuw.github.io/uwm/
- Code: https://github.com/WEIRDLabUW/unified-world-model @ `b7d4067c8859775e5b08f681e76d538015cdd445`

Note several unrelated 2026 papers use similar framing (2606.01027, 2604.26694);
everything here is strictly from 2504.02792 and its repo.

## 1. Forward noising — DDPM, NOT flow matching

```
q(x_t | x_{t-1}) = N(x_t | sqrt(1-β_t) x_{t-1}, β_t I)
x_t = sqrt(ᾱ_t) x_0 + sqrt(1-ᾱ_t) ε
a_{t_a}     = sqrt(ᾱ_{t_a}) a  + sqrt(1-ᾱ_{t_a}) ε_a
o'_{t_{o'}} = sqrt(ᾱ_{t_o'}) o' + sqrt(1-ᾱ_{t_o'}) ε_{o'}
```

One shared schedule (`DDIMScheduler`, `squaredcos_cap_v2`,
`num_train_timesteps=100`). **Epsilon-prediction** throughout. No flow matching
anywhere.

## 2. Time convention — **t=0 CLEAN, t=T NOISE — the OPPOSITE of ours**

> "Time-steps closer to T (fully noised) indicate full masking, while timesteps
> closer to 0 (unnoised) indicate no masking."

Discrete integers in [0, T), T=100. **Our flow convention is t=0 noise, t=1
clean.** Any port must invert the axis, and the corner semantics invert with it:

| UWM | meaning | our flow equivalent |
|---|---|---|
| `t = T` | fully noised / **marginalized** | `t = 0` |
| `t = 0` | clean / **conditioned** | `t = 1` |

## 3. Modality times — two

`t_a` (action) and `t_{o'}` (next observation). Network is
`s_θ(o, a_{t_a}, o'_{t_o'}, t_a, t_o')`.

**Asymmetry worth noting:** the *current* observation `o` is never noised and
carries no timestep — it is clean conditioning. The 2D time space is
(action, next-obs); current obs sits outside it.

## 4. Training distribution — **INDEPENDENT, NO DELIBERATE DIAGONAL MASS**

This was the critical question. Paper III-B: "we **independently sample** action
timestep t_a and next observation timestep t_{o'}"; Appendix -A 2: "independently
from the uniform distribution U[0,T)". Code confirms — two separate
`torch.randint` calls, no coupling, no mixture:

```python
action_t   = torch.randint(0, num_train_steps, (batch_size,))
next_obs_t = torch.randint(0, num_train_steps, (batch_size,))
```

**The training measure is the plain product `U{0..99} ⊗ U{0..99}`.** The diagonal
is hit only by chance, at rate 1/100.

> Consequence: the undocumented `sample_joint` denoises along the exact diagonal
> — the *least*-trained path. Equally, each axis endpoint gets only ~1% of
> training mass. The scheme works only because the network generalizes smoothly
> across conditioning time.

The only deliberate deviation is a deterministic clamp for action-free video:
`action_t[~action_mask] = num_train_steps - 1`. And note Appendix -A 2: "The
action diffusion loss is computed **across all samples in a batch**" — for
action-free video the model regresses the *imputed pure noise* as the action
target; the action loss is **not** masked out. (The PAD baseline does mask it.)

## 5. Conditioning — AdaLN on a concatenated global vector

Current-obs features and **both** timestep embeddings concatenate into one global
vector driving AdaLN in every DiT block. Noisy actions and noisy next-obs are
**in-context tokens**, not AdaLN conditioning. Full bidirectional attention.

## 6. Tokenization

- **Action**: chunk of 16; shallow MLP shared across timesteps → 16 tokens.
- **Next-obs**: latent diffusion — frozen SDXL VAE (224×224×3 → 28×28×4), 3D conv
  patchifier (2,4,4). Diffusion happens entirely in VAE latent space.
- **Current obs**: ResNet-18, fine-tuned — a *different* pathway from the VAE.
- **Registers**: 8 learnable tokens, appended, discarded at output. Ablated and
  reported to help; hypothesis is they provide "an intermediary medium to
  exchange information" between modalities.
- **Low-dim state**: encoder exists but `use_low_dim=False` in the released
  config. Modalities are exactly {action, next-obs-video} — there is no separate
  "state" diffusion modality.

## 7. Loss

```
ℓ(θ) = E[ w_a ||ε^θ_a − ε_a||² + w_{o'} ||ε^θ_{o'} − ε_{o'}||² ]
```

**Paper/code discrepancy:** the paper introduces weights `w_a, w_{o'}` but the
code hardcodes both to 1 with `F.mse_loss` (mean over elements), so the effective
balance is an implicit per-element normalization. No SNR weighting.

## 8. Time embeddings — SHARED featurizer, then a JOINT MLP over the pair

The single most implementation-critical finding.

```python
class DualTimestepEncoder(nn.Module):
    self.sinusoidal_pos_emb = SinusoidalPosEmb(embed_dim)   # ZERO params, shared
    self.proj = nn.Sequential(Linear(embed_dim*2, hidden), Mish(), Linear(hidden, embed_dim))
    def forward(self, t1, t2):
        return self.proj(torch.cat([sinu(t1), sinu(t2)], dim=-1))
```

Three parts: (a) the sinusoidal featurizer is **parameterless**, so "shared" is
nearly vacuous; (b) there is **no separate learned MLP per modality**; (c) the
two are **concatenated and passed through a single joint MLP**, producing **one
fused vector representing the pair jointly**.

> **Implication for our port.** If per-modality time embeddings are two
> independent vectors that are summed or separately injected, that is **not**
> UWM. UWM's AdaLN sees a *joint nonlinear function* of (t_a, t_o'), strictly
> more expressive than additive per-modality embeddings, and that is what lets
> the network learn corner-specific behavior ("when t_o' is at max, ignore video
> tokens").

Our existing `TwoTimePINTDenoiser` uses the additive/linear-mix form
(`c = action_embed + (w·e_s + (1−w)·e_a)`). It is **zero-parameter and exactly
diagonal-compatible**, which UWM's joint MLP is not — but it is *less expressive
off-diagonal*. This is a real, documented divergence to be declared, not hidden.

## 9. Sampler — DDIM

`num_inference_steps=10` over 100 train steps. Paper Eqs. 2–5 are written in
ancestral DDPM form but the code ships DDIM — a minor paper/code mismatch.

Non-obvious detail: with diffusers' "leading" spacing, `set_timesteps(10)` gives
`[90,80,...,10,0]`. So the inference "T" corner is **90**, while the training
clamp for action-free video is **99** — a **90-vs-99 mismatch** acknowledged
nowhere. Presumably benign (both deep in the near-pure-noise regime) but real.

## 10–13. The four endpoints — all by setting timesteps only

| endpoint | construction |
|---|---|
| **policy** `p(a\|o)` | `t_{o'} = T`, `o'_T ~ N(0,I)`; denoise action T→0 |
| **video prediction** `p(o'\|o)` | `t_a = T` fixed; denoise next-obs |
| **forward dynamics** `p(o'\|o,a)` | `t_a = 0`, clean action; denoise next-obs |
| **inverse dynamics** `p(a\|o,o')` | `t_{o'} = 0`, clean next-obs; denoise action |

No architectural change, no retraining, no classifier-free guidance. The deployed
policy is `sample_marginal_action`: `next_obs_t` pinned at 90, the next-obs
sample drawn once and **never updated**.

**Joint generation is CODE-ONLY** (`sample_joint`, diagonal `t_a = t_{o'} = t`),
used for training-time visualization, never for evaluation. The paper enumerates
exactly four modes and does not describe it.

## 14. Interior paths — **NO. This is the open territory.**

All four paper modes **pin one modality's time at a corner (T or 0) and sweep the
other** — these are **axis-parallel edges** of the [0,T]² square. The
undocumented `sample_joint` rides the **main diagonal**. Nothing else exists.

> "Fixing the timestep for either t_a or t_{o'} to T marginalizes the
> corresponding variable, while setting the timestep to 0 performs conditioning."

The one gesture at a continuum is the masking analogy — "Time-steps closer to T
indicate full masking" — but it is about the *semantics* of intermediate times,
not traversing them. Training covers the interior uniformly; **neither paper nor
code ever runs a sampler along an interior path.**

**This is an explicit, quotable gap, and it is precisely our registered Stage E.**

## 15. Code beyond the paper

1. `sample_joint` — unpublished diagonal sampler, visualization only.
2. Loss weights `w_a, w_{o'}` in paper, absent from code (hardcoded 1:1).
3. Eqs. 2–5 ancestral DDPM; code is DDIM.
4. The 90-vs-99 clamp mismatch.
5. `use_low_dim` / `use_language` paths exist but disabled in the released config.

## 16. Conditioning masks × modality times — **the mask IS a time clamp**

UWM has **no attention mask and no mask token** for modality masking. Masking is
implemented *entirely* by clamping a diffusion timestep to T:

```python
action_t[~action_mask] = self.num_train_steps - 1
```

By contrast the PAD and GR1 baselines in the same repo use genuine attention
masks. In UWM the conditioning mask does not "interact with" modality times — it
**is** a modality time. That is why one mechanism serves both training-time
masking and inference-time marginalization.

## 17. Bottom line for our program

1. **Convention inverts** (t=0 clean → our t=1 clean). Corner semantics invert.
2. **Faithful UWM training has NO diagonal mass** — our `twotime_flow.py`
   defaults to `p_diag=0.5`. Faithful = `p_diag=0.0`; the §20 mixture is a
   *rescue*, not the primary.
3. **Time embedding is a joint MLP over the pair**, not additive per-modality.
   Our zero-param design diverges here — declared, and a documented limitation.
4. **Interior-path traversal is unclaimed by UWM.** That is our Stage E.

## Could not determine

Continuous-time/flow variant (does not exist); any interior-path sampler
(absent); numeric `w_a, w_{o'}`; rationale for 90-vs-99; v1/v2 diffs (v3 only
audited); whether released checkpoints match this exact code path.
