# Stage B — EC-Diffuser architecture / tokenization audit

Traced from production code at HEAD `ff66cc4`. Every claim below was read out of
the source or the checkpoint, not inferred from a name.

## 1. Model path

| component | path |
|---|---|
| denoiser | `diffuser/diffuser/models/pint.py` → `AdaLNPINTDenoiser` (L35) |
| transformer | `diffuser/diffuser/models/transformer_modules.py` → `AdaLNParticleTransformer` (L406) |
| generative wrapper | `diffuser/diffuser/models/flow_matching.py` → `ConditionalFlowMatching` (L13) |
| evaluator | `experiments/policy_improvement/phase2_diagnostics.py` |

## 2. Tensor shapes (read from the canonical checkpoint)

| quantity | value |
|---|--:|
| horizon `H` (`block_size`) | 5 |
| `transition_dim` | 483 |
| `action_dim` | 3 |
| observation dim | 480 |
| `features_dim` (per particle) | 10 |
| n particles | **48** = (483−3)/10 |
| `projection_dim` | 512 |
| transformer blocks (in checkpoint) | 8 |
| heads | 8 |
| total params | 60,649,640 |
| checkpoint internal step | **499000** (file is named `state_400000.pt` — name ≠ step) |

`multiview = True`: the checkpoint has `view1_encoding`/`view2_encoding` and
**no** `particle_encoding`, so the 48 particles are 2 views × 24 particles.

## 3. Token construction — the central fact

`AdaLNPINTDenoiser.forward` (pint.py L163-196):

```
actions      = x[:, :, :3]                    # [bs, T, 3]
x_particles  = x[:, :, 3:].view(bs, T, 48, 10)

action_particle     = action_projection(actions) + action_encoding
new_state_particles = particle_projection(x_particles) + view{1,2}_encoding

x_cat = cat([action_particle.unsqueeze(2), new_state_particles], dim=2)
        # -> [bs, T, 1+48, 512]
```

**Token index 0 is the action; indices 1..48 are DLP particles.** This holds at
every one of the 5 horizon positions, so the sequence the transformer sees is
`[bs, 49, 5, 512]` after the permute.

Decoding is **split, with no shared head**:

```
action_decoder_out   = action_decoder(particles_trans[:, :, 0, :])    # token 0
particle_decoder_out = particle_decoder(particles_trans[:, :, 1:, :]) # tokens 1..48
x_out = cat([action_decoder_out, particle_decoder_out], dim=-1)
```

## 4. Time conditioning reaches the network by TWO paths

This is the load-bearing detail for any two-time retrofit.

**Path 1 — additive, per token** (pint.py L192):
```
t_embed = time_mlp(time)                 # [bs, 512]
x_proj  = x_cat + t_embed[:, None, None, :]   # broadcast to ALL tokens
```

**Path 2 — AdaLN modulation** (transformer_modules.py L494):
```
c = action_embed + t_embed.unsqueeze(1)  # combined conditioning
... block(x, c) for every block ... ; ln_f(x, c)
```

Note that `c` mixes the **projected action tokens** with the time embedding, and
a single `c` modulates every token in every block. A faithful two-time model must
decide what to do with Path 2, because it is inherently shared across modalities.

`time_mlp` is a **single shared network** (`SinusoidalPosEmb → Linear → GELU →
Linear`, L88-93). There is exactly one of them — which is what makes a
zero-added-parameter two-time construction possible (§17 of the directive).

## 5. Flow convention — VERIFIED, not assumed

`ConditionalFlowMatching._compute_flow_loss` (flow_matching.py L305-333):

```
x0_local = torch.randn_like(x1)                 # x0 = NOISE
xt       = (1-t)*x0_local + t*x1_local          # L330
target_velocity = x1_local - x0_local           # L332
prediction = self._call_model(xt, cond, time * self.time_scale)
```

⇒ **t=0 is noise, t=1 is clean data**; target is `data − noise`. `time_scale` =
1000.0. `time` has shape `(B,)` — **one scalar per batch element**, broadcast by
`.view(-1,1,1)` across both horizon and all 483 channels. State and action
therefore share a single time *by construction* in the baseline.

## 6. Sampler

`conditional_sample` (L385-452): left-endpoint fixed-step forward Euler.

```
x  = randn(...)                       # start at t=0 (noise)
dt = 1/steps
for step in range(steps):
    time     = step/steps
    velocity = model(x, cond, time*time_scale) * conditioning_mask
    x        = x + dt*velocity
    apply_conditioning(x, cond)
```

Evaluator uses `steps = 4` (NFE=4). No value head; `sort_by_value` is inert.

## 7. Conditioning / inpainting

`_apply_conditioning` (L264): for each conditioned timestep, **observation
coordinates only** are overwritten — `x[:, t, action_dim:] = value`. Actions are
never conditioned. `_make_conditioning_mask` (L270) marks those observation
entries `False` so they are excluded from the loss.

The evaluator conditions timestep 0 with the current observation and the last
row with the goal observation (goal-conditioned inpainting).

## 8. Terminal action target

`_make_conditioning_mask` L275-279 documents it: the `t=H-1` action target is a
**normalized zero placeholder, not an executed action** — constructed by
`GoalDataset.__getitem__` as goal-observation + `normalize(zeros)`. It is NOT
sequence padding. Baseline keeps it supervised (`mask_terminal_action` defaults
off). Per the directive this stays as-is.

## 9. Registered buffer hazard (still live)

`loss_weight_matrix` is a **registered buffer of shape (5, 483)** serialized into
every checkpoint, so `load_state_dict` overwrites the constructor's configured
value. `rebuild_loss_weight_matrix()` (L105) exists to repair this after load.
Any new config-dependent buffer in this program must be audited the same way
(§55).

## 10. Positional structure

`positional_bias=False` for this config ⇒ `pos_emb` is a learned parameter of
shape `(1, block_size, n_embed)` added over the **time/horizon** axis only
(transformer_modules.py L499-501). Particle index carries **no** positional
embedding — the particle axis is permutation-structured, not ordered. This
matters for register-token placement (§37): a register added along the particle
axis inherits no positional identity.

## 11. Existing direct-action variant

**None found in production code.** A grep across the repo for
`direct_action|query_token|action_query|no_action_noise` returns only a prose
mention in `experiments/topconf/CURRENT_EVIDENCE_SYNTHESIS.md` describing the
token split above. See `direct_action/DIRECT_ACTION_RECOVERY.md`.
