# Two-Time Flow — Formalization, Architecture, and Experiment Design

Zero-GPU design document. **No two-time training has been run.**

---

## §4. Current architecture trace (exact)

Verified against source, not inferred.

```
RAW BATCH
  trajectories : [B, H=5, D=483]      D = action_dim(3) + obs_dim(480)
  conditions   : {0: [B,480], 4: [B,480]}     observations only

SPLIT                                               (pint.py:165-167)
  actions      = x[:, :, :3]                        [B,5,3]
  x_particles  = x[:, :, 3:].view(B,5,48,10)        48 = 24 particles x 2 views

PROJECTION / TOKENIZATION                           (pint.py:170-181)
  A   = action_projection(actions)   + action_encoding        [B,5,256]
  S_k = particle_projection(x_part)  + view{1,2}_encoding     [B,5,48,256]
  x_cat = cat([A.unsqueeze(2), S], dim=2)                     [B,5,49,256]
        -> TOKEN 0 IS THE ACTION; TOKENS 1..48 ARE STATE PARTICLES

TIME CONDITIONING                                   (pint.py:189-191)
  e = time_mlp(t)                                             [B,256]
  x_proj = x_cat + e[:,None,None,:]        broadcast to ALL 49 tokens

TRANSFORMER                                         (transformer_modules.py:493)
  c = action_embed + e[:,None,:]           AdaLN conditioning, all blocks
  particles_trans = F(x_proj.permute(0,2,1,3), A, e)

DECODE                                              (pint.py:207-212)
  action_out   = action_decoder(  particles_trans[:,:,0,:] )   [B,5,3]
  particle_out = particle_decoder(particles_trans[:,:,1:,:])   [B,5,480]
  x_out = cat([action_out, particle_out], -1)                  [B,5,483]

MASK / CONDITIONING                                 (flow_matching.py:211-221)
  x[:, t, 3:] = value        for t in {0, 4}       OBSERVATIONS ONLY
  mask[:, t, 3:] = False     -> those coords carry zero loss
```

**Two findings that shaped the design.**

1. **State and action are literally separate transformer tokens.** Action is token
   index 0 with its own projection/encoding/decoder; state particles are tokens
   1..48. This makes per-block time conditioning surgical rather than structural.
2. **`e` enters twice** — the token-wise add *and* the AdaLN conditioning vector
   `c`, which modulates every block via `gamma`/`beta`. A correct retrofit must
   handle both paths; handling only the token add would leave the whole network
   silently conditioned on a single time.

## §6. Two-time flow math

`t = 0` is noise, `t = 1` is clean (Tal's convention).

```
x^s(t_s) = (1 − t_s)·z^s + t_s·x^s_1        state block  (dims 3:483)
x^a(t_a) = (1 − t_a)·z^a + t_a·x^a_1        action block (dims 0:3)

v^s* = x^s_1 − z^s                          targets are path-independent
v^a* = x^a_1 − z^a                          (linear interpolant)

v_θ( x^s(t_s), x^a(t_a), t_s, t_a, cond ) -> (v̂^a, v̂^s)
```

Loss keeps the current masked form; only the corruption and time conditioning
change:

```
L = Σ_blocks  masked_mean( |v̂ − v*| ⊙ W ⊙ mask )
```

**Note (our convention):** the repo's `_compute_flow_loss` uses
`target = noise − data`, the sign-flipped convention. The implementation follows
the repo, not the sketch above; the unit tests assert the repo convention.

## §9. What t_state means for conditioned tokens — IMPORTANT

The conditioning mask overwrites `x[:, t, 3:]` for `t ∈ {0, 4}` **after**
corruption and re-imposes it at every solver step. Therefore:

| token group | is it corrupted by t_state? | in loss? |
|---|---|---|
| **current observation** (row 0, obs slice) | **NO — clamped clean at every step** | no (masked) |
| **goal** (row 4, obs slice) | **NO — clamped clean at every step** | no (masked) |
| **predicted state trajectory** (rows 1–3, obs slice) | **YES — this is what t_state governs** | yes |
| **action tokens** (rows 0–4, action slice) | **YES — governed by t_action** | yes (row 4 is the zero placeholder) |

**So the semantic statement must be about the *unconditioned* predicted state
trajectory (rows 1–3), not about the observation or goal.** It would be wrong to
say "the goal becomes noisy at low t_state" — it never does. Every diagram and
figure must reflect this.

## §5/§7. The three semantic regimes

| regime | condition | meaning (for the *predicted* state rows) |
|---|---|---|
| **joint generation** | `t_s = t_a` | current EC-Diffuser; one clock for both |
| **inverse-model / plan-to-action** | `t_s > t_a` | predicted state plan is cleaner than actions ⇒ infer actions that execute a given plan |
| **world-model / forward-dynamics** | `t_a > t_s` | actions are cleaner than the predicted future state ⇒ predict the state that follows given actions |

Wording discipline: "inverse-model regime" / "plan-conditioned action inference",
**not** classical inverse dynamics — EC-Diffuser is a trajectory-level joint model
conditioned on a goal, not a per-step `(s_t, s_{t+1}) → a_t` map.

## §8. Warm-start: algebraic proof + empirical test

**Original** (pint.py:190-191, transformer_modules.py:493):

```
e      = time_mlp(t)
token0 = A   + e ;   token_k = S_k + e   (k ≥ 1)
c      = action_embed + e
```

**Two-time** (`twotime_denoiser.py`), reusing the *same* `time_mlp`:

```
e_s = time_mlp(t_s) ;  e_a = time_mlp(t_a)
token0 = A   + e_a
token_k = S_k + e_s        (k ≥ 1)
c      = action_embed + [ w·e_s + (1−w)·e_a ]
```

**Proof at `t_s = t_a = t`:**

1. `time_mlp` is deterministic ⇒ `e_s = e_a = e`.
2. token 0: `A + e_a = A + e` ✓ ; tokens k≥1: `S_k + e_s = S_k + e` ✓
3. AdaLN: `w·e + (1−w)·e = e·(w + 1 − w) = e` ⇒ `c = action_embed + e` ✓
   **for any `w`**
4. `F`, its weights, decoders and encodings are untouched.

Every input to `F` is identical ⇒ output identical. **QED**

Two consequences: `w` is a **pure off-diagonal knob** that provably cannot affect
the diagonal; and there are **zero new parameters**, so the canonical `state_dict`
loads strictly.

**Empirical confirmation:** diagonal `max|base − two| = 0.000e+00` (bit-exact);
off-diagonal genuinely differs (`3.78e-02`); parameter delta **0**; measured
CPU fwd+bwd ratio **0.989×** at the canonical architecture (256-dim, 6 layers,
8.10 M params) — overhead within noise.

## §19. Unit tests — 22/22 PASS

`test_two_time.py`, CPU only:

| # | test | result |
|--:|---|---|
| 1 | diagonal corruption == ordinary CFM | PASS (0.000e+00) |
| 2 | extreme inverse `t_s=1,t_a=0`: state clean, action pure noise | PASS |
| 3 | extreme world `t_s=0,t_a=1`: action clean, state pure noise | PASS |
| 4 | conditioning mask exactly imposed for all `(t_s,t_a)` | PASS |
| 5 | velocity targets correct per block | PASS |
| 6 | shapes | PASS `(4,5,483)` |
| 7 | `t_action` and `t_state` each change the output | PASS |
| 8 | zero added params; diagonal **bit-exact** vs parent | PASS (0.000e+00) |
| 9 | finite nonzero grads in diagonal / inverse / world | PASS |
| 10 | all 5 schedules: `(0,0)→(1,1)`, monotone, in `[0,1]` | PASS |

## §10. Training distribution — computed, not described

400k samples each.

| | P(exactly diagonal) | P(t_s>t_a) | P(t_a>t_s) | marginal SD | mean \|t_s−t_a\| | boundary mass |
|---|--:|--:|--:|--:|--:|--:|
| **A** independent uniform | **0.0000** | 0.499 | 0.501 | 0.289 | 0.334 | — |
| **B** mixture ⅓/⅓/⅓ | **0.333** | 0.334 | 0.333 | 0.289 | 0.222 | — |
| **B'** mixture p_diag=0.5 | **0.499** | 0.252 | 0.249 | 0.289 | 0.167 | — |
| **C** offset `t ± δ/2` | 0.0000 | 0.498 | 0.502 | **0.304** | 0.229 | **0.124** |

**Recommendation: Option B (or B′).**

- **A is disqualified for warm-start**: the diagonal has *exactly zero* mass, so
  fine-tuning would never train the configuration we deploy and must be expected
  to drift off the bit-exact starting policy. This is the catastrophic-forgetting
  risk in §11, made concrete.
- **C distorts the problem**: clipping puts **12.4%** of mass on the boundary and
  inflates the marginal SD from 0.289 to 0.304, so it changes marginal training
  difficulty while *looking* symmetric — exactly the failure the brief warns about.
- **B keeps marginals uniform (SD 0.289, identical to A) while retaining explicit
  diagonal mass**, and is symmetric between the two off-diagonal regimes.
- **Precedent**: DM1 forces `r=t` for **ρ=0.5** of samples and MVP/IVC adds a
  dedicated diagonal loss — two independent groups found it necessary to
  over-train the degenerate sub-manifold of a two-time scheme.

`p_diag` is the single most important knob and should be **predeclared, not
tuned**: B (⅓) if we want maximal off-diagonal signal, B′ (½) if diagonal
preservation is the priority. Given warm-start starts from a *competent* diagonal
policy we care about not destroying, **B′ (p_diag = 0.5) is the safer first
choice**, with the diagonal-preservation control (§12 CONTROL 3) as the check.

## §11/§15. Inference schedules — one-parameter family

```
t_state(s)  = s^(2^(−b))
t_action(s) = s^(2^(+b))
```

`b = 0` diagonal; `b > 0` state-leading (inverse-biased); `b < 0` action-leading
(world-biased). Endpoints `(0,0)→(1,1)` exactly for every `b`; monotone; within
`[0,1]`; **`b → −b` exactly swaps the two roles**; NFE unchanged (a path
reparameterization, not extra network calls).

| b | γ_s | γ_a | mean(t_s−t_a) | max lead | regime |
|--:|--:|--:|--:|--:|---|
| −1.0 | 2.000 | 0.500 | −0.330 | 0.473 | strong action-leading |
| −0.5 | 1.414 | 0.707 | −0.170 | 0.250 | mild action-leading |
| 0.0 | 1.000 | 1.000 | 0.000 | 0.000 | diagonal |
| +0.5 | 0.707 | 1.414 | +0.170 | 0.250 | mild state-leading |
| +1.0 | 0.500 | 2.000 | +0.330 | 0.473 | strong state-leading |

Recommended set: **S0…S4 = b ∈ {0, ±0.5, ±1}**. The piecewise schedule
(`t_s=min(1,2s)`, `t_a=max(0,2s−1)`) is retained as an interpretable **extreme
control**, not part of the main family — it gives a clean "state fully clean
before action starts" reference point.

This yields the figure the brief wants: *error / control vs schedule bias `b`*,
a single axis, rather than five unrelated handcrafted curves.

## §14. Two distinct questions — kept separate

**Question A — conditional capability.** At a *fixed* `(t_s, t_a)` cell: does
cleaner state context reduce action error, and does cleaner action context reduce
state error? Produces the 2D heatmaps (§12 of the brief). **This can establish the
inverse/world-model interpretation even if no inference path improves control.**

**Question B — generation path.** Starting at `(0,0)` and ending at `(1,1)`: does
the *path* change final generation and control? Produces the scheduler result.

These must not be merged. A is a statement about the learned conditional; B is a
statement about sampling dynamics. A positive A with a null B would still be a
publishable mechanistic finding.

## §12. Predeclared controls

| id | what | purpose |
|---|---|---|
| **CONTROL 1** | canonical Flow checkpoint, unmodified | reference policy |
| **CONTROL 2** | two-time retrofit **at init**, diagonal schedule | must reproduce CONTROL 1 **exactly** (already proven bit-exact) |
| **CONTROL 3** | after fine-tune, **diagonal** schedule | isolates "did fine-tuning itself change the policy?" |
| **EXP-INV** | same fine-tuned ckpt, state-leading `b>0` | inverse-model regime |
| **EXP-WLD** | same fine-tuned ckpt, action-leading `b<0` | world-model regime |

Without CONTROL 3, any schedule effect is confounded with the effect of
fine-tuning. **One checkpoint, multiple schedules** (§13) removes model-to-model
training noise from the schedule comparison.
