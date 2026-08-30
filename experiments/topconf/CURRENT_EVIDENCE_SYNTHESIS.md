# Current evidence synthesis

Verified against artifacts on commit `2b34ffb`, 2026-08-30.

## Verification of the frozen evidence (§1)

| item | claimed | verified |
|---|---|---|
| seed 42 checkpoint | `861dc344…` | **`861dc34434474455`** ✓ |
| seed 43 checkpoint | `c8e00ead…` | **`c8e00eadfed9b8a0`** ✓ |
| seed 44 checkpoint | `c2c13f55…` | **`c2c13f557aca7cf0`** ✓ |
| internal step | 499000 | **499000** ✓ (file named `state_400000.pt`) |
| EMA present | yes | **yes** ✓ |
| parameter count | ~60.65M | EMA state_dict 60,649,640 elements (incl. buffers); `.parameters()` = 60,646,925 ✓ |
| episode set | `replicate0_n96` | stored sha `35144910b1471b7b…` ✓ |

## Architecture audit (§9) — favorable for a projected method

`AdaLNPINTDenoiser.forward` (`diffuser/diffuser/models/pint.py:144-215`):

- the **action is its own token at index 0**, produced by a dedicated
  `action_projection` + learned `action_encoding`;
- state particles go through a separate `particle_projection` (+ view encodings);
- tokens are concatenated `[action_token, particle_tokens]` and attended jointly;
- **decoding is split**: `action_decoder(particles_trans[:,:,0,:])` and
  `particle_decoder(particles_trans[:,:,1:,:])` — *no shared output head*.

Consequence: an action-specific conditioning signal (e.g. step size) could be
injected into the action token or the action decoder **without touching the
state pathway**. A hook already exists — `_time_embedding(self, time, interval)`
— which AdaLNPINT currently rejects (`"does not support interval conditioning"`,
line 141) but a sibling denoiser implements (line 230). This is the natural
extension point, and it means a projected method is architecturally cheap.

**Level 0** (code-verified).

## The central tension this synthesis exposes

The evidence, taken together, does **not** currently support the motivating story
for an action-projected shortcut method. Specifically:

1. **Offline, NFE1's action is already as good as NFE2's.** Recorded-replay
   action error: 0.01785/0.01742/0.01944 (NFE1) vs 0.01717/0.01605/0.01761
   (NFE2). On seed 42 NFE1 is *better*. The gap is ≤0.002 in action units.
2. **The action distribution is nearly unchanged.** 8 noises/condition, seed 44:
   pairwise sample spread 0.00837 (NFE1) vs 0.00713 (NFE2); mean action magnitude
   0.20726 vs 0.20631. No collapse, no large multimodality difference.
3. **Yet closed-loop NFE1 is ~6 pp worse** (directionally, 3/3 seeds, §1.4).

So the ~6 pp control penalty is **not** explained by the action endpoint's
distance to the demonstrated action, nor by its sampling distribution, on
in-distribution replay. A method that targets the one-step action endpoint is
therefore aiming at a quantity that is *already accurate*.

### What could explain the penalty instead

- **(H-A) Closed-loop compounding.** Offline replay scores a single decision from
  a recorded state. Control executes 100 sequential decisions from *self-induced*
  states. Small action differences compound, and the state branch — which is
  ~49% worse at NFE1 (0.2096 vs 0.1076) — may matter through the trajectory the
  policy actually visits, even though it does not affect the action *within* one
  step.
- **(H-B) Off-distribution states.** The replay diagnostic only ever queries
  recorded, in-distribution states. NFE1 may degrade specifically on the
  off-distribution states a policy drives itself into.
- **(H-C) Evaluator noise.** The +6 pp had only 1 of 3 CIs excluding zero.

**H-A/H-B are the same in effect: the action is fine where we measured it, and
the measurement location is the problem.** This is testable offline and cheaply
(see roadmap), and it must be resolved before any method commitment.

## Status of the candidate method

**Not supported yet.** The diagnostics intended to motivate a projected action
shortcut instead falsified its premise: there is no measurable one-step action
deficit to compress away on in-distribution replay.

This does not kill the project — the *phenomenon* (state and action blocks having
radically different NFE requirements, §1.6) is verified and replicated across
three checkpoints. But the method must target the mechanism that actually causes
the closed-loop penalty, and we have not yet identified it.
