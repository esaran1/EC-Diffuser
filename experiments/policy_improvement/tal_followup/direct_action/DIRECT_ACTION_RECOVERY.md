# Stage C — direct-action / query-token recovery

**Finding: NO direct-action implementation, checkpoint, or result exists anywhere
in this repository. The finding cannot be *recovered*; it would have to be
*built*.**

This is a provenance result, and it changes the program's plan, so it is stated
before anything else.

## 1. Search performed (exhaustive, zero GPU)

| where | how | result |
|---|---|---|
| working tree | `grep -rln "direct_action\|query_token\|action_query\|no_action_noise\|action_token"` over `*.py`/`*.md` | 1 hit, prose only (see §2) |
| all git history, all branches | `git log --all --grep=` for direct-action/query/action-token | 5 hits, all unrelated (seed-launch commits, "solver bias direction") |
| every file ever added | `git log --all --diff-filter=A --name-only` filtered for direct/query | 5 hits, all `loss_balance_audit` direction-audit files |
| checkpoints on disk | `ls data/panda_push/ \| grep -i "direct\|query\|noact"` | none |
| branches | `git branch -a` → `fast-generative-policies`, `flow-matching`, `main` (+origins) | no candidate branch |

## 2. The one prose hit, and what it actually says

`experiments/topconf/CURRENT_EVIDENCE_SYNTHESIS.md` L21-32 describes the token
split (action at index 0, split decoders, no shared head) and concludes that an
action-specific conditioning signal *could* be injected without touching the
state pathway — "the natural extension point, and it means a projected method is
architecturally cheap."

That is an **architectural suitability claim**, code-verified at Level 0. It is
**not** a report of a trained direct-action model, and it makes no performance
comparison.

## 3. Consequence for the program

Tal's observation — that retaining generative state tokens while replacing the
noised action trajectory with an action/query token "also works with only modest
degradation" — has **no corresponding artifact here**. Possible explanations:

- it was observed in a different codebase or by another person;
- it is a conceptual expectation rather than a measured result;
- it was measured somewhere not shared with this repository.

Per directive §14 the rule is: *if neither implementation nor checkpoint exists,
implement the minimal architecture matching Tal's description.* That is a
**from-scratch build plus training**, not a recovery — and it must be costed
against the 10.0 GPU-h cap accordingly. It is priority 2 behind UWM (§6).

**It must not be reported as reproducing a prior finding**, because there is no
prior finding in the record to reproduce.

## 4. Do not confuse it with a diffusion endpoint (directive §15)

A learned query token carrying **no action observation** is architecturally
distinct from either conventional endpoint:

- `t_action = 0` feeds **noise** into the action channel;
- `t_action = 1` feeds the **clean action** (unavailable at inference).

A query token feeds a *learned constant*. Equivalence to `t_action = 0` would
hold only if the model were invariant to the realized noise draw — which is not
given and would have to be demonstrated, not assumed. The two models stay
separate in all analyses.

## 5. Existing evidence that bears on H1 (not a substitute for the experiment)

`experiments/loss_balance_audit/ENRICHED_SOLVER_BIAS_AUDIT.md` (Classification D)
measured, on frozen canonical checkpoints:

> The action prediction degrades ~5-6× MORE than the state prediction

under solver/discretization error — the *opposite* of the audit's §18 hypothesis.

Read conservatively, this is consistent with the action channel being the
**fragile** part of the joint denoising process rather than the robust part. That
is suggestive for H1 ("what is action denoising buying us?") but it is an offline
geometric measurement on a diagonal-trained model, and directive §48 forbids
treating an offline metric as a control claim. Logged as motivation, not
evidence.

## 6. Minimal build specification (if executed)

Matching Tal's description with the smallest possible diff to production:

| element | choice | rationale |
|---|---|---|
| state pathway | **unchanged** — still noised, still generative | isolates the action intervention |
| action input | learned embedding replacing `action_projection(noised_action)` at token 0 | the query token |
| per-horizon | one query token per horizon position (H=5), each learned | matches existing token layout |
| action output | `action_decoder(token 0)` predicts the **action directly** (not a velocity) | "directly map the action-token output to the action" |
| action loss | same L1/L2 family as production, on the direct action | comparability |
| state loss | **unchanged** flow-velocity loss | unchanged |
| time conditioning | state time only; query token still receives `t_embed` via the existing shared path | avoids inventing a second convention |
| params added | H × projection_dim = 5 × 512 = 2,560 (+0 if tied across H) | negligible vs 60.6M |

Open ambiguity to resolve before building: whether the action head should predict
the action at **all H positions** or only the executed first action. Production
executes `actions[0,0]` MPC-style (`exe_steps: 1`, per
`experiments/isaacgym_flow_diagnosis.md` L223), so the first position is what
control actually consumes.
