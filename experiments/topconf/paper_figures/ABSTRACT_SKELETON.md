# Abstract Skeleton

Every bracket is a slot backed by a frozen number. Nothing here needs E1.

**[Setup]** Fast generative policies are commonly evaluated by comparing success
rates across sampler budgets on a fixed benchmark of scenarios.

**[Observation]** In an entity-centric diffusion/flow manipulation benchmark on GPU
physics, we find that fixing the policy, the scenario, and the policy's sampling
noise does **not** fix the outcome. Across 8 physics realizations of 96 scenarios,
outcomes bifurcate rather than jitter: successful realizations end 0.018 m from
goal, failed ones 0.304 m, against a 0.04 m threshold.

**[Consequence]** This produces an evaluator resolution floor. Single-realization
evaluation resolves differences no finer than ~7.6-8.1 pp (95% half-width, N=96).
Across 12 calibrated NFE contrasts, **14.8%** of nested single-realization views
are strict sign reversals against a 3-realization reference, and **25.0%** disagree in
practical category (predeclared +/-5 pp). Per-contrast sign disagreement ranges
0%-56%, concentrated on small effects.

**[Method]** We give a calibration procedure that estimates the resolution floor
from same-policy repeats and predicts, before running a comparison, how many
physics realizations it needs. Calibration is held out: variance is estimated from
checkpoints excluding the comparison being predicted.

**[Result — positive]** Held-out signal-to-resolution ratio predicts **sign**
reliability monotonically, 66.7% to 97.2%.
(Exact ties are not counted as reversals; see the ledger.)

**[Result — negative, must be stated]** The same calibration does **not** predict
*practical-category* reliability, which stays flat at 72-78% across every ratio
bin. Knowing an effect is resolvable in sign does not make its magnitude
trustworthy.

**[Scope — must be stated]** One simulator family, one task family, three
checkpoints, two task complexities. Physics nondeterminism in this simulator is
vendor-documented; our contribution is its measured consequence for policy
comparison, not its discovery. Of 12 contrasts, **0 have CIs excluding zero** — we
report a resolution limit, not an equivalence result.
