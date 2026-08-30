# Arm-neutral solver-bias evaluation: the state finding does not survive

**No training. No loss change. No new solver. No simulator-authored targets.**
Compute: **1580 s = 0.44 GPU-h** (projected 0.434; inside the 0.5 h limit).
12,672 network evaluations = 3 seeds × 8 noises × (16 + 512) NFE.

Scripts: `arm_neutral_eval.py`, `analyze_arm_neutral.py`
Data: `arm_neutral_endpoints.npz` (53 MB, sha256
`12720b9b02e507cb63df089f0a580364807610a63d4007828d044d2d2674783b`)

**HEADLINE: you were right. Against a target that neither solver authored, the
E16 state advantage DISAPPEARS and slightly reverses (S-C/S-D). The action
effect SURVIVES and grows (A-A). "E16 imagines better" is withdrawn.**

---

## 1. Target-provenance audit (Phase 0, code inspection only)

Every prior imagination experiment advanced the environment with **a model's own
generated action**, then scored predictions against the resulting state.

| Experiment | Current source | Goal source | Executed action source | Future target source | Arm-neutral? |
|---|---|---|---|---|---|
| NFE imagination sweep | Isaac Gym rollout | env `desired_goal` | **Flow @ lowest NFE (NFE 1)** | Isaac Gym after that action | **NO** |
| Matched-NFE solver study | Isaac Gym rollout | env | **Flow Euler@2** (`ARMS[0]`) | Isaac Gym after that action | **NO** |
| Converged-reference study | Isaac Gym rollout | env | **Flow Euler@8** (`ARMS[0]`) | Isaac Gym after that action | **NO** |
| Coarse-Euler manifold study | Isaac Gym rollout | env | **Flow Euler@8** | Isaac Gym after that action | **NO** |
| Multi-noise solver-bias | Isaac Gym rollout | env | **Flow Euler@16, noise 0** | Isaac Gym after that action | **NO** |
| Enriched solver-bias | Isaac Gym rollout | env | **Flow Euler@16, noise 0** | Isaac Gym after that action | **NO** |
| Gaussian-vs-Flow imagination | Isaac Gym rollout | env | **each model advanced its OWN rollout** | own-action-authored | **NO — self-scored** |
| **This experiment** | **recorded replay** | **recorded replay** | **none (no stepping)** | **recorded `obs[s+1]`** | **YES** |

The Gaussian-vs-Flow case is a *different* confound worth stating precisely:
Gaussian and Flow were never compared against a shared target. Each model
advanced its own separate trajectory and was scored against its own
self-authored future. That is symmetric rather than one-sided, but it is still
not an external-target comparison.

## 2. Claims corrected as a result (Phase 1 / Phase 18)

**WITHDRAWN — "E16 imagines better" / "coarse Euler predicts the true future
better."** Now measured against a non-authored target: **the effect vanishes**
(§8). Prior statements should read: *"Against an E16-authored rollout target,
E16 scored better than E512."*

**WITHDRAWN — "action degrades 5-6× more than state" as previously derived.**
The ratio was computed from a confounded state baseline. The arm-neutral ratio
is different in kind (§13), because the state effect is no longer positive.

**CORRECTED — "the state comparison is unaffected because both arms score
against the same realised next state."** This was my error. Both arms shared the
target, but the target itself was `Env(s_current, a_E16)`, so it was E16-authored.
Sharing a biased target does not remove the bias.

**PRESERVED — all raw measurements.** No number in any prior report is retracted;
only interpretations change. The prior E16-vs-E512 numbers remain valid
descriptions of behaviour *under an E16-authored rollout*.

**DOWNGRADED — Gaussian-vs-Flow imagination claims.** Not unbiased generative
comparisons (see §19).

## 3. Replay-data split status (Phase 3) — NOT held out

Inspected the training pipeline: `SequenceDataset` loads the replay buffer with
`max_n_episodes=5000` and builds indices over **every** episode. There is **no
train/validation/test split anywhere** in the dataset class or the
`pandapush_flow_single_gpu` config.

**These 96 transitions were in the Flow training set.** Per instruction I do not
call them held out. This is an **arm-neutral in-distribution replay diagnostic**,
which is exactly right for a solver-comparison question and claims no
generalization.

## 4. Exact temporal / action indexing (Phase 4, traced not inferred)

From `diffuser/diffuser/datasets/sequence.py` (`GoalDataset.__getitem__`), for a
segment starting at `s` with horizon 5:

```
observations = [obs[s], obs[s+1], obs[s+2], obs[s+3], GOAL]      # note: last slot is the
actions      = [act[s], act[s+1], act[s+2], act[s+3], 0]         #       episode goal, not obs[s+4]
conditions   = {0: observations[0], 4: observations[-1]}
trajectory   = concat([actions, observations], axis=-1)          # [3 action | 480 obs]
```

Therefore:

- **generated t=1 ↔ recorded `obs[s+1]`** — the state target
- **generated `action[0]` ↔ recorded `act[s]`** — the action target
- **`act[s]` drives `obs[s] → obs[s+1]`**, i.e. precisely the scored transition
- t=4 is conditioned on the **episode goal**, *not* `obs[s+4]` — so the goal is
  not a 4-step-ahead state, confirming the earlier finding that goal progress at
  t=1 is minimal.

No implementation offset was found.

## 5. Frozen arm-neutral sample set

| property | value |
|---|---|
| source | `panda_push_replay_buffer_dlp.pkl`, observations sha256 `8506fdd2…`, actions sha256 `b339a59c…` |
| conditions | **96, each from a DISTINCT episode** (not adjacent windows) |
| episode pool | successful episodes only (1950 of 2000), so the recorded goal is achieved |
| timesteps | uniform in [0, 95); realised range 0–92 |
| sampling seed | 20260902 (deterministic, predeclared) |
| noise-bank seed | 20260901 |
| split status | **in-distribution, not held out** |
| selection | **independent of solver performance** |

Latents are already DLP-encoded in the replay file, so **no encoding and no
Isaac Gym stepping was performed at all** (Phase 7 satisfied by construction).

## 6. Preflight cache validation (Phase 9) — PASSED before GPU work

```
{'n_keys': 42, 'all_keys_have_write_path': True, 'save_reload_roundtrip': 'OK',
 'empty_lists': 0, 'length_agreement': 'OK', 'hash_producible': True}
```

Static check that every key has ≥1 write path; dummy tensors at full expected
shapes; full save → reload → schema-validate cycle; then a 2-condition ×
2-noise **smoke test that ran end to end (24 s)**. Only then was the full run
launched. It completed first try — the previous 0.28 GPU-h of crashed attempts
was not repeated.

## 7. Within-run reproducibility

Every tensor used in every paired comparison — current, goal, targets, both arms'
full transition tensors, noise hashes — was produced and cached in **one
process**. No cross-process pairing. Post-run, the artifact was reloaded from
disk and metrics recomputed from the file alone (action L2 reproduced exactly at
0.03909).

## 8-9. STATE: E16 vs E512 against the RECORDED next state

| seed | E16 | E512 | paired Δ (E512−E16) [95% CI] | median | E16 wins | relative |
|---|--:|--:|---|--:|--:|--:|
| 42 | 0.02607 | **0.02589** | **−0.00018** [−0.00045, **+0.00009**] | −0.00006 | **49.3%** | −0.69% |
| 43 | 0.02554 | **0.02510** | **−0.00044** [−0.00076, −0.00012] | −0.00004 | **49.5%** | −1.72% |
| 44 | 0.02577 | **0.02567** | **−0.00010** [−0.00038, **+0.00018**] | +0.00004 | **50.7%** | −0.37% |

**The E16 state advantage is gone.**

- The sign has **flipped**: Δ is negative on all three seeds, i.e. E512 is
  nominally *closer* to the recorded next state.
- **E16 wins 49.3-50.7% of noises — indistinguishable from chance** (was 59-66%).
- Two of three CIs span zero; the magnitudes (≤0.00044) are ~3-15× smaller than
  the confounded effect (+0.00104 to +0.00150).
- Medians are ≈0 on all seeds.

For scale: the recorded current→t1 distance is 0.03144, and both arms sit at
~0.026, so both genuinely predict better than copying — but they are **equally
good**.

## 10-12. ACTION: E16 vs E512 against the RECORDED dataset action

| seed | L2 E16 | L2 E512 | paired Δ [95% CI] | E16 wins | relative |
|---|--:|--:|---|--:|--:|
| 42 | **0.03909** | 0.04652 | **+0.00743** [+0.00621, +0.00865] | **71.0%** | **+19.01%** |
| 43 | **0.03805** | 0.04588 | **+0.00783** [+0.00654, +0.00911] | **72.8%** | **+20.57%** |
| 44 | **0.03615** | 0.04382 | **+0.00766** [+0.00643, +0.00892] | **72.5%** | **+21.19%** |

Magnitude vs direction decomposition:

| seed | L1 E16/E512 | \|magnitude\| error E16/E512 | direction cosine E16/E512 |
|---|--:|--:|--:|
| 42 | 0.05212 / 0.06250 | 0.02373 / 0.02700 | **0.9708 / 0.9548** |
| 43 | 0.05027 / 0.06122 | 0.02269 / 0.02693 | 0.9747 / 0.9598 |
| 44 | 0.04762 / 0.05839 | 0.02157 / 0.02549 | 0.9765 / 0.9619 |

**The action effect survives the removal of the confound and is slightly larger**
(+19-21% vs the confounded +16%), with E16 winning **71-73%** of matched noises
(vs 65% before). Both magnitude error (+14-19%) and direction cosine (−0.016)
degrade, so it is not purely a scaling artifact. Cosines remain high (0.95-0.98)
for both arms — the actions are broadly right, and E512 is consistently a little
worse.

## 13. Arm-neutral action/state sensitivity ratio

| seed | state relative | action relative |
|---|--:|--:|
| 42 | **−0.69%** | **+19.01%** |
| 43 | **−1.72%** | **+20.57%** |
| 44 | **−0.37%** | **+21.19%** |

The ratio is not meaningfully expressible: the state denominator is negative and
near zero, so the previously reported "5-6×" has no arm-neutral analogue.
Per Phase 12's caution about unstable baselines, I report the two numbers
separately. **In absolute normalized terms: accurate integration costs ~20% of
action accuracy and ~0% of state accuracy.**

## 14. Joint state/action co-occurrence

| seed | Spearman ρ | both improve | both worsen | state↑ action↓ | state↓ action↑ |
|---|--:|--:|--:|--:|--:|
| 42 | +0.180 (p=5.5e-7) | 16.4% | 36.7% | 34.2% | 12.6% |
| 43 | +0.215 (p=1.9e-9) | 16.1% | 38.4% | 34.4% | 11.1% |
| 44 | +0.186 (p=2.1e-7) | 15.4% | 38.5% | 34.0% | 12.1% |

Weakly positive (~4% of variance) — the two blocks move together only slightly.
Notably **~34% of samples improve in state while worsening in action**, roughly
triple the reverse (~12%). The solver perturbs the two output blocks
substantially independently, consistent with the divergent aggregate results.

## 15. STATE classification: **S-C — the state gap DISAPPEARS**

(bordering S-D: the sign reverses on all three seeds, and one CI excludes zero,
but the magnitudes are tiny and two CIs span zero, so I classify it as
"disappears" rather than claim a reversal.)

## 16. ACTION classification: **A-A — large E16 advantage SURVIVES**

Indeed slightly amplified once the E16-authored target is removed.

## 17. Is "E16 imagines better" defensible? **NO — withdrawn**

Against a target neither solver authored, E16 and E512 predict the next state
**equally well** (win rate ≈ 50%, |Δ| ≤ 0.00044, sign favouring E512). The
earlier advantage was an artifact of scoring against a rollout that E16 itself
generated.

Defensible replacement wording:

> Under an E16-authored rollout target, E16 scored better than E512. Against
> recorded transitions that neither solver authored, the two arms predict the
> next state equally well, while E16 matches the recorded action markedly better.

## 18. Is action/state decoupling real? **YES — and it is now the finding**

The decoupling is *stronger* arm-neutral than it appeared before, and cleaner:
accurate integration leaves next-state prediction essentially unchanged while
degrading action prediction by ~20%, on 3/3 seeds, with ~71-73% of individual
noises favouring E16 and only weak per-sample correlation (ρ ≈ 0.19) between the
two effects.

This is a genuine property of the discrete map: **the action block of the joint
transition tensor is far more sensitive to integration accuracy than the state
block.** The action block is 3 of 483 coordinates and is conditioned at no
timestep.

It still does **not** explain control-vs-imagination saturation — if anything it
sharpens the puzzle, since control saturates by NFE 2-4 while the action channel
keeps changing out to NFE 512. Whether a ~20% action-error increase matters
in closed loop is untested here.

## 19. Do Gaussian-vs-Flow imagination claims need re-evaluation? **YES**

Per §1, Gaussian and Flow were each scored against **self-authored** futures.
Prior statements that Gaussian imagination is better than Flow are **not
unbiased generative-state comparisons** and should be marked accordingly.

**Cost estimate for a Gaussian-neutral comparison (not run, per instruction):**
the Gaussian model uses 100 NFE and the same batch; one 8-noise pass over the
same 96 recorded conditions ≈ 3 × 8 × 100 NFE-equivalents ≈ **0.08 GPU-h** for a
single-checkpoint arm (one Gaussian checkpoint exists, so no seed replication).
Cheap — but it is **not** part of the approved scope and I did not run it.

## 20. Loss-hypothesis status: **LOWER PRIORITY** (unchanged)

Nothing implicates the objective. The surviving effect is a property of the
sampler acting on the action block. If anything the arm-neutral result weakens
the loss case further: the *state* block — where the dimensional-imbalance
argument focused — shows no solver sensitivity at all against an honest target.

## 21. Exactly ONE next experiment

**Test whether the ~20% action-error increase has any closed-loop consequence,
by evaluating control success at Euler@16 vs Euler@512 on the frozen paired
Isaac Gym episode set.**

This is the one question the whole action finding hinges on and cannot be
answered offline: a 20% increase in per-step action error against recorded
demonstrations may be entirely absorbed by closed-loop feedback, or may degrade
success. Cost: two arms × 96 paired episodes; E512 is 32× the inference of E16,
so this needs a cost estimate and **explicit approval before running** — it is a
control benchmark, which the current authorization excludes.

Rationale for choosing it: the state finding is now settled (no effect), the
action finding is robust and arm-neutral, and its *only* remaining interpretive
gap is whether it matters for the actual task. Every offline analysis that could
be done on this question has now been done.

---

## HARD STOP OBSERVED

No training. No loss modification. No simulator-authored targets. No MeanFlow.
No VP. No solver changes. No new architecture. No control benchmark run.
