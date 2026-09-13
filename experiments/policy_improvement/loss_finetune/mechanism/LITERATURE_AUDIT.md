# Literature / code-context audit (directive section 28)

**Question:** is masking non-executed or padded terminal actions already standard
practice in related sequence-action policies?

**Answer: yes — masking padded/invalid action positions out of the loss is
established practice, and the accompanying normalization question has been
treated elsewhere as a correctness issue.** This makes our intervention a
*convention alignment*, not a novel technique. Novelty is **not** claimed.

Web access was available; this audit is based on public sources retrieved
2026-09-13, listed below.

## 1. Masking padded actions is standard

- **LeRobot** carries `action_is_pad`, "a bool mask of shape (B, horizon) that
  marks timesteps padded because the episode ended before `horizon` steps," and
  its policies exclude those entries from the loss. The mask exists precisely to
  stop padded action positions from contributing gradient.
- **Diffusion Policy** (Chi et al., RSS 2023) documents that handling padding at
  the start and end of each demonstration episode, according to the observation
  and action horizons, matters for performance; several implementations pad
  no-op actions at episode end to indicate termination.

So the *idea* that a non-executed, synthetic action slot should be excluded from
supervision is well established in this literature. Our arm B is the same idea
applied to EC-Diffuser's goal-conditioning row.

## 2. The normalization question has been treated as a BUG elsewhere

This is the most directly relevant finding, and it bears on our denominator
caveat from an independent direction.

LeRobot issue **#3433** ("Incorrect loss normalization when masking padded
actions in SmolVLA (Same issue as ACT)") states:

> "`.mean()` divides by the total number of elements (including the zeroed-out
> padding), the per-scalar denominator is larger than the number of valid
> entries, so the reported loss (and therefore the effective gradient signal) is
> underestimated proportional to the padding fraction."

The fix is to average over **valid entries only**. The same correction was
already applied to ACT, Diffusion and MultiTaskDiT in PR **#3377**.

**Mapping onto our arms:**

| | denominator | LeRobot's view |
|---|---|---|
| our **B** (registered masking) | active/valid coordinates only (46464) | the **fixed**, correct convention |
| a hypothetical "mask numerator, keep full denominator" | all coordinates (46560) | the **bug** they removed |

Our **B′** control arm is deliberately constructed to be that second, buggy-style
variant — masked numerator with the unmasked denominator — purely to isolate the
0.207% scaling. It is a scientific control, not a proposed method.

Two consequences worth stating:

1. Our registered semantics (commit `58d07c4`, "restrict that count to
   unconditioned elements when masking is present") independently matches what
   the LeRobot maintainers concluded was correct. That is reassuring about the
   registered design.
2. The community regards this denominator choice as mattering enough to file and
   fix — which is a reason to *measure* our 0.207% rather than wave it away on
   theory. Note the magnitudes differ hugely: their padding fraction can be a
   large share of a batch, whereas ours is 96/46560 = 0.206%.

## 3. What this does and does not license

**Does:**
- Describing arm B as aligning EC-Diffuser with a convention already used
  elsewhere for padded/invalid action positions.
- Noting that our denominator handling matches the corrected convention.

**Does not:**
- Any novelty claim. Masking invalid action targets is not new.
- Any claim that other codebases specifically mask a *goal-conditioning row's*
  dummy action. Our terminal slot is not padding (see
  `TERMINAL_TARGET_PROVENANCE.md`): it is a constructed constant paired with the
  goal observation, present in **every** window, not only at episode boundaries.
  That specific construction is an EC-Diffuser design detail, and I found no
  public discussion of it.
- Any claim about absence. A handful of repositories and issues cannot establish
  that something is unprecedented, and no such inference is made here.

## 4. Honest limitation

This audit covers public documentation, one upstream issue and one referenced PR.
It is not an exhaustive survey, and I did not read the LeRobot or Diffusion
Policy source trees directly (not vendored in this repo). The conclusions above
are restricted to what the retrieved sources actually state.

## Sources

- [LeRobot — Adding a Policy (docs, `action_is_pad`)](https://huggingface.co/docs/lerobot/bring_your_own_policies)
- [LeRobot issue #3433 — Incorrect loss normalization when masking padded actions](https://github.com/huggingface/lerobot/issues/3433)
- [Diffusion Policy (project page)](https://diffusion-policy.cs.columbia.edu/)
- [Diffusion Policy (arXiv:2303.04137)](https://arxiv.org/html/2303.04137v5)
- [real-stanford/diffusion_policy (code)](https://github.com/real-stanford/diffusion_policy)
