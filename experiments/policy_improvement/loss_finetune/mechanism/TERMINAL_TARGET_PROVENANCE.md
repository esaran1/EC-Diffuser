# What the terminal action target actually is

**Finding: the terminal action slot is a deliberately constructed dummy paired
with the goal observation. It is not padding, not the last executed action, and
not a bug.** It is a perfectly constant target — one unique value across every
window in the dataset.

This was established during Stage D and it reframes how the A-vs-B result should
be described.

## The construction, from production code

The dataset is `GoalDataset` (verified at runtime, not assumed).
`GoalDataset.__getitem__` builds every training trajectory as:

```python
observations = np.concatenate([
    self.fields.normed_observations[path_ind, start:actual_end-1],
    goal])                                   # <- GOAL as the final observation
actions = np.concatenate([
    self.fields.normed_actions[path_ind, start:actual_end-1],
    self.normalizer.normalize(              # <- explicitly constructed ZERO
        np.zeros((1, self.action_dim), ...), 'actions')])
```

So the **last row of every trajectory is the goal-conditioning row**: the goal
observation, paired with a dummy zero action. The real action at that dataset
index is discarded.

`get_conditions` then conditions on `{0: observations[0], H-1: observations[-1]}`
— i.e. the model is *conditioned* on that goal observation, while the action slot
beside it carries the dummy.

## Measurements

| quantity | value |
|---|---|
| unique terminal action rows over 5,000 sampled windows | **1** |
| terminal action value (normalized) | **[0, 0, 0.706]** |
| per-dimension std across windows | **[0, 0, 0]** |
| equals `normalizer.normalize(zeros)` | **True** |
| terminal observation equals `cond[H-1]` (the goal) | **True** |
| for contrast, t=1 action per-dim std | [0.232, 0.269, 0.294] |

The value 0.706 is simply where a raw zero maps under the gripper dimension's
`[-0.36, 0.06]` range; the first two dimensions are symmetric so zero maps to
zero.

## A contradiction I had to resolve

An intermediate check suggested the constant came from **padding**: 6,000 of
198,000 windows (3.03%) have `end > path_length`, and `GoalDataset.__getitem__`
does pad those. But a *normal* window (episode 0, 0→5, well inside the episode)
showed the same constant terminal action, while the raw dataset action at that
index was `[0.085, -1.0, 0.048]`.

Padding therefore could not be the explanation. Reading the actual `GoalDataset`
implementation — rather than the base `SequenceDataset` I had been looking at —
showed the terminal row is *always* overwritten by construction, padded or not.
The padding path is real but irrelevant to this question.

Recording this because the intermediate, wrong explanation was plausible and
would have produced a confidently incorrect claim in the report.

## Why this matters for the A-vs-B result

It explains the offline numbers precisely:

- Arm A supervises a **constant** target at t=H-1, so it learns it almost
  exactly: replay action error at that slot is **0.0006**.
- Arms B/C/D mask it, so they stop emitting it: error **0.72214**, identical to
  five decimals across arms and seeds.
- The executed action at t=0 is essentially unchanged (0.0160–0.0164 across all
  arms and seeds), so nothing about real action prediction improved.

It also sharpens the claim language. The intervention removes supervision on a
**constructed constant placeholder that occupies the action slot of the
goal-conditioning row** — a slot that is never executed and carries no
behavioural information. That is a more precise and more defensible description
than "the terminal zero-action target," and it makes the ablation framing
natural: masking removes a degenerate regression target, not real supervision.

It remains **not** called a bug. The construction is deliberate: the row exists
to carry the goal observation, and the action slot has to contain something.
Whether supervising it is *harmful* is exactly the empirical question this
program is testing, and the answer so far is a modest replicated control gain
with no improvement in executed-action accuracy.
