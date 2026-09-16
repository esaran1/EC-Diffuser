# The adaptation criterion (§23) — registered before reading the field results

The pilot's §23 question is *when has the model actually adapted to the
off-diagonal training distribution?* This must be decided by a rule fixed in
advance, because the closed warm-start branch failed precisely by reading
structure into off-diagonal numbers after the fact.

## What "adapted" means here

The closed branch established that a **diagonal-trained** model queried
off-diagonal degrades almost perfectly with distance from the trained diagonal:

> corr(action error, cosine-to-diagonal conditioning) = −0.995 / −0.999 / −1.000

So the diagnostic signature of *non*-adaptation is a near-perfect monotone
dependence of off-diagonal error on |t_s − t_a|.

## The registered comparison

Both arms are measured on the **same** 5×5 grid with identical replay states,
identical noise, and identical conditioning. The control (`p_diag = 1.0`) has by
construction **never seen an off-diagonal time**, so it is the correct reference
for "what off-diagonal looks like without adaptation."

Two predeclared quantities, for the action block (which control consumes):

1. **off/diag ratio** — mean off-diagonal error divided by mean diagonal error.
2. **corr(|t_s − t_a|, error)** over off-diagonal cells only.

## Predeclared readings

| outcome | criterion |
|---|---|
| **ADAPTED** | UWM off/diag ratio **materially lower** than CTL's, and UWM's corr(dist, err) materially weaker in magnitude than CTL's |
| **NOT ADAPTED** | UWM's off-diagonal behaviour statistically indistinguishable from CTL's — i.e. training on off-diagonal times bought nothing |
| **PARTIAL** | one of the two moves, the other does not |

"Materially" is deliberately not given a numeric threshold here: this is a
one-seed offline diagnostic with no resolution model, and inventing a threshold
would be false precision. The report will state the measured numbers and let the
direction and magnitude speak, explicitly flagging that a single seed cannot
establish generality.

## What adaptation does NOT license

Adaptation is a **precondition** for the Stage-E field study to be meaningful —
it says the off-diagonal inputs are no longer out-of-distribution by
construction. It is **not** evidence that any inference path improves control.
Per §48 only closed-loop evaluation at R=3 can support a control claim, and per
§49 a difference below ~3 pp is UNRESOLVED at routine resolution.

## Early-stopping rule (§23)

The field diagnostic is run at checkpoints 500 / 1000 / 2000 / final(5000) of the
UWM arm. If the off/diag ratio has plateaued by 2000, the 5000-step result is
reported but **no extension to 10000 is authorized**. An extension requires the
ratio to still be improving strongly at 5000 *and* the projection to fit the cap.
