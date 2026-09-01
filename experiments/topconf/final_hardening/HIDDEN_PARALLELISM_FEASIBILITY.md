# Hidden-parallelism feasibility (§3-§8) — E1/E2/E3

## Can it be tested from cache? **NO** (§4)

The evaluator maps scenarios to GPU slots **deterministically and identically in
every run**:

```python
index = list(range(batch.start, batch.stop))   # scenario i -> slot (i mod 16)
```

So across all ~60 cached evaluation runs, **scenario i has always occupied slot
i mod 16**. Slot is a deterministic function of scenario ID. There is **zero**
slot variation in cache, and the two are **perfectly confounded**.

### Observational check (reported as motivation, not evidence)

Per-slot physics-sensitivity counts on the R=8 same-arm bank
(6 scenarios per slot): `[2,1,3,5,4,4,3,1,3,4,1,3,0,1,1,3]`, overall sensitive
fraction 0.406. Observed variance of per-slot counts 2.129 vs binomial
expectation 1.447 — **ratio 1.47**.

Suggestive, but **uninterpretable**: slot ≡ scenario identity here, so this
cannot separate a slot effect from scenario difficulty. It motivates E1; it is
not evidence for it.

## Implementation — a clean single-variable change, with one real risk

Permuting **within** each 16-slot batch keeps the batch composition, `num_envs`,
horizon, physics config, checkpoint, policy and CRN bank identical. One line:

```python
index = [perm[j] for j in range(batch.start, batch.stop)]
```

**The risk (must be fixed before any science run).** The CRN wrapper seeds by
`(batch_start, decision)`, which is slot-independent — good. But the noise tensor
is drawn for the whole batch at once, so **scenario in slot k receives noise
row k**. A naive permutation would therefore change *both* the slot *and* the
policy-noise row, confounding the very thing E1 tests.

Fix: apply the same permutation to the noise rows so noise stays attached to the
**scenario**. This must be validated by a zero-GPU assertion — under the identity
permutation the permuted code path must reproduce cached results **bit-exactly**
— before spending any GPU.

## Cost (measured throughput: NFE4 = 133 s, NFE2 = 122 s per 96-episode run)

| experiment | design | GPU-h |
|---|---|--:|
| **E1** same-arm slot sensitivity | seed 42, NFE4, 3 permutations | **0.111** |
| **E1+** as above, 4 permutations | | 0.148 |
| **E2** policy-ranking × slot | NFE2+NFE4, 3 perms, seed 42 | **0.212** |
| E2 full | NFE2+NFE4, 4 perms, 3 seeds | 0.850 |
| **E3** `num_envs` sensitivity | 1/8/16/32/96 | not costed — see below |

## E3 (`num_envs`) — second priority, as the brief anticipated

Changing `num_envs` alters batch grouping, GPU scheduling, memory layout and
possibly environment construction simultaneously. It is **not** a single-variable
manipulation, so a positive result would be uninterpretable and a null result
weak. **Slot permutation at fixed `num_envs` is the clean first test**, exactly as
§8 argues. Recommend E3 only if E1 is positive and a mechanism question follows.

## What each outcome would mean

**E1 positive** (same policy, same scenarios, different slots → materially
different outcomes): establishes that GPU-parallel evaluation *configuration* is
a hidden experimental variable, not a throughput knob. This is qualitatively
stronger than "run-to-run noise exists" and is unclaimed in the literature.

**E1 null** (slot assignment does not matter beyond ordinary run-to-run
variation): also valuable and publishable — it **localizes** the nondeterminism
to execution-order/atomics rather than slot placement, and it lets us state that
the effect is not an artifact of scenario packing. It strengthens the existing
paper rather than adding to it.

**E2** is only worth running if E1 is positive; it is the headline version
(*apparent policy difference changes because scenarios were packed differently*).
