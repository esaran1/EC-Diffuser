# Stage-B probe R-ratios are INVALID — and why the scientific results are not

**The Stage-B 500-step probe produced `DENOMINATOR-NOT-NEGLIGIBLE`
(R_param 0.917, R_output 0.833, R_t0 0.740). That verdict is VOID.** The probe
does not reproduce the scientific arm B, so its numerator and denominator measure
incommensurable things.

**The scientific runs are unaffected**, and the denominator question is answered
directly and validly by the four-seed B′ branch (§4 below).

## 1. How it was caught

The registered design said `B_probe` "should reproduce the stored valid B seed-42
checkpoint at step 500 within the deterministic tolerance." It does not:

| comparison | L2 distance |
|---|--:|
| probe B(500) vs **scientific** B(500) | **1.70** |
| scientific B(500) vs scientific A(500) | **0.0996** |

The probe-vs-scientific gap is **17× larger than the entire A-vs-B scientific
difference**. Any ratio mixing probe distances (numerator) with scientific
distances (denominator) is meaningless.

The verdict also contradicted the Stage-A theory sharply — a 2.06e-3 loss
scaling cannot move an Adam-trained model 92% as far as the masking intervention
— which is what prompted the check rather than acceptance.

## 2. Root cause: leftover gradients from `train.py`'s pre-training test pass

`train.py` runs a forward/backward sanity check before training:

```python
print('Testing forward...', end=' ', flush=True)      # train.py L156-159
batch = utils.batchify(dataset[0])
loss, _ = diffusion.loss(*batch)
loss.backward()                                       # <-- populates .grad
```

and `Trainer.train()` calls `zero_grad()` **after** `optimizer.step()`, not before
the accumulation loop (`training.py` L266). So in every scientific run:

```
step 1 gradient = g_test + g_batch1 + g_batch2
```

whereas `bprime_probe.py`, which drives the Trainer directly and skips that test
pass, uses:

```
step 1 gradient =          g_batch1 + g_batch2
```

One contaminated step is enough: Adam carries it in `m` and `v` forever, so the
trajectories separate immediately and never re-converge.

Confirmed at step 10, long before step 500:

| | displacement from canonical |
|---|--:|
| probe B @10 | 1.287e-02 |
| scientific B early10 | 1.100e-02 |
| **probe vs scientific @10** | **1.535e-02** — larger than either displacement |

By step 500 the probe had moved 1.62 from canonical vs the scientific run's 1.35.

`batches_drawn` also differs (probe 1000, scientific 1002 at the "500" label),
a separate off-by-one: the periodic saver fires inside the loop *after* the step
counter increments, so `state_500.pt` actually holds 501 optimizer steps. Minor
next to the gradient contamination, but it means the two checkpoints are not even
the same training length.

## 3. Why the scientific results are NOT affected

Every scientific arm — A, B, C, D, all four B′ replicates — runs through
`train.py` and therefore receives the **identical** pre-training test-pass
gradient. It is a shared, constant feature of the protocol, not a differential
between arms. The paired A-vs-B and A-vs-B′ contrasts are unaffected.

It is, separately, a small blemish in the pipeline worth recording: step 1 of
every fine-tune includes a stray gradient from `dataset[0]`. It does not
invalidate anything here because it is common to all arms, and it is **not**
fixed now — changing it would alter the pinned training path mid-program.

## 4. The denominator question is answered anyway — by the four-seed B′ branch

Because the gate failed, the orchestrator ran the full four-seed B′ branch, which
uses the real `train.py` path and is therefore directly comparable to A and B:

| seed | A | B | B′ | B−A | **B′−A** | B′−B |
|---|--:|--:|--:|--:|--:|--:|
| 42 | 0.7750 | 0.8042 | 0.7937 | +2.92 | **+1.87** | −1.04 |
| 43 | 0.7729 | 0.7729 | 0.7896 | +0.00 | **+1.67** | +1.67 |
| 44 | 0.7500 | 0.7896 | 0.7792 | +3.96 | **+2.92** | −1.04 |
| 45 | 0.7833 | 0.8042 | 0.7917 | +2.08 | **+0.83** | −1.25 |
| **mean** | | | | **+2.24** | **+1.82** | **−0.42** |

B′ has the **masked numerator** and **A's denominator**. It retains most of the
benefit (+1.82 pp, positive in **4/4** seeds), and B′−B is only −0.42 pp.

**The gain travels with the masking, not with the 0.207% renormalization.** Had
the renormalization been the operative factor, B′ should have looked like A
(≈0 pp); it does not.

This is a stronger test than the probe would have been: it measures the actual
scientific endpoint (closed-loop control across four paired seeds) rather than a
parameter-distance proxy at 500 steps.

## 5. Disposition

- Stage-B probe R-ratios and the `DENOMINATOR-NOT-NEGLIGIBLE` verdict: **VOID**,
  preserved but not used for any claim.
- The probe artifacts are kept under `probe_runs/` for provenance.
- The denominator classification is taken from the four-seed B′ branch instead.
- The GPU cost of the probe is charged to the ledger as spent-but-unusable; the
  B′ branch it triggered is charged as valid science, since it answers the
  question the probe was meant to answer.
