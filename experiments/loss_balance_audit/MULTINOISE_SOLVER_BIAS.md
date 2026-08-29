# Multi-noise E16 vs E512: does stochastic dispersion explain the coarse-Euler advantage?

**No training. No loss change. No new solver. No simulator reruns beyond this
diagnostic.** Compute: **1531 s = 0.43 GPU-h** (estimated 0.41 before running,
under the 0.5 h stop threshold).

Scripts: `multinoise_solver_bias.py`, `analyze_multinoise.py`, `plot_multinoise.py`
Data: `multinoise_endpoints.npz`, `multinoise_protocol.json`, `multinoise_analysis.json`
Figure: `experiments/figures/multinoise_solver_bias.png`

**Answer: NO. Classification B — similar-or-greater dispersion with a SHIFTED
CENTRE. Extra diversity does not explain the effect; 95% of the mean gap
survives in a dispersion-robust centre estimate.**

---

## 1. Exact noise-bank construction

One `torch.Generator(device="cpu")` seeded **20260830** per training seed. At
each rollout step it draws `N_NOISE = 8` tensors of shape
`(num_envs, horizon, transition_dim)` **in order**; noise index *i* is then
reused verbatim by every arm at that step. The generator is advanced only by
these draws, so the bank is exactly reproducible from the seed plus the draw
order (recorded in `multinoise_protocol.json`).

Sample set: frozen episode set `35144910`, 16 episodes × 6 rollout steps =
**96 conditions per seed**, × 8 noises × 2 arms × 3 seeds = **4,608 generated
trajectories**.

## 2. Confirmation that E16 and E512 share identical z per pair

Two independent checks:

| check | s42 | s43 | s44 |
|---|--:|--:|--:|
| E16 run twice on the same x0 (determinism) | **0.0** | **0.0** | **0.0** |
| same-index E16↔E512 distance | 0.01508 | 0.01471 | 0.01519 |
| **cross-index** E16↔E512 distance | 0.04006 | 0.03521 | 0.03571 |
| ratio cross/same | **2.66×** | **2.39×** | **2.35×** |

Endpoints sharing a noise index are 2.4–2.7× closer than endpoints from
different indices. If the arms had received different noise, the two rows would
coincide. **Pairing is verified at all three levels: same condition, same
checkpoint, same initial noise.**

## 3. Within-run reproducibility protocol

Isaac Gym/DLP observations are **not bit-reproducible across processes** (max
|Δchamfer| ≈ 0.05, established in the previous audit). Therefore **every**
endpoint and **every** ground-truth future used in a paired comparison here was
generated inside this single run. No new per-example value is paired against any
previously cached per-example value.

Aggregate replication check (descriptive only): the paired E512−E16 mean is
**+0.00154** here, vs +0.00161 (manifold audit) and +0.00210 (converged-reference
study). **Three independent runs, same sign, same order of magnitude.**

## 4-7. Primary results, per seed

| seed | arm | mean single-sample | best-of-8 | dispersion | medoid error |
|---|---|--:|--:|--:|--:|
| 42 | euler16 | **0.04380** | **0.03211** | 0.03925 | **0.03801** |
| 42 | euler512 | 0.04544 | 0.03321 | **0.04304** | 0.03945 |
| 43 | euler16 | **0.04296** | 0.03147 | 0.03806 | **0.03676** |
| 43 | euler512 | 0.04419 | **0.03135** | **0.04181** | 0.03775 |
| 44 | euler16 | **0.04266** | **0.03179** | 0.03761 | **0.03728** |
| 44 | euler512 | 0.04443 | 0.03294 | **0.04129** | 0.03926 |

Three-seed summary (sd across seeds ≤ 0.0009 on every metric):

| metric | E16 | E512 | difference |
|---|--:|--:|--:|
| **A. mean single-sample error** (primary) | **0.04314** | 0.04469 | **+0.00154** |
| **B. best-of-8** (coverage only) | **0.03179** | 0.03250 | +0.00071 |
| **C. dispersion** (mean pairwise) | 0.03831 | **0.04205** | **+0.00374** |
| **D. medoid error** (centre) | **0.03735** | 0.03882 | **+0.00147** |

Medoid is used as the centre estimate rather than a coordinate mean, because
particle sets are permutation-invariant and naive coordinate averaging across
unaligned particle indices is not defined. No barycenter is claimed.

## 8-9. Paired per-noise E512 − E16 (the strongest comparison)

| seed | mean Δ | median Δ | 95% CI | E16 wins |
|---|--:|--:|---|--:|
| 42 | +0.00164 | +0.00154 | [+0.00128, +0.00200] | **66.0%** of 768 |
| 43 | +0.00122 | +0.00127 | [+0.00087, +0.00158] | **62.0%** of 768 |
| 44 | +0.00177 | +0.00164 | [+0.00143, +0.00211] | **68.4%** of 768 |
| pooled (descriptive) | +0.00154 | — | [+0.00134, +0.00175] | 65.5% of 2,304 |

**Mean and median agree closely on every seed**, so the effect is not driven by
a heavy tail. **E16 wins on ~2 of every 3 individual noise draws** — a broad
shift across the noise distribution, not a minority of outliers. Every per-seed
CI excludes zero.

## 10-11. Endpoint displacement and its correlation with degradation

Mean |E16 − E512| for the same z: **0.01568 / 0.01556 / 0.01539** (seeds
42/43/44) — i.e. accurate integration moves the endpoint by ~0.0155, about **10×
the size of the resulting prediction change** (+0.00154).

Spearman correlation of displacement against Δ ground-truth error:

| seed | ρ | p |
|---|--:|--:|
| 42 | +0.067 | 0.064 |
| 43 | +0.034 | 0.348 |
| 44 | +0.041 | 0.256 |

**Essentially no correlation** (ρ ≤ 0.07, none significant at 0.05). The samples
most altered by accurate integration are **not** the ones that degrade most.
Descriptive only; no causal claim.

This matters: it means the degradation is not a simple "bigger move ⇒ bigger
error" effect. The shift is systematic in *direction*, not proportional to
magnitude.

## 12. Best-of-K curve (secondary, existing samples only)

Expected best-of-K over random subsets of the 8 draws:

| K | euler16 | euler512 |
|--:|--:|--:|
| 1 | **0.04323** | 0.04475 |
| 2 | **0.03794** | 0.03923 |
| 4 | **0.03441** | 0.03538 |
| 8 | **0.03179** | 0.03250 |

**E16 leads at every K and E512 never overtakes**, though the gap narrows
(+0.00152 → +0.00071). E512's greater dispersion does buy it slightly faster
gains from extra draws, but not enough to catch up within K = 8. K was not tuned;
all four values are reported.

Per §8, best-of-8 is **not** presented as evidence about distribution quality —
it is reported here only beside mean error, dispersion and medoid error.

## 13. Per-seed consistency

Every finding replicates on all three independently trained checkpoints:
E512 worse on mean error (3/3), worse on medoid (3/3), more dispersed (3/3),
E16 favoured on a majority of individual noises (3/3, 62–68%). Best-of-8 is the
only metric that flips on one seed (s43: E512 0.03135 vs E16 0.03147) — the
weakest and noisiest metric, as expected.

## 14. Classification: **B — SIMILAR (here greater) DISPERSION, SHIFTED CENTRE**

The protocol's outcome A required E512 to show *similar or better best-of-8*
alongside a larger spread. **E512 is more dispersed (+0.00374, ~10% wider) but
its best-of-8 is also worse (+0.00071, 2 of 3 seeds)**, so outcome A's signature
is not met.

The decisive quantity is the medoid:

> **medoid gap +0.00147 / mean gap +0.00154 = 95%**

A dispersion-robust estimate of where the sample cloud is *centred* retains
essentially the entire effect. If wider spread were the explanation, the centre
would coincide and only the mean would move. It does not.

Not C (E512's dispersion is higher, not lower). Not D (the effect is robust:
3/3 seeds, all CIs exclude zero, replicated across three runs).

## 15. Does stochastic diversity explain the coarse-Euler advantage?

**No.** Three independent reasons:

1. **95% of the mean gap survives in the medoid**, which is insensitive to spread.
2. **E512's best-of-8 is also worse**, so its extra samples do not reach closer to
   the observed future even when given 8 chances.
3. **E16 wins ~65% of individual paired noise draws** — a broad location shift,
   not a variance artifact.

E512 *is* genuinely more dispersed, and that is a real effect of accurate
integration worth recording. It simply is not what produces the prediction gap.

## 16. Strongest interpretation actually supported

**The finite-step Euler map places the centre of its conditional sample cloud
closer to the observed future than the converged ODE flow does, and this holds
per-noise, per-seed, and after controlling for dispersion.**

This is the §12 pattern the protocol flagged as high-importance: lower mean,
better medoid, advantage across most paired z. It is **not** explained by
"E512 is more diverse."

Conceptual framing (§13 of the protocol, offered as interpretation only): Flow
Matching trains `v_theta` for continuous-time transport, but deployment applies
a finite-step discrete map. The deployed policy is effectively
**(model + integrator + NFE)**, not `v_theta` alone. These results are consistent
with the discretisation acting as an inductive bias that happens to align with
the supervised conditional target. **No theoretical novelty is claimed**, and no
method is proposed.

## 17. Exact limitations

- **There is only ONE observed future per conditioning context.** We can measure
  distance to that observed future, sample diversity, and coverage of it. We
  **cannot** measure likelihood under the true conditional future distribution,
  its calibration, or whether E512's additional diversity is semantically valid.
  Establishing those would require multiple valid futures under essentially
  identical conditions, or another defensible conditional-distribution evaluation.
- **E512 being worse against the observed future does NOT establish that E16 is
  the better generative model.** It establishes that E16 is better aligned with
  this particular supervised target.
- Effect sizes are small: +0.00154 against a metric whose held-out-real baseline
  offset is ~0.004–0.006. Real and replicated, but small.
- Correlational analyses (§10-11) are descriptive; no causal claim.
- Cross-process latent replay is not bit-reproducible; only within-run pairing is
  valid. Prior-run comparisons above are aggregate and descriptive.
- Best-of-K depends jointly on distribution quality, spread and K, and is
  reported only as a coverage diagnostic.

## 18. Loss-hypothesis status: **LOWER PRIORITY** (unchanged)

Nothing here implicates the training objective. The effect is a property of the
*sampler*, and it points toward the discrete map rather than the loss. Combined
with the small converged Flow-vs-Gaussian residual whose CI overlaps Gaussian,
loss modification remains unmotivated. **No loss change made.**

## 19. Exactly ONE next experiment

**Characterise the direction of the E16→E512 shift by decomposing the
displacement against the goal, using only the already-cached endpoints.**

For each condition and noise, the E16 and E512 endpoints and the goal state are
all in hand. Project the displacement `(E512 − E16)` onto the direction toward
the conditioning goal, and separately measure each arm's distance to the goal.
Pure CPU on `multinoise_endpoints.npz`, minutes, **zero GPU**.

Rationale: §14/§16 establish that coarse Euler shifts the sample-cloud centre in
a systematically useful direction, and §10-11 show the shift is directional
rather than magnitude-driven. The single most informative follow-up is *which
way* it points. If E16 endpoints are systematically nearer the conditioned goal,
the discretisation is biasing generation toward the conditioning signal — a
concrete, checkable mechanism. If not, the direction is task-specific in some
other sense and that also narrows the space. This must be settled before any
claim about what the discrete map contributes, and it costs nothing.

---

## HARD STOP OBSERVED

No training. No loss modification. No new solver. No MeanFlow. No VP.
No policy changes.
