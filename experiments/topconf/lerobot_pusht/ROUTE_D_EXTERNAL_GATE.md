# Route-D external gate — decision

## Verdict: **D-GO-CONDITIONAL**

The independent pretrained Diffusion Policy shows the phenomenon in its
**saturation** form — offline metric keeps improving while closed-loop utility
does not — but at n=50 the ordering differences are inside screening noise, so a
focused confirmatory comparison is required before this can be called a
replicated mis-ranking.

## Why not D-GO-STRONG

Claim B in its strong form ("offline error *mis-ranks* budgets") requires a
resolvable ordering inversion. We do not have one:

- all seven closed-loop CIs overlap (§ resolvability table)
- NFE2 vs NFE100 McNemar **p = 1.000**
- Spearman = −0.018 is computed from seven points whose y-values are not
  individually distinguishable

Reporting −0.018 as evidence of mis-ranking would be exactly the "trivial
noise-level ranking swap" §19/§ interpretation-rules forbid.

## Why not D-WEAKENED

The external policy does **not** align offline quality with behaviour either. The
offline metric improves 21% from NFE2 to NFE100 and unambiguously prefers NFE100;
behaviour does not follow it — NFE2 ties or beats NFE100 on both closed-loop
measures at 35.3× less compute. That is a genuine dissociation, just not yet a
*resolved inversion*.

## Cross-system comparison (§25)

| | EC-Diffuser Flow | LeRobot Diffusion Policy |
|---|---|---|
| model | joint state+action | action-only |
| generative family | Conditional Flow Matching | DDPM |
| architecture | entity-centric transformer | 1D conv UNet |
| environment | Isaac Gym PushCube | gym-pusht PushT |
| simulator | GPU, **nondeterministic** | CPU pymunk, **deterministic** |
| offline optimum | NFE 2 (action) | NFE 100 |
| behavioural saturation | from NFE 2 | from NFE 2 |
| offline→control alignment | **no** | **no** |

Both systems agree on the *shared* question: **the offline action metric does not
identify the efficient behavioural operating point.** They disagree on the shape
— EC-Diffuser's offline action error is non-monotone (best at 2, degrading to
512); PushT's is monotone-improving. Per §18/§20 the identical curve shape was
never required.

Both saturate behaviourally at **NFE 2**, from opposite offline-metric directions.
That is the strongest cross-system statement available.

## Answers to the required questions

1. **hashes** — model `995d14d3…` (1,050,862,408 B), config `d391a7bf…`,
   train_config `500ea79b…`; see PROVENANCE.md
2. **revision** — `84a7c23178445c6bbf7e1a884ff497017910f653`; **lerobot 0.3.2**
3. **harness reproduction** — PASSED, seed-exact: ours 0.9863/90% vs published
   0.9525/80% on seeds 1000-1009, 7/10 episodes identical
4. **stochasticity ledger** — outcome **B**: policy diffusion sampling only;
   PushT physics deterministic (see DETERMINISM_AUDIT.md)
5. **timestep schedules** — stored per arm; first t moves 99→50→0 as N→1
6. **UNet calls** — `calls/plan == len(timesteps) == N` verified for all 8 arms
7. **offline curve** — 254.99 / 5.92 / 4.77 / 4.78 / 4.73 / 4.75 / 4.67 / 4.65
8. **offline uncertainty** — sd over K=4 ≈ 0.88-1.69 (NFE≥2); 34.17 at NFE1
9. **success** — 0 / 88 / 84 / 88 / 86 / 90 / 92 / 86 %
10. **avg max reward** — 0.088 / 0.967 / 0.921 / 0.963 / 0.937 / 0.960 / 0.979 / 0.928
11. **latency (ms)** — 12.7 / 19.9 / 34.3 / 40.2 / 74.1 / 139.8 / 340.6 / 702.8
12. **ranking comparison** — Spearman(offline, success) = −0.018;
    Spearman(offline, max_reward) = +0.214; optima at NFE100 vs NFE50
13. **Claim B replicated?** — **partially**: the dissociation replicates in its
    saturation form; a resolved *inversion* does not, at n=50
14. **PushT needs its own calibration?** — **No.** Physics is deterministic; only
    policy sampling varies. Isaac Gym's R=3 protocol is deliberately not imported
15. **GPU cost** — 0.638 (sweep) + ~0.13 (setup/validation/offline) ≈ **0.77
    GPU-h**, under the 0.80 cap
16. **decision** — **D-GO-CONDITIONAL**

## 17. Strongest claim now supportable

> In two independently authored generative policies — a joint entity-state/action
> Flow model in a GPU-simulated pushing task, and an action-only DDPM policy in a
> deterministic 2D pushing task — offline demonstrated-action error fails to
> identify the efficient closed-loop operating point. Both policies reach their
> behavioural plateau at two network evaluations, while their offline metrics
> point elsewhere (EC-Diffuser's action error degrades beyond NFE2; Diffusion
> Policy's keeps improving to NFE100 for 35.3× the latency and no measurable
> behavioural gain).

Not supportable: that offline error *inverts* the behavioural ranking in PushT.

## 18. Exactly ONE next experiment

**Confirmatory paired comparison of NFE2 vs NFE50 on PushT at n = 500**, the two
budgets that matter (behavioural plateau vs behavioural optimum, 17× latency
apart). At n=50 the observed 4 pp gap has a CI half-width of roughly ±10 pp; n=500
brings it to about ±3 pp, matching the published evaluation size and making the
comparison decisive in the only place the screen suggested a real difference.

Estimated cost: 500 episodes × (1.38 + 11.33 s/ep) ≈ **1.77 GPU-h**. This exceeds
the Stage-1 cap and is **not run**; it requires separate approval.

**HARD STOP.** No further episodes, budgets, checkpoints, policies, Gaussian
arm, training, or paper expansion.
