# MeanFlow closure check — arXiv:2512.02012

**Verdict: the closure STANDS. No new MeanFlow GPU experiments.**

## The paper

*Improved Mean Flows: On the Challenges of Fastforward Generative Models* —
Geng, Lu, Wu, Shechtman, Kolter, He. arXiv:2512.02012 (v1 Dec 2025, v2 May 2026).
https://arxiv.org/abs/2512.02012 · code (JAX/TPU, MIT):
https://github.com/Lyy-iiis/imeanflow

First-party follow-up from the original MeanFlow lab. Two contributions:
re-parameterizing the network-dependent MF target into a standard regression on
instantaneous velocity (replacing the *conditional* velocity inside the JVP with
a *predicted marginal* one, to cut variance), and treating guidance as an
explicit conditioning variable. Headline: 1.72 FID at 1-NFE on ImageNet 256×256
vs 3.43 for original MF-XL/2.

## Why it does not reopen our branch

**1. Different failure mode — beware a lexical trap.** The paper talks about a
"boundary" prominently, but it means the **r→t** boundary where average velocity
collapses to instantaneous velocity, used as a free way to obtain `v_θ` with no
extra parameters. Its *diagnosed* pathology is high gradient variance and a
non-decreasing loss curve — a training-dynamics complaint. Our hypothesized
failure was at the one-step **deployment** boundary. Shared word, different
claim.

**2. No rescue evidence.** The paper improves already-working models (original MF
FID 32.69 → 29.42 with the boundary variant). There is no ablation anywhere
showing a diverged or non-functional run restored. Our MeanFlow run was a
*categorical* failure — 0% closed-loop success. A variance-reduction fix that
buys ~3 FID on a functioning baseline is not evidence it converts 0% into a
working policy.

**3. No control results.** ImageNet class-conditional generation only; FID/IS
metrics; zero mentions of robotics, policy, action or manipulation. Models are
89M–610M params trained 240–800 epochs on TPUs — a regime that tells us nothing
about a 20k-step PandaPush policy run.

**4. Cheapness is the only criterion it passes**, and it is the weakest one.

## What it DOES do for us

iMF independently corroborates that vanilla MeanFlow training is genuinely
unstable and that its target is ill-posed as a regression problem. That is a
useful **citation supporting why our attempt struggled and why we stopped** — it
strengthens the closure write-up rather than undermining it.

## Correction to our own records — carry this forward

Our notes describe the MeanFlow failure as "IVC-style", implying a deployment
boundary. The IVC terminology traces to *Mean Flow Policy with Instantaneous
Velocity Constraint* (arXiv:2602.13810), whose boundary is **r = t**, not the
(r=0, t=1) deployment boundary.

> Any future write-up must not call our failure "IVC-style deployment-boundary".
> Those are two different boundaries. State the observed symptom (0% closed-loop
> success at the one-step deployment setting) and keep the mechanism explicitly
> unresolved.

## Related work found incidentally (NOT audited, NOT actioned)

- **arXiv:2602.13810** — MeanFlow policy with an auxiliary IVC loss
  (`ℒ_MF + λ·ℒ_IVC`, λ=1.0) enforcing `u_θ(a(t),t,t) = v`. Robomimic + OGBench.
- **arXiv:2602.13718 "HybridFlow"** — reports a MeanFlow-for-manipulation failure
  resembling ours (validation loss plateauing at a floor, multi-step inference
  not helping) and attributes it to inference-time distribution shift — genuinely
  a deployment-side diagnosis. Remedy is a 2-NFE inference schedule.

Both are potential **citations**, not experiments. Testing either needs fresh GPU
runs on a closed branch. Neither was verified beyond a single fetch. **Branch
remains closed.**
