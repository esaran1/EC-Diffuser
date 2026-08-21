# Paired Isaac Gym NFE study

Date: 2026-08-20. Branch `fast-generative-policies`.

**Question.** What is the minimum Flow NFE that reliably retains or exceeds the
canonical Gaussian EC-Diffuser performance on Isaac Gym 3-cube PushCube?

**Context.** The diagnosis in `experiments/isaacgym_flow_diagnosis.md` is closed:
no vanilla-Flow implementation failure, no action-normalization failure, no
undertraining problem, and no contact-control failure. This study asks the
original low-NFE question in that now-validated environment. **No training was
run.** NFE is an inference-time solver override.

## 1. Design

| Element | Value |
|---|---|
| Arms | Flow at 1, 2, 4, 8, 16 solver steps; Gaussian at 100 |
| Flow checkpoint | `data/panda_push/flow/3C_dlp_adalnpint_randcolor_H5_T4_seed42`, EMA, step 499k |
| Gaussian checkpoint | `.../3C_adalnpintlarge_dlp_randcolor_H5_T100/state_1200000.pt`, EMA |
| Episode sets | **3 independently generated sets of 96 episodes** |
| Total | 6 arms x 3 sets x 96 = **1,728 episodes** |
| Task | 3-cube PushCube, random colours, 100-step episodes |

### 1.1 These are evaluation replicates, not training seeds

**One** trained Flow checkpoint and **one** Gaussian checkpoint are used
throughout. Between replicates only the *episodes* change; between Flow arms
only the *solver-step count* changes. Nothing about training varies.

So the between-replicate spread reported in §4 measures **evaluation-sampling
variance only**. It says nothing about training-seed variance, and no claim in
this document should be read as covering it. Establishing training-seed
variance would require retraining, which was explicitly out of scope.

### 1.2 Pairing is enforced, not assumed

Each replicate's initial and goal cube states are recorded once, frozen to disk,
and hashed with SHA256. Every arm loads that same file, and each result records
the hash. The aggregator **refuses to report** if the arms within a replicate do
not share one hash.

Episode-set hashes:

| Replicate | SHA256 (prefix) |
|---|---|
| 0 | `35144910b1471b7b` |
| 1 | *(recorded at run time)* |
| 2 | *(recorded at run time)* |

### 1.3 Model calls are verified, not requested

A forward hook on the denoiser counts actual calls. Every arm is checked against
its requested NFE and flagged on mismatch. A CPU unit test
(`tests/test_nfe_study_integrity.py`) independently pins that the solver honours
1/2/4/8/16 — including 16, which exceeds the checkpoint's trained default of 4 —
and that the override reaches flow wrappers while leaving `GaussianDiffusion`
untouched, so the reference arm cannot silently drop below 100 steps.

## 2. Statistical power, stated up front

At 288 paired episodes per arm (3 x 96), with a baseline near 0.85-0.88, the
approximate power to detect a true difference is:

| True difference | Power |
|---|--:|
| 3 points | ~0.20 |
| 5 points | ~0.44-0.54 |
| 7 points | ~0.75-0.86 |
| 10 points | ~0.98 |

**This study can detect differences of roughly 7 points or larger. It cannot
resolve 3-point differences.** Any "no significant difference" below must be
read as "not larger than about 5-7 points", not as "identical".


## 3. Results

*(filled in when all 18 runs complete)*

## 4. Between-replicate variance

*(filled in when all 18 runs complete)*

## 5. Does the NFE trend replicate?

This is the section that most affects how the study should be read.

The two completed replicates **disagreed on the sign of the NFE trend**:

| Replicate | Spearman rho | p | success at NFE 1/2/4/8/16 |
|---|--:|--:|---|
| 0 | −0.369 | 0.541 | 0.854 0.885 0.885 0.844 0.854 |
| 1 | **+1.000** | **<0.001** | 0.760 0.865 0.885 0.896 0.917 |

Same checkpoint, same six arms, same protocol — only the 96 episodes differ.

Replicate 0 alone supports "the NFE axis is flat".
Replicate 1 alone supports "success rises monotonically with NFE".

Both would have been reported confidently from a single 96-episode run. This is
the concrete justification for the three-replicate design, and it means **no
per-set trend may be quoted on its own**; only the pooled curve is defensible.

## 6. Answer to the study question

*(filled in when all 18 runs complete)*

## 7. Is 3-cube saturated?

*(filled in when all 18 runs complete)*

## 8. Compute

*(filled in when all 18 runs complete)*
