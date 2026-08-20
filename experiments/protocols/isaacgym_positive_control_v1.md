# Protocol: Isaac Gym Gaussian positive control (PREDECLARED, NOT RUN)

Status: **predeclared, GPU-blocked.** Written before any rollout so the decision
rule cannot be chosen after seeing the result.

## 1. Why this runs first

This is the only arm in the project whose expected result is published. Until it
reproduces, a Flow failure in Isaac Gym has at least five explanations —
checkpoint, DLP encoder, env config, controller, policy — and none can be
separated. Every downstream item (3–7) produces an ambiguous null if the
pipeline is broken.

## 2. Fixed configuration

| Item | Value |
|---|---|
| Checkpoint | `ecdiffuser-data/pretrained_models/panda_push/diffusion/3C_adalnpintlarge_dlp_randcolor_H5_T100/state_1200000.pt` |
| Weights | EMA |
| Diffusion | `models.GaussianDiffusion`, 100 steps, `predict_epsilon=false` |
| Denoiser | `models.AdaLNPINTDenoiser`, 12 layers, hidden 512, 8 heads |
| DLP encoder | `ecdiffuser-data/latent_rep_chkpts/dlp_push_6C/dlp_panda_push.pth` |
| Env config | `env_config/generalization_num_cubes` (`clipActions` 1.0, `actionScale` 1.0) |
| Entities | 3 cubes, random color |
| Horizon | 5 |
| Episodes | 100, fixed recorded initial and goal states |
| Episode set | hashed with SHA256 and recorded in the result file |

## 3. Pairing requirement

The recorded episode set is created **once** and reused by every later arm —
Flow, the training-step sweep, and any A/B/C arm. Each result file records
`episode_set_sha256`, and a mismatch invalidates the comparison. This is the
same discipline used for the cube-double sweep, applied from the first
measurement rather than retrofitted.

## 4. Metrics

- Success rate (fraction of goals reached) with a Clopper-Pearson 95% interval
- Per-cube final goal distance
- Planner latency and verified denoiser calls per plan (must equal 100)
- Action saturation: fraction of steps with any component at the clip limit
- EE z-trajectory over the episode (tests Defect A, see §6)
- Concurrent GPU processes at launch, recorded in the result file

## 5. Preregistered decision rule

- **Gaussian reproduces its published success rate** -> the pipeline is
  validated. Proceed to item 5 (failure localization) and item 4 (DLP
  visualization) on the same fixed episodes, then Flow.
- **Gaussian fails** -> **STOP.** Debug the pipeline. Do not run any Flow arm,
  do not run the training-step sweep, and do not train the standardized or VP
  arms. A Flow number measured against a broken pipeline is not evidence.
- **Gaussian partially reproduces** (materially below published, clearly above
  zero) -> treat as failure for gating purposes and localize the discrepancy
  before proceeding.

## 6. Defect A rides along at no extra cost

`experiments/isaacgym_debug_investigation.md` §2 establishes that the z action
range is asymmetric, so a normalized-zero output decodes to a downward push of
-0.1498. The EE z-trajectory is therefore recorded in this same run. If the
Gaussian control passes and the arm nonetheless trends downward into the table,
that is a specific, measured mechanism to carry into item 5 — and it costs
nothing extra to collect here.

## 7. Compute

~0.5–1 GPU-h, no training. Must not launch while another process holds GPU
memory; record `nvidia-smi --query-compute-apps` output in the result file.

## 8. Explicitly excluded

No training. No new method. No Flow arm until this passes. No A/B/C
standardization arms — those are gated on this result and on resolving the
loss-balance confound described in the investigation's item 7.
