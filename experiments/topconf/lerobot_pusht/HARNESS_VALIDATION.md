# Harness validation — PASSED (with a reproducibility finding)

## The LeRobot version incompatibility (reproducibility finding, NOT a policy result)

The first harness run produced `avg_max_reward = 0.0087` and 0% success — the
policy appeared completely broken.

**Root cause.** Under **lerobot 0.4.4**, loading this March-2025 checkpoint emits:

```
WARNING:root:Unexpected key(s) when loading model:
 ['normalize_inputs.buffer_observation_image.mean',
  'normalize_inputs.buffer_observation_image.std',
  'normalize_inputs.buffer_observation_state.max',
  'normalize_inputs.buffer_observation_state.min',
  'normalize_targets.buffer_action.max',
  'normalize_targets.buffer_action.min',
  'unnormalize_outputs.buffer_action.max',
  'unnormalize_outputs.buffer_action.min']
```

In 0.4.4 the policy exposes only a single `diffusion` child module —
normalization was moved out of the policy into a processor pipeline — so all
eight normalization buffers were **silently discarded**. The policy then emitted
**normalized actions in [0,1]** instead of pixel coordinates in [0,512]:

| | action range over 12 steps | resulting agent_pos |
|---|---|---|
| lerobot 0.4.4 (buffers dropped) | min [0.589, 0.446] max [1.0, 1.0] | `[0.675, 0.621]` |
| expected | pixel coords, action space `Box(0, 512)` | `[131, 258]` at reset |

**Fix: pin `lerobot==0.3.2`.** Top-level modules become
`['normalize_inputs', 'normalize_targets', 'unnormalize_outputs', 'diffusion']`,
all eight buffers load with **no warning**, and
`unnormalize_outputs.buffer_action.max = [511., 511.]` — the published semantics.

Checkpoint stat values recovered:
image mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`;
state min `[13.456, 32.938]` max `[496.146, 510.958]`;
action min `[12., 25.]` max `[511., 511.]`.

**This is a checkpoint/library version-compatibility issue, not a defect in
current LeRobot and not a scientific result about the policy.** It is recorded
because an unvalidated harness would have produced an entirely fabricated NFE
curve. No PyPI release contemporaneous with the March-2025 checkpoint exists
(earliest is 0.3.2, 2025-08-01), so 0.3.2 is the closest release preserving the
legacy normalization-buffer contract.

## Validation against the published baseline

Compared on the **exact same seeds** the published run used, rather than against
the aggregate — a much sharper test.

| seed | ours max_reward | published | ours success | published |
|--:|--:|--:|:--:|:--:|
| 1000 | 1.0000 | 0.9812 | True | False |
| 1001 | 1.0000 | 1.0000 | True | True |
| 1002 | 1.0000 | 1.0000 | True | True |
| 1003 | 1.0000 | 1.0000 | True | True |
| 1004 | 1.0000 | 1.0000 | True | True |
| 1005 | 0.9748 | 1.0000 | True | True |
| 1006 | 0.9981 | 1.0000 | True | True |
| 1007 | 1.0000 | 1.0000 | True | True |
| 1008 | 0.8900 | 1.0000 | **False** | True |
| 1009 | 1.0000 | 0.5435 | True | **False** |

- ours (seeds 1000-1009): **avg_max_reward 0.9863**, success 90%
- published, same ten seeds: **avg_max_reward 0.9525**, success 80%
- published, full 500: avg_max_reward 0.9551, `pc_success` 65.4%
- **7/10 episode outcomes identical**

Differences are consistent with diffusion-policy sampling stochasticity at n=10.
Per the protocol, the harness is **not** rejected for a several-point success
difference; reward distribution, action scale, episode length and preprocessing
all match. **GATE PASSED.**

## NFE accounting verified empirically

A forward hook on `policy.diffusion.unet` counted **`calls_per_plan = 100.00`**
at the default budget, and `len(scheduler.timesteps) == num_inference_steps` for
every budget. Both required identities hold:

```
UNet forwards per plan == requested num_inference_steps == len(scheduler.timesteps)
```
