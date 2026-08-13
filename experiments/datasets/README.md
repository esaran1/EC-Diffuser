# Dataset protocol

Schema version: `fast-generative-policy-dataset-v1`.

This directory contains small, immutable metadata only. Source datasets, converted
arrays, images, checkpoints, and simulator recordings are never committed.

## Common logical schema

Every task adapter must expose an episode table with:

- `observation`: policy-visible state or encoded visual observation;
- `action`: benchmark-native action after documented, invertible normalization;
- `goal`: explicit goal state, future-state relabel, or fixed task goal;
- `episode_id`, `timestep`, `success`, and `task_id`;
- `entity`: object identifiers and object state when the benchmark exposes them.

Optional fields are `rgb`, `depth`, `segmentation`, `proprioception`, `reward`,
`termination`, and `simulator_state`. A task adapter must document which optional
fields are absent. Algorithms share one task adapter and one split; Diffusion,
Flow Matching, Improved MeanFlow, and Shortcut Models do not get method-specific
datasets.

## Split and transformation rules

1. Preserve source files byte-for-byte and record their SHA256.
2. Split only at episode boundaries, before window extraction.
3. Fit normalization on the training split only.
4. Store source episode indices and deterministic transform parameters.
5. Never infer success from reward when the benchmark provides an official success.
6. Never expose privileged simulator state as a policy observation.
7. Keep rejected episodes in the audit manifest with an explicit reason.
8. Use one goal-relabeling policy per task, frozen before method comparison.

The current inventory is in `phase6_dataset_inventory.json`. The local PushCube
health report can be reproduced with:

```bash
python diffuser/scripts/audit_pushcube_dataset.py \
  ecdiffuser-data/push_cubes/3C_randcolor/panda_push_replay_buffer_dlp.pkl \
  --output experiments/datasets/pushcube_3c_randcolor_health.json
```

Bounded task-level state subsets were downloaded under the Git-ignored `data/`
tree and audited. MimicGen RGB is contained in its selected HDF5; DexJoCo camera
videos were deliberately not downloaded. No source array was duplicated.
