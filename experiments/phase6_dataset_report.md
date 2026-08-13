# Phase 6 dataset decision

Status: bounded task-level data and integrity audit complete. No RL training or
demonstration collection was started. The suite is approved for bounded pilots;
full runs remain behind the measured-compute gate.

## Classification

- **B — EC-Diffuser PushCube-3 random-color DLP.** The 811,232,768-byte
  pickle contains 2,000 episodes and 200,000 transitions. It has 1,950 full
  successes, no NaN/Inf, no action-limit violations, and no exact duplicate
  state/action/goal episodes. Episodes 54 and 1327 reach clipped object states
  at +/-1.1 and are predeclared for exclusion. The source has no split.
- **B — OGBench Puzzle-4x4 play (state), proposed Tier A.** Official train and
  validation NPZs are 262,414,247 and 26,216,338 bytes. The play stream must be
  episodized, goal-relabeled, and windowed deterministically.
- **B — MimicGen Three-Piece Assembly D1 large-interpolation, proposed Tier B.**
  The official 4,597,953,838-byte HDF5 has 1,000 generated demos, 514,005
  transitions, and a 700-step evaluation horizon. Only 984 contain a positive
  stored reward; 16 zero-reward demos are quarantined pending simulator replay.
- **B — DexJoCo Hammer-Nail rand_full, proposed Tier B.** The task subset is
  635,864,128 bytes: 100 successful episodes, 27,571 frames, 22-D actions,
  23-D policy state, and two 640x640 AV1 camera streams. It requires a split,
  training-only normalization, and fixed-task goal mapping.

There are no Category D tasks in the proposed native-simulator suite. Therefore
no RL expert algorithm or RL pilot is justified. The migrated Isaac Lab PushCube
task remains Category C and deferred: direct legacy visual/DLP transfer scored
0/16, so demonstrations would be required only if that migration is promoted to
a paper benchmark.

## Local integrity and coverage

PushCube object initial positions cover approximately x=[-0.150, 0.150] and
y=[-0.199, 0.200] for all three cubes. Goals cover approximately
x=[-0.249, 0.150], y=[-0.200, 0.200]. Initial object-goal distance is
0.1945 +/- 0.0927 m; median final distance is 0.01048 m. Action norms have
median 0.0224, P90 0.7787, P99 1.1294; only 0.0482% of action components are
saturated. The three entities have comparable initial/goal-distance coverage.

Weaknesses that must not be hidden:

- 50 failed demonstrations are present; 48 are valid and useful for hindsight
  goal training, while two contain clipped states.
- terminals are zero everywhere; episode boundaries come from the array axes;
- goals and DLP goals are time-constant;
- there are no stored RGB frames, explicit episode seeds, or source split;
- overlap with historical simulator evaluation resets cannot be excluded because
  those reset/goal manifests were never stored.

The deterministic non-copying split manifest excludes only episodes 54 and 1327
and assigns the remaining 1,998 episodes at seed 2606: 1,598 train, 200
validation, and 200 offline test. This split is for future final training; it does
not retroactively change the historical EC-Diffuser/Flow results.

## Storage and compute gate

The bounded download was 4,893,501,470 bytes total: OGBench train/validation,
one MimicGen task HDF5, and DexJoCo state/action metadata without camera videos.
OGBench arrays expand to 1,057,058,100 bytes in memory. All downloaded numerical
data are finite; no exact episode duplicates were found, and OGBench train and
validation have zero exact episode overlap. No data-collection pilot applies
because official demonstrations exist.

Phase 7 is not approved or frozen. Its proposed full matrix is intentionally
costed in `experiments/phase7_experiment_matrix.json` and must not be launched as
written on one GPU without pilot screening.

## Adapter pilots

- OGBench: 997,000 horizon-5 windows; 83-D observations, 5-D actions;
  normalizer SHA256 `2f5ec416b1f407928f1e55b615a7b5a1229580247b4812b18cff0ff1b33a265e`.
- MimicGen: 447,389 horizon-10 windows; 51-D official low-dimensional
  observations, 7-D actions; normalizer SHA256
  `c9c3f08a23f17ce31f33c4babfe72690ff3adaaf6bb3b5932b58ed29c415f3e4`.
- DexJoCo: 22,147 horizon-30 windows; 23-D non-privileged observations,
  22-D actions; normalizer SHA256
  `dc2c868abef76e88897f63fbe34129ef760a0c7ac600a2836bcf32aeb0bd09d7`.

All adapters passed deterministic split, leakage, conditioning, finite-value,
and repository-batch-contract tests. The modern state benchmarks use a shared
flat sequence backbone; PINT remains specific to entity-structured PushCube.
