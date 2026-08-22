# Repository snapshot for the pre-submission audit

Recorded 2026-08-22. All values obtained by direct command execution; nothing
inferred.

## Repository state

| Item | Value |
|---|---|
| Branch | `fast-generative-policies` |
| HEAD | `6d75e92bcce4bdb6f7cd6ef0f4a650da532c5055` |
| `git status --porcelain` | empty (clean tree) |

Command: `git rev-parse --abbrev-ref HEAD; git rev-parse HEAD; git status --porcelain`

Recent commits (`git log --oneline -15`):

```
6d75e92 isaacgym: generalize the probe by cube count and add cross-difficulty analysis
4af59f9 isaacgym: verify the 5-cube probe is valid before spending GPU
3f5042a isaacgym: 4-cube probe complete -- Regime B, useful headroom
0e4b743 isaacgym: add cubes-completed distribution and paired per-object analysis
b0b2aae isaacgym: add the 4-cube headroom probe (not yet run)
6492021 isaacgym: verify 4-cube probe validity before spending GPU
ccf4768 isaacgym: NFE study complete -- two calls match Gaussian at a hundred
14496fa isaacgym: NFE study replicate 1 complete, pooled n=192
7ab6a43 isaacgym: record the replicate disagreement in the NFE study report
3a5816c isaacgym: check whether the NFE trend itself replicates
```

## Environment

| Item | Value |
|---|---|
| Python | 3.8.20 |
| PyTorch | 2.1.2+cu121 |
| CUDA (torch) | 12.1 |
| cuDNN | 8902 |
| Platform | Linux-7.0.0-28-generic-x86_64-with-glibc2.17 |
| GPU | NVIDIA GeForce RTX 4080 (16376 MiB) |
| NVIDIA driver | 580.173.02 |
| Isaac Gym | `/home/jren313/software/isaacgym/python/isaacgym` (Preview 4) |
| Conda env | `ecdiffuser-linux` |

## Artifact hashes (SHA256, full file)

| Artifact | Bytes | SHA256 |
|---|--:|---|
| Gaussian checkpoint | 485,446,653 | `7116627cdd841f76035f06dcbc09c66ff84e0a58d8c4146d42264cba2d7492f5` |
| Flow checkpoint | 485,430,114 | `861dc34434474455a25dc3a15ea4e1754066202df538364cf41114b42f4fcc3b` |
| DLP encoder | 36,853,253 | `a8a1113048df79c0fd00cdd4539779a7b3c588cefc4e5f26eb50c0482f756236` |
| Policy training data | 811,232,768 | `7abf83b82fcf2bae801ddae6fa6138d505f4957570d638e37da1e5a5290baf12` |

Paths:

- Gaussian: `ecdiffuser-data/pretrained_models/panda_push/diffusion/3C_adalnpintlarge_dlp_randcolor_H5_T100/state_1200000.pt`
- Flow: `data/panda_push/flow/3C_dlp_adalnpint_randcolor_H5_T4_seed42/state_400000.pt`
- DLP: `ecdiffuser-data/latent_rep_chkpts/dlp_push_6C/dlp_panda_push.pth`
- Policy data: `ecdiffuser-data/push_cubes/3C_randcolor/panda_push_replay_buffer_dlp.pkl`

**Note on the Flow checkpoint filename.** `state_400000.pt` is an *epoch label*,
not a step count. Its internal `step` field is **499000** (verified by loading
the file and reading `d['step']`). This offset is documented in
`experiments/isaacgym_flow_diagnosis.md` §11.

## Running work at audit start

The 5-cube probe (PID 3725052, launched from commit `6d75e92`) was **still
running** when this audit began: `[env] num_objects=5 horizon=200`, episode set
`f8dff00dfd7b1752`, Gaussian arm partially complete. Per the audit directive its
files and processes were not touched, and no competing GPU work was launched.
Its partial outputs are **excluded from all audit evidence** until it exits.
