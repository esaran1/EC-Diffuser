# Policy breadth feasibility gate

2026-08-30 · commit `73ff86d` · **No GPU used. Nothing downloaded. No training.**
All external claims verified by HEAD requests and raw source inspection.

---

# 1. Executive verdict

**BREADTH-GO-3 → `lerobot/diffusion_pusht`.**

Not GO-1 (HRI-EU Flow Matching is **rejected** — it contains an inference bug and
has no config-level NFE control). Not GO-2 in its Stanford form (same weights,
heavier plumbing). The winner is the **LeRobot** packaging of Diffusion Policy on
PushT, because varying inference compute on fixed pretrained weights is a
**declared config field**, not a source edit.

# 2. What must generalize for Route D to work

Only **Claim B** must cross policies: *offline action-reconstruction error does
not reliably rank inference budgets by closed-loop utility.* Claim A (semantic
state/action asymmetry) is structurally impossible in an action-only policy and
stays an EC-Diffuser case study. Claim D/E (simulator-realization uncertainty) is
Isaac-Gym-side and does not need the external policy — indeed PushT's determinism
makes it a **contrast case**, which is scientifically better than a replication.

# 3. EC-Diffuser Gaussian audit — **cannot provide an NFE curve**

`diffusion.py:196-215`: `p_sample_loop` iterates
`for i in reversed(range(0, self.n_timesteps))`, and `posterior_variance`,
`posterior_mean_coef1/2` are all precomputed at construction from
`cosine_beta_schedule(n_timesteps)` (lines 69-99). **There is no DDIM path and no
step-subsampling anywhere in the repository.**

Reducing steps would require implementing a new sampler family → sampler-family
confound. **Per §3's instruction, no "Gaussian NFE curve" is invented.**

Valid comparisons only: Gaussian **@100 fixed** as a reference operating point,
and joint state/action offline errors at that fixed budget (identical metric path
to Flow — already done in earlier work).

Checkpoint inventory (all local, verified):

| checkpoint | n_timesteps | note |
|---|--:|---|
| `panda_push/.../3C_adalnpintlarge_dlp_randcolor_H5_T100` | 100 | the canonical Gaussian arm |
| `panda_push/.../PushT_3C_dlp_pintlarge_H5_T5` | **5** | trained at 5 — no truncation axis |
| `kitchen/.../kitchen_1C_dlp_40kp_0.25anchor_s_H5_T5` | **5** | trained at 5 — no truncation axis |

**One Gaussian checkpoint per task → checkpoint-level evidence only**, never
algorithm-level. Classified as **INTERNAL CROSS-GENERATIVE CONTROL**, not
independent breadth (§3).

Also audited and rejected as free axes: `shortcut_benchmark` and
`improved_meanflow_benchmark_mb8` contain only `state_0.pt` at
`n_train_steps=1001` — throughput benchmarks, not trained models.

# 4. HRI Flow Matching audit — **REJECT**

Repo is public and the Drive checkpoints resolve. But `examples/flow_pusht.py:211-223`:

```python
timehorion = 1                       # hardcoded local, not config/CLI
for i in range(timehorion):
    noise = torch.rand(...)          # BUG: uniform, training used randn
    ...
```

Three disqualifiers:
1. `timehorion` is a **hardcoded local** (1 in `flow_pusht.py`/`flow_mimic.py`, 16
   in `flow_kitchen.py`) → NFE control requires source edits.
2. **Train/test noise mismatch**: training draws `torch.randn`, inference draws
   `torch.rand` (uniform [0,1]) — in *all four* scripts.
3. Fresh noise is re-drawn **inside** the integration loop → not a valid ODE solve.

Any NFE curve measured here would characterize a defect, not the method. This is
the same class of silent bug our own action-path audits were built to catch.

# 5. Original Diffusion Policy audit — **viable, second choice**

Public PushT checkpoint `epoch=0550-test_mean_score=0.969.ckpt`
(**1,044,185,793 B**, HTTP 200); `pusht.zip` **30,988,725 B**. Same weights class
as LeRobot. Drawback: `eval.py` accepts only `--checkpoint/--output_dir/--device`
with config baked into the pickled payload → NFE variation requires in-process cfg
patching. **6-10 h effort vs 2-4 h.** Retain as a corroborating replication.

# 6. LeRobot Diffusion Policy audit — **WINNER (independently verified)**

I verified every load-bearing claim myself rather than trusting the audit:

| item | verified value |
|---|---|
| `model.safetensors` | **1,050,862,408 B**, HTTP 200 (`x-linked-size` confirms) |
| `noise_scheduler_type` | `"DDPM"` |
| `num_train_timesteps` | **100** |
| `num_inference_steps` | **`null`** (declared field, defaults to 100) |
| dataset `lerobot/pusht` | **7.7 MB** |
| published baseline | 65.4% over 500 episodes |

Mechanism, from `modeling_diffusion.py`:
```
228  if config.num_inference_steps is None:
229      self.num_inference_steps = self.noise_scheduler.config.num_train_timesteps
256  self.noise_scheduler.set_timesteps(self.num_inference_steps)
258  for t in self.noise_scheduler.timesteps:
260      model_output = self.unet(...)
```
**Exactly one UNet call per timestep → 1 forward = 1 NFE.** `set_timesteps`
strides the *same* ancestral DDPM chain on the *same* weights: **no sampler-family
switch, no retraining.** This is the clean single-variable manipulation Route D
requires.

### Correction to the audit's divisor caveat

The audit warned that non-divisors of 100 silently change the step count. I
reproduced `set_timesteps`' striding: `len(timesteps)` **always equals `n`**. The
divisor issue affects **stride uniformity** (`step_ratio` floors), not the NFE
count, so the NFE axis is never mislabeled. I will still restrict to divisors
**{100, 50, 25, 20, 10, 5, 4, 2, 1}** for uniform spacing. Note `n=1` yields a
single step at `t=0`, which must be sanity-checked empirically before use.

# 7. Consistency Policy audit — **REJECT**

README documents training an EDM teacher then distilling the student yourself. **No
checkpoint URLs exist.** Violates the no-retraining constraint outright.

# 8. FlowPolicy / ManiFlow / other

- **FlowPolicy (AAAI 2025):** no policy checkpoints; repo contains only
  `third_party/VRL3/vrl3_ckpts/*.pt` (expert *data-generation* policies). README:
  "You could generate demonstrations by yourself." Vendors mujoco-py 2.1.2.14 +
  gym 0.21.0 — very difficult to install in 2026. **REJECT.**
- **ManiFlow (CoRL 2025):** datasets only, no policy checkpoints; needs
  RoboTwin + Adroit + DexArt + MetaWorld. **REJECT.**
- **OneDP:** no official repo located. **Unavailable.**
- Also: FlowPolicy/ManiFlow are already *few-step by design*, so truncating a
  single pretrained iterative policy is not even the right question for them.

# 9. Candidate comparison table

| Candidate | Type | Joint s/a? | Ckpt | Size | Env | Sim | NFE variable on fixed weights? | Offline metric | Closed-loop | Retrain? | Setup h | GPU-h | Verdict |
|---|---|---|---|--:|---|---|---|---|---|---|--:|--:|---|
| **lerobot/diffusion_pusht** | DDPM policy | no | ✅ | 1.05 GB | PushT | pymunk CPU **deterministic** | ✅ **config field** | write (~2 h) | ✅ | no | 2-4 | ~0.3 | **GO** |
| real-stanford/diffusion_policy | DDPM policy | no | ✅ | 1.04 GB | PushT | pymunk CPU | ✅ (cfg patch) | write | ✅ | no | 6-10 | ~0.3 | backup |
| EC-Diffuser Gaussian | DDPM, joint | **yes** | ✅ local | — | PushCube | Isaac Gym | ❌ **no DDIM/striding** | ✅ | ✅ | no | 0 | ~0.3 | fixed-budget ref only |
| HRI-EU/flow_matching | Flow | no | ✅ | ~GB | PushT/Kitchen/Mimic | mixed | ❌ hardcoded + **noise bug** | ✗ | ✅ | no | 20+ | — | **REJECT** |
| Consistency-Policy | consistency | no | ❌ | — | PushT/Mimic | — | — | — | ✅ | **yes** | — | — | **REJECT** |
| FlowPolicy | consistency-FM | no | ❌ | — | Adroit/MetaWorld | mujoco-py | — | — | ✅ | **yes** | — | — | **REJECT** |
| ManiFlow | FM | no | ❌ | — | many | — | — | — | ✅ | **yes** | — | — | **REJECT** |

# 10. Winning external policy

**`lerobot/diffusion_pusht`** — independently authored (HuggingFace/LeRobot),
independently trained weights, different architecture (1D conv UNet vs our
entity-centric transformer), different generative family (DDPM vs Flow), different
environment (PushT), different simulator (pymunk CPU). Satisfies every §12
independence requirement.

# 11. Exact scientific role

Tests **Claim B only**: does an offline action-reconstruction metric rank
inference budgets the same way closed-loop success does? It **cannot** test Claim
A (action-only policy, no state branch) and must not be asked to.

**Bonus role:** PushT is **deterministic** — pure CPU pymunk, fixed `dt`,
`np.random.RandomState(seed)` resets. That makes it a *contrast* to Isaac Gym for
Claim D: nested realization variance is a **contact-rich GPU-physics** property,
not a universal one. That is a stronger, more falsifiable statement than asserting
it everywhere.

# 12. Minimal external experiment

NFE ∈ {1, 2, 4, 5, 10, 20, 50, 100} (divisors). For each:
(a) offline action MSE vs logged expert actions on held-out PushT windows;
(b) closed-loop PushT score, 50 episodes initially (not 500);
(c) UNet call count asserted; (d) planner-only latency.
Determinism check first: same seed + same actions → bit-identical? If yes, R=1
suffices and **no replication protocol is needed** (§23).

# 13. Expected download/storage

**~1.06 GB total** (1.05 GB checkpoint + 7.7 MB dataset). Well under the 2 GB
feasibility gate — but per §18 I have **downloaded nothing**; this needs approval.

# 14. Expected setup effort

**2-4 h.** LeRobot installs via pip; the only bespoke work is the offline
action-error evaluator (neither repo ships one — both `eval.py` report closed-loop
reward only). Dataset exposes expert actions directly.

# 15. Expected GPU cost

**~0.3 GPU-h** for the minimal experiment (8 budgets × 50 episodes + offline
sweep). PushT is CPU-physics, so GPU is only the UNet.

# 16. Simulator repeatability requirements

**Do not port R=3 from Isaac Gym** (§23). Test PushT determinism first. Evidence
says deterministic (CPU pymunk, fixed dt, seeded RandomState) → likely R=1 valid.
If confirmed, that is a **finding**, not an inconvenience.

# 17. Evaluation-method prior-art correction

I withdraw the earlier phrasing that this ground is "unoccupied."
- **GPUSimBench (2607.13059)** owns GPU-simulator nondeterminism, four
  stochasticity regimes, divergence measured in **cm of physical state**.
- **N-SCORE (2603.13616)** owns statistically rigorous sequential policy
  comparison — **assuming i.i.d. evaluation data**.
- **Stochastic-simulation guarantees (2309.10874)** owns finite-sample failure
  probability / risk bounds under stochastic simulators.
- **RoboDojo (2607.04434)** standardizes protocols (50 eps × 3 seeds, mean±sd),
  no CIs, no nondeterminism discussion.

A hierarchical/clustered bootstrap is **standard statistics** and must not be
presented as a new method.

# 18. Exact evaluator novelty that remains

Narrow and **empirical**, not statistical:

1. **Magnitude in success-rate units.** GPUSimBench measures cm; we measure that
   8 identical repeats span **10.4 pp** (SD 4.23 pp) with everything fixed, and
   that **39/96 episodes are physics-sensitive** while 0/96 fail robustly.
2. **A demonstrated wrong conclusion.** Recomputing NFE4−NFE1 from single
   realizations of our own data: seed 43 gives **+6.2, +5.2, or −3.1 pp** —
   **the sign flips** with the realization drawn, versus +2.8 pp at R=3. This is
   the §19/§25-E demonstration, already in hand at zero GPU cost.
3. **Empirical resolution curves** (7.55 → 4.36 → 3.38 → 2.67 pp for R=1/3/5/8)
   letting a practitioner pick R for a target effect size.
4. **Contact-triggered attribution** (200× amplification at contact onset).
5. **A deterministic contrast case** (PushT) showing where R=1 *is* sufficient.

# 19. Claim matrix

| Claim | Evidence now | External replication | Status |
|---|---|---|---|
| **A** semantic state/action NFE asymmetry | Flow ×3 seeds, Level 1 | needs another *joint* policy — none feasible | **case study, not universal** |
| **B** offline metrics mis-rank budgets | EC-Diffuser only | **LeRobot DP feasible** | **the breadth claim** |
| **C** low budgets alter occupancy | EC-Diffuser, 28% NFE-specific | optional | **secondary/demoted** |
| **D** R=1 understates uncertainty | strong, Isaac Gym | PushT = contrast | **strong** |
| **E** nested protocol corrects rankings | sign-flip demo in hand | — | **strong, empirical only** |

# 20. Minimum top-conference experimental package

1. Flow ×3 checkpoints (have) · 2. calibrated control (have) · 3. arm-neutral
semantic curves (have) · 4. Gaussian@100 fixed-budget control (cheap, valid) ·
5. **LeRobot DP external replication of Claim B** (new) · 6. 4/5-cube cross-task ·
7. offline-vs-control across both policies · 8. noise calibration (have) ·
9. **sign-flip demonstration** (have) · 10. positioning vs SANTS / Efficient-WAM /
Flash-WAM / GPUSimBench / N-SCORE / 2309.10874.

# 21. Full compute roadmap (rebuilt, not inherited)

**MUST-HAVE**

| # | purpose | GPU-h | unlocks |
|---|---|--:|---|
| 1 | PushT determinism check + harness validation vs 65.4% | 0.1 | trust |
| 2 | LeRobot DP: 8 budgets × offline action MSE | 0.1 | Claim B offline |
| 3 | LeRobot DP: 8 budgets × 50 closed-loop episodes | 0.3 | **Claim B** |
| 4 | Gaussian@100 joint offline metrics (in-repo) | 0.3 | Axis-1 control |
| 5 | replicate occupancy/disagreement on seeds 43,44 | 0.25 | Claim C rigor |
| | **subtotal** | **~1.05** | |

**NICE-TO-HAVE**

| # | purpose | GPU-h |
|---|---|--:|
| 6 | 4/5-cube calibrated NFE pairs (fixed-H) | ~2.8 |
| 7 | LeRobot DP at 500 episodes for headline | ~0.6 |
| | **subtotal** | **~3.4** |

**Total ~4.5 GPU-h** — the earlier 3.4 figure was optimistic; it omitted the
external policy entirely.

# 22. Five reviewer kill shots

1. *"Claim A is one model in one environment."* — Conceded; it is framed as a
   case study, with Claim B carrying breadth.
2. *"SANTS already showed non-monotonic depth-vs-utility."* — Theirs is open-loop
   on fixed demos with action denoising frozen; ours is closed-loop.
3. *"GPUSimBench owns simulator nondeterminism."* — They measure cm; we measure
   success-rate resolution and a reversed conclusion.
4. *"Hierarchical bootstrap isn't new."* — Agreed, and we don't claim it. The
   contribution is the measured magnitude and the sign-flip demonstration.
5. *"Two policies is still thin."* — Fair. Mitigated by two *simulators* with
   opposite determinism properties, which is the more informative axis.

# 23. GO / NO-GO

**BREADTH-GO-3, CONDITIONAL on a ~1.06 GB download.**

External breadth is genuinely feasible for **Claim B only**, at ~1.05 GPU-h
must-have. Claim A remains an honest single-system case study. If the ~1.06 GB
download is not approved, or if the LeRobot harness fails to reproduce 65.4%,
fall back to **Route E (internal report)** rather than padding a case study.
