# Control-Frequency Audit — PandaPush vs OGBench

Traced from configs and source, not from documentation or recollection.
Purpose: test whether *"Panda may hide action-generation improvements because
control frequency is low"* is **quantitatively** plausible.

---

## 1. PandaPush (Isaac Gym) — **3 Hz**, stated and verified

Config (`env_config/generalization_num_cubes/IsaacPandaPushConfig.yaml`, the file
actually used for 3-cube via `Args.env_config_dir`):

| line | key | value |
|--:|---|---|
| 26 | `controlFrequency` | **3  # Hz** |
| 48 | `sim.dt` | `0.01667`  (= 1/60) |
| 49 | `sim.substeps` | `2` |
| 11 | `episodeLength` | `[30, 50, 100]  # [1C, 2C, 3C]` |

The value is not merely declared — it is consumed:

- `isaac_panda_push_env.py:235` — `self.control_timestep = 1.0 / self.cfg["env"]["controlFrequency"]`
- `isaac_panda_push_env.py:932` — `for i in range(int(self.control_timestep / self.sim_timestep)):`
  wraps the OSC torque computation, i.e. the inner physics loop per policy action.

**Derivation:**

```
sim dt                = 0.01667 s          -> 60 Hz sim interaction
substeps = 2                                -> 120 Hz effective physics
control_timestep      = 1 / 3 = 0.3333 s
sim steps per action  = int(0.3333 / 0.01667) = 19
EFFECTIVE POLICY CONTROL RATE = 3 Hz
```

**One policy action is held for ~19 simulation steps (~0.33 s).**
A 100-step 3-cube episode therefore spans **33.3 s** of simulated time.

## 2. OGBench manipspace (MuJoCo) — **20 Hz**, verified from source

Installed locally: `ogbench-1.2.1` at
`data/envs/ogbench-eval/lib/python3.8/site-packages/ogbench`.

- `manipspace/envs/env.py:19-20` — base defaults `physics_timestep=0.002`,
  `control_timestep=0.002`, documented at `env.py:29` as *"Timestep used for
  control. This is the actual control frequency."*
- `manipspace/envs/manipspace_env.py:26-27` — **the operative override**:
  `physics_timestep=0.002`, `control_timestep=0.05`
- `manipspace/envs/cube_env.py:9` — `class CubeEnv(ManipSpaceEnv)`, and its
  `super().__init__(*args, **kwargs)` at line 42 passes no timestep override, so
  cube tasks inherit `0.05`.

**Derivation:**

```
physics_timestep = 0.002 s   -> 500 Hz physics
control_timestep = 0.05  s   -> 20 Hz  POLICY CONTROL RATE
n_substeps       = 0.05/0.002 = 25
```

## 3. Comparison

| | PandaPush 3-cube | OGBench cube (manipspace) |
|---|--:|--:|
| physics | 120 Hz (60 Hz × 2 substeps) | 500 Hz |
| **policy control** | **3 Hz** | **20 Hz** |
| sim steps per action | 19 | 25 |
| seconds per action | 0.333 | 0.050 |
| episode length | 100 steps = 33.3 s | task-dependent |

**OGBench manipspace runs the policy 6.7× more often than PandaPush.**

## 4. Interpretation — what this does and does not license

**Established:** Tal's premise is quantitatively correct. PandaPush is a **3 Hz**
benchmark; each action commits the arm for a third of a second, during which an
OSC controller runs ~19 physics steps to a target the policy cannot revise.

**Plausible consequence, NOT established:** at 3 Hz, a large fraction of task
outcome is determined by the *low-frequency waypoint sequence* — closer to the
state-plan — while fine action detail is smoothed by the OSC controller and by the
long hold. An improvement that sharpens action generation would then have limited
room to show up in success rate. This is a **hypothesis (H5)**, and it is
consistent with our own frozen NFE results: the arm-neutral curve found action
replay error minimized at **NFE 2** while state error kept improving to NFE 32, and
closed-loop control was famously hard to move — 0/12 contrasts had CIs excluding
zero.

**Not claimed:** that higher control frequency *causes* a larger two-time benefit.
That requires the OGBench experiment, and OGBench remains a **secondary** step
behind the PandaPush mechanism study (brief §15). Prior OGBench work in this repo
failed with a documented goal-stitching pathology (99.2% of conditioning goals
beyond the trajectory endpoint), so its harness must be re-audited before reuse.

**Caveat on the comparison:** the two benchmarks differ in simulator, controller
(OSC vs OGBench's actuation), action space, and task, so 3 Hz vs 20 Hz is a
control-rate comparison, not a controlled experiment. It supports prioritizing a
higher-frequency benchmark for transfer; it does not by itself predict effect size.

## Sources

`env_config/generalization_num_cubes/IsaacPandaPushConfig.yaml`;
`isaac_panda_push_env.py:235,932`;
`ogbench/manipspace/envs/env.py:19-20,29`;
`ogbench/manipspace/envs/manipspace_env.py:26-27`;
`ogbench/manipspace/envs/cube_env.py:9,42`.
