"""Section-4 audit: is ANY other loss-relevant config field silently dropped?

Compares, for each arm config:
  (a) the diffusion-wrapper constructor's full parameter list
  (b) what diffusion_wrapper_kwargs() actually forwards
  (c) the fields the arm config actually SETS

An omission is a field the config sets, that the constructor accepts, that the
whitelist does NOT forward -> it would silently fall back to a default.
CPU-only; imports nothing that needs a GPU.
"""
import inspect, json, os, sys, importlib, importlib.util
sys.path.insert(0, os.path.abspath("diffuser"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from diffuser.configuration import diffusion_wrapper_kwargs

# Resolve the wrapper class exactly as train.py does: from the config's
# `diffusion` field, so this audit follows the real indirection rather than a
# hardcoded class name that could drift away from what training constructs.
_cfgmod = importlib.import_module("config.pandapush_lossft_armB")
_diff_path = _cfgmod.base["diffusion"]["diffusion"]
# Resolve exactly as diffuser.utils.config.import_class does: repo_name is the
# first segment of that module's __name__ ('diffuser'), joined with the config
# string's module part. Replicated inline rather than importing utils.config,
# which would pull in utils.rendering and its GPU-side deps.
_m, _c = ".".join(_diff_path.split(".")[:-1]), _diff_path.split(".")[-1]
WrapperCls = getattr(importlib.import_module("diffuser." + _m), _c)
print("wrapper class:", _diff_path)
sig = inspect.signature(WrapperCls.__init__)
ctor_params = [p for p in sig.parameters if p not in ("self", "model")]

# Also the trainer, since step_start_ema/lr_warmup_steps are loss-relevant plumbing.
from diffuser.utils.training import Trainer
tr_params = [p for p in inspect.signature(Trainer.__init__).parameters
             if p not in ("self", "diffusion_model", "dataset", "renderer")]

class NS:
    pass

report = {"ctor_params": ctor_params, "trainer_params": tr_params, "arms": {}}
for arm in ("A", "B", "C", "D"):
    mod = importlib.import_module(f"config.pandapush_lossft_arm{arm}")
    d = mod.base["diffusion"]
    ns = NS()
    for k, v in d.items():
        setattr(ns, k, v)
    ns.n_diffusion_steps = d.get("n_diffusion_steps", 4)
    fwd = diffusion_wrapper_kwargs(ns, observation_dim=480, action_dim=4)
    set_fields = set(d.keys())
    # fields the config sets that the CONSTRUCTOR would accept
    ctor_relevant = set_fields & set(ctor_params)
    omitted = sorted(ctor_relevant - set(fwd.keys()))
    report["arms"][arm] = {
        "forwarded": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(type(v)))
                      for k, v in fwd.items()},
        "config_fields_ctor_accepts": sorted(ctor_relevant),
        "OMITTED_FROM_WHITELIST": omitted,
        "mask_terminal_action": fwd.get("mask_terminal_action", "<<ABSENT>>"),
        "lambda_action": fwd.get("lambda_action", "<<ABSENT>>"),
        # trainer-side: does train.py read them via getattr with a default?
        "trainer_side_set": sorted(set_fields & set(tr_params)),
    }

# `n_diffusion_steps` is reported as "omitted" but is a benign ALIAS, not a
# defect: the constructor resolves n_solver_steps / n_diffusion_steps /
# n_timesteps by precedence into one value, and the whitelist forwards the
# config's value as n_timesteps=4. With the other two left None, n_timesteps
# wins and all three attributes are set to 4. Verified by reading
# ConditionalFlowMatching._resolve_constructor_steps. Cleared, not waived.
BENIGN_ALIASES = {"n_diffusion_steps"}
real = {a: sorted(set(report["arms"][a]["OMITTED_FROM_WHITELIST"]) - BENIGN_ALIASES)
        for a in report["arms"]}
for a in report["arms"]:
    report["arms"][a]["OMITTED_EXCLUDING_BENIGN_ALIASES"] = real[a]
report["benign_aliases_cleared"] = sorted(BENIGN_ALIASES)
any_omission = any(real[a] for a in real)
report["ANY_OMISSION"] = any_omission
report["ANY_REAL_OMISSION"] = any_omission
print(json.dumps(report, indent=2))
out = "experiments/policy_improvement/loss_finetune/rerun/WHITELIST_AUDIT.json"
json.dump(report, open(out, "w"), indent=2)
print("\nANY_OMISSION:", any_omission)
