"""Validate the final-checkpoint fix (§1). CPU only, tiny model.

1. a run configured for N steps produces an actual N-step final checkpoint
2. adding the final save does not change training or batch order
3. the final saved weights equal the in-memory weights at termination
"""
import os, sys, tempfile, shutil
import numpy as np, torch
sys.path.insert(0, "diffuser")
from diffuser.utils.arrays import set_global_device  # noqa: E402
set_global_device("cpu")
import wandb  # noqa: E402
wandb.init(mode="disabled")
from diffuser.utils.training import Trainer  # noqa: E402
src = open("experiments/policy_improvement/validate_resume.py").read().split("def flat")[0]
exec(src)

N = 10


def flat(sd):
    return torch.cat([v.detach().float().reshape(-1) for v in sd.values()
                      if torch.is_tensor(v) and v.is_floating_point()])


def run(logdir, do_final_save):
    t = make_trainer(logdir)
    seen = []
    orig = t.dataloader
    def spy():
        for b in orig:
            seen.append(float(b.trajectories.sum())); yield b
    t.dataloader = spy()
    # emulate train.py: epochs of 5 steps, save_freq mimicking the old bug
    t.save_freq = 0
    for _ in range(N // 5):
        t.train(n_train_steps=5)
    if do_final_save:
        t.save('final')
    return t, seen


tmp = tempfile.mkdtemp(prefix="finalsave_")
try:
    d1, d2 = os.path.join(tmp, "nosave"), os.path.join(tmp, "withsave")
    os.makedirs(d1); os.makedirs(d2)
    a, seen_a = run(d1, False)
    b, seen_b = run(d2, True)

    print("=== 2. final save is training-inert ===")
    fa, fb = flat(a.model.state_dict()), flat(b.model.state_dict())
    ea, eb = flat(a.ema_model.state_dict()), flat(b.ema_model.state_dict())
    same_w = torch.equal(fa, fb); same_e = torch.equal(ea, eb)
    same_batches = seen_a == seen_b
    print(f"  model weights identical : {same_w}  max|d|={float((fa-fb).abs().max()):.3e}")
    print(f"  EMA weights identical   : {same_e}  max|d|={float((ea-eb).abs().max()):.3e}")
    print(f"  batch order identical   : {same_batches} ({len(seen_a)} batches)")
    print(f"  step identical          : {a.step == b.step} ({a.step} vs {b.step})")

    print("\n=== 1. N-step run produces an N-step final checkpoint ===")
    p = os.path.join(d2, "state_final.pt")
    exists = os.path.exists(p)
    data = torch.load(p, map_location="cpu") if exists else {}
    print(f"  state_final.pt exists   : {exists}")
    print(f"  configured N            : {N}")
    print(f"  checkpoint internal step: {data.get('step')}")
    step_ok = data.get("step") == N

    print("\n=== 3. saved weights == in-memory weights at termination ===")
    sm = flat(data["model"]); se = flat(data["ema"])
    w_ok = torch.equal(sm, fb); e_ok = torch.equal(se, eb)
    print(f"  model matches in-memory : {w_ok}  max|d|={float((sm-fb).abs().max()):.3e}")
    print(f"  EMA matches in-memory   : {e_ok}  max|d|={float((se-eb).abs().max()):.3e}")
    print(f"  has optimizer state     : {'optimizer' in data}")
    print(f"  has rng state           : {'rng' in data}")

    ok = all([same_w, same_e, same_batches, exists, step_ok, w_ok, e_ok])
    print(f"\nRESULT: {'FINAL-SAVE FIX VALIDATED' if ok else 'VALIDATION FAILED'}")
    sys.exit(0 if ok else 1)
finally:
    shutil.rmtree(tmp, ignore_errors=True)
