"""§5: the run lock must not alter training semantics or consume RNG."""
import os, sys, tempfile, shutil
import numpy as np, torch
sys.path.insert(0, "diffuser")
from diffuser.utils.arrays import set_global_device  # noqa: E402
set_global_device("cpu")
import wandb  # noqa: E402
wandb.init(mode="disabled")
from diffuser.utils.run_lock import RunLock  # noqa: E402
src = open("experiments/policy_improvement/validate_resume.py").read().split("def flat")[0]
exec(src)

def flat(sd):
    return torch.cat([v.detach().float().reshape(-1) for v in sd.values()
                      if torch.is_tensor(v) and v.is_floating_point()])

def run(logdir, use_lock):
    lock = RunLock(logdir).acquire() if use_lock else None
    t = make_trainer(logdir)
    seen = []
    orig = t.dataloader
    def spy():
        for b in orig:
            seen.append(float(b.trajectories.sum())); yield b
    t.dataloader = spy(); t.save_freq = 0
    t.train(n_train_steps=10)
    if lock: lock.release()
    return t, seen

tmp = tempfile.mkdtemp()
try:
    a, sa = run(os.path.join(tmp, "nolock"), False)
    b, sb = run(os.path.join(tmp, "lock"), True)
    fa, fb = flat(a.model.state_dict()), flat(b.model.state_dict())
    ea, eb = flat(a.ema_model.state_dict()), flat(b.ema_model.state_dict())
    checks = [
        ("minibatch identities identical", sa == sb, f"{len(sa)} batches"),
        ("model weights identical", torch.equal(fa, fb), f"max|d|={float((fa-fb).abs().max()):.3e}"),
        ("EMA weights identical", torch.equal(ea, eb), f"max|d|={float((ea-eb).abs().max()):.3e}"),
        ("step identical", a.step == b.step, f"{a.step} vs {b.step}"),
    ]
    ok = True
    print("=== lock is training-inert? ===")
    for n, p, d in checks:
        ok &= p; print(f"  {'PASS' if p else 'FAIL'}  {n:32s} {d}")
    print(f"\nRESULT: {'LOCK IS TRAINING-INERT' if ok else 'LOCK ALTERS TRAINING'}")
    sys.exit(0 if ok else 1)
finally:
    shutil.rmtree(tmp, ignore_errors=True)
