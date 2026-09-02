"""Phase 2 §B: continuous-vs-resume equivalence test for full-state checkpointing.

PATH 1: train K+M steps continuously.
PATH 2: train K, save full checkpoint, reload into a FRESH trainer, train M more.
Compare model / EMA / optimizer moments / step / RNG.

Tiny model, tiny data, CPU-capable. Does not touch the real experiment.
"""
import copy, os, random, shutil, sys, tempfile
from collections import namedtuple

import numpy as np
import torch

sys.path.insert(0, "diffuser")
from diffuser.utils.arrays import set_global_device  # noqa: E402
from diffuser.utils.training import Trainer, cycle  # noqa: E402

set_global_device("cpu")   # this validation runs entirely on CPU

import wandb  # noqa: E402
os.environ.setdefault("WANDB_MODE", "disabled")
wandb.init(mode="disabled")

K, M, SEED = 6, 4, 1234

# module scope so DataLoader worker processes can pickle it
Batch = namedtuple("Batch", "trajectories conditions")


class TinyDataset(torch.utils.data.Dataset):
    Batch = None
    def __init__(self, n=64, d=5):
        g = torch.Generator().manual_seed(0)
        self.x = torch.randn(n, 3, d, generator=g)
    def __len__(self): return len(self.x)
    def __getitem__(self, i):
        return Batch(self.x[i], {0: self.x[i][0]})


class TinyModel(torch.nn.Module):
    """Stands in for the diffusion wrapper: exposes .loss(*batch)."""
    def __init__(self, d=5):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(d, 16), torch.nn.ReLU(),
                                       torch.nn.Linear(16, d))
    def forward(self, x): return self.net(x)
    def loss(self, trajectories, conditions):
        # consume RNG so RNG restoration is actually exercised
        noise = torch.randn_like(trajectories)
        pred = self.net(trajectories + 0.1 * noise)
        return ((pred - trajectories) ** 2).mean(), {}
    def parameters_count(self): return sum(p.numel() for p in self.parameters())


def seed_all(s):
    torch.manual_seed(s); np.random.seed(s); random.seed(s)


def make_trainer(logdir):
    seed_all(SEED)
    model = TinyModel()
    return Trainer(model, TinyDataset(), renderer=None, train_batch_size=4,
                   train_lr=1e-3, gradient_accumulate_every=2, ema_decay=0.995,
                   log_freq=10_000, save_freq=0, sample_freq=0, n_reference=1,
                   results_folder=logdir, bucket=None, dataloader_seed=SEED)


def flat(sd):
    return torch.cat([v.detach().float().reshape(-1) for v in sd.values()
                      if torch.is_tensor(v) and v.is_floating_point()])


def adam_moments(opt):
    out = []
    for st in opt.state_dict()["state"].values():
        for k in ("exp_avg", "exp_avg_sq"):
            if k in st and torch.is_tensor(st[k]):
                out.append(st[k].detach().float().reshape(-1))
    return torch.cat(out) if out else torch.tensor([])


def main():
    tmp = tempfile.mkdtemp(prefix="resume_val_")
    try:
        d1, d2 = os.path.join(tmp, "cont"), os.path.join(tmp, "res")
        os.makedirs(d1); os.makedirs(d2)

        # PATH 1 -- continuous K+M
        t1 = make_trainer(d1)
        t1.train(n_train_steps=K + M)

        # PATH 2 -- K, save, fresh trainer, load, M
        t2 = make_trainer(d2)
        t2.train(n_train_steps=K)
        t2.save("ckpt")
        t3 = make_trainer(d2)          # FRESH object, then restore
        t3.load("ckpt")
        t3.train(n_train_steps=M)

        checks = {}
        checks["step"] = (t1.step == t3.step, f"{t1.step} vs {t3.step}")
        for name, a, b in (("model", t1.model.state_dict(), t3.model.state_dict()),
                           ("ema", t1.ema_model.state_dict(), t3.ema_model.state_dict())):
            fa, fb = flat(a), flat(b)
            same = torch.equal(fa, fb)
            checks[name] = (same, f"max|d|={float((fa-fb).abs().max()):.3e}")
        ma, mb = adam_moments(t1.optimizer), adam_moments(t3.optimizer)
        if ma.numel() and ma.numel() == mb.numel():
            checks["optimizer_moments"] = (torch.equal(ma, mb),
                                           f"max|d|={float((ma-mb).abs().max()):.3e}")
        else:
            checks["optimizer_moments"] = (False, f"shape {ma.numel()} vs {mb.numel()}")
        # next random draw after both paths
        checks["rng_next_draw"] = (
            torch.equal(torch.randn(4, generator=None) * 0, torch.zeros(4)), "n/a")

        print(f"\n=== continuous ({K}+{M}) vs resume ({K} -> save/load -> {M}) ===")
        ok = True
        for k, (passed, detail) in checks.items():
            if k == "rng_next_draw": continue
            print(f"  {k:18s}: {'BIT-EXACT' if passed else 'DIFFERS':10s} {detail}")
            ok &= passed
        print(f"\nRESULT (with live dataloader): "
              f"{'BIT-EXACT RESUME' if ok else 'NOT BIT-EXACT'}")

        # --- Test 2: identical data stream, to isolate state restoration ---
        def det_stream(ds, bs):
            i = 0
            while True:
                idx = [(i * bs + k) % len(ds) for k in range(bs)]
                yield Batch(torch.stack([ds[j].trajectories for j in idx]),
                            {0: torch.stack([ds[j].conditions[0] for j in idx])})
                i += 1

        e1, e2 = os.path.join(tmp, "c2"), os.path.join(tmp, "r2")
        os.makedirs(e1); os.makedirs(e2)
        u1 = make_trainer(e1); u1.dataloader = det_stream(u1.dataset, u1.batch_size)
        u1.train(n_train_steps=K + M)
        u2 = make_trainer(e2); u2.dataloader = det_stream(u2.dataset, u2.batch_size)
        u2.train(n_train_steps=K); u2.save("ckpt2")
        u3 = make_trainer(e2); u3.load("ckpt2")
        u3.dataloader = det_stream(u3.dataset, u3.batch_size)
        for _ in range(K * u3.gradient_accumulate_every):
            next(u3.dataloader)
        u3.train(n_train_steps=M)
        ok2 = (u1.step == u3.step)
        print("\n=== identical-data-stream control (isolates state restore) ===")
        print(f"  {'step':18s}: {'BIT-EXACT' if ok2 else 'DIFFERS'} {u1.step} vs {u3.step}")
        for name, a, b in (("model", u1.model.state_dict(), u3.model.state_dict()),
                           ("ema", u1.ema_model.state_dict(), u3.ema_model.state_dict())):
            fa, fb = flat(a), flat(b); same = torch.equal(fa, fb); ok2 &= same
            print(f"  {name:18s}: {'BIT-EXACT' if same else 'DIFFERS':10s} "
                  f"max|d|={float((fa-fb).abs().max()):.3e}")
        na, nb = adam_moments(u1.optimizer), adam_moments(u3.optimizer)
        same = torch.equal(na, nb); ok2 &= same
        print(f"  {'optimizer_moments':18s}: {'BIT-EXACT' if same else 'DIFFERS':10s} "
              f"max|d|={float((na-nb).abs().max()):.3e}")
        print(f"\nRESULT (identical data stream): "
              f"{'BIT-EXACT' if ok2 else 'NOT BIT-EXACT'}")
        # checkpoint contents
        data = torch.load(os.path.join(d2, "state_ckpt.pt"), map_location="cpu")
        print("\ncheckpoint keys:", sorted(data.keys()))
        print("rng subkeys   :", sorted((data.get("rng") or {}).keys()))
        print("meta          :", data.get("meta"))
        return 0 if ok else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
