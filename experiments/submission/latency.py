"""Deployment latency (offline, no simulator): network forward and full policy
sampling per batch size / NFE / precision. CUDA-event timing after warm-up.
Deployment sampling uses torch.randn noise (no evaluator seeding overhead)."""
import json, os, sys, time
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = "/home/jren313/EC-Diffuser-1"
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "diffuser")]
os.chdir(ROOT)
import diffuser.utils as utils  # noqa: E402

DATASET = "ecdiffuser-data/push_cubes/3C_randcolor/panda_push_replay_buffer_dlp.pkl"
exp = utils.load_diffusion("data", "panda_push", "canon/seed42", epoch="400000", seed=42,
                           is_diffusion=True, override_dataset_path=DATASET)
diff = exp.ema; diff.eval(); dev = "cuda:0"; H = diff.horizon; m = diff.model
orig = m.forward; state = {"prec": "fp32"}


def fwd(*a, **k):
    if state["prec"] == "fp16":
        with torch.autocast("cuda", dtype=torch.float16):
            return orig(*a, **k).float()
    return orig(*a, **k)


m.forward = fwd
it = exp.dataset[0]
c1 = {k: torch.as_tensor(v).float().to(dev)[None] for k, v in it[1].items()}


def timed(fn, reps, warm=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return float(np.median(ts)), float(np.percentile(ts, 95))


rows = []
with torch.no_grad():
    for prec in ("fp16", "fp32"):
        state["prec"] = prec
        for b in (1, 4, 16, 96):
            c = {k: v.expand(b, -1).contiguous() for k, v in c1.items()}
            x = torch.randn(b, H, diff.transition_dim, device=dev); t = torch.full((b,), 500.0, device=dev)
            f50, f95 = timed(lambda: m(x, c, t), 50 if b < 96 else 20)
            rows.append(dict(kind="forward", precision=prec, batch=b, nfe=1, ms_p50=f50, ms_p95=f95))
            for nfe in (1, 2, 3, 4, 8, 100):
                reps = 30 if nfe < 100 else (10 if b < 96 else 3)
                s50, s95 = timed(lambda: diff.conditional_sample(c, n_steps=nfe, verbose=False), reps,
                                 warm=3 if nfe == 100 else 10)
                rows.append(dict(kind="sample", precision=prec, batch=b, nfe=nfe, ms_p50=s50, ms_p95=s95,
                                 ms_per_nfe=s50 / nfe, max_hz_policy_only=1000.0 / s50))
            print(prec, b, "fwd %.2f ms" % f50, flush=True)
meta = dict(ckpt="s42_400k", gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
            sdpa=all(bl.attn.use_sdpa for bl in m.particle_transformer.blocks),
            note="batch dim = robots/envs controlled in parallel; one plan per control step")
json.dump(dict(meta=meta, rows=rows), open(os.path.join(HERE, "results/latency.json"), "w"), indent=1)
import csv  # noqa: E402
with open(os.path.join(HERE, "results/latency.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r})); w.writeheader(); w.writerows(rows)
for r in rows:
    if r["batch"] == 1:
        print(r)
