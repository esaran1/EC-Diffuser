"""Perception latency: DLP encoding of the 2 camera views (128x128) per control
step, exactly as isaac_env_wrappers._extract_dlp_features, batch 1/16. CUDA events."""
import json, os, sys
import numpy as np
import torch
ROOT = "/home/jren313/EC-Diffuser-1"; HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [ROOT, os.path.join(ROOT, "diffuser")]; os.chdir(ROOT)
from dlp_utils import load_pretrained_rep_model, get_dlp_rep  # noqa: E402
m = load_pretrained_rep_model(dir_path="ecdiffuser-data/latent_rep_chkpts/dlp_push_6C", model_type="dlp").cuda().eval()
rows = []
for b in (1, 16):
    img = torch.randint(0, 255, (b, 3, 128, 128), device="cuda", dtype=torch.uint8)

    def enc():
        with torch.no_grad():
            for _ in range(2):        # two views, encoded separately as in the wrapper
                get_dlp_rep(m.encode_all(img.to(torch.float32) / 255, deterministic=True))
    for _ in range(10):
        enc()
    ts = []
    for _ in range(50):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); enc(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    rows.append(dict(batch=b, ms_p50_two_views=float(np.median(ts)), ms_p95=float(np.percentile(ts, 95))))
    print(rows[-1])
json.dump(rows, open(os.path.join(HERE, "results/latency_dlp.json"), "w"), indent=1)
