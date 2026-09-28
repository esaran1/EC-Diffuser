"""Attention-concentration half of the register diagnostic (CPU, canonical EMA).

Per layer: mass each KEY token receives (mean over heads, queries, batch),
relative to its uniform share. A register-style pathology is a small set of
tokens receiving many times their share across layers (attention sinks),
especially low-information ones. Token order in the attention matrix:
index = n*T + t, n=0 action, n=1..48 DLP particles (2 views x 24), T=5.
"""
import sys, numpy as np, torch
sys.path.insert(0, "diffuser")
from diffuser.utils.serialization import load_config
from diffuser.models.pint import AdaLNPINTDenoiser
from diffuser.models.flow_matching import ConditionalFlowMatching
R = "3C_dlp_adalnpint_randcolor_H5_T4_seed42"
torch.manual_seed(0); torch.set_num_threads(16)
mc = load_config("data/panda_push/uwm_ctl", R, "model_config.pkl")
dc = load_config("data/panda_push/uwm_ctl", R, "dataset_config.pkl")
dc._dict["dataset_path"] = "ecdiffuser-data/push_cubes/3C_randcolor/panda_push_replay_buffer_dlp.pkl"
ds = dc(seed=42)
m = AdaLNPINTDenoiser(**mc._dict)
w = ConditionalFlowMatching(m, horizon=5, observation_dim=480, action_dim=3, n_timesteps=4,
                            loss_type="l1", action_weight=10)
ck = torch.load("data/panda_push/flow/%s/state_400000.pt" % R, map_location="cpu")
w.load_state_dict(ck["ema"]); w.eval()
store = {}
for i, blk in enumerate(m.particle_transformer.blocks):
    orig = blk.attn.forward
    def f(x, c=None, return_attention=False, _o=orig, _i=i):
        y, a = _o(x, c, return_attention=True); store[_i] = a.detach(); return y
    blk.attn.forward = f
rng = np.random.RandomState(20260928); idx = rng.choice(len(ds), 16, replace=False)
x1 = torch.stack([torch.as_tensor(ds[int(i)].trajectories) for i in idx]).float()
cond = {k: torch.stack([torch.as_tensor(ds[int(i)].conditions[k]) for i in idx]).float()
        for k in ds[int(idx[0])].conditions}
N, T = 49, 5
print("layer  t    action_share/uniform  max_particle/uniform  top3 key tokens (n) share/uniform   entropy/max")
res = {}
for tv in (0.1, 0.5, 0.9):
    x0 = torch.randn_like(x1); xt = (1 - tv) * x0 + tv * x1; w._apply_conditioning(xt, cond)
    with torch.no_grad():
        m(xt, cond, torch.full((16,), tv * 1000.0))
    for L in sorted(store):
        a = store[L]                                   # B, H, NT, NT
        recv = a.mean(dim=(0, 1, 2))                   # mass received per key, sums to 1
        per_tok = recv.view(N, T).sum(-1).numpy()      # per token type (summed over horizon)
        share = per_tok * N                            # 1.0 == uniform
        ent = (-(a.clamp_min(1e-12) * a.clamp_min(1e-12).log()).sum(-1)).mean().item() / np.log(N * T)
        top = np.argsort(-share)[:3]
        res.setdefault(tv, []).append((L, share[0], share[1:].max(), np.median(share[1:]), ent))
        if L in (0, 3, 6, 9, 11):
            print("%4d  %.1f   %8.2f            %8.2f            %s   %.3f" % (
                L, tv, share[0], share[1:].max(),
                " ".join("%d:%.2f" % (n, share[n]) for n in top), ent))
print("\nsummary across all layers and times:")
allr = [r for v in res.values() for r in v]
print("  action token share/uniform: min %.2f  median %.2f  max %.2f" % (
    min(r[1] for r in allr), np.median([r[1] for r in allr]), max(r[1] for r in allr)))
print("  max particle share/uniform: min %.2f  median %.2f  max %.2f" % (
    min(r[2] for r in allr), np.median([r[2] for r in allr]), max(r[2] for r in allr)))
print("  normalized attention entropy: min %.3f  median %.3f" % (
    min(r[4] for r in allr), np.median([r[4] for r in allr])))
