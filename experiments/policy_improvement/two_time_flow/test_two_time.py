"""§19 unit tests for the two-time flow design. CPU only, no training."""
import sys
import numpy as np
import torch

sys.path.insert(0, "diffuser")
sys.path.insert(0, "experiments/policy_improvement/two_time_flow")
from diffuser.models.pint import AdaLNPINTDenoiser          # noqa: E402
from twotime_denoiser import TwoTimePINTDenoiser            # noqa: E402

AD, OD, H = 3, 480, 5
TD = AD + OD
KW = dict(horizon=H, transition_dim=TD, action_dim=AD, features_dim=10,
          hidden_dim=64, projection_dim=64, n_heads=4, n_layers=2, dropout=0.0,
          multiview=True, max_particles=None, positional_bias=False, device="cpu")


def corrupt(x1, z, t_s, t_a, action_dim=AD):
    """Two-time linear CFM corruption. t=0 noise, t=1 clean."""
    ts = t_s.view(-1, 1, 1); ta = t_a.view(-1, 1, 1)
    xt = torch.empty_like(x1)
    xt[..., :action_dim] = (1 - ta) * z[..., :action_dim] + ta * x1[..., :action_dim]
    xt[..., action_dim:] = (1 - ts) * z[..., action_dim:] + ts * x1[..., action_dim:]
    return xt


def targets(x1, z, action_dim=AD):
    """v* = x1 - z, computed blockwise (identical formula, split for clarity)."""
    v = x1 - z
    return v[..., :action_dim], v[..., action_dim:]


def apply_cond(x, cond, action_dim=AD):
    for t, val in cond.items():
        x[:, t, action_dim:] = val
    return x


results = []
def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")


B = 4
torch.manual_seed(0)
x1 = torch.randn(B, H, TD)
z = torch.randn(B, H, TD)
cond = {0: torch.randn(B, OD), H - 1: torch.randn(B, OD)}

print("\n=== 1. DIAGONAL EQUIVALENCE (corruption) ===")
t = torch.rand(B)
xt_two = corrupt(x1, z, t, t)
xt_one = (1 - t.view(-1, 1, 1)) * z + t.view(-1, 1, 1) * x1
check("diagonal corruption == ordinary CFM", torch.allclose(xt_two, xt_one, atol=0),
      f"max|d|={float((xt_two-xt_one).abs().max()):.3e}")

print("\n=== 2. EXTREME INVERSE REGIME (t_s=1, t_a=0) ===")
one, zero = torch.ones(B), torch.zeros(B)
xt = corrupt(x1, z, one, zero)
check("state block == clean data", torch.allclose(xt[..., AD:], x1[..., AD:], atol=0))
check("action block == pure noise", torch.allclose(xt[..., :AD], z[..., :AD], atol=0))

print("\n=== 3. EXTREME WORLD REGIME (t_s=0, t_a=1) ===")
xt = corrupt(x1, z, zero, one)
check("action block == clean data", torch.allclose(xt[..., :AD], x1[..., :AD], atol=0))
check("state block == pure noise", torch.allclose(xt[..., AD:], z[..., AD:], atol=0))

print("\n=== 4. CONDITIONING MASK INVARIANT TO (t_s,t_a) ===")
ok = True
for ts_, ta_ in [(0., 0.), (1., 0.), (0., 1.), (.5, .3), (1., 1.)]:
    xt = corrupt(x1, z, torch.full((B,), ts_), torch.full((B,), ta_))
    xt = apply_cond(xt.clone(), cond)
    for tt, val in cond.items():
        ok &= torch.allclose(xt[:, tt, AD:], val, atol=0)
check("conditioned obs exactly imposed for all (t_s,t_a)", ok)

print("\n=== 5. VELOCITY TARGETS ===")
va, vs = targets(x1, z)
check("action target == x1-z on action block",
      torch.allclose(va, (x1 - z)[..., :AD], atol=0))
check("state target == x1-z on state block",
      torch.allclose(vs, (x1 - z)[..., AD:], atol=0))
check("targets independent of t_s,t_a (linear path)", True, "(by construction)")

print("\n=== 6. SHAPES ===")
torch.manual_seed(1); net = TwoTimePINTDenoiser(**KW).eval()
with torch.no_grad():
    out = net(x1, None, torch.rand(B) * 1000, time_action=torch.rand(B) * 1000)
check("output shape == input shape", tuple(out.shape) == (B, H, TD), str(tuple(out.shape)))

print("\n=== 7. TIME EMBEDDINGS REACH THE RIGHT TOKENS ===")
t_fixed = torch.full((B,), 300.0)
with torch.no_grad():
    o1 = net(x1, None, t_fixed, time_action=torch.full((B,), 100.0))
    o2 = net(x1, None, t_fixed, time_action=torch.full((B,), 900.0))
check("changing t_action changes output", not torch.equal(o1, o2),
      f"max|d|={float((o1-o2).abs().max()):.3e}")
with torch.no_grad():
    o3 = net(x1, None, torch.full((B,), 100.0), time_action=t_fixed)
    o4 = net(x1, None, torch.full((B,), 900.0), time_action=t_fixed)
check("changing t_state changes output", not torch.equal(o3, o4),
      f"max|d|={float((o3-o4).abs().max()):.3e}")

print("\n=== 8. DIAGONAL CHECKPOINT COMPATIBILITY ===")
torch.manual_seed(2); base = AdaLNPINTDenoiser(**KW).eval()
torch.manual_seed(2); two = TwoTimePINTDenoiser(**KW).eval()
missing = two.load_state_dict(base.state_dict(), strict=True)
tt = torch.rand(B) * 1000
with torch.no_grad():
    a = base(x1, None, tt); b = two(x1, None, tt, time_action=tt)
check("zero added parameters",
      sum(p.numel() for p in base.parameters()) == sum(p.numel() for p in two.parameters()))
check("diagonal output BIT-EXACT vs parent", torch.equal(a, b),
      f"max|d|={float((a-b).abs().max()):.3e}")

print("\n=== 9. GRADIENTS IN ALL THREE REGIMES ===")
for name, ts_, ta_ in [("diagonal", .5, .5), ("inverse", .9, .1), ("world", .1, .9)]:
    net.zero_grad(set_to_none=True)
    xt = corrupt(x1, z, torch.full((B,), ts_), torch.full((B,), ta_))
    out = net(xt, None, torch.full((B,), ts_ * 1000), time_action=torch.full((B,), ta_ * 1000))
    va_, vs_ = targets(x1, z)
    loss = (out[..., :AD] - va_).abs().mean() + (out[..., AD:] - vs_).abs().mean()
    loss.backward()
    def gnorm(mod):
        tot, fin = 0.0, True
        for prm in mod.parameters():
            if prm.grad is not None:
                fin &= bool(torch.isfinite(prm.grad).all())
                tot += float(prm.grad.abs().sum())
        return tot, fin
    na, fa_ok = gnorm(net.action_projection)
    ns, fs_ok = gnorm(net.particle_projection)
    check(f"{name}: action+state grads finite & nonzero",
          bool(fa_ok and fs_ok and na > 0 and ns > 0),
          f"|g_action|={na:.3e} |g_state|={ns:.3e}")

print("\n=== 10. SAMPLER ENDPOINTS / MONOTONICITY ===")
def diagonal(s):  return s, s
def state_lead(s):
    return np.minimum(1.0, 2*s), np.maximum(0.0, 2*s - 1)
def action_lead(s):
    return np.maximum(0.0, 2*s - 1), np.minimum(1.0, 2*s)
def power(s, gs, ga): return s**gs, s**ga

s = np.linspace(0, 1, 201)
sched = {"S0 diagonal": diagonal(s),
         "S1 state-lead (power g_s=0.5)": power(s, 0.5, 1.0),
         "S2 state-lead (piecewise)": state_lead(s),
         "S3 action-lead (power g_a=0.5)": power(s, 1.0, 0.5),
         "S4 action-lead (piecewise)": action_lead(s)}
for nm, (ts_, ta_) in sched.items():
    ok = (abs(ts_[0]) < 1e-12 and abs(ta_[0]) < 1e-12 and
          abs(ts_[-1]-1) < 1e-12 and abs(ta_[-1]-1) < 1e-12 and
          np.all(np.diff(ts_) >= -1e-12) and np.all(np.diff(ta_) >= -1e-12) and
          ts_.min() >= -1e-12 and ts_.max() <= 1+1e-12 and
          ta_.min() >= -1e-12 and ta_.max() <= 1+1e-12)
    lead = "state-leading" if (ts_ >= ta_ - 1e-12).all() else (
           "action-leading" if (ta_ >= ts_ - 1e-12).all() else "mixed")
    check(f"{nm}: (0,0)->(1,1), monotone, in [0,1]", bool(ok), lead)

n_fail = sum(1 for _, ok, _ in results if not ok)
print(f"\n{'='*60}\n{len(results)-n_fail}/{len(results)} PASSED"
      f"{'' if n_fail==0 else f'  -- {n_fail} FAILED'}\n{'='*60}")
sys.exit(1 if n_fail else 0)
