"""Diagnostic-only ODE solvers for the Flow field. Canonical code is untouched.

All solvers integrate dx/dt = v_theta(x, t) on t in [0,1] with the SAME
conditioning discipline as the canonical sampler (flow_matching.py:358-373):
the velocity is masked by the conditioning mask and conditioning is re-imposed
after every state update.

NFE accounting -- number of velocity-network forward calls per trajectory:
    euler     : n_steps          (1 eval / step)
    midpoint  : 2 * n_steps      (2 evals / step)
    heun      : 2 * n_steps      (2 evals / step)
    rk4       : 4 * n_steps      (4 evals / step)
"""
import torch

NFE_PER_STEP = {"euler": 1, "midpoint": 2, "heun": 2, "rk4": 4}


def _setup(model, cond, x0):
    x = x0.clone()
    model._apply_conditioning(x, cond)
    cmask = model._make_conditioning_mask(x, cond)
    return x, cmask


def _v(model, x, cond, t_scalar, cmask, counter):
    t = x.new_full((x.shape[0],), float(t_scalar))
    out = model.model(x, cond, t * model.time_scale)
    counter[0] += 1
    return out * cmask.to(x.dtype)


def integrate(model, cond, x0, method, n_steps, collect=None):
    """Returns (endpoint, nfe_used, diagnostics)."""
    x, cmask = _setup(model, cond, x0)
    dt = 1.0 / float(n_steps)
    counter = [0]
    diag = {"update_norm": [], "cos_consecutive": [], "rel_v_change": []}
    prev_v = None

    for k in range(n_steps):
        t = k * dt
        if method == "euler":
            v = _v(model, x, cond, t, cmask, counter)
            step = dt * v
            v_ref = v
        elif method == "midpoint":
            v1 = _v(model, x, cond, t, cmask, counter)
            xm = x + 0.5 * dt * v1
            model._apply_conditioning(xm, cond)
            v2 = _v(model, xm, cond, t + 0.5 * dt, cmask, counter)
            step = dt * v2
            v_ref = v1
            if collect is not None:
                diag["rel_v_change"].append(
                    ((v2 - v1).norm(dim=(1, 2)) / (v1.norm(dim=(1, 2)) + 1e-9)).mean().item())
        elif method == "heun":
            v1 = _v(model, x, cond, t, cmask, counter)
            xe = x + dt * v1
            model._apply_conditioning(xe, cond)
            v2 = _v(model, xe, cond, t + dt, cmask, counter)
            step = dt * 0.5 * (v1 + v2)
            v_ref = v1
            if collect is not None:
                diag["rel_v_change"].append(
                    ((v2 - v1).norm(dim=(1, 2)) / (v1.norm(dim=(1, 2)) + 1e-9)).mean().item())
        elif method == "rk4":
            k1 = _v(model, x, cond, t, cmask, counter)
            xa = x + 0.5 * dt * k1; model._apply_conditioning(xa, cond)
            k2 = _v(model, xa, cond, t + 0.5 * dt, cmask, counter)
            xb = x + 0.5 * dt * k2; model._apply_conditioning(xb, cond)
            k3 = _v(model, xb, cond, t + 0.5 * dt, cmask, counter)
            xc = x + dt * k3; model._apply_conditioning(xc, cond)
            k4 = _v(model, xc, cond, t + dt, cmask, counter)
            step = dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            v_ref = k1
        else:
            raise ValueError(f"unknown method {method}")

        if collect is not None:
            diag["update_norm"].append(step.norm(dim=(1, 2)).mean().item())
            if prev_v is not None:
                diag["cos_consecutive"].append(torch.nn.functional.cosine_similarity(
                    v_ref.flatten(1), prev_v.flatten(1), dim=1).mean().item())
            prev_v = v_ref.clone()

        x = x + step
        model._apply_conditioning(x, cond)

    return x, counter[0], diag
