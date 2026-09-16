"""Path-correct two-time sampler (directive section 25).

Each modality's velocity is scaled by ITS OWN time's increment (chain rule).
The baseline scalar-time Euler update is the special case f_s = f_a = identity.

Derivation and test rationale: TWO_TIME_SAMPLER.md. Zero GPU.
"""
import torch


# ----------------------------------------------------------------- schedules
def p0_diagonal(u):
    return u, u


def p1_state_leads(u):
    return 1.0 - (1.0 - u) ** 2, u ** 2


def p2_action_leads(u):
    return u ** 2, 1.0 - (1.0 - u) ** 2


def p3_state_first(u):
    return min(2.0 * u, 1.0), max(2.0 * u - 1.0, 0.0)


def p4_action_first(u):
    return max(2.0 * u - 1.0, 0.0), min(2.0 * u, 1.0)


PATHS = {
    "P0_diagonal": p0_diagonal,
    "P1_state_leads": p1_state_leads,
    "P2_action_leads": p2_action_leads,
    "P3_state_first": p3_state_first,
    "P4_action_first": p4_action_first,
}


def schedule_points(path, steps):
    """Exact (t_s, t_a) at each of steps+1 knots of u in [0,1]."""
    f = PATHS[path] if isinstance(path, str) else path
    return [f(k / float(steps)) for k in range(steps + 1)]


def validate_schedule(path, steps=64, tol=1e-9):
    """Endpoints, monotonicity and range. Returns (ok, detail)."""
    pts = schedule_points(path, steps)
    ts = [p[0] for p in pts]
    ta = [p[1] for p in pts]
    if abs(ts[0]) > tol or abs(ta[0]) > tol:
        return False, "start not (0,0): ({:.3g},{:.3g})".format(ts[0], ta[0])
    if abs(ts[-1] - 1.0) > tol or abs(ta[-1] - 1.0) > tol:
        return False, "end not (1,1): ({:.3g},{:.3g})".format(ts[-1], ta[-1])
    for seq, nm in ((ts, "t_s"), (ta, "t_a")):
        if any(b < a - tol for a, b in zip(seq, seq[1:])):
            return False, "{} not monotone".format(nm)
        if min(seq) < -tol or max(seq) > 1.0 + tol:
            return False, "{} outside [0,1]".format(nm)
    return True, "ok"


# ------------------------------------------------------------------ sampler
@torch.no_grad()
def two_time_sample(velocity_fn, x_init, action_dim, path="P0_diagonal",
                    steps=4, apply_cond=None, cond_mask=None):
    """Integrate a two-time flow along `path` with forward Euler.

    velocity_fn(x, t_s, t_a) -> velocity, same shape as x. t_s/t_a are python
    floats; the caller is responsible for any time_scale and for broadcasting.

    The state block advances by the t_s increment and the action block by the
    t_a increment -- never by a shared du. This is the chain rule of section 2
    in TWO_TIME_SAMPLER.md and is the whole point of this function.

    No clean target is ever referenced here: x moves only under velocity_fn.
    """
    pts = schedule_points(path, steps)
    x = x_init.clone()
    if apply_cond is not None:
        x = apply_cond(x)
    for k in range(steps):
        ts0, ta0 = pts[k]
        ts1, ta1 = pts[k + 1]
        dts, dta = ts1 - ts0, ta1 - ta0
        v = velocity_fn(x, ts0, ta0)
        if cond_mask is not None:
            v = v * cond_mask.to(v.dtype)
        # chain rule: each block scaled by its OWN time increment
        x[..., :action_dim] = x[..., :action_dim] + dta * v[..., :action_dim]
        x[..., action_dim:] = x[..., action_dim:] + dts * v[..., action_dim:]
        if apply_cond is not None:
            x = apply_cond(x)
    return x


def model_calls(steps):
    """Model evaluations used by one rollout (section 27 fair accounting)."""
    return steps
