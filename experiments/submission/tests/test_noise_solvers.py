"""Cheap CPU tests: batch-invariant noise and solver correctness on analytic fields."""
import math, os, sys
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
ROOT = os.path.abspath(os.path.join(HERE, "../../.."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "diffuser")]
from noise import batch_noise, episode_seed  # noqa: E402
from diffuser.models.flow_matching import ConditionalFlowMatching  # noqa: E402


def test_noise_batch_invariance():
    shape = (5, 7)
    full = batch_noise(1, 123, list(range(96)), 3, shape, "cpu")
    for groups in ([list(range(96))], [list(range(i, i + 16)) for i in range(0, 96, 16)],
                   [[i] for i in range(96)], [list(range(95, -1, -1))]):
        for g in groups:
            part = batch_noise(1, 123, g, 3, shape, "cpu")
            assert torch.equal(part, full[g]), "noise depends on batch grouping"
    assert not torch.equal(batch_noise(2, 123, [0], 3, shape, "cpu"), full[[0]])   # realization
    assert not torch.equal(batch_noise(1, 124, [0], 3, shape, "cpu"), full[[0]])   # set
    assert not torch.equal(batch_noise(1, 123, [0], 4, shape, "cpu"), full[[0]])   # step
    assert not torch.equal(full[0], full[1])                                       # episode


class Field(torch.nn.Module):
    """v(x, t) for analytic ODEs; t arrives multiplied by time_scale."""
    def __init__(self, kind, scale):
        super().__init__(); self.kind, self.scale = kind, scale
        self.p = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64)); self.calls = 0
    def forward(self, x, cond, t):
        self.calls += 1
        tt = (t / self.scale).view(-1, 1, 1)
        return 0.7 * x if self.kind == "lin" else tt * x   # exact: e^0.7, e^0.5


def make(kind):
    m = Field(kind, 1000.0)
    w = ConditionalFlowMatching(m, horizon=3, observation_dim=2, action_dim=1, time_scale=1000.0)
    return w, m


def run(kind, solver, n):
    w, m = make(kind)
    torch.manual_seed(0)
    cond = {0: torch.zeros(4, 2, dtype=torch.float64)}
    x0 = torch.randn(4, 3, 3, dtype=torch.float64)
    out = w.conditional_sample(cond, n_steps=n, noise=x0, solver=solver, verbose=False).trajectories
    exact = math.exp(0.7 if kind == "lin" else 0.5)
    free = torch.ones_like(x0, dtype=torch.bool); free[:, 0, 1:] = False      # conditioned entries
    err = ((out - x0 * exact)[free]).abs().max().item()
    assert torch.all(out[:, 0, 1:] == 0)                                      # conditioning held
    return err, m.calls


def test_solver_orders_and_calls():
    for kind in ("lin", "time"):
        for solver, order, cps in (("euler", 1, 1), ("midpoint", 2, 2), ("rk4", 4, 4)):
            e1, c1 = run(kind, solver, 8)
            e2, c2 = run(kind, solver, 16)
            assert c1 == 8 * cps and c2 == 16 * cps
            rate = math.log2(e1 / e2)
            assert abs(rate - order) < 0.35, (kind, solver, rate)
    assert run("lin", "rk4", 1)[0] < run("lin", "midpoint", 2)[0] < run("lin", "euler", 4)[0]


def test_default_path_unchanged():
    w, _ = make("lin")
    cond = {0: torch.zeros(2, 2, dtype=torch.float64)}
    torch.manual_seed(5); a = w.conditional_sample(cond, n_steps=4, verbose=False).trajectories
    torch.manual_seed(5); x0 = torch.randn(2, 3, 3, dtype=torch.float64)
    b = w.conditional_sample(cond, n_steps=4, noise=x0, verbose=False).trajectories
    assert torch.equal(a, b)
    x = x0.clone(); x[:, 0, 1:] = 0
    for k in range(4):                                     # reference forward-Euler loop
        v = 0.7 * x; v[:, 0, 1:] = 0; x = x + 0.25 * v; x[:, 0, 1:] = 0
    assert torch.equal(a, x)



def test_colors_scenario_property():
    from noise import episode_colors
    a = [episode_colors(7, e, 6, 3) for e in range(96)]
    assert a == [episode_colors(7, e, 6, 3) for e in range(96)]
    assert all(len(set(c)) == 3 and all(0 <= x < 6 for x in c) for c in a)
    assert len({tuple(c) for c in a}) > 30                   # per-episode diversity
    assert a != [episode_colors(8, e, 6, 3) for e in range(96)]
    print("colors ok")


if __name__ == "__main__":
    test_noise_batch_invariance(); test_solver_orders_and_calls(); test_default_path_unchanged(); test_colors_scenario_property()
    print("ALL TESTS PASSED")

