"""Real-path pairing verification (replaces the bypassing verify_pairing.py).

The old test constructed the diffusion wrapper directly with explicit kwargs,
bypassing `diffusion_wrapper_kwargs` -- the function that was silently dropping
the loss knobs. It could never have caught the defect it was meant to catch.

This version consumes the per-arm outputs of `realpath_objective_test.py`, which
drives the production construction path, and checks BOTH halves of pairing:

  COMMON INVARIANTS -- everything that must be identical across arms
  INTENDED DIFFERENCES -- exactly the registered loss changes, and nothing else

Note the design point from the directive: a correct pairing test must NOT demand
that the loss itself be identical. Identical losses are what the broken run
produced. What must be identical is the model, the data, and the prediction;
what must differ is the objective.
"""
import itertools
import json
import os
import sys

R = os.path.dirname(os.path.abspath(__file__))
ARMS = ("A", "B", "C", "D")

INTENT = {
    "A": {"mask": False, "lam": 1.0},
    "B": {"mask": True, "lam": 1.0},
    "C": {"mask": True, "lam": 2.0},
    "D": {"mask": True, "lam": 5.0},
}

data = {a: json.load(open(os.path.join(R, "objtest_{}.json".format(a)))) for a in ARMS}
failures = []
checks = []


def check(name, ok, detail=""):
    checks.append({"check": name, "ok": bool(ok), "detail": detail})
    if not ok:
        failures.append("{}: {}".format(name, detail))


# ---------------- COMMON INVARIANTS ----------------------------------------
preds = {data[a]["prediction_sha16"] for a in ARMS}
check("same forward predictions before update", len(preds) == 1, str(sorted(preds)))

params = {data[a]["model_params_sha16_before"] for a in ARMS}
check("same initialized weights (canonical)", len(params) == 1, str(sorted(params)))

for key in ("horizon", "action_dim"):
    vals = {data[a]["objective"][key] for a in ARMS}
    check("same {}".format(key), len(vals) == 1, str(sorted(vals)))

obs_sums = {round(data[a]["objective"]["observation_block_sum"], 9) for a in ARMS}
check("observation weighting identical across arms", len(obs_sums) == 1, str(sorted(obs_sums)))

unw_obs = {round(data[a]["losses"]["unweighted_observation_loss"], 12) for a in ARMS}
check("same unweighted observation error", len(unw_obs) == 1, str(sorted(unw_obs)))

unw_act_bcd = {round(data[a]["losses"]["unweighted_action_loss"], 12) for a in "BCD"}
check("B/C/D same unweighted action error (weighting differs, errors do not)",
      len(unw_act_bcd) == 1, str(sorted(unw_act_bcd)))

# ---------------- INTENDED DIFFERENCES -------------------------------------
for a in ARMS:
    o = data[a]["objective"]
    want = INTENT[a]
    check("arm {} mask_terminal_action == {}".format(a, want["mask"]),
          bool(o["mask_terminal_action"]) == want["mask"],
          "actual {}".format(o["mask_terminal_action"]))
    check("arm {} lambda_action == {}".format(a, want["lam"]),
          abs(float(o["lambda_action"]) - want["lam"]) < 1e-12,
          "actual {}".format(o["lambda_action"]))
    check("arm {} terminal mask in force in real loss path".format(a),
          bool(data[a]["terminal_mask_probe"]["terminal_action_masked"]) == want["mask"],
          str(data[a]["terminal_mask_probe"]))

# lambda must scale the action block exactly, relative to B (mask on, lambda 1)
base_action = data["B"]["objective"]["action_block_sum"]
for a in ("C", "D"):
    want = INTENT[a]["lam"] * base_action
    got = data[a]["objective"]["action_block_sum"]
    check("arm {} action block == {}x B".format(a, INTENT[a]["lam"]),
          abs(got - want) < 1e-9, "got {} want {}".format(got, want))

# weighted action contribution must scale exactly with lambda on a frozen batch
base_contrib = data["B"]["losses"]["weighted_action_contribution"]
for a in ("C", "D"):
    want = INTENT[a]["lam"] * base_contrib
    got = data[a]["losses"]["weighted_action_contribution"]
    check("arm {} weighted action contribution == {}x B".format(a, INTENT[a]["lam"]),
          abs(got - want) <= 1e-9 * max(1.0, abs(want)),
          "got {} want {}".format(got, want))

# A vs B differ ONLY by terminal-action supervision
check("A has active terminal action contribution",
      data["A"]["losses"]["terminal_action_contribution"] > 0,
      str(data["A"]["losses"]["terminal_action_contribution"]))
check("B/C/D terminal action contribution is exactly zero",
      all(data[a]["losses"]["terminal_action_contribution"] == 0.0 for a in "BCD"))
check("masking reduces the loss denominator (supervision removed)",
      data["B"]["losses"]["conditioning_mask_denominator"]
      < data["A"]["losses"]["conditioning_mask_denominator"],
      "A {} -> B {}".format(data["A"]["losses"]["conditioning_mask_denominator"],
                            data["B"]["losses"]["conditioning_mask_denominator"]))

# ---------------- ONE-STEP DIVERGENCE SENTINEL ------------------------------
after = {a: data[a]["one_step"]["model_params_sha16_after_one_step"] for a in ARMS}
for x, y in itertools.combinations(ARMS, 2):
    check("one-step divergence {} vs {}".format(x, y), after[x] != after[y],
          "both {}".format(after[x]) if after[x] == after[y] else "distinct")

for a in ARMS:
    g = data[a]["one_step"]["grad_norm"]
    check("arm {} gradient finite and nonzero".format(a),
          g > 0 and g == g and g != float("inf"), str(g))

verdict = "PAIRED" if not failures else "FAILED"
report = {
    "verdict": verdict,
    "n_checks": len(checks),
    "n_failed": len(failures),
    "failures": failures,
    "checks": checks,
    "arms": {a: {"objective": data[a]["objective"],
                 "losses": data[a]["losses"],
                 "one_step": data[a]["one_step"]} for a in ARMS},
}
out = os.path.join(R, "PAIRING_REALPATH.json")
json.dump(report, open(out, "w"), indent=2, sort_keys=True)

for c in checks:
    print("  [{}] {}".format("PASS" if c["ok"] else "FAIL", c["check"]))
print("\n{} checks, {} failed -> {}".format(len(checks), len(failures), verdict))
print("written:", out)
sys.exit(0 if verdict == "PAIRED" else 1)
