"""E1 analysis — frozen BEFORE results exist. Consumes the three run artifacts
and emits the full predeclared table plus the automatic classification.

Classification thresholds are PREDECLARED in E1_ANALYSIS_PLAN.md:
  A NUMERICAL ONLY   : <=1 scenario discordant per permutation AND |dsucc| < 2.0 pp
  B BEHAVIOURAL      : >=2 discordant scenarios OR |dsucc| >= 2.0 pp
  C COMPARISON-RELEVANT: success range across mappings >= 5.0 pp
"""
import argparse, json, os
import numpy as np

D = "experiments/topconf/final_hardening/e1_results"
DISC_A, DSUCC_A, RANGE_C = 1, 0.02, 0.05
CONT = ["goal_success_frac", "cubes_placed", "avg_obj_dist", "max_obj_dist", "n_contacted"]


def load(name, d=D):
    return json.load(open(os.path.join(d, f"e1_{name}.json")))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", default=D); cli = ap.parse_args()
    runs = {}
    for n in ("identity", "perm1", "perm2"):
        runs[n] = load(n, cli.dir)
    out = {"thresholds": {"discordant_max_for_A": DISC_A, "dsucc_A_pp": 100*DSUCC_A,
                          "range_C_pp": 100*RANGE_C}}

    print("=== 8. SUCCESS PER MAPPING ===")
    S = {}
    for n, r in runs.items():
        S[n] = {e["episode"]: float(e["success"]) for e in r["episodes"]}
        print(f"  {n:9s} success = {r['success_rate']:.4f}  ({int(sum(S[n].values()))}/96)"
              f"  calls/plan={r.get('calls_per_plan','?')}  perm_sha={r.get('perm_sha256','?')[:12]}")
    rates = {n: runs[n]["success_rate"] for n in runs}
    rng_pp = 100*(max(rates.values())-min(rates.values()))
    out["success"] = rates; out["range_pp"] = rng_pp
    print(f"\n  success RANGE across mappings: {rng_pp:.2f} pp")

    print("\n=== 9. IDENTITY vs PERMUTATION (scenario-paired) ===")
    eps = sorted(S["identity"]); disc_tot = 0
    for n in ("perm1", "perm2"):
        a = np.array([S["identity"][e] for e in eps]); b = np.array([S[n][e] for e in eps])
        d = b - a
        gain = int(((a == 0) & (b == 1)).sum()); loss = int(((a == 1) & (b == 0)).sum())
        disc = gain + loss; disc_tot = max(disc_tot, disc)
        out[f"{n}_vs_identity"] = {"dsucc": float(d.mean()), "discordant": disc,
                                   "identity_fail_perm_ok": gain, "identity_ok_perm_fail": loss,
                                   "changed_scenarios": [int(e) for e, x in zip(eps, d) if x != 0]}
        print(f"  {n}: dsucc = {100*d.mean():+.2f} pp | discordant {disc}/96 "
              f"(id-fail->ok {gain}, id-ok->fail {loss})")
        if disc:
            print(f"      changed scenarios: {out[f'{n}_vs_identity']['changed_scenarios']}")

    print("\n=== 10. CONTINUOUS TASK METRICS (mean over 96 scenarios) ===")
    print(f"{'metric':22s} " + " ".join(f"{n:>10s}" for n in runs) + f" {'max|delta|':>11s}")
    cont = {}
    for k in CONT:
        vals = {}
        for n, r in runs.items():
            try: vals[n] = float(np.mean([e[k] for e in r["episodes"]]))
            except KeyError: vals[n] = float("nan")
        md = max(abs(vals[n]-vals["identity"]) for n in ("perm1", "perm2"))
        cont[k] = {**vals, "max_abs_delta": md}
        print(f"{k:22s} " + " ".join(f"{vals[n]:10.4f}" for n in runs) + f" {md:11.4f}")
    out["continuous"] = cont

    print("\n=== 11. AFFECTED-SCENARIO ANALYSIS ===")
    changed = set()
    for n in ("perm1", "perm2"):
        changed |= set(out[f"{n}_vs_identity"]["changed_scenarios"])
    p_any = np.array([np.mean([S[n][e] for n in runs]) for e in eps])
    out["n_scenarios_changed_any"] = len(changed)
    print(f"  scenarios whose outcome changed under ANY mapping: {len(changed)}/96")
    print(f"  scenarios identical across all three mappings   : {96-len(changed)}/96")
    if changed:
        print(f"  ids: {sorted(changed)}")

    print("\n=== 13. CLASSIFICATION (predeclared) ===")
    dmax = max(abs(out[f"{n}_vs_identity"]["dsucc"]) for n in ("perm1", "perm2"))
    if rng_pp >= 100*RANGE_C:
        cls = "C - LARGE / POLICY-COMPARISON-RELEVANT"
    elif disc_tot > DISC_A or dmax >= DSUCC_A:
        cls = "B - BEHAVIOURALLY DETECTABLE"
    else:
        cls = "A - NUMERICAL ONLY (E1 NULL for the paper hypothesis)"
    out["classification"] = cls
    print(f"  max discordant = {disc_tot}, max |dsucc| = {100*dmax:.2f} pp, range = {rng_pp:.2f} pp")
    print(f"  -> {cls}")
    print(f"\n  E2 GATE: {'JUSTIFIED' if cls.startswith('C') else 'NOT justified'}")
    json.dump(out, open(os.path.join(cli.dir, "e1_analysis.json"), "w"), indent=2)
    print(f"\nwrote {cli.dir}/e1_analysis.json")


if __name__ == "__main__":
    main()
