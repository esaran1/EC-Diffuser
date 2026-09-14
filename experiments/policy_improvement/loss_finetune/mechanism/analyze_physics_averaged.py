"""Physics-averaged robustness analysis (directive sections 16-17).

For each policy P and fine-tuning seed s, average the ORIGINAL accepted control
evaluation with the NEW independent simulator realization:

    Jbar(P, s) = mean(original macro, repeat macro)          # R = 2

then form the paired treatment effect WITHIN each seed:

    Delta_s = Jbar(B, s) - Jbar(A, s)

The pair is the unit. Cross-seed comparisons (B_43 vs A_44) are never causal
contrasts. Per-E-set effects use both repeats.

With R=2 and n_seed=4 this is descriptive. No hypothesis test is performed and no
significance threshold is defined; the identical-policy evaluator spread
(1.87 pp macro range, 0.80 pp SD) is reported only as context.
"""
import json
import os
import statistics as st

P2 = "experiments/policy_improvement/phase2"
SETS = ["E0", "E1s", "E2s", "E3s", "E4s"]
SEEDS = [42, 43, 44, 45]
ORIG_TAG = {42: ("lossft_A", "lossft_B"), 43: ("lossft_r1_A", "lossft_r1_B"),
            44: ("lossft_r2_A", "lossft_r2_B"), 45: ("lossft_r3_A", "lossft_r3_B")}


def load(tag):
    p = os.path.join(P2, "{}_diagnostics.json".format(tag))
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    rows = {}
    missing = []
    for s in SEEDS:
        ta, tb = ORIG_TAG[s]
        entry = {}
        for arm, tag in (("A", ta), ("B", tb)):
            orig = load(tag)
            rep = load("lossft_rep2_{}{}".format(arm, s))
            if orig is None or rep is None:
                missing.append("seed{} arm{}".format(s, arm))
                continue
            o_sets = {k: orig["control"][k]["success"] for k in SETS}
            r_sets = {k: rep["control"][k]["success"] for k in SETS}
            entry[arm] = {
                "orig_macro": orig["control_mean"],
                "repeat_macro": rep["control_mean"],
                "Jbar": 0.5 * (orig["control_mean"] + rep["control_mean"]),
                "per_set_avg": {k: 0.5 * (o_sets[k] + r_sets[k]) for k in SETS},
                "orig_sets": o_sets, "repeat_sets": r_sets,
                "orig_valid": orig.get("n_valid_sets"),
                "repeat_valid": rep.get("n_valid_sets"),
            }
        if len(entry) == 2:
            rows[s] = entry

    if missing:
        print("INCOMPLETE -- missing:", ", ".join(missing))
    if not rows:
        print("no complete physics-averaged pairs yet")
        return

    print("=" * 86)
    print("PHYSICS-AVERAGED PAIRED EFFECT  (R=2 simulator realizations per policy)")
    print("=" * 86)
    print("{:<7}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>12}".format(
        "seed", "A orig", "A rep", "A Jbar", "B orig", "B rep", "B Jbar", "Delta (pp)"))
    deltas = {}
    for s in sorted(rows):
        A, B = rows[s]["A"], rows[s]["B"]
        d = B["Jbar"] - A["Jbar"]
        deltas[s] = d
        print("{:<7}{:>10.4f}{:>10.4f}{:>10.4f}{:>10.4f}{:>10.4f}{:>10.4f}{:>+12.2f}".format(
            s, A["orig_macro"], A["repeat_macro"], A["Jbar"],
            B["orig_macro"], B["repeat_macro"], B["Jbar"], 100 * d))

    dl = [deltas[s] for s in sorted(deltas)]
    n = len(dl)
    print("\n  paired deltas (pp): {}".format(["{:+.2f}".format(100 * v) for v in dl]))
    print("  mean   {:+.2f} pp".format(100 * st.mean(dl)))
    print("  median {:+.2f} pp".format(100 * st.median(dl)))
    if n > 1:
        sd = st.stdev(dl)
        print("  sample SD {:.2f} pp    min {:+.2f}   max {:+.2f}".format(
            100 * sd, 100 * min(dl), 100 * max(dl)))
        tcrit = {2: 12.706, 3: 4.303, 4: 3.182}.get(n)
        if tcrit and sd > 0:
            half = tcrit * sd / (n ** 0.5)
            print("  DESCRIPTIVE 95% interval (n={}): {:+.2f} to {:+.2f} pp"
                  .format(n, 100 * (st.mean(dl) - half), 100 * (st.mean(dl) + half)))
            print("  (descriptive only; R=2 and n_seed=4 cannot support a test)")
    pos = sum(1 for v in dl if v > 0)
    zer = sum(1 for v in dl if v == 0)
    neg = sum(1 for v in dl if v < 0)
    print("  signs: {} positive, {} zero, {} negative".format(pos, zer, neg))

    print("\n  leave-one-seed-out mean paired delta:")
    for s in sorted(deltas):
        rest = [deltas[k] for k in deltas if k != s]
        print("    drop seed{}: {:+.2f} pp".format(s, 100 * st.mean(rest)))

    print("\n  per-E-set mean B-A (both repeats averaged):")
    for k in SETS:
        vals = [rows[s]["B"]["per_set_avg"][k] - rows[s]["A"]["per_set_avg"][k]
                for s in sorted(rows)]
        up = sum(1 for v in vals if v > 0)
        print("    {:<5} mean {:+.2f} pp   range {:+.2f} .. {:+.2f}   seeds B>A {}/{}"
              .format(k, 100 * st.mean(vals), 100 * min(vals), 100 * max(vals),
                      up, len(vals)))

    # --- registered descriptive classification ------------------------------
    mean_pos = st.mean(dl) > 0
    nonneg = sum(1 for v in dl if v >= 0)
    loo_all_pos = all(st.mean([deltas[k] for k in deltas if k != s]) > 0
                      for s in deltas)
    if mean_pos and nonneg >= 3 and loo_all_pos:
        verdict = "CONTROL-EFFECT-ROBUST-PRELIMINARY"
    elif st.mean(dl) <= 0 or pos < max(1, n // 2):
        verdict = "CONTROL-EFFECT-NOT-ROBUST"
    else:
        verdict = "CONTROL-EFFECT-UNRESOLVED"
    print("\n  mean>0: {}   seeds nonneg: {}/{}   leave-one-out all positive: {}"
          .format(mean_pos, nonneg, n, loo_all_pos))
    print("  DESCRIPTIVE CLASSIFICATION: {}".format(verdict))

    out = {"per_seed": rows, "deltas": deltas, "mean": st.mean(dl),
           "median": st.median(dl), "sd": st.stdev(dl) if n > 1 else None,
           "n_positive": pos, "n_zero": zer, "n_negative": neg,
           "verdict": verdict}
    path = ("experiments/policy_improvement/loss_finetune/mechanism/"
            "PHYSICS_AVERAGED.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    print("\nwritten", path)


if __name__ == "__main__":
    main()
