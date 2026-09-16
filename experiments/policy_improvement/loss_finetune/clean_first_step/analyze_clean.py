"""Clean-first-step analysis: primary statistic, and legacy-vs-clean comparison.

Primary Stage-E statistic, per fine-tuning seed:

    Delta_clean(s) = macro(B_clean_s) - macro(A_clean_s)

Then the core causal comparison against the legacy protocol, per seed:

    Delta_legacy_orig(s)   original single-realization legacy result
    Delta_legacy_R2(s)     physics-averaged legacy result (R=2)
    Delta_clean(s)         corrected protocol

    Delta_clean - Delta_legacy_R2

Question: does removing the contaminated first update preserve, reduce,
eliminate, or reverse the mask effect?

Applies the PREDECLARED trainer-confound classification. No fixed pp cutoff is
invented anywhere; the identical-policy evaluator spread (1.87 pp macro range,
0.80 pp SD) is context only.
"""
import json
import os
import statistics as st

P2 = "experiments/policy_improvement/phase2"
SETS = ["E0", "E1s", "E2s", "E3s", "E4s"]
SEEDS = [42, 43, 44, 45]
LEGACY = {42: ("lossft_A", "lossft_B"), 43: ("lossft_r1_A", "lossft_r1_B"),
          44: ("lossft_r2_A", "lossft_r2_B"), 45: ("lossft_r3_A", "lossft_r3_B")}


def load(tag):
    p = os.path.join(P2, "{}_diagnostics.json".format(tag))
    return json.load(open(p)) if os.path.exists(p) else None


def macro(tag):
    d = load(tag)
    return d["control_mean"] if d else None


def main():
    rows, missing = {}, []
    for s in SEEDS:
        ta, tb = LEGACY[s]
        entry = {}
        entry["legacy_A_orig"] = macro(ta)
        entry["legacy_B_orig"] = macro(tb)
        entry["legacy_A_rep"] = macro("lossft_rep2_A{}".format(s))
        entry["legacy_B_rep"] = macro("lossft_rep2_B{}".format(s))
        entry["clean_A"] = macro("lossft_clean_A{}".format(s))
        entry["clean_B"] = macro("lossft_clean_B{}".format(s))
        for k, v in entry.items():
            if v is None:
                missing.append("seed{} {}".format(s, k))
        rows[s] = entry

    have_clean = all(rows[s]["clean_A"] is not None and rows[s]["clean_B"] is not None
                     for s in SEEDS)
    have_r2 = all(rows[s]["legacy_A_rep"] is not None and rows[s]["legacy_B_rep"] is not None
                  for s in SEEDS)

    if missing:
        print("INCOMPLETE -- missing: {}".format(", ".join(missing[:12])))
        if not have_clean:
            print("clean four-seed result not available yet")

    # ---- legacy physics-averaged -----------------------------------------
    if have_r2:
        print("=" * 84)
        print("LEGACY PHYSICS-AVERAGED (R=2)")
        print("=" * 84)
        print("%-6s%10s%10s%10s%10s%12s" % ("seed", "A orig", "A rep", "B orig",
                                            "B rep", "Delta (pp)"))
        d_r2 = {}
        for s in SEEDS:
            e = rows[s]
            ja = 0.5 * (e["legacy_A_orig"] + e["legacy_A_rep"])
            jb = 0.5 * (e["legacy_B_orig"] + e["legacy_B_rep"])
            d_r2[s] = jb - ja
            e["Jbar_A"], e["Jbar_B"], e["delta_legacy_R2"] = ja, jb, jb - ja
            print("%-6d%10.4f%10.4f%10.4f%10.4f%+12.2f" % (
                s, e["legacy_A_orig"], e["legacy_A_rep"],
                e["legacy_B_orig"], e["legacy_B_rep"], 100 * d_r2[s]))
        v = list(d_r2.values())
        print("\n  mean %+.2f  median %+.2f  SD %.2f  range %+.2f..%+.2f  "
              "signs +%d/0%d/-%d" % (
                  100 * st.mean(v), 100 * st.median(v), 100 * st.stdev(v),
                  100 * min(v), 100 * max(v),
                  sum(1 for x in v if x > 0), sum(1 for x in v if x == 0),
                  sum(1 for x in v if x < 0)))

    # ---- clean ------------------------------------------------------------
    if have_clean:
        print("\n" + "=" * 84)
        print("CLEAN-FIRST-STEP FOUR-SEED RESULT (primary Stage-E statistic)")
        print("=" * 84)
        print("%-6s%12s%12s%14s" % ("seed", "A_clean", "B_clean", "Delta (pp)"))
        d_cl = {}
        for s in SEEDS:
            e = rows[s]
            d_cl[s] = e["clean_B"] - e["clean_A"]
            e["delta_clean"] = d_cl[s]
            print("%-6d%12.4f%12.4f%+14.2f" % (
                s, e["clean_A"], e["clean_B"], 100 * d_cl[s]))
        v = list(d_cl.values())
        n = len(v)
        print("\n  mean %+.2f  median %+.2f  SD %.2f  range %+.2f..%+.2f  "
              "signs +%d/0%d/-%d" % (
                  100 * st.mean(v), 100 * st.median(v), 100 * st.stdev(v),
                  100 * min(v), 100 * max(v),
                  sum(1 for x in v if x > 0), sum(1 for x in v if x == 0),
                  sum(1 for x in v if x < 0)))
        print("  leave-one-seed-out mean:")
        for s in SEEDS:
            rest = [d_cl[k] for k in d_cl if k != s]
            print("    drop seed%d: %+.2f pp" % (s, 100 * st.mean(rest)))

        # ---- legacy vs clean ---------------------------------------------
        if have_r2:
            print("\n" + "=" * 84)
            print("LEGACY (R=2) vs CLEAN -- the core causal comparison")
            print("=" * 84)
            print("%-6s%16s%16s%18s" % ("seed", "Delta_legacy_R2",
                                        "Delta_clean", "clean - legacy"))
            diffs = []
            for s in SEEDS:
                e = rows[s]
                dd = e["delta_clean"] - e["delta_legacy_R2"]
                diffs.append(dd)
                print("%-6d%+16.2f%+16.2f%+18.2f" % (
                    s, 100 * e["delta_legacy_R2"], 100 * e["delta_clean"], 100 * dd))
            print("\n  mean legacy_R2 %+.2f pp   mean clean %+.2f pp   "
                  "mean shift %+.2f pp" % (
                      100 * st.mean([rows[s]["delta_legacy_R2"] for s in SEEDS]),
                      100 * st.mean([rows[s]["delta_clean"] for s in SEEDS]),
                      100 * st.mean(diffs)))

            # ---- PREDECLARED classification ------------------------------
            mean_clean = st.mean([rows[s]["delta_clean"] for s in SEEDS])
            mean_leg = st.mean([rows[s]["delta_legacy_R2"] for s in SEEDS])
            nonneg = sum(1 for s in SEEDS if rows[s]["delta_clean"] >= 0)
            pos = sum(1 for s in SEEDS if rows[s]["delta_clean"] > 0)
            loo_pos = all(st.mean([rows[k]["delta_clean"] for k in SEEDS if k != s]) > 0
                          for s in SEEDS)
            if mean_clean > 0 and nonneg >= 3 and loo_pos:
                confound = "TRAINER-CONFOUND-UNLIKELY"
            elif mean_clean <= 0 or pos < 2:
                confound = "TRAINER-CONFOUND-DOMINANT"
            elif mean_leg > 0 and mean_clean < 0.5 * mean_leg:
                confound = "TRAINER-CONFOUND-CONTRIBUTORY"
            else:
                confound = "TRAINER-CONFOUND-UNRESOLVED"

            if mean_clean > 0 and nonneg >= 3 and loo_pos:
                mask = "CLEAN-TERMINAL-MASK-SUPPORTED"
            elif mean_clean <= 0 and pos < 2:
                mask = "CLEAN-TERMINAL-MASK-NOT-SUPPORTED"
            else:
                mask = "CLEAN-TERMINAL-MASK-UNRESOLVED"

            print("\n  clean mean>0: %s   seeds nonneg: %d/4   "
                  "leave-one-out all positive: %s" % (mean_clean > 0, nonneg, loo_pos))
            print("  TRAINER CONFOUND : %s" % confound)
            print("  CLEAN MASK       : %s" % mask)
            rows["classification"] = {"trainer_confound": confound,
                                      "clean_mask": mask,
                                      "mean_clean": mean_clean,
                                      "mean_legacy_R2": mean_leg}

    out = "experiments/policy_improvement/loss_finetune/clean_first_step/CLEAN_RESULTS.json"
    with open(out, "w") as fh:
        json.dump({str(k): v for k, v in rows.items()}, fh, indent=2, sort_keys=True, default=str)
    print("\nwritten", out)


if __name__ == "__main__":
    main()
