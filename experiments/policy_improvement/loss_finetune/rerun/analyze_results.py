"""Assemble the corrected B/C/D result tables for the final report.

Reads the valid canonical/A references and the corrected B/C/D diagnostics,
and emits the tables the directive asks for: full control table, primary
contrasts (A-B, B-C, B-D, C-D descriptive), replay results, and the runtime
objective manifests that prove each arm trained the objective it claims.

Deliberately does NOT rank arms by raw total loss: the arms optimize
differently-weighted objectives, so their total losses are not comparable as
"better optimization" (directive section 22). Ranking uses semantic/unweighted
errors, replay, and closed-loop control.
"""
import json
import os
import statistics as st

P = "experiments/policy_improvement/phase2"
RUNROOT = "data/panda_push"
SUB = "3C_dlp_adalnpint_randcolor_H5_T4_seed42"
SETS = ["E0", "E1s", "E2s", "E3s", "E4s"]

CANONICAL = {"E0": 0.7500, "E1s": 0.7500, "E2s": 0.8125, "E3s": 0.7396,
             "E4s": 0.8021, "macro": 0.7708, "SD": 0.0338}
CANONICAL_REPLAY = {"ema_action": 0.02689, "ema_state": 0.20532}


def load(arm):
    path = os.path.join(P, "lossft_{}_diagnostics.json".format(arm))
    if not os.path.exists(path):
        return None
    return json.load(open(path))


def manifest(arm):
    path = os.path.join(RUNROOT, "lossft_arm{}".format(arm), SUB,
                        "RUNTIME_OBJECTIVE_MANIFEST.json")
    if not os.path.exists(path):
        return None
    return json.load(open(path))


def main():
    data = {a: load(a) for a in "ABCD"}
    available = [a for a in "ABCD" if data[a]]

    print("=" * 78)
    print("RUNTIME OBJECTIVE MANIFESTS (what each arm ACTUALLY constructed)")
    print("=" * 78)
    print("{:<5}{:<8}{:<8}{:<12}{:<12}{:<12}{}".format(
        "arm", "mask", "lambda", "mat_sum", "act_blk", "obs_blk", "verdict"))
    for a in "ABCD":
        m = manifest(a)
        if not m:
            if a == "A":
                # Arm A ran before the runtime-manifest mechanism existed. Its
                # objective is verified instead by the stored loss_weight_matrix
                # in its own checkpoint, which the corrected production path
                # reproduces byte-identically (sha16 f6f56c359875b7f4) -- see
                # the section-10 "A would remain A" check.
                print("{:<5}{:<8}{:<8}{:<12.1f}{:<12.1f}{:<12.1f}{}".format(
                    "A", "False", "1.0", 2442.0, 42.0, 2400.0,
                    "PRE-MANIFEST (verified via stored matrix sha f6f56c359875b7f4)"))
            else:
                print("{:<5}(no manifest - arm not yet run)".format(a))
            continue
        o = m["actual"]
        print("{:<5}{:<8}{:<8}{:<12.1f}{:<12.1f}{:<12.1f}{}".format(
            a, str(o["mask_terminal_action"]), o["lambda_action"],
            o["loss_weight_matrix_sum"], o["action_block_sum"],
            o["observation_block_sum"], m["verdict"]))

    print()
    print("=" * 78)
    print("FULL CONTROL TABLE (NFE=4, five fixed scenario seeds, n=96 each)")
    print("=" * 78)
    header = "{:<12}".format("policy") + "".join("{:>9}".format(s) for s in SETS)
    print(header + "{:>9}{:>9}".format("macro", "SD"))
    print("{:<12}".format("canonical")
          + "".join("{:>9.4f}".format(CANONICAL[s]) for s in SETS)
          + "{:>9.4f}{:>9.4f}".format(CANONICAL["macro"], CANONICAL["SD"]))
    rows = {}
    for a in available:
        c = data[a]["control"]
        vals = [c[s]["success"] for s in SETS]
        rows[a] = vals
        print("{:<12}".format("arm " + a)
              + "".join("{:>9.4f}".format(v) for v in vals)
              + "{:>9.4f}{:>9.4f}".format(data[a]["control_mean"], data[a]["control_sd"]))

    # validity
    print("\nset validity / contention:")
    for a in available:
        c = data[a]["control"]
        statuses = {s: c[s]["status"] for s in SETS}
        mid = {s: c[s].get("midset_max_foreign_mib", "n/a") for s in SETS}
        allvalid = all(v == "VALID" for v in statuses.values())
        print("  arm {}: {}  n_valid_sets={}  midset_max_foreign_mib={}".format(
            a, "ALL VALID" if allvalid else statuses,
            data[a].get("n_valid_sets"), mid))

    print()
    print("=" * 78)
    print("PRIMARY CONTRASTS")
    print("=" * 78)
    contrasts = [("A", "B", "terminal zero-action supervision ablation"),
                 ("B", "C", "2x action-side weighting"),
                 ("B", "D", "5x action-side weighting"),
                 ("C", "D", "DESCRIPTIVE: moderate vs stronger tested weight")]
    for x, y, label in contrasts:
        if x not in rows or y not in rows:
            print("  {} vs {}: not available yet".format(x, y))
            continue
        dm = data[y]["control_mean"] - data[x]["control_mean"]
        print("\n  {} -> {}  ({})".format(x, y, label))
        print("    macro {:.4f} -> {:.4f}   delta {:+.4f} ({:+.2f} pp)".format(
            data[x]["control_mean"], data[y]["control_mean"], dm, 100 * dm))
        per = []
        for i, s in enumerate(SETS):
            d = rows[y][i] - rows[x][i]
            per.append("{} {:+.4f}".format(s, d))
        print("    per-set: " + "  ".join(per))
        signs = [rows[y][i] - rows[x][i] for i in range(len(SETS))]
        print("    sets improved {}/{}, unchanged {}, worse {}".format(
            sum(1 for v in signs if v > 0), len(SETS),
            sum(1 for v in signs if v == 0), sum(1 for v in signs if v < 0)))

    print()
    print("=" * 78)
    print("REPLAY (LIVE and EMA) + SEMANTIC LOSSES")
    print("=" * 78)
    print("{:<12}{:>12}{:>12}{:>12}{:>12}".format(
        "policy", "LIVE_act", "LIVE_state", "EMA_act", "EMA_state"))
    print("{:<12}{:>12}{:>12}{:>12.5f}{:>12.5f}".format(
        "canonical", "-", "-", CANONICAL_REPLAY["ema_action"],
        CANONICAL_REPLAY["ema_state"]))
    for a in available:
        d = data[a]
        print("{:<12}{:>12.5f}{:>12.5f}{:>12.5f}{:>12.5f}".format(
            "arm " + a,
            d["LIVE"]["replay_action_error"], d["LIVE"]["replay_state_error"],
            d["EMA"]["replay_action_error"], d["EMA"]["replay_state_error"]))
    print("\n  !! replay ACTION error is NOT comparable between A/canonical and")
    print("     B/C/D: replay_error averages over ALL horizon steps including")
    print("     t=H-1, the slice B/C/D no longer supervise, so the active set")
    print("     differs. B vs C vs D ARE mutually comparable (same mask).")
    print("     Replay STATE error (t=1:4) is comparable everywhere.")
    print("     Replay is not equivalent to closed-loop control and is not used")
    print("     to rank arms. See REPLAY_METRIC_CAVEAT.md.")

    print("\nunweighted (semantic) losses -- comparable across arms;")
    print("raw TOTAL loss is NOT comparable because the objectives are weighted differently.")
    print("{:<12}{:>14}{:>14}{:>14}".format("policy", "action_loss", "state_loss", "action_grad"))
    for a in available:
        d = data[a]["EMA"]
        print("{:<12}{:>14.6f}{:>14.6f}{:>14.6f}".format(
            "arm " + a, d["action_loss"], d["state_loss"], d["action_grad"]))

    # evaluator variability context
    var_path = "experiments/policy_improvement/loss_finetune/rerun/EVALUATOR_VARIABILITY.json"
    if os.path.exists(var_path):
        v = json.load(open(var_path))
        print()
        print("=" * 78)
        print("DESCRIPTIVE EVALUATOR VARIABILITY (context only, NOT a threshold)")
        print("=" * 78)
        print("  four independent evaluations of ONE byte-identical policy:")
        print("  macro range {:.2f} pp, empirical SD {:.2f} pp".format(
            v["macro_range_pp"], 100 * v["macro_empirical_sd"]))
        print("  per-set ranges: " + ", ".join(
            "{} {:.2f} pp".format(k, v["per_set_range_pp"][k]) for k in SETS))

    out = {
        "canonical": CANONICAL,
        "arms": {a: {"control": {s: data[a]["control"][s]["success"] for s in SETS},
                     "macro": data[a]["control_mean"],
                     "sd": data[a]["control_sd"],
                     "LIVE": data[a]["LIVE"], "EMA": data[a]["EMA"],
                     "manifest": manifest(a)} for a in available},
    }
    json.dump(out, open("experiments/policy_improvement/loss_finetune/rerun/RESULTS.json", "w"),
              indent=2, sort_keys=True)
    print("\nwritten RESULTS.json")


if __name__ == "__main__":
    main()
