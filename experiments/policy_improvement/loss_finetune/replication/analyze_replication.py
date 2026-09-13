"""Paired four-seed analysis of the terminal-action mask (A vs B).

The replication unit is a PAIRED A/B fine-tuning run. The primary statistic is

    Delta_r = macro(B_r) - macro(A_r)

computed WITHIN each fine-tuning seed. Cross-seed comparisons (e.g. B_43 vs
A_44) are never used as causal contrasts.

The action-error decomposition deliberately separates the executed action (t=0),
the nonterminal actions, and the terminal placeholder (t=H-1). The whole-horizon
action error is NOT used as an A/B headline: A is trained to emit the terminal
zero placeholder and B is not, so that aggregate is not comparable across arms.
"""
import json
import os
import statistics as st

P = "experiments/policy_improvement/phase2"
SETS = ["E0", "E1s", "E2s", "E3s", "E4s"]

# (label, fine-tune seed, diagnostics tag for A, tag for B)
REPLICATES = [
    ("original", 42, "lossft_A", "lossft_B"),
    ("rep1", 43, "lossft_r1_A", "lossft_r1_B"),
    ("rep2", 44, "lossft_r2_A", "lossft_r2_B"),
    ("rep3", 45, "lossft_r3_A", "lossft_r3_B"),
]


def load(tag):
    p = os.path.join(P, "{}_diagnostics.json".format(tag))
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    rows = []
    for label, seed, ta, tb in REPLICATES:
        A, B = load(ta), load(tb)
        if not A or not B:
            print("replicate {} (seed {}): incomplete, skipping".format(label, seed))
            continue
        a = [A["control"][s]["success"] for s in SETS]
        b = [B["control"][s]["success"] for s in SETS]
        rows.append({
            "label": label, "seed": seed,
            "A": a, "B": b,
            "A_macro": A["control_mean"], "B_macro": B["control_mean"],
            "A_sd": A["control_sd"], "B_sd": B["control_sd"],
            "delta": B["control_mean"] - A["control_mean"],
            "per_set_delta": [bb - aa for aa, bb in zip(a, b)],
            "A_valid": A.get("n_valid_sets"), "B_valid": B.get("n_valid_sets"),
            "A_diag": A, "B_diag": B,
        })

    if not rows:
        print("no complete replicates yet")
        return

    print("=" * 92)
    print("PER-REPLICATE FIVE-SET CONTROL (paired within fine-tuning seed)")
    print("=" * 92)
    for r in rows:
        print("\n  {} (fine-tuning seed {})   valid sets A={} B={}".format(
            r["label"], r["seed"], r["A_valid"], r["B_valid"]))
        hdr = "    {:<4}".format("arm") + "".join("{:>9}".format(s) for s in SETS)
        print(hdr + "{:>9}{:>9}".format("macro", "SD"))
        print("    {:<4}".format("A") + "".join("{:>9.4f}".format(v) for v in r["A"])
              + "{:>9.4f}{:>9.4f}".format(r["A_macro"], r["A_sd"]))
        print("    {:<4}".format("B") + "".join("{:>9.4f}".format(v) for v in r["B"])
              + "{:>9.4f}{:>9.4f}".format(r["B_macro"], r["B_sd"]))
        print("    {:<4}".format("B-A") + "".join("{:>+9.4f}".format(v) for v in r["per_set_delta"])
              + "{:>+9.4f}".format(r["delta"]))
        up = sum(1 for v in r["per_set_delta"] if v > 0)
        tie = sum(1 for v in r["per_set_delta"] if v == 0)
        dn = sum(1 for v in r["per_set_delta"] if v < 0)
        print("    sets B>A: {}  tied: {}  B<A: {}".format(up, tie, dn))

    print()
    print("=" * 92)
    print("FOUR-SEED SUMMARY TABLE")
    print("=" * 92)
    print("{:<10}{:>6}{:>10}{:>10}{:>11}{:>9}{:>9}{:>7}{:>6}{:>7}".format(
        "seed", "ft", "A macro", "B macro", "B-A (pp)", "A SD", "B SD",
        "B>A", "tie", "B<A"))
    for r in rows:
        up = sum(1 for v in r["per_set_delta"] if v > 0)
        tie = sum(1 for v in r["per_set_delta"] if v == 0)
        dn = sum(1 for v in r["per_set_delta"] if v < 0)
        print("{:<10}{:>6}{:>10.4f}{:>10.4f}{:>+11.2f}{:>9.4f}{:>9.4f}{:>7}{:>6}{:>7}".format(
            r["label"], r["seed"], r["A_macro"], r["B_macro"], 100 * r["delta"],
            r["A_sd"], r["B_sd"], up, tie, dn))

    deltas = [r["delta"] for r in rows]
    n = len(deltas)
    print()
    print("  paired deltas (pp): {}".format(["{:+.2f}".format(100 * d) for d in deltas]))
    print("  mean   {:+.2f} pp".format(100 * st.mean(deltas)))
    print("  median {:+.2f} pp".format(100 * st.median(deltas)))
    if n > 1:
        sd = st.stdev(deltas)
        print("  sample SD {:.2f} pp".format(100 * sd))
        print("  min {:+.2f} pp   max {:+.2f} pp".format(100 * min(deltas), 100 * max(deltas)))
        # DESCRIPTIVE interval only; n is tiny.
        tcrit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(n)
        if tcrit and sd > 0:
            half = tcrit * sd / (n ** 0.5)
            print("  DESCRIPTIVE paired 95% interval (n={}, t={}): {:+.2f} to {:+.2f} pp"
                  .format(n, tcrit, 100 * (st.mean(deltas) - half),
                          100 * (st.mean(deltas) + half)))
            print("  (descriptive only -- n is far too small for a hypothesis test)")
    pos = sum(1 for d in deltas if d > 0)
    zer = sum(1 for d in deltas if d == 0)
    neg = sum(1 for d in deltas if d < 0)
    print("  sign consistency: {} positive, {} zero, {} negative (of {})".format(
        pos, zer, neg, n))

    # leave-one-seed-out: is the aggregate driven by the original seed?
    print("\n  leave-one-seed-out mean paired delta:")
    for i, r in enumerate(rows):
        rest = deltas[:i] + deltas[i + 1:]
        if rest:
            print("    drop {:<9} mean {:+.2f} pp".format(r["label"], 100 * st.mean(rest)))

    print()
    print("=" * 92)
    print("PER-EVALUATION-SET SUMMARY (mean B-A across fine-tuning seeds)")
    print("=" * 92)
    print("  {:<6}{:>12}{:>16}{:>10}".format("set", "mean B-A pp", "range pp", "seeds B>A"))
    for i, s in enumerate(SETS):
        vals = [r["per_set_delta"][i] for r in rows]
        up = sum(1 for v in vals if v > 0)
        print("  {:<6}{:>+12.2f}{:>16}{:>10}".format(
            s, 100 * st.mean(vals),
            "{:+.2f} .. {:+.2f}".format(100 * min(vals), 100 * max(vals)),
            "{}/{}".format(up, len(vals))))

    out = {
        "replicates": [{k: v for k, v in r.items() if not k.endswith("_diag")}
                       for r in rows],
        "paired_deltas": deltas,
        "mean_delta": st.mean(deltas),
        "median_delta": st.median(deltas),
        "sd_delta": st.stdev(deltas) if n > 1 else None,
        "n_positive": pos, "n_zero": zer, "n_negative": neg,
    }
    json.dump(out, open("experiments/policy_improvement/loss_finetune/replication/"
                        "REPLICATION_SUMMARY.json", "w"), indent=2, sort_keys=True)
    print("\nwritten REPLICATION_SUMMARY.json")


if __name__ == "__main__":
    main()
