"""Build the evaluation provenance registry (directive section 4).

Walks every control-evaluation artifact in phase2/, resolves each to the exact
checkpoint bytes it evaluated, and classifies validity. The point is to identify
which pairs of evaluations are genuine SAME-CHECKPOINT repeats -- the only pairs
that measure evaluator variability rather than policy difference.

A pair qualifies as repeatability data only if checkpoint SHA, EMA/live
convention, NFE, evaluator protocol and validity all match, with no contention.
Cross-policy pairs (A vs B, seed42 vs seed43, legacy vs clean) are explicitly
NOT eligible.

Zero GPU.
"""
import hashlib
import json
import os

P2 = "experiments/policy_improvement/phase2"
OUT = "experiments/policy_improvement/evaluator_resolution"
SETS = ["E0", "E1s", "E2s", "E3s", "E4s"]
DATA = "data/panda_push"

# tag -> (loadpath relative to data/panda_push, epoch, which weights the
# evaluator used). phase2_diagnostics.py evaluates exp.ema for control.
KNOWN_FAMILY = {
    "canonical_anchor": "canonical",
    "lossft_A": "legacy_A", "lossft_B": "legacy_B",
    "lossft_C": "legacy_C", "lossft_D": "legacy_D",
    "lossft_r1_A": "legacy_A", "lossft_r1_B": "legacy_B",
    "lossft_r2_A": "legacy_A", "lossft_r2_B": "legacy_B",
    "lossft_r3_A": "legacy_A", "lossft_r3_B": "legacy_B",
    "lossft_rep2_A42": "legacy_A", "lossft_rep2_B42": "legacy_B",
    "lossft_rep2_A43": "legacy_A", "lossft_rep2_B43": "legacy_B",
    "lossft_rep2_A44": "legacy_A", "lossft_rep2_B44": "legacy_B",
    "lossft_rep2_A45": "legacy_A", "lossft_rep2_B45": "legacy_B",
    "lossft_clean_A42": "clean_A", "lossft_clean_B42": "clean_B",
    "lossft_clean_A43": "clean_A", "lossft_clean_B43": "clean_B",
    "lossft_clean_A44": "clean_A", "lossft_clean_B44": "clean_B",
    "lossft_clean_A45": "clean_A", "lossft_clean_B45": "clean_B",
    "lossft_bprime_42": "bprime", "lossft_bprime_43": "bprime",
    "lossft_bprime_44": "bprime", "lossft_bprime_45": "bprime",
}

_sha_cache = {}


def sha256_of(path):
    if path in _sha_cache:
        return _sha_cache[path]
    if not os.path.exists(path):
        _sha_cache[path] = None
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    _sha_cache[path] = h.hexdigest()
    return _sha_cache[path]


def classify(tag, d, fname):
    """Validity bucket. Never silently promote an invalid artifact."""
    if "INVALID" in fname:
        return ("INVALID_CONTENDED" if "contended" in fname.lower()
                else "INVALID_OTHER")
    if "SUSPECT" in fname:
        return "INVALID_OTHER"
    ctrl = d.get("control")
    if not ctrl:
        return "UNKNOWN_EXCLUDED"
    if d.get("control_mean") is None or d.get("n_valid_sets") != len(SETS):
        return "INVALID_OTHER"
    if not all(ctrl.get(s, {}).get("status") == "VALID" for s in SETS):
        return "INVALID_OTHER"
    if any(ctrl.get(s, {}).get("midset_max_foreign_mib", 0) for s in SETS):
        return "INVALID_CONTENDED"
    # Current protocol = the five fixed E sets, n=96, evaluated on EMA.
    if any(ctrl.get(s, {}).get("n") != 96 for s in SETS):
        return "VALID_HISTORICAL_SUPPORTING"
    return "VALID_CURRENT_PROTOCOL"


def main():
    os.makedirs(OUT, exist_ok=True)
    records = []
    for fname in sorted(os.listdir(P2)):
        if not fname.endswith(".json") or "_diagnostics" not in fname:
            continue
        path = os.path.join(P2, fname)
        try:
            d = json.load(open(path))
        except Exception as exc:
            records.append({"file": fname, "bucket": "UNKNOWN_EXCLUDED",
                            "error": str(exc)})
            continue
        tag = fname.split("_diagnostics")[0]
        loadpath = d.get("loadpath")
        ckpt = None
        if loadpath:
            ckpt = os.path.join(DATA, loadpath, "state_{}.pt".format(
                d.get("epoch", "final")))
            if not os.path.exists(ckpt):
                alt = os.path.join(DATA, loadpath, "state_final.pt")
                ckpt = alt if os.path.exists(alt) else ckpt
            # The kwargs-defect runs were later RE-RUN correctly at the same
            # loadpath, and their original checkpoints were preserved beside it
            # under *_INVALID_kwargs_bug/. Resolving the recorded loadpath would
            # therefore attribute the OLD evaluation to the NEW checkpoint bytes.
            # Redirect to the preserved directory so the SHA matches what this
            # evaluation actually ran on.
            if "INVALID_kwargs_bug" in fname:
                arm_dir = loadpath.split("/")[0]
                preserved = loadpath.replace(
                    arm_dir, arm_dir + "_INVALID_kwargs_bug", 1)
                cand = os.path.join(DATA, preserved, "state_final.pt")
                if os.path.exists(cand):
                    ckpt = cand
        ctrl = d.get("control", {})
        rec = {
            "file": fname,
            "tag": tag,
            "family": KNOWN_FAMILY.get(tag, "unknown"),
            "loadpath": loadpath,
            "checkpoint_path": ckpt,
            "checkpoint_sha256": sha256_of(ckpt) if ckpt else None,
            "epoch": d.get("epoch"),
            "weights": "EMA",          # phase2_diagnostics control() uses exp.ema
            "nfe": 4,
            "n_valid_sets": d.get("n_valid_sets"),
            "macro": d.get("control_mean"),
            "macro_sd_across_sets": d.get("control_sd"),
            "per_set": {s: ctrl.get(s, {}).get("success") for s in SETS},
            "per_set_n": {s: ctrl.get(s, {}).get("n") for s in SETS},
            "per_set_status": {s: ctrl.get(s, {}).get("status") for s in SETS},
            "per_set_foreign_mib": {
                s: ctrl.get(s, {}).get("midset_max_foreign_mib") for s in SETS},
            "timestamp_iso": (ctrl.get("E0", {}).get("pre_snapshot", {})
                              .get("iso")),
            "episode_level_data": any(
                k in ctrl.get("E0", {}) for k in
                ("successes", "success_vector", "per_episode", "episode_ids")),
            "path": path,
        }
        rec["bucket"] = classify(tag, d, fname)
        records.append(rec)

    # ---- evaluation-level validity ----------------------------------------
    # A record can be scientifically invalid (it answered the wrong question)
    # while its EVALUATION was executed perfectly. The kwargs-defect B/C/D runs
    # are exactly that: their checkpoints were byte-identical to legacy A42, so
    # as loss-arm results they were void, but as repeat evaluations of ONE fixed
    # policy they are valid and usable here. Contended runs are not.
    for r in records:
        ps = r.get("per_set_status") or {}
        pn = r.get("per_set_n") or {}
        pf = r.get("per_set_foreign_mib") or {}
        r["evaluation_valid"] = bool(
            r.get("macro") is not None
            and r.get("n_valid_sets") == len(SETS)
            and all(ps.get(s) == "VALID" for s in SETS)
            and all(pn.get(s) == 96 for s in SETS)
            and not any(pf.get(s) or 0 for s in SETS)
            # Degenerate policies (macro 0.0 or 1.0) are pinned at the range
            # boundary, where success-rate variance is structurally near zero.
            # Their repeat spread says nothing about resolution near the ~78%
            # regime every policy comparison in this project lives in. Excluded
            # from repeatability estimation, not from the registry.
            and 0.0 < r["macro"] < 1.0
            # A file flagged SUSPECT was set aside for a reason; do not quietly
            # readmit it as repeatability data.
            and "SUSPECT" not in r["file"])

    # ---- same-checkpoint repeat groups ------------------------------------
    groups = {}
    for r in records:
        if not r.get("evaluation_valid"):
            continue
        sha = r.get("checkpoint_sha256")
        if not sha:
            continue
        groups.setdefault((sha, r["weights"], r["nfe"]), []).append(r["tag"])
    repeats = {k[0][:16]: sorted(v) for k, v in groups.items() if len(v) > 1}

    summary = {}
    for r in records:
        summary[r.get("bucket", "UNKNOWN_EXCLUDED")] = summary.get(
            r.get("bucket", "UNKNOWN_EXCLUDED"), 0) + 1

    out = {
        "n_records": len(records),
        "bucket_counts": summary,
        "same_checkpoint_repeat_groups": repeats,
        "episode_level_data_available": any(r.get("episode_level_data")
                                            for r in records),
        "records": records,
    }
    with open(os.path.join(OUT, "evaluation_registry.json"), "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)

    print("records: {}".format(len(records)))
    for k in sorted(summary):
        print("  {:<30} {}".format(k, summary[k]))
    print("\nsame-checkpoint repeat groups (evaluation-valid; scientific bucket may differ):")
    if not repeats:
        print("  none")
    for sha, tags in sorted(repeats.items()):
        print("  {}  n={}  {}".format(sha, len(tags), tags))
    print("\nepisode-level success vectors present in artifacts: {}".format(
        out["episode_level_data_available"]))
    print("\nwritten {}/evaluation_registry.json".format(OUT))


if __name__ == "__main__":
    main()
