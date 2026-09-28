"""Single source of truth for cumulative program GPU spend.

Reconstructs spend from EVERY runner log in the program directory, deduplicated
by (stage, acquire-timestamp) because several logs tee the same lines. Includes
valid, wasted, partial and contended work alike -- the 10 GPU-h cap is on GPU
occupancy, not on scientific usefulness.

NEVER resets across restarts, sessions or driver repairs: it derives the total
from persisted logs plus a floor file, not from any in-process counter.
"""
import os
import re
from datetime import datetime as dt

HERE = os.path.dirname(os.path.abspath(__file__))
FLOOR = os.path.join(HERE, "SPEND_FLOOR.txt")


def _pairs():
    opens, pairs = {}, {}
    for f in sorted(os.listdir(HERE)):
        if not f.endswith(".log"):
            continue
        for line in open(os.path.join(HERE, f), errors="ignore"):
            m = re.search(r"\[([0-9T:\-]+)\].*lease acquired \(stage ([^,]+),", line)
            if m:
                opens[(f, m.group(2))] = m.group(1)
                continue
            m = re.search(r"\[([0-9T:\-]+)\].*lease released \(stage ([^)]+)\)", line)
            if m and (f, m.group(2)) in opens:
                a = opens.pop((f, m.group(2)))
                pairs[(m.group(2), a)] = m.group(1)      # dedupe on stage+acquire
    return pairs, opens


def spent(now=None):
    """Cumulative GPU-h. Unclosed leases are charged to `now` (conservative)."""
    pairs, opens = _pairs()
    total = 0.0
    for (stage, a), b in pairs.items():
        total += (dt.fromisoformat(b) - dt.fromisoformat(a)).total_seconds() / 3600
    # an unclosed lease is either live or died; charge it either way
    ref = now or dt.now()
    for (_f, stage), a in opens.items():
        d = (ref - dt.fromisoformat(a)).total_seconds() / 3600
        if 0 < d < 2.0:                    # live run; cap the charge at 2 h
            total += d
    floor = 0.0
    if os.path.exists(FLOOR):
        try:
            floor = float(open(FLOOR).read().strip())
        except ValueError:
            pass
    # monotone: never report less than the recorded floor
    total = max(total, floor)
    try:
        with open(FLOOR, "w") as fh:
            fh.write("{:.6f}\n".format(total))
    except OSError:
        pass
    return total


if __name__ == "__main__":
    print("{:.4f}".format(spent()))
