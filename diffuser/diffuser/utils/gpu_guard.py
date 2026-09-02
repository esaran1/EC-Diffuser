"""GPU contention guard (infrastructure only).

Records a launch-time GPU snapshot and detects foreign compute processes that
appear later. Performs NO tensor work and never touches model computation:
it only shells out to nvidia-smi and compares PID sets.
"""
import json
import os
import subprocess
import time


def _query(fields, extra=None):
    cmd = ["nvidia-smi", f"--query-{'compute-apps' if extra else 'gpu'}={fields}",
           "--format=csv,noheader,nounits"]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=20)
        return [l.strip() for l in out.decode().splitlines() if l.strip()]
    except Exception:
        return []


def compute_apps():
    """[(pid, process_name, used_mib)] for every process holding GPU memory."""
    rows = []
    for line in _query("pid,process_name,used_memory", extra=True):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            try:
                rows.append((int(parts[0]), parts[1], int(float(parts[2]))))
            except ValueError:
                continue
    return rows


def snapshot():
    return {
        "timestamp": time.time(),
        "iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gpu": _query("index,name,memory.used,memory.total,utilization.gpu"),
        "compute_apps": [
            {"pid": p, "name": n, "used_mib": m} for p, n, m in compute_apps()
        ],
    }


def require_uncontended(min_free_mib_threshold=1000, allow_pids=()):
    """Raise unless the GPU is free of foreign compute processes at launch."""
    snap = snapshot()
    mine = set(allow_pids) | {os.getpid()}
    foreign = [a for a in snap["compute_apps"]
               if a["pid"] not in mine and a["used_mib"] >= min_free_mib_threshold]
    snap["foreign_at_launch"] = foreign
    if foreign:
        raise RuntimeError(
            "GPU is contended at launch; refusing to start. Foreign processes: "
            + json.dumps(foreign)
        )
    return snap


class ContentionMonitor:
    """Detects a foreign GPU-heavy process appearing after launch.

    check() is called between optimizer steps and does no GPU work; it is
    rate-limited so the nvidia-smi call cannot perturb throughput.
    """

    def __init__(self, baseline_pids=None, min_mib=1000, period_s=60.0):
        self.baseline = set(baseline_pids or ()) | {os.getpid()}
        self.min_mib = min_mib
        self.period_s = period_s
        self._last = 0.0
        self.event = None

    def check(self, force=False):
        now = time.time()
        if not force and now - self._last < self.period_s:
            return None
        self._last = now
        foreign = [
            {"pid": p, "name": n, "used_mib": m}
            for p, n, m in compute_apps()
            if p not in self.baseline and m >= self.min_mib
        ]
        if foreign:
            self.event = {"iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
                          "foreign": foreign}
            return self.event
        return None
