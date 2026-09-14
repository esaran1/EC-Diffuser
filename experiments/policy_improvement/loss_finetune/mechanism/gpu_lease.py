"""Global GPU lease for the terminal-mask mechanism program.

WHY THIS EXISTS. Run-directory locks are not sufficient: two stages with
different run directories can each acquire their own lock and launch GPU work
simultaneously, contaminating each other even though both jobs are ours. This
lease is a single, program-wide mutex covering ALL GPU work in this program.

Properties (all required by the directive):

  atomic acquisition   O_CREAT|O_EXCL on one fixed path; the filesystem decides
                       the winner, no read-then-write window
  identity recorded    PID, stage id, timestamp, hostname, command
  single holder        one path, one holder, no re-entrancy
  clean release        released on normal exit and on SIGINT/SIGTERM
  stale policy         EXPLICIT: a lock whose PID is alive is NEVER removed,
                       regardless of age. A lock whose PID is gone is reported as
                       stale and only broken with --force (or force=True), which
                       records who broke it and what it contained.
  TOCTOU closed        the caller checks GPU cleanliness, acquires the lease,
                       then RE-CHECKS cleanliness while holding it. If a foreign
                       job appeared in between, the lease is released and the
                       launch is abandoned.

Usage from the single orchestrator:

    with GpuLease("stageB:probe-Bprime") as lease:
        ...launch exactly one GPU process...

or from shell:

    python gpu_lease.py acquire --stage stageB --command "..."   # runs it
"""
import argparse
import errno
import json
import os
import signal
import socket
import subprocess
import sys
import time

LOCK_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".gpu_program.lock")
FOREIGN_MIB_THRESHOLD = 1000


def foreign_mib(exclude_pids=()):
    """Total GPU memory held by compute apps that are not ours."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return -1                      # unknown: treat as dirty, never as clean
    total = 0
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        try:
            pid, mib = [p.strip() for p in line.split(",")]
            pid, mib = int(pid), int(mib)
        except ValueError:
            continue
        if pid in exclude_pids:
            continue
        if mib >= FOREIGN_MIB_THRESHOLD:
            total += mib
    return total


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except OSError as exc:
        return exc.errno == errno.EPERM   # exists but not ours to signal
    return True


def read_lock():
    try:
        with open(LOCK_PATH) as fh:
            return json.load(fh)
    except Exception:
        return None


class LeaseBusy(RuntimeError):
    pass


class GpuLease(object):
    """Program-wide GPU mutex. One holder at a time, across all stages."""

    def __init__(self, stage, force=False):
        self.stage = stage
        self.force = force
        self.acquired = False
        self._prev_handlers = {}

    # -- acquisition -------------------------------------------------------
    def acquire(self):
        payload = {
            "pid": os.getpid(),
            "stage": self.stage,
            "host": socket.gethostname(),
            "acquired_at": time.time(),
            "acquired_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "command": " ".join(sys.argv),
        }
        try:
            fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            holder = read_lock()
            if holder and pid_alive(int(holder.get("pid", -1))):
                raise LeaseBusy(
                    "GPU lease held by pid {} (stage {!r}, since {}); refusing to "
                    "break a LIVE lock".format(holder.get("pid"),
                                               holder.get("stage"),
                                               holder.get("acquired_iso")))
            if not self.force:
                raise LeaseBusy(
                    "GPU lease at {} looks STALE (holder pid {} not alive, stage "
                    "{!r}, since {}). Re-run with force to break it.".format(
                        LOCK_PATH, holder.get("pid") if holder else "?",
                        holder.get("stage") if holder else "?",
                        holder.get("acquired_iso") if holder else "?"))
            payload["broke_stale_lock"] = holder
            os.replace(LOCK_PATH, LOCK_PATH + ".broken.{}".format(int(time.time())))
            fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)

        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, indent=1)
        self.acquired = True
        self._install_handlers()
        return self

    def release(self):
        if not self.acquired:
            return
        holder = read_lock()
        if holder and int(holder.get("pid", -1)) == os.getpid():
            try:
                os.unlink(LOCK_PATH)
            except OSError:
                pass
        self.acquired = False
        self._restore_handlers()

    # -- signal safety -----------------------------------------------------
    def _install_handlers(self):
        def handler(signum, frame):
            self.release()
            prev = self._prev_handlers.get(signum)
            if callable(prev):
                prev(signum, frame)
            else:
                sys.exit(128 + signum)
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                self._prev_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass

    def _restore_handlers(self):
        for sig, prev in self._prev_handlers.items():
            try:
                signal.signal(sig, prev)
            except (ValueError, OSError):
                pass
        self._prev_handlers = {}

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()
        return False


def wait_stable_clean(probes=10, interval=18, max_wait_s=None, log=print):
    """Require `probes` consecutive clean readings. Any foreign job resets it."""
    streak = 0
    started = time.time()
    while True:
        mib = foreign_mib()
        if mib == 0:
            streak += 1
            if streak >= probes:
                log("[gpu_lease] GPU stable-clean for {}s".format(probes * interval))
                return True
            time.sleep(interval)
        else:
            if streak:
                log("[gpu_lease] clean streak broken at {} (foreign {} MiB)".format(
                    streak, mib))
            streak = 0
            time.sleep(60)
        if max_wait_s is not None and time.time() - started > max_wait_s:
            log("[gpu_lease] gave up waiting for a stable-clean GPU")
            return False


def run_guarded(stage, command, probes=10, interval=18, max_wait_s=None,
                force=False, log=print):
    """stable-clean -> acquire lease -> RE-CHECK cleanliness -> run one GPU job.

    The re-check after acquisition is what closes the TOCTOU race: a foreign job
    may appear between the last clean probe and the launch.
    """
    if not wait_stable_clean(probes, interval, max_wait_s, log):
        return 3
    lease = GpuLease(stage, force=force)
    try:
        lease.acquire()
    except LeaseBusy as exc:
        log("[gpu_lease] {}".format(exc))
        return 4
    try:
        mib = foreign_mib()
        if mib != 0:
            log("[gpu_lease] foreign job ({} MiB) appeared between the clean check "
                "and lease acquisition; abandoning this launch".format(mib))
            return 5
        log("[gpu_lease] lease acquired (stage {}, pid {}); launching".format(
            stage, os.getpid()))
        return subprocess.call(command, shell=True)
    finally:
        lease.release()
        log("[gpu_lease] lease released (stage {})".format(stage))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="wait clean, take the lease, run one command")
    r.add_argument("--stage", required=True)
    r.add_argument("--command", required=True)
    r.add_argument("--probes", type=int, default=10)
    r.add_argument("--interval", type=int, default=18)
    r.add_argument("--max-wait", type=float, default=None)
    r.add_argument("--force", action="store_true")
    s = sub.add_parser("status")
    sub.add_parser("clean-check")
    cli = ap.parse_args()

    if cli.cmd == "status":
        h = read_lock()
        if not h:
            print("no lease held")
            return 0
        alive = pid_alive(int(h.get("pid", -1)))
        print(json.dumps({**h, "holder_alive": alive}, indent=1))
        return 0
    if cli.cmd == "clean-check":
        print(foreign_mib())
        return 0
    return run_guarded(cli.stage, cli.command, cli.probes, cli.interval,
                       cli.max_wait, cli.force)


if __name__ == "__main__":
    sys.exit(main())
