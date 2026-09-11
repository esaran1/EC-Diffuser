"""Atomic single-run lock keyed to an output directory.

Infrastructure only: performs no tensor work, consumes no RNG, and does not
touch the training stream. Prevents two processes writing the same log /
checkpoint directory (we already had one duplicate-launch race).
"""
import errno
import json
import os
import time


class RunLockError(RuntimeError):
    pass


class RunLock:
    """O_CREAT|O_EXCL lock file inside the run directory.

    Stale policy is explicit and conservative: a lock whose recorded PID is
    still alive is NEVER treated as stale. A lock whose PID is gone is reported
    as stale but still requires force=True to break, so an active run can never
    be silently overwritten.
    """

    def __init__(self, run_dir, name=".run.lock"):
        self.run_dir = str(run_dir)
        self.path = os.path.join(self.run_dir, name)
        self.acquired = False

    @staticmethod
    def _alive(pid):
        try:
            os.kill(int(pid), 0)
        except (OSError, ValueError, TypeError):
            return False
        return True

    def read(self):
        try:
            with open(self.path) as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return None

    def acquire(self, force=False):
        os.makedirs(self.run_dir, exist_ok=True)
        payload = json.dumps({
            "pid": os.getpid(),
            "timestamp": time.time(),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "run_dir": self.run_dir,
        }, indent=1)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            holder = self.read()
            pid = (holder or {}).get("pid")
            if holder is not None and self._alive(pid):
                raise RunLockError(
                    "run directory {} is locked by live PID {} since {}".format(
                        self.run_dir, pid, holder.get("iso")))
            if not force:
                raise RunLockError(
                    "run directory {} holds a STALE lock (PID {} not running). "
                    "Re-run with force=True only if you are certain no writer "
                    "is active.".format(self.run_dir, pid))
            os.unlink(self.path)
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        with os.fdopen(fd, "w") as fh:
            fh.write(payload)
        self.acquired = True
        return self

    def release(self):
        if not self.acquired:
            return
        holder = self.read()
        if holder and holder.get("pid") == os.getpid():
            try:
                os.unlink(self.path)
            except OSError:
                pass
        self.acquired = False

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()
        return False
