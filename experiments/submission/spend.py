"""Fresh-phase GPU spend (cap 20 GPU-h): same lease-pair accounting as the closed
program (tal_followup/uwm/program_spend.py) but over experiments/submission/ logs
only, with its own monotone floor."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../policy_improvement/tal_followup/uwm"))
import program_spend as _ps  # noqa: E402
_ps.ROOT = HERE
_ps.FLOOR = os.path.join(HERE, "SPEND_FLOOR.txt")
CAP = 20.0


def spent():
    return _ps.spent()


if __name__ == "__main__":
    print("{:.4f}".format(spent()))
