"""Unit naming/validity shared by sub_eval (GPU) and run.py (no Isaac import)."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
UNITS = os.path.join(HERE, "results/units")
CALLS = {"euler": 1, "midpoint": 2, "rk4": 4}


def unit_name(ckpt, ne, nE, nfe, solver, prec, R, s):
    return "{}_c{}_b{}_{}{}_{}_R{}_{}".format(ckpt, ne, nE, solver, nfe, prec, R, s)


def is_valid(name):
    p = os.path.join(UNITS, name + ".json")
    return os.path.exists(p) and json.load(open(p)).get("status") == "VALID"
