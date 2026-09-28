"""Fixed scenario sets for N-cube generalization. Same five seeds as the
3-cube sets; record_episode_set reused VERBATIM. Never regenerates an
existing file. Usage: gen_sets.py N [set names...]"""
import os, pickle, sys
import isaacgym  # noqa: F401
sys.path.insert(0, "experiments/scripts")
import diffuser.utils as utils  # noqa: E402
from diffuser.eval_utils import setup_isaac_env  # noqa: E402
from isaacgym_control import Args, record_episode_set  # noqa: E402
OUT = "experiments/isaacgym_episode_sets"
SEEDS = {"E0": (20260820, "replicate0_n96.pkl"), "E1s": (20261820, "replicate1_n96.pkl"),
         "E2s": (20262820, "replicate2_n96.pkl"), "E3s": (20263820, "replicate3_n96.pkl"),
         "E4s": (20264820, "replicate4_n96.pkl")}
n = int(sys.argv[1]); names = sys.argv[2:] or list(SEEDS)
todo = [(k,) + SEEDS[k] for k in names if not os.path.exists("%s/c%d_%s" % (OUT, n, SEEDS[k][1]))]
if not todo:
    print("all requested sets exist"); sys.exit(0)
args = Args(); args.num_entity = n
utils.set_global_device(args.device)
env = setup_isaac_env(args); env.horizon = args.max_episode_length
for name, seed, f in todo:
    p = record_episode_set(env, 96, seed=seed)
    p["num_entity"] = n
    path = "%s/c%d_%s" % (OUT, n, f)
    pickle.dump(p, open(path, "wb"))
    print("%s N=%d seed=%d -> %s sha %s  init %s" % (name, n, seed, path, p["sha256"][:16], p["init"].shape))
