"""Extract every REAL executed action exactly as GoalDataset uses it
(normed_actions[p, start:actual_end-1] => indices 0..L_p-2), plus the
constructed terminal stop target normalize(zeros). CPU only."""
import sys, numpy as np
sys.path.insert(0, "diffuser")
from diffuser.utils.serialization import load_config
R = "3C_dlp_adalnpint_randcolor_H5_T4_seed42"
dc = load_config("data/panda_push/uwm_ctl", R, "dataset_config.pkl")
dc._dict["dataset_path"] = "ecdiffuser-data/push_cubes/3C_randcolor/panda_push_replay_buffer_dlp.pkl"
ds = dc(seed=42)
f = ds.fields
A = np.concatenate([f.normed_actions[p, :int(f.path_lengths[p]) - 1]
                    for p in range(len(f.path_lengths))]).astype(np.float32)
term = ds.normalizer.normalize(np.zeros((1, ds.action_dim), dtype=np.float32), "actions").astype(np.float32)
np.savez_compressed("experiments/policy_improvement/tal_followup/action_latent/actions.npz",
                    actions=A, terminal=term, path_lengths=np.asarray(f.path_lengths))
print("real actions", A.shape, "min", A.min(0), "max", A.max(0), "std", A.std(0))
print("terminal (normalized zero):", term)
