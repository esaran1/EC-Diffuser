"""Submission-phase closed-loop evaluator (batch-invariant policy noise).

One process = one checkpoint x cube count x batch size; loops over units
(scenario set, realization, NFE/solver/precision config). Each unit writes
results/units/<name>.json (+ .npz with per-episode arrays) and is skipped if a
VALID result already exists, so the runner can simply re-launch after contention.

Differences from phase2_diagnostics.control (kept unchanged for old results):
  * policy noise x0 is drawn per (realization, set, episode, control step) via
    noise.batch_noise -> independent of batch size/grouping, and realizations
    now differ in policy noise (old R1/R2 shared NOISE_SEED + batch offset);
  * solver (euler/midpoint/rk4) and precision (fp32/fp16, SDPA) per unit;
  * synchronized per-call sampling latency.
"""
import argparse, hashlib, json, os, pickle, sys, threading, time

import isaacgym  # noqa: F401
import numpy as np
import torch

ROOT = "/home/jren313/EC-Diffuser-1"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(ROOT, "experiments/scripts"), os.path.join(ROOT, "diffuser")]
import diffuser.utils as utils  # noqa: E402
from diffuser.eval_utils import setup_isaac_env  # noqa: E402
from diffuser.utils import gpu_guard  # noqa: E402
from isaacgym_control import Args, array_to_state_dict, entity_positions  # noqa: E402
from noise import batch_noise, set_key, episode_colors  # noqa: E402
from isaacgym import gymapi  # noqa: E402

from units import UNITS, CALLS, unit_name, is_valid  # noqa: E402
SETS = {"E0": "replicate0_n96.pkl", "E1s": "replicate1_n96.pkl", "E2s": "replicate2_n96.pkl",
        "E3s": "replicate3_n96.pkl", "E4s": "replicate4_n96.pkl"}
SUCCESS_THRESH = 0.04
DATASET = "ecdiffuser-data/push_cubes/3C_randcolor/panda_push_replay_buffer_dlp.pkl"


def sha256(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def install_precision(model):
    """Network-only autocast switch; solver state stays FP32 (output cast back)."""
    if getattr(model, "_prec_wrapped", False):
        return
    orig = model.forward

    def fwd(*a, **k):
        if model._prec == "fp16":
            with torch.autocast("cuda", dtype=torch.float16):
                return orig(*a, **k).float()
        return orig(*a, **k)
    model.forward = fwd
    model._prec, model._prec_wrapped = "fp32", True


def rollout(diff, env, args, ep, set_id, R, nfe, solver, dev, norm):
    nE = env.num_envs; keys = ep["keys"]; n_ep = len(ep["init"])
    H, T = args.horizon, env.horizon
    succ, final_d = np.zeros(n_ep), np.zeros(n_ep)
    acts = np.zeros((n_ep, T, diff.action_dim), np.float32)
    dtr = np.zeros((n_ep, T), np.float32); lat = []
    for bs in range(0, n_ep, nE):
        idx = list(range(bs, min(bs + nE, n_ep))); na = len(idx)
        idx += [idx[-1]] * (nE - na)
        inner = env.env
        if inner.random_obj_colors:          # per-episode colors (see noise.episode_colors)
            for i, e in enumerate(idx):
                for j, key in zip(episode_colors(set_id, e, inner.num_colors, inner.num_objects),
                                  inner._obj_id_dict):
                    inner.gym.set_rigid_body_color(inner.envs[i], inner._obj_id_dict[key], 0,
                                                   gymapi.MESH_VISUAL, inner.object_colors[j])
        rc, inner.random_obj_colors = inner.random_obj_colors, False
        try:
            obs = env.reset(set_init_states=array_to_state_dict(ep["init"][idx], keys, dev),
                            set_goal_states=array_to_state_dict(ep["goal"][idx], keys, dev))
        finally:
            inner.random_obj_colors = rc
        gp = np.asarray(ep["goal"][idx])
        for t in range(T):
            o = np.asarray(obs["achieved_goal"]).reshape(nE, -1)
            g = np.asarray(obs["desired_goal"]).reshape(nE, -1)
            c = {0: torch.as_tensor(norm.normalize(o, "observations"), device=dev, dtype=torch.float32),
                 H - 1: torch.as_tensor(norm.normalize(g, "observations"), device=dev, dtype=torch.float32)}
            x0 = batch_noise(R, set_id, idx, t, (H, diff.transition_dim), dev)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.no_grad():
                s = diff.conditional_sample(c, n_steps=nfe, noise=x0, solver=solver, verbose=False)
            torch.cuda.synchronize(); lat.append((time.perf_counter() - t0) * 1e3)
            a = norm.unnormalize(s.trajectories[:, 0, :diff.action_dim].cpu().numpy(), "actions")
            acts[bs:bs + na, t] = a[:na]
            obs, _, _, _ = env.step(torch.as_tensor(a, device=dev, dtype=torch.float32))
            st = entity_positions(env)
            dtr[bs:bs + na, t] = np.linalg.norm(st[:, 1:, :2] - gp, axis=-1).max(axis=1)[:na]
        final_d[bs:bs + na] = dtr[bs:bs + na, -1]
    succ = (final_d <= SUCCESS_THRESH).astype(np.int8)
    return succ, final_d, acts, dtr, np.asarray(lat)


def monitored(fn):
    me = os.getpid(); mon = {"max": 0, "events": [], "stop": False}

    def poll():
        while not mon["stop"]:
            try:
                if not gpu_guard.monitoring_ok():
                    mon["events"].append({"monitoring_lost": True, "iso": time.strftime("%H:%M:%S")})
                for pid_, nm_, mib_ in gpu_guard.compute_apps():
                    if pid_ != me and mib_ >= 1000:
                        mon["max"] = max(mon["max"], mib_)
                        mon["events"].append({"pid": pid_, "name": nm_, "mib": mib_,
                                              "iso": time.strftime("%H:%M:%S")})
            except Exception:
                pass
            time.sleep(5)
    pre = gpu_guard.snapshot()
    foreign_pre = [a for a in pre["compute_apps"] if a["pid"] != me and a["used_mib"] >= 1000]
    if foreign_pre:
        return None, {"status": "ABORTED_CONTENDED", "foreign": foreign_pre}
    th = threading.Thread(target=poll, daemon=True); th.start()
    out = fn()
    mon["stop"] = True; th.join(timeout=8)
    post = gpu_guard.snapshot()
    foreign_post = [a for a in post["compute_apps"] if a["pid"] != me and a["used_mib"] >= 1000]
    bad = bool(foreign_post) or bool(mon["events"])
    return out, {"status": "INVALID_CONTENDED_DURING" if bad else "VALID",
                 "midset_events": mon["events"][:20], "midset_max_foreign_mib": mon["max"],
                 "foreign_post": foreign_post,
                 "foreign_small_pre": [a for a in pre["compute_apps"] if a["pid"] != me]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="short name, e.g. s42_400k")
    ap.add_argument("--loadpath", required=True)
    ap.add_argument("--epoch", required=True)
    ap.add_argument("--train_seed", type=int, required=True)
    ap.add_argument("--num_entity", type=int, default=3)
    ap.add_argument("--num_envs", type=int, default=16,
                    help="16 = canonical layout (shipped config / training collection); rendering\n"
                         "depends on env grid position, so other values change the scenario")
    ap.add_argument("--configs", required=True, help="nfe:solver:prec,...  e.g. 4:euler:fp16")
    ap.add_argument("--R", default="1,2")
    ap.add_argument("--sets", default="E0,E1s,E2s,E3s,E4s")
    ap.add_argument("--max_episodes", type=int, default=None, help="first K episodes of each set")
    cli = ap.parse_args()
    os.makedirs(UNITS, exist_ok=True)
    cfgs = [(int(n), s, p) for n, s, p in (c.split(":") for c in cli.configs.split(","))]
    Rs = [int(r) for r in cli.R.split(",")]; sets = cli.sets.split(",")
    sfx = "" if cli.max_episodes is None else "_k{}".format(cli.max_episodes)
    todo = [(s, R, c) for s in sets for R in Rs for c in cfgs
            if not is_valid(unit_name(cli.ckpt, cli.num_entity, cli.num_envs, c[0], c[1], c[2], R, s) + sfx)]
    if not todo:
        print("nothing to do"); return 0

    os.chdir(ROOT)
    args = Args(); utils.set_global_device(args.device)
    args.num_entity = cli.num_entity; args.num_envs = cli.num_envs
    env = setup_isaac_env(args); env.horizon = args.max_episode_length
    exp = utils.load_diffusion("data", args.dataset, cli.loadpath, epoch=cli.epoch, seed=args.seed,
                               is_diffusion=True, override_dataset_path=DATASET)
    dev = args.device; norm = exp.dataset.normalizer; diff = exp.ema; diff.eval()
    ck_path = os.path.realpath(os.path.join(ROOT, "data", args.dataset, cli.loadpath,
                                            "state_{}.pt".format(cli.epoch)))
    meta = dict(ckpt=cli.ckpt, train_seed=cli.train_seed, loadpath=cli.loadpath, epoch=cli.epoch,
                ckpt_path=ck_path, ckpt_sha256=sha256(ck_path), internal_step=int(exp.trainer.step),
                weights="ema", num_entity=cli.num_entity, batch=cli.num_envs,
                episode_length=int(env.horizon), noise="per_episode_v2", colors="per_episode_v1", sdpa=True,
                env_spacing=float(env.env.cfg["env"]["envSpacing"]),
                layout="canonical_16" if cli.num_envs == 16 and env.env.cfg["env"]["envSpacing"] == 1.5
                else "noncanonical")
    install_precision(diff.model)
    for b in diff.model.particle_transformer.blocks:
        assert b.attn.use_sdpa, "SDPA expected to be the default"

    cache = {}
    for s, R, (nfe, solver, prec) in todo:
        name = unit_name(cli.ckpt, cli.num_entity, cli.num_envs, nfe, solver, prec, R, s) + sfx
        if s not in cache:
            fn = SETS[s] if cli.num_entity == 3 else "c{}_{}".format(cli.num_entity, SETS[s])
            ep = pickle.load(open(os.path.join(ROOT, "experiments/isaacgym_episode_sets", fn), "rb"))
            if cli.max_episodes:
                ep = dict(ep, init=ep["init"][:cli.max_episodes], goal=ep["goal"][:cli.max_episodes])
            cache[s] = (fn, ep)
        fn, ep = cache[s]
        diff.model._prec = prec
        t0 = time.time()
        out, val = monitored(lambda: rollout(diff, env, args, ep, set_key(ep["sha256"]), R,
                                             nfe, solver, dev, norm))
        rec = dict(meta, unit=name, set=s, set_file=fn, set_sha256=ep["sha256"], realization=R,
                   nfe=nfe, solver=solver, calls_per_plan=nfe * CALLS[solver], precision=prec,
                   wall_s=time.time() - t0, **val)
        if out is not None:
            succ, fd, acts, dtr, lat = out
            rec.update(n=int(len(succ)), successes=int(succ.sum()), success=float(succ.mean()),
                       lat_ms_mean=float(lat.mean()), lat_ms_p50=float(np.median(lat)),
                       lat_ms_p95=float(np.percentile(lat, 95)), n_calls_timed=int(len(lat)))
            np.savez_compressed(os.path.join(UNITS, name + ".npz"), success=succ, final_d=fd,
                                actions=acts, d=dtr.astype(np.float16), lat_ms=lat.astype(np.float32))
        json.dump(rec, open(os.path.join(UNITS, name + ".json"), "w"), indent=1)
        print("{} {} success={} lat={:.1f}ms wall={:.0f}s".format(
            name, rec["status"], rec.get("success"), rec.get("lat_ms_mean", float("nan")), rec["wall_s"]),
            flush=True)
        if rec["status"] != "VALID":
            print("contention: stopping so the runner can wait for a clean GPU", flush=True)
            return 7
    return 0


if __name__ == "__main__":
    sys.exit(main())
