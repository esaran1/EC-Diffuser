import warnings
warnings.filterwarnings('ignore')
import os
import json
import time
from diffuser.eval_utils import setup_isaac_env, evaluate_policy, wandb_log_eval_stats
from diffuser.utils.arrays import set_global_device
from diffuser.configuration import diffusion_wrapper_kwargs, flow_sampling_kwargs
import diffuser.utils as utils
import wandb
from diffuser.utils.args import ArgsParser
#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

args = ArgsParser().parse_args('diffusion')
## set default device for data and models to be args.device
set_global_device(args.device)

#-----------------------------------------------------------------------------#
#---------------------------------- dataset ----------------------------------#
#-----------------------------------------------------------------------------#
dataset_config = utils.Config(
    args.loader,
    savepath=(args.savepath, 'dataset_config.pkl'),
    dataset_path=args.dataset_path,
    dataset_name=args.dataset,
    horizon=args.horizon,
    obs_only=args.obs_only,
    action_only=args.action_only,
    normalizer=args.normalizer,
    particle_normalizer=args.particle_normalizer,
    preprocess_fns=args.preprocess_fns,
    use_padding=args.use_padding,
    max_path_length=args.max_path_length,
    overfit=args.overfit,
    single_view=(args.input_type == 'dlp' and not args.multiview),
)


render_config = utils.Config(
    args.renderer,
    savepath=(args.savepath, 'render_config.pkl'),
    env=None,
    num_entity=args.num_entity,
    particle_dim=args.features_dim,
    single_view=(args.input_type == 'dlp' and not args.multiview),
)

dataset = dataset_config()
renderer = render_config()

observation_dim = dataset.observation_dim
action_dim = dataset.action_dim

#-----------------------------------------------------------------------------#
#------------------------------ model & trainer ------------------------------#
#-----------------------------------------------------------------------------#
model_config = utils.Config(
    args.model,
    savepath=(args.savepath, 'model_config.pkl'),
    features_dim=args.features_dim,
    action_dim=action_dim,
    hidden_dim=args.hidden_dim,
    projection_dim=args.projection_dim,
    n_head=args.n_heads,
    n_layer=args.n_layers,
    dropout=args.dropout,
    block_size=args.horizon,
    positional_bias=args.positional_bias,
    max_particles=args.max_particles,
    multiview=args.multiview,
    device=args.device,
)

diffusion_config = utils.Config(
    args.diffusion,
    savepath=(args.savepath, 'diffusion_config.pkl'),
    device=args.device,
    **diffusion_wrapper_kwargs(args, observation_dim, action_dim)
)

trainer_config = utils.Config(
    utils.Trainer,
    savepath=(args.savepath, 'trainer_config.pkl'),
    train_batch_size=args.batch_size,
    train_lr=args.learning_rate,
    gradient_accumulate_every=args.gradient_accumulate_every,
    ema_decay=args.ema_decay,
    sample_freq=args.sample_freq,
    save_freq=args.save_freq,
    label_freq=int(args.n_train_steps // args.n_saves),
    save_parallel=args.save_parallel,
    results_folder=args.savepath,
    bucket=args.bucket,
    n_reference=args.n_reference,
    # Paired training stream: with a seeded generator every arm of a loss
    # screen draws the identical batch sequence, so the only difference
    # between arms is the registered loss change. None (default) reproduces
    # the previous unseeded behaviour.
    dataloader_seed=getattr(args, 'dataloader_seed', None),
    # Fine-tuning knobs. Defaults reproduce the canonical trainer exactly:
    #   step_start_ema=2000 (canonical), lr_warmup_steps=0 (canonical).
    # A warm-start fine-tune MUST pass step_start_ema=0, otherwise step_ema()
    # hard-copies LIVE->EMA for the whole run and destroys the canonical EMA.
    step_start_ema=getattr(args, 'step_start_ema', 2000),
    lr_warmup_steps=getattr(args, 'lr_warmup_steps', 0),
)

#-----------------------------------------------------------------------------#
#-------------------------------- instantiate --------------------------------#
#-----------------------------------------------------------------------------#

model = model_config()

diffusion = diffusion_config(model)

trainer = trainer_config(diffusion, dataset, renderer)

plan_args = ArgsParser().parse_args('plan', savepath=args.savepath)

logger_config = utils.Config(
    utils.Logger,
    renderer=renderer,
    logpath=plan_args.savepath,
    vis_freq=plan_args.vis_freq,
    max_render=plan_args.max_render,
)

## policies are wrappers around an unconditional diffusion model
policy_config = utils.Config(
    plan_args.policy,
    diffusion_model=diffusion,
    normalizer=dataset.normalizer,
    preprocess_fns=plan_args.preprocess_fns,
    verbose=False,
    horizon=plan_args.horizon,
    **flow_sampling_kwargs(diffusion, plan_args.n_diffusion_steps)
)

logger = logger_config()
policy = policy_config()



#-----------------------------------------------------------------------------#
#---------------------------- create environments ----------------------------#
#-----------------------------------------------------------------------------#
env = setup_isaac_env(args)
if args.input_type == 'dlp':
    renderer.env = env

#-----------------------------------------------------------------------------#
#------------------------ test forward & backward pass -----------------------#
#-----------------------------------------------------------------------------#
print('Testing forward...', end=' ', flush=True)
batch = utils.batchify(dataset[0])
loss, _ = diffusion.loss(*batch)
loss.backward()
print('✓')

#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#
wandb_run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            group=args.wandb_group_name,
            config=args,
            sync_tensorboard=False,
            settings=wandb.Settings(start_method="fork"),
        )
savepath_folder_name = args.savepath.split('/')[-1]
wandb_run.name = savepath_folder_name

n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)

# --- optional GPU contention guard (infrastructure only) --------------------
# Enabled with --require_uncontended_gpu; refuses to start on a shared GPU and
# writes a full-state checkpoint then exits cleanly if a foreign GPU process
# appears mid-run. Does not touch model computation (proven in
# experiments/policy_improvement/validate_resume.py and guard_noeffect probe).
if getattr(args, 'require_uncontended_gpu', False):
    from diffuser.utils import gpu_guard
    launch_snapshot = gpu_guard.require_uncontended()
    with open(os.path.join(args.savepath, 'gpu_launch_snapshot.json'), 'w') as fh:
        json.dump(launch_snapshot, fh, indent=1)
    print(f'[ gpu_guard ] launch snapshot: {launch_snapshot["gpu"]}', flush=True)
    trainer.contention_monitor = gpu_guard.ContentionMonitor(
        baseline_pids=[a['pid'] for a in launch_snapshot['compute_apps']])

# ---------------------------------------------------------------------------
# Warm start: initialize the trainable network from the canonical `model` and
# the EMA from the canonical `ema`, i.e. continuation semantics. We deliberately
# do NOT copy ema -> model; they differ materially (relative RMS ~3.66%).
# Adam is NOT restored: the canonical checkpoint predates full-state saving and
# contains only {step, model, ema}, so all arms restart Adam identically.
if getattr(args, 'warm_start_checkpoint', None):
    import torch   # after isaacgym has been imported by eval_utils
    _wpath = args.warm_start_checkpoint
    _wdata = torch.load(_wpath, map_location=args.device)
    trainer.model.load_state_dict(_wdata['model'])
    trainer.ema_model.load_state_dict(_wdata['ema'])
    # `loss_weight_matrix` is a registered BUFFER, so the canonical checkpoint
    # carries its own lambda=1 copy and the load above silently overwrites the
    # correctly lambda-scaled matrix built by the constructor. The scalar
    # `lambda_action` attribute survives, so the arm would look configured
    # while training the baseline objective -- the same byte-identical-arm
    # symptom as the kwargs whitelist defect, from a different cause. Rebuild
    # from the configured knobs and log the correction.
    for _which, _mod in (('model', trainer.model), ('ema', trainer.ema_model)):
        _before, _after = _mod.rebuild_loss_weight_matrix()
        print(f'[ warm_start ] {_which}: loss_weight_matrix rebuilt after load '
              f'(sum {_before} -> {_after}, lambda_action={_mod.lambda_action})',
              flush=True)
    trainer.step = 0                      # fine-tuning step counter starts at 0
    _has_opt = 'optimizer' in _wdata
    print(f'[ warm_start ] {_wpath} (canonical step {_wdata.get("step")}) '
          f'-> model+ema loaded; optimizer_in_ckpt={_has_opt}; Adam restarted',
          flush=True)
    del _wdata

# ---------------------------------------------------------------------------
# MANDATORY fail-loud objective verification, immediately before optimizer
# step 0. `diffusion_wrapper_kwargs` is a whitelist: anything it omits silently
# falls back to a constructor default. That once made four loss arms train the
# same objective and produce byte-identical checkpoints, undetected until after
# the compute was spent. This asserts that the objective the trainer ACTUALLY
# constructed matches the parsed config, and aborts before any training if not.
# It never warns and continues.
from diffuser.utils.objective_guard import assert_objective
_expected_objective = {
    'mask_terminal_action': bool(getattr(args, 'mask_terminal_action', False)),
    'lambda_action': float(getattr(args, 'lambda_action', 1.0)),
    # lambda_action scales the ENTIRE action block, the inherited t=0
    # first-action weight included, so the expected first-action entry is
    # action_weight * lambda (e.g. 10 -> 20 at lambda=2), not action_weight.
    'first_action_weight': float(args.action_weight) * float(getattr(args, 'lambda_action', 1.0)),
}
# Probe the conditioning mask with the REAL batch size and the REAL cond
# timesteps, so the manifest records the denominator this run actually trains
# with rather than a unit-batch stand-in. Cond timesteps come from a real
# dataset item, not an assumption.
_probe_batch = utils.batchify(dataset[0])
_cond_timesteps = tuple(sorted(int(_t) for _t in _probe_batch[1]))
_manifest = assert_objective(
    trainer.model,                       # the wrapper the trainer optimizes
    _expected_objective,
    run_id=args.savepath.rstrip('/').split('/')[-1],
    manifest_path=None,                  # written below, after trainer fields
    cond_timesteps=_cond_timesteps,
    probe_batch=int(args.batch_size),
)
# step_start_ema is trainer-side, not wrapper-side, but a wrong value silently
# destroys the warm-started EMA, so it is recorded and checked here too.
_sse_expected = int(getattr(args, 'step_start_ema', 2000))
_sse_actual = int(trainer.step_start_ema)
_manifest['trainer'] = {
    'step_start_ema_expected': _sse_expected,
    'step_start_ema_actual': _sse_actual,
    'lr_warmup_steps': int(getattr(trainer, 'lr_warmup_steps', 0)),
    'dataloader_seed': getattr(args, 'dataloader_seed', None),
    'train_lr': float(args.learning_rate),
}
if _sse_expected != _sse_actual:
    raise RuntimeError(
        f'step_start_ema mismatch: config {_sse_expected}, trainer {_sse_actual}')
with open(os.path.join(args.savepath, 'RUNTIME_OBJECTIVE_MANIFEST.json'), 'w') as fh:
    json.dump(_manifest, fh, indent=2, sort_keys=True)
print(f'[ objective_guard ] step_start_ema expected={_sse_expected} '
      f'actual={_sse_actual}', flush=True)

from diffuser.utils.run_lock import RunLock
_run_lock = RunLock(args.savepath).acquire()
print(f'[ run_lock ] acquired {_run_lock.path} pid={os.getpid()}', flush=True)

# MANDATORY actual-run divergence evidence. The preflight one-step sentinel is
# NOT a substitute for checking the real process: run a short prefix of genuine
# optimizer steps and save the weights, so an arm that silently reverted to the
# canonical matrix is caught after ~10 steps instead of after 2,000. Written for
# every warm-start run; the cross-arm comparison is done by the runner.
_EARLY_STEPS = 10
if getattr(args, 'warm_start_checkpoint', None):
    trainer.train(n_train_steps=_EARLY_STEPS)
    trainer.save(f'early{_EARLY_STEPS}')
    _early_manifest = {
        'run_id': args.savepath.rstrip('/').split('/')[-1],
        'steps': int(trainer.step),
        'mask_terminal_action': bool(trainer.model.mask_terminal_action),
        'lambda_action': float(trainer.model.lambda_action),
        'loss_weight_matrix_sum': float(trainer.model.loss_weight_matrix.sum().item()),
        'checkpoint': os.path.join(args.savepath, f'state_early{_EARLY_STEPS}.pt'),
    }
    with open(os.path.join(args.savepath, 'EARLY_DIVERGENCE_LOCAL.json'), 'w') as fh:
        json.dump(_early_manifest, fh, indent=2, sort_keys=True)
    print(f'[ early_divergence ] saved {_EARLY_STEPS}-step checkpoint at '
          f'internal step {trainer.step}, matrix sum '
          f'{_early_manifest["loss_weight_matrix_sum"]}', flush=True)
    # Re-verify the objective AFTER real optimizer steps: nothing in training
    # should have restored a foreign matrix.
    assert_objective(
        trainer.model, _expected_objective,
        run_id=_early_manifest['run_id'] + f'-after{_EARLY_STEPS}steps',
        manifest_path=os.path.join(args.savepath,
                                   f'RUNTIME_OBJECTIVE_MANIFEST_after{_EARLY_STEPS}.json'),
        cond_timesteps=_cond_timesteps, probe_batch=int(args.batch_size))

for i in range(n_epochs):
    print(f'Epoch {i} / {n_epochs} | {args.savepath}')
    _remaining = args.n_steps_per_epoch
    if i == 0 and getattr(args, 'warm_start_checkpoint', None):
        _remaining = args.n_steps_per_epoch - _EARLY_STEPS   # keep total at n_train_steps
    trainer.train(n_train_steps=_remaining)

    if getattr(trainer, 'contention_event', None) is not None:
        with open(os.path.join(args.savepath, 'contention_event.json'), 'w') as fh:
            json.dump(trainer.contention_event, fh, indent=1)
        print('[ gpu_guard ] contention detected; full-state checkpoint written, '
              'stopping cleanly.', flush=True)
        break

    if args.eval_freq and i % args.eval_freq == 0:
        plan_args.savepath = logger.savepath = os.path.join(args.savepath, f'epoch_{i}')
        os.makedirs(plan_args.savepath, exist_ok=True)
        stat_save_path = os.path.join(plan_args.savepath, 'eval_stats.pkl')
        eval_stat_dict = evaluate_policy(policy, env, plan_args, logger, num_eval_episodes=100, exe_steps=plan_args.exe_steps, stat_save_path=stat_save_path)
        wandb_log_eval_stats(env, eval_stat_dict, plan_args)

#-----------------------------------------------------------------------------#
#------------------------------- final save ----------------------------------#
#-----------------------------------------------------------------------------#
# The periodic saver fires on `step % save_freq == 0` and labels by
# `step // label_freq * label_freq`. The loop runs steps 0..N-1, so a save
# scheduled exactly at step N never fires and the true final weights are lost.
# Save them explicitly here. This runs AFTER the training loop has finished,
# so it cannot affect training, batch order, or any RNG stream consumed by
# training.
if getattr(trainer, 'contention_event', None) is None:
    trainer.save('final')
    print(f'[ train ] saved FINAL checkpoint at step {trainer.step} '
          f'-> {args.savepath}/state_final.pt', flush=True)
_run_lock.release()
print('[ run_lock ] released', flush=True)

# §13 completion sentinel: written ONLY after a verified final save, so a
# scheduler can advance on artifact evidence instead of process-name matching
# (a stale regex previously blocked the workflow for hours).
if getattr(trainer, 'contention_event', None) is None:
    import hashlib as _hl
    _fp = os.path.join(args.savepath, 'state_final.pt')
    _ck = torch.load(_fp, map_location='cpu')
    _sent = {
        'run_dir': args.savepath,
        'pid': os.getpid(),
        'final_internal_step': int(_ck['step']),
        'expected_steps': int(args.n_train_steps),
        'step_matches': bool(int(_ck['step']) == int(args.n_train_steps)),
        'checkpoint_path': _fp,
        'checkpoint_sha256_16': _hl.sha256(open(_fp,'rb').read()).hexdigest()[:16],
        'end_timestamp': time.time(),
        'end_iso': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'exit_status': 'clean',
    }
    del _ck
    with open(os.path.join(args.savepath, 'RUN_COMPLETE.json'), 'w') as fh:
        json.dump(_sent, fh, indent=1)
    print(f"[ sentinel ] RUN_COMPLETE.json written "
          f"(step {_sent['final_internal_step']}, matches={_sent['step_matches']})",
          flush=True)