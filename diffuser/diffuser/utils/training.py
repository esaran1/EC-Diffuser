import os
import copy
import math
import numbers
import random
import numpy as np
import torch
import einops
import pdb

from .arrays import batch_to_device, to_np, to_device, apply_dict
from .timer import Timer
from .cloud import sync_logs
import wandb
from tqdm import tqdm

def cycle(dl):
    while True:
        for data in dl:
            yield data


class ResumableRandomSampler(torch.utils.data.RandomSampler):
    """RandomSampler whose FIRST epoch can start at a sample offset.

    Draws from the generator in exactly the same order as RandomSampler (the
    permutation is drawn lazily on the first index request, after the
    DataLoader iterator has drawn its base seed), so a resumed stream is
    identical to an uninterrupted one. Only the first epoch is offset.
    """

    def __init__(self, data_source, generator, skip_first_samples=0):
        super().__init__(data_source, generator=generator)
        self._skip = int(skip_first_samples)

    def __iter__(self):
        skip, self._skip = self._skip, 0
        it = super().__iter__()
        for _ in range(skip):
            next(it)
        yield from it


def reconstruct_dataloader_generator(seed, n_items, batch_size, batches_drawn,
                                     saved_state=None):
    """Exact dataloader-generator state at the START of the in-flight epoch.

    torch 2.1.2 with a supplied generator consumes, per epoch and in order:
      1. the DataLoader iterator's base seed  (dataloader.py: random_(generator=g))
      2. RandomSampler's permutation           (sampler.py:164, used)
      3. a second RandomSampler permutation    (sampler.py:165, `[:n % n]`, discarded)
    The generator is dedicated to this DataLoader, so its state after E full
    epochs is a pure function of (seed, E). Returns (generator_at_epoch_start,
    batches_into_epoch). If `saved_state` is given, verifies that replaying the
    in-flight epoch reproduces the state the interrupted run recorded, and
    raises if it does not: an unverifiable resume is refused, not guessed.
    """
    per_epoch = math.ceil(n_items / batch_size)
    epochs, into = divmod(int(batches_drawn), per_epoch)
    g = torch.Generator()
    g.manual_seed(int(seed))
    for _ in range(epochs):
        torch.empty((), dtype=torch.int64).random_(generator=g)
        torch.randperm(n_items, generator=g)
        torch.randperm(n_items, generator=g)
    epoch_start = g.get_state().clone()
    if saved_state is not None:
        saved = saved_state.cpu() if hasattr(saved_state, "cpu") else saved_state
        probe = torch.Generator()
        probe.set_state(epoch_start)
        candidates = []
        if into == 0:
            candidates.append(probe.get_state().clone())
        torch.empty((), dtype=torch.int64).random_(generator=probe)
        torch.randperm(n_items, generator=probe)
        candidates.append(probe.get_state().clone())
        torch.randperm(n_items, generator=probe)
        candidates.append(probe.get_state().clone())
        if not any(torch.equal(c, saved) for c in candidates):
            raise RuntimeError(
                "exact-resume check FAILED: replaying dataloader seed {} for {} "
                "batches does not reproduce the recorded generator state. "
                "Refusing to resume with an unverified data stream.".format(
                    seed, batches_drawn))
    g.set_state(epoch_start)
    return g, into

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        renderer,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        results_folder='./results',
        n_reference=8,
        bucket=None,
        max_grad_norm=None,
        collect_step_diagnostics=False,
        adam_betas=(0.9, 0.999),
        lr_warmup_steps=0,
        dataloader_seed=None,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
        if dataloader_seed is not None and (
            isinstance(dataloader_seed, bool) or not isinstance(dataloader_seed, int)
        ):
            raise TypeError("dataloader_seed must be an integer or None")
        dataloader_generator = None
        if dataloader_seed is not None:
            dataloader_generator = torch.Generator()
            dataloader_generator.manual_seed(dataloader_seed)
        self.dataloader_seed = dataloader_seed
        self.dataloader_generator = dataloader_generator
        self.batches_drawn = 0
        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=1,
            shuffle=True, pin_memory=True, generator=dataloader_generator,
        ))
        self.dataloader_vis = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        ))
        self.renderer = renderer
        if (
            not isinstance(adam_betas, (tuple, list))
            or len(adam_betas) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, numbers.Real)
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) < 1.0
                for value in adam_betas
            )
        ):
            raise ValueError("adam_betas must contain two finite values in [0, 1)")
        if isinstance(lr_warmup_steps, bool) or not isinstance(lr_warmup_steps, int):
            raise TypeError("lr_warmup_steps must be a non-negative integer")
        if lr_warmup_steps < 0:
            raise ValueError("lr_warmup_steps must be a non-negative integer")
        self.train_lr = float(train_lr)
        self.adam_betas = tuple(float(value) for value in adam_betas)
        self.lr_warmup_steps = lr_warmup_steps
        self.optimizer = torch.optim.Adam(
            diffusion_model.parameters(), lr=train_lr, betas=self.adam_betas
        )

        self.logdir = results_folder
        self.bucket = bucket
        self.n_reference = n_reference
        if max_grad_norm is not None:
            if isinstance(max_grad_norm, bool) or not isinstance(max_grad_norm, numbers.Real):
                raise TypeError("max_grad_norm must be a finite positive number")
            if not math.isfinite(float(max_grad_norm)) or float(max_grad_norm) <= 0.0:
                raise ValueError("max_grad_norm must be a finite positive number")
        if not isinstance(collect_step_diagnostics, bool):
            raise TypeError("collect_step_diagnostics must be boolean")
        self.max_grad_norm = None if max_grad_norm is None else float(max_grad_norm)
        self.collect_step_diagnostics = collect_step_diagnostics
        self.train_history = []

        self.reset_parameters()
        self.step = 0

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train(self, n_train_steps, front_bg=None, side_bg=None, latent_rep_model=None):
        timer = Timer()
        if not hasattr(self, "train_history"):
            self.train_history = []
        for step in range(n_train_steps):
            step_loss = None
            step_infos = {}
            for i in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)
                self.batches_drawn += 1
                batch = batch_to_device(batch)

                loss, infos = self.model.loss(*batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'non-finite training loss at step {self.step}: {loss}')
                scaled_loss = loss / self.gradient_accumulate_every
                scaled_loss.backward()
                detached_loss = loss.detach() / self.gradient_accumulate_every
                step_loss = detached_loss if step_loss is None else step_loss + detached_loss
                for key, value in infos.items():
                    detached_value = value.detach() if torch.is_tensor(value) else value
                    contribution = detached_value / self.gradient_accumulate_every
                    step_infos[key] = step_infos.get(key, 0) + contribution

            gradient_checks = [
                (name, torch.isfinite(parameter.grad).all())
                for name, parameter in self.model.named_parameters()
                if parameter.grad is not None
            ]
            gradients_are_finite = (
                not gradient_checks
                or torch.stack([check for _, check in gradient_checks]).all().item()
            )
            if not gradients_are_finite:
                nonfinite_gradients = [
                    name for name, check in gradient_checks if not check.item()
                ]
                raise FloatingPointError(
                    f'non-finite gradients at step {self.step}: {nonfinite_gradients[:5]}'
                )

            should_diagnose = bool(
                getattr(self, "collect_step_diagnostics", False)
                and self.log_freq
                and self.step % self.log_freq == 0
            )
            parameters_with_grad = [
                parameter for parameter in self.model.parameters()
                if parameter.grad is not None
            ]
            if should_diagnose:
                gradient_l2_preclip = math.sqrt(sum(
                    float(parameter.grad.detach().double().square().sum())
                    for parameter in parameters_with_grad
                ))
                gradient_max_abs_preclip = max(
                    float(parameter.grad.detach().abs().max())
                    for parameter in parameters_with_grad
                ) if parameters_with_grad else 0.0
                parameter_l2 = math.sqrt(sum(
                    float(parameter.detach().double().square().sum())
                    for parameter in self.model.parameters()
                ))
                parameter_snapshots = [
                    parameter.detach().clone()
                    for parameter in self.model.parameters()
                ]

            max_grad_norm = getattr(self, "max_grad_norm", None)
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    parameters_with_grad, max_grad_norm, error_if_nonfinite=True
                )
            if should_diagnose:
                gradient_l2_postclip = math.sqrt(sum(
                    float(parameter.grad.detach().double().square().sum())
                    for parameter in parameters_with_grad
                ))
                gradient_max_abs_postclip = max(
                    float(parameter.grad.detach().abs().max())
                    for parameter in parameters_with_grad
                ) if parameters_with_grad else 0.0

            lr_warmup_steps = getattr(self, "lr_warmup_steps", 0)
            if lr_warmup_steps:
                learning_rate = self.train_lr * min(
                    float(self.step) / lr_warmup_steps, 1.0
                )
                for parameter_group in self.optimizer.param_groups:
                    parameter_group["lr"] = learning_rate
            self.optimizer.step()
            if should_diagnose:
                step_infos["learning_rate"] = self.optimizer.param_groups[0]["lr"]
                update_l2 = math.sqrt(sum(
                    float((parameter.detach() - before).double().square().sum())
                    for parameter, before in zip(
                        self.model.parameters(), parameter_snapshots
                    )
                ))
                del parameter_snapshots
                step_infos.update({
                    "gradient_l2_preclip": gradient_l2_preclip,
                    "gradient_max_abs_preclip": gradient_max_abs_preclip,
                    "gradient_l2_postclip": gradient_l2_postclip,
                    "gradient_max_abs_postclip": gradient_max_abs_postclip,
                    "parameter_l2": parameter_l2,
                    "update_l2": update_l2,
                    "update_to_parameter_ratio": (
                        update_l2 / parameter_l2 if parameter_l2 else 0.0
                    ),
                })
            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.save_freq and self.step % self.save_freq == 0:
                label = self.step // self.label_freq * self.label_freq
                self.save(label)

            if self.log_freq and self.step % self.log_freq == 0:
                infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in step_infos.items()])
                interval_seconds = timer()
                print(f'{self.step}: {step_loss:8.4f} | {infos_str} | t: {interval_seconds:8.4f}', flush=True)
                record = {
                    "step": int(self.step),
                    "loss": float(step_loss.item()),
                    "interval_seconds": float(interval_seconds),
                }
                record.update({
                    key: float(value.item()) if torch.is_tensor(value) else float(value)
                    for key, value in step_infos.items()
                })
                self.train_history.append(record)
                wandb.log({'step': self.step, 'loss': step_loss, **step_infos})

            if self.step == 0 and self.sample_freq:
                self.render_reference(self.n_reference, front_bg=front_bg, side_bg=side_bg)

            if self.sample_freq and self.step % self.sample_freq == 0:
                self.render_samples(front_bg=front_bg, side_bg=side_bg)

            # --- contention guard (infrastructure only; no tensor work) ---
            monitor = getattr(self, 'contention_monitor', None)
            _force = os.environ.get('EC_TEST_FORCE_CONTENTION_STEP')
            if _force is not None and self.step == int(_force):
                monitor = monitor or type('M', (), {'check': lambda _s: None})()
                _forced = {'forced_test_interrupt': True, 'step': self.step}
            else:
                _forced = None
            if monitor is not None:
                event = _forced or monitor.check()
                if event is not None:
                    self.save(f'contention_{self.step}')
                    self.contention_event = event
                    print(f'[ contention ] foreign GPU process detected at step '
                          f'{self.step}: {event}', flush=True)
                    print('[ contention ] wrote emergency full-state checkpoint; '
                          'exiting cleanly', flush=True)
                    self.step += 1
                    return list(self.train_history)

            self.step += 1
        return list(self.train_history)

    def evaluate(self, n_eval_steps):
        '''
            evaluate model on validation set
        '''
        self.model.eval()
        self.ema_model.eval()
        with torch.no_grad():
            all_losses = []
            all_infos = []
            for step in tqdm(range(n_eval_steps), desc='eval'):
                batch = next(self.dataloader)
                batch = batch_to_device(batch)
                loss, infos = self.model.loss(*batch)
                all_losses.append(loss.item())
                all_infos.append(infos)
            for key in infos:
                mean_val = np.mean([info[key].cpu().numpy() for info in all_infos])
                print(f'{key}: {mean_val:8.4f}')
                wandb.log({'step':self.step, key: mean_val})
            print(f'loss: {np.mean(all_losses):8.4f}')
            wandb.log({'step':self.step, 'loss': np.mean(all_losses)})

    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            # --- full-state resume payload (infrastructure only; does not
            # affect model computation, loss, or optimizer hyperparameters) ---
            'optimizer': self.optimizer.state_dict(),
            'rng': {
                'torch_cpu': torch.get_rng_state(),
                'torch_cuda_all': (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available() else None
                ),
                'numpy': np.random.get_state(),
                'python': random.getstate(),
                'dataloader_generator': (
                    self.dataloader_generator.get_state()
                    if getattr(self, 'dataloader_generator', None) is not None
                    else None
                ),
            },
            'batches_drawn': self.batches_drawn,
            'meta': {
                'dataloader_seed': self.dataloader_seed,
                'train_lr': self.train_lr,
                'adam_betas': self.adam_betas,
                'lr_warmup_steps': self.lr_warmup_steps,
                'gradient_accumulate_every': self.gradient_accumulate_every,
                'batch_size': self.batch_size,
                'full_state_version': 1,
            },
        }
        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}', flush=True)
        if self.bucket is not None:
            sync_logs(self.logdir, bucket=self.bucket, background=self.save_parallel)

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])
        # --- full-state restore; older checkpoints simply lack these keys ---
        if 'optimizer' in data:
            self.optimizer.load_state_dict(data['optimizer'])
        rng = data.get('rng')
        if rng is not None:
            if rng.get('torch_cpu') is not None:
                torch.set_rng_state(rng['torch_cpu'].cpu()
                                    if hasattr(rng['torch_cpu'], 'cpu')
                                    else rng['torch_cpu'])
            if rng.get('torch_cuda_all') is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([t.cpu() if hasattr(t, 'cpu') else t
                                              for t in rng['torch_cuda_all']])
            if rng.get('numpy') is not None:
                np.random.set_state(rng['numpy'])
            if rng.get('python') is not None:
                random.setstate(rng['python'])
            gen_state = rng.get('dataloader_generator')
            if getattr(self, 'dataloader_generator', None) is not None:
                # EXACT resume of the data stream. The generator is replayed
                # from dataloader_seed to the start of the in-flight epoch and
                # the already-consumed batches of that epoch are skipped, so
                # the resumed batch sequence is identical to an uninterrupted
                # run. Verified against the recorded generator state; a
                # mismatch raises rather than resuming on a different stream.
                drawn = int(data.get('batches_drawn', 0))
                gen, into = reconstruct_dataloader_generator(
                    self.dataloader_seed, len(self.dataset), self.batch_size,
                    drawn, saved_state=gen_state)
                self.dataloader_generator = gen
                self.dataloader = cycle(torch.utils.data.DataLoader(
                    self.dataset, batch_size=self.batch_size, num_workers=1,
                    sampler=ResumableRandomSampler(
                        self.dataset, generator=gen,
                        skip_first_samples=into * self.batch_size),
                    pin_memory=True, generator=gen,
                ))
                self.batches_drawn = drawn
                self.resumed_exact = {'batches_drawn': drawn,
                                      'batches_into_epoch': into,
                                      'generator_verified': gen_state is not None}

    #-----------------------------------------------------------------------------#
    #--------------------------------- rendering ---------------------------------#
    #-----------------------------------------------------------------------------#

    def render_reference(self, batch_size=10, front_bg=None, side_bg=None):
        '''
            renders training points
        '''

        ## get a temporary dataloader to load a single batch
        dataloader_tmp = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, num_workers=0, shuffle=False, pin_memory=True
        ))
        batch = dataloader_tmp.__next__()
        dataloader_tmp.close()

        ## get trajectories and condition at t=0 from batch
        trajectories = to_np(batch.trajectories)
        conditions = to_np(batch.conditions[0])[:,None]

        ## [ batch_size x horizon x observation_dim ]
        normed_observations = trajectories[:, :, self.dataset.action_dim:]
        observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')
        savepath = os.path.join(self.logdir, f'_sample-reference.png')
        self.renderer.composite(savepath, observations, front_bg=front_bg, side_bg=side_bg)

    def render_samples(self, batch_size=2, n_samples=2, front_bg=None, side_bg=None):
        '''
            renders samples from (ema) diffusion model
        '''
        for i in range(batch_size):

            ## get a single datapoint
            batch = self.dataloader_vis.__next__()
            conditions = to_device(batch.conditions)

            ## repeat each item in conditions `n_samples` times
            conditions = apply_dict(
                einops.repeat,
                conditions,
                'b d -> (repeat b) d', repeat=n_samples,
            )

            ## [ n_samples x horizon x (action_dim + observation_dim) ]
            samples = self.ema_model(conditions)
            trajectories = to_np(samples.trajectories)

            ## [ n_samples x horizon x observation_dim ]
            normed_observations = trajectories[:, :, self.dataset.action_dim:]

            # [ 1 x 1 x observation_dim ]
            normed_conditions = to_np(batch.conditions[0])[:,None]

            ## [ n_samples x (horizon + 1) x observation_dim ]
            normed_observations = np.concatenate([
                np.repeat(normed_conditions, n_samples, axis=0),
                normed_observations
            ], axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

            savepath = os.path.join(self.logdir, f'sample-{self.step}-{i}.png')
            self.renderer.composite(savepath, observations, front_bg=front_bg, side_bg=side_bg)
