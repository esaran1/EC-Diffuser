import os
import copy
import math
import numbers
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
        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=1, shuffle=True, pin_memory=True
        ))
        self.dataloader_vis = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        ))
        self.renderer = renderer
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

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

            self.optimizer.step()
            if should_diagnose:
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
            'ema': self.ema_model.state_dict()
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
