from collections import namedtuple
import numpy as np
np.set_printoptions(precision=3, suppress=True)
import torch
import pdb

from .preprocessing import get_preprocess_fn
from .d4rl import load_environment, sequence_dataset
from .normalization import DatasetNormalizer
from .buffer import ReplayBuffer


Batch = namedtuple('Batch', 'trajectories conditions')
ValueBatch = namedtuple('ValueBatch', 'trajectories conditions values')

class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_path='', dataset_name='panda_push', horizon=64, obs_only=False,
        normalizer='LimitsNormalizer', particle_normalizer='ParticleGaussianNormalizer', preprocess_fns=[], max_path_length=1000,
        max_n_episodes=5000, termination_penalty=0, use_padding=True, overfit=False, action_only=False, single_view=False, **kwargs):
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, dataset_name)
        self.dataset_path = dataset_path
        self.horizon = horizon
        self.obs_only = obs_only
        self.action_only = action_only
        self.max_path_length = max_path_length
        self.use_padding = use_padding

        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        assert dataset_path, 'Dataset path must be provided'
        fields.load_paths_from_pickle(dataset_path, single_view=single_view and 'kitchen' not in dataset_path)
        if overfit:
            fields._count = 1
        fields.finalize()
        self.successful_episode_idxes = fields.successful_episode_idxes
        self.normalizer = DatasetNormalizer(fields, normalizer, particle_normalizer=particle_normalizer, path_lengths=fields['path_lengths'])
        self.indices = self.make_indices(fields.path_lengths, horizon)

        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        if 'kitchen' in dataset_path:
            self.normalize(keys=['observations', 'actions'])
        else:
            self.normalize()
        self.particle_dim = fields.observations.shape[-1]
        self.observation_dim = fields.normed_observations.shape[-1]
        self.action_dim = fields.normed_actions.shape[-1]
        print(f'[ datasets/sequence ] Dataset fields: {self.fields}')
        print(f'[ datasets/sequence ] Dataset normalizer: {self.normalizer}')

    def normalize(self, keys=['observations', 'actions', 'goals']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            if key == 'observations' or key == 'goals':
                array = self.fields[key]    # (n_episodes, max_path_length, n_entities, dim)
                normed = self.normalizer(array, 'observations')
            else:
                array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
                normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)
    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        return {0: observations[0]}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        path_ind, start, end = self.indices[idx]

        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]

        conditions = self.get_conditions(observations)
        if self.obs_only:
            trajectories = observations
        else:
            trajectories = np.concatenate([actions, observations], axis=-1)
        batch = Batch(trajectories, conditions)
        return batch


class GoalDataset(SequenceDataset):

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = path_length - 1
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices


    def __getitem__(self, idx):
        path_ind, start, end = self.indices[idx]
        path_length = self.fields.path_lengths[path_ind]
        # hindsight goals for unsuccessful episodes
        if path_ind in self.successful_episode_idxes:
            goal = self.fields.normed_goals[path_ind, path_length-1:path_length]
        else:
            goal = self.fields.normed_observations[path_ind, path_length-1:path_length]

        # Calculate actual end and determine if padding is needed
        path_length = self.fields.path_lengths[path_ind]
        actual_end = min(end, path_length)
        padding_needed = end - actual_end

        # Fetch observations and actions
        if self.action_only:
            observations = np.concatenate([self.fields.normed_observations[path_ind, start:start+1].repeat(actual_end-1-start, axis=0), 
                                           goal])
        else:
            observations = np.concatenate([self.fields.normed_observations[path_ind, start:actual_end-1],
                                           goal])
        actions = np.concatenate([self.fields.normed_actions[path_ind, start:actual_end-1],
                                  self.normalizer.normalize(np.zeros((1, self.action_dim), dtype=self.fields.normed_actions.dtype), 'actions')])
        # Handle padding
        if padding_needed > 0:
            # Pad observations with the last observation repeated
            last_obs = observations[-1]
            observations = np.vstack([observations] + [last_obs] * padding_needed)
            # Pad actions with zeros
            last_action = actions[-1]
            actions = np.vstack([actions] + [last_action] * padding_needed)

        conditions = self.get_conditions(observations)
        if self.obs_only:
            trajectories = observations
        else:
            trajectories = np.concatenate([actions, observations], axis=-1)
        batch = Batch(trajectories, conditions)
        return batch

    def get_conditions(self, observations):
        '''
            condition on both the current observation and the last observation in the plan
        '''
        return {
            0: observations[0].copy(),
            self.horizon - 1: observations[-1].copy(),
        }

class LatentActionGoalDataset(GoalDataset):
    """GoalDataset whose action channel is a FROZEN learned latent.

    Real actions and the constructed terminal stop target are both mapped
    through the same frozen encoder, so the stop semantics are preserved in
    latent space. Precomputed once; __getitem__ stays deterministic.
    """

    def __init__(self, *args, action_vae_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        import torch
        from diffuser.models.latent_action import FrozenActionCodec
        if action_vae_path is None:
            raise ValueError("action_vae_path is required")
        codec = FrozenActionCodec(action_vae_path)
        self.action_vae_sha16 = codec.sha16
        na = self.fields.normed_actions
        flat = torch.as_tensor(na.reshape(-1, na.shape[-1]), dtype=torch.float32)
        z = torch.cat([codec.encode(c) for c in flat.split(65536)]).numpy()
        self.fields.normed_actions = z.reshape(na.shape).astype(na.dtype)
        raw_term = self.normalizer.normalize(
            np.zeros((1, self.action_dim), dtype=na.dtype), 'actions')
        self._latent_terminal = codec.encode(
            torch.as_tensor(raw_term, dtype=torch.float32)).numpy().astype(na.dtype)

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        path_ind, start, end = self.indices[idx]
        n_real = min(end, self.fields.path_lengths[path_ind]) - 1 - start
        traj = batch.trajectories.copy()
        # rows >= n_real hold the constructed terminal (and its padding copies)
        traj[n_real:, :self.action_dim] = self._latent_terminal
        return Batch(traj, batch.conditions)
