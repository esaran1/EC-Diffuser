"""Frozen learned action representation for EC-Diffuser (Tal: fully latent).

Pipeline: normalized raw action -> frozen VAE encoder (posterior mean) ->
latent, rescaled to [-1, 1] by the latent's own dataset limits -> EC-Diffuser
generates in that space -> frozen decoder -> normalized raw action. The VAE is
trained separately and never updated here.
"""
import hashlib
import torch
from torch import nn

from .flow_matching import ConditionalFlowMatching


class _VAE(nn.Module):
    def __init__(self, a_dim=3, z_dim=3, h=128):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(a_dim, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(),
                                 nn.Linear(h, 2 * z_dim))
        self.dec = nn.Sequential(nn.Linear(z_dim, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(),
                                 nn.Linear(h, a_dim))


class FrozenActionCodec(nn.Module):
    def __init__(self, path):
        super().__init__()
        ck = torch.load(path, map_location="cpu")
        self.z_dim = int(ck["z_dim"])
        self.vae = _VAE(3, self.z_dim)
        self.vae.load_state_dict(ck["state_dict"])
        for p in self.vae.parameters():
            p.requires_grad_(False)
        self.vae.eval()
        self.register_buffer("z_min", ck["z_min"].float())
        self.register_buffer("z_max", ck["z_max"].float())
        self.sha16 = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]

    @torch.no_grad()
    def encode(self, a):
        mu = self.vae.enc(a).chunk(2, -1)[0]
        return 2.0 * (mu - self.z_min) / (self.z_max - self.z_min) - 1.0

    @torch.no_grad()
    def decode(self, z):
        mu = (z + 1.0) * 0.5 * (self.z_max - self.z_min) + self.z_min
        return self.vae.dec(mu)


class LatentActionFlowMatching(ConditionalFlowMatching):
    """Training is unchanged (the dataset already yields latent actions); the
    sampler decodes the action slice so every consumer sees normalized raw
    actions, exactly as for the raw-action model."""

    action_space = "latent"

    def __init__(self, *args, action_vae_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        if action_vae_path is None:
            raise ValueError("action_vae_path is required")
        # Held OUTSIDE the module tree: not in state_dict, not in parameters(),
        # so checkpoints are structurally identical to the raw-action model and
        # the strict warm start from canonical is unaffected. Moved lazily.
        object.__setattr__(self, "_codec", FrozenActionCodec(action_vae_path))
        self.action_vae_path = action_vae_path
        self.action_vae_sha16 = self._codec.sha16
        if self._codec.z_dim != self.action_dim:
            raise ValueError("latent dim {} != action slot dim {}".format(
                self._codec.z_dim, self.action_dim))

    @property
    def codec(self):
        dev = self.loss_weight_matrix.device
        if self._codec.z_min.device != dev:
            self._codec.to(dev)
        return self._codec

    def decode_actions(self, traj):
        out = traj.clone()
        out[..., :self.action_dim] = self.codec.decode(traj[..., :self.action_dim]).to(traj.dtype)
        return out

    @torch.no_grad()
    def conditional_sample(self, cond, *args, **kwargs):
        s = super().conditional_sample(cond, *args, **kwargs)
        return type(s)(self.decode_actions(s.trajectories), s.values, s.chains)
