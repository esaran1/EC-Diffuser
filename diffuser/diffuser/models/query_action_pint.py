"""Direct-action / query-token PINT denoiser (tal_followup Stage C).

Tal's construction: retain noisy/generative STATE tokens, but do NOT feed a
conventionally noised action trajectory. Instead use a learned action/query
token, and map the action-token output directly to the action.

    STANDARD:  noisy state + noisy action  -> denoiser -> state head + action head
    THIS:      noisy state + QUERY token   -> denoiser -> state head + direct action

IMPORTANT (directive section 15): a learned query token carrying no action
observation is NOT the same as t_action=0 (which feeds NOISE) or t_action=1
(which feeds the CLEAN action). Equivalence would have to be demonstrated, not
assumed. This class is kept separate from the independent-time UWM model in all
analyses.

Weight-compatible with AdaLNPINTDenoiser: it adds only the query embedding, and
every other parameter loads from a canonical checkpoint unchanged.
"""
import torch
from torch import nn

from .pint import AdaLNPINTDenoiser


class QueryActionPINTDenoiser(AdaLNPINTDenoiser):
    """PINT denoiser whose action slot is a learned query, not a noised action.

    tie_across_horizon:
        False -> one learned query per horizon position (H x projection_dim)
        True  -> a single shared query broadcast over the horizon
    """

    # parameters absent from a canonical AdaLNPINTDenoiser checkpoint
    warm_start_new_params = ("action_query",)

    def __init__(self, *args, tie_across_horizon=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.tie_across_horizon = bool(tie_across_horizon)
        h = 1 if self.tie_across_horizon else int(kwargs.get("block_size", 5))
        # small init, matching the existing encoding parameters' scale
        self.action_query = nn.Parameter(0.02 * torch.randn(1, h, self.projection_dim))

    def forward(self, x, cond, time, return_attention=False, *, interval=None):
        bs, T, f = x.size()
        # The action slice of x is IGNORED: the query token replaces it. State
        # particles are untouched and remain fully generative.
        x_particles = x[:, :, self.action_dim:].view(bs, T, -1, self.features_dim)

        q = self.action_query
        if self.tie_across_horizon:
            action_particle = q.expand(bs, T, self.projection_dim)
        else:
            if q.shape[1] < T:
                raise ValueError(
                    "action_query has {} horizon slots but got T={}".format(
                        q.shape[1], T))
            action_particle = q[:, :T, :].expand(bs, T, self.projection_dim)
        action_particle = action_particle + self.action_encoding.repeat(bs, T, 1)

        state_particles = self.particle_projection(x_particles)
        if self.multiview:
            n_p = state_particles.size(2) // 2
            v1 = state_particles[:, :, :n_p, :] + self.view1_encoding.repeat(bs, T, n_p, 1)
            v2 = state_particles[:, :, n_p:, :] + self.view2_encoding.repeat(bs, T, n_p, 1)
            new_state_particles = torch.cat([v1, v2], dim=2)
        else:
            new_state_particles = state_particles + self.particle_encoding.repeat(
                bs, T, state_particles.size(2), 1)

        x_cat = torch.cat([action_particle.unsqueeze(2), new_state_particles], dim=2)
        t_embed = self._time_embedding(time, interval)
        x_proj = x_cat + t_embed[:, None, None, :]
        x_proj = x_proj.permute(0, 2, 1, 3)

        if return_attention:
            particles_trans, attn = self.particle_transformer(
                x_proj, action_particle, t_embed, return_attention=True)
        else:
            particles_trans = self.particle_transformer(x_proj, action_particle, t_embed)
        particles_trans = particles_trans.permute(0, 2, 1, 3)

        action_out = self.action_decoder(particles_trans[:, :, 0, :])
        particle_out = self.particle_decoder(particles_trans[:, :, 1:, :])
        particle_out = particle_out.view(bs, T, -1)
        x_out = torch.cat([action_out, particle_out], dim=-1)
        return (x_out, attn) if return_attention else x_out
