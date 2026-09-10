"""Two-time PINT denoiser: separate flow times for state and action tokens.

Weight-compatible retrofit of AdaLNPINTDenoiser. Adds ZERO parameters: the
existing `time_mlp` is reused for both t_state and t_action.

Diagonal equivalence: when t_state == t_action == t, this must reproduce the
parent's output EXACTLY, because
  e_s = e_a = time_mlp(t)                       (same weights, same input)
  per-token add reduces to the original broadcast add
  c = action_embed + (w*e_s + (1-w)*e_a) = action_embed + e   for any w
"""
import torch

from diffuser.models.pint import AdaLNPINTDenoiser


class TwoTimePINTDenoiser(AdaLNPINTDenoiser):
    """AdaLNPINTDenoiser accepting (t_state, t_action).

    Call as forward(x, cond, time, time_action=None). When `time_action` is
    None the module is byte-identical to the parent (single-time) behaviour.
    """

    def __init__(self, *args, adaln_state_weight=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0.0 <= float(adaln_state_weight) <= 1.0:
            raise ValueError("adaln_state_weight must lie in [0, 1]")
        # w in c = action_embed + (w*e_s + (1-w)*e_a). Any w preserves the
        # diagonal, so this is a pure off-diagonal design choice.
        self.adaln_state_weight = float(adaln_state_weight)

    def forward(self, x, cond, time, time_action=None, return_attention=False,
                *, interval=None):
        if time_action is None:
            return super().forward(x, cond, time, return_attention=return_attention,
                                   interval=interval)

        bs, T, f = x.size()
        actions = x[:, :, :self.action_dim]
        x_particles = x[:, :, self.action_dim:].view(bs, T, -1, self.features_dim)

        action_particle = self.action_projection(actions)
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

        # --- the two-time part: shared time_mlp, per-token-group injection ---
        e_state = self._time_embedding(time, interval)          # [bs, D]
        e_action = self._time_embedding(time_action, interval)  # [bs, D]

        x_proj = torch.empty_like(x_cat)
        x_proj[:, :, 0, :] = x_cat[:, :, 0, :] + e_action[:, None, :]
        x_proj[:, :, 1:, :] = x_cat[:, :, 1:, :] + e_state[:, None, None, :]

        w = self.adaln_state_weight
        t_embed = w * e_state + (1.0 - w) * e_action

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
