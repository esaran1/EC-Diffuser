"""Looped EC-Diffuser denoiser (adapted from Looped-DiT, arXiv:2609.40305).

Blocks [loop_start, loop_end) are applied n_loops times per denoising step with
shared weights; blocks before run once (prelude), blocks after run once per
decode (coda). XSA is applied in the looped blocks. No new parameters: a
canonical AdaLNPINTDenoiser checkpoint loads strictly. With n_loops=1 and
xsa=False the forward is identical to the parent.
"""
import torch
from .pint import AdaLNPINTDenoiser
from .flow_matching import ConditionalFlowMatching


class LoopedPINTDenoiser(AdaLNPINTDenoiser):
    def __init__(self, *args, loop_start=4, loop_end=8, n_loops=2, loop_xsa=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.loop_start, self.loop_end, self.n_loops = int(loop_start), int(loop_end), int(n_loops)
        for i in range(self.loop_start, self.loop_end):
            self.particle_transformer.blocks[i].attn.xsa = bool(loop_xsa)
        self.collect_loops = False
        self._loop_outputs = []

    def _decode(self, h, c, bs, T):
        pt = self.particle_transformer
        for blk in pt.blocks[self.loop_end:]:
            h = blk(h, c)
        out = pt.head(pt.ln_f(h, c)).permute(0, 2, 1, 3)
        a = self.action_decoder(out[:, :, 0, :])
        p = self.particle_decoder(out[:, :, 1:, :]).reshape(bs, T, -1)
        return torch.cat([a, p], dim=-1)

    def forward(self, x, cond, time, return_attention=False, *, interval=None):
        bs, T, f = x.size()
        actions = x[:, :, :self.action_dim]
        x_particles = x[:, :, self.action_dim:].view(bs, T, -1, self.features_dim)
        action_particle = self.action_projection(actions) + self.action_encoding.repeat(bs, T, 1)
        sp = self.particle_projection(x_particles)
        if self.multiview:
            n = sp.size(2) // 2
            sp = torch.cat([sp[:, :, :n] + self.view1_encoding.repeat(bs, T, n, 1),
                            sp[:, :, n:] + self.view2_encoding.repeat(bs, T, n, 1)], dim=2)
        else:
            sp = sp + self.particle_encoding.repeat(bs, T, sp.size(2), 1)
        t_embed = self._time_embedding(time, interval)
        h = (torch.cat([action_particle.unsqueeze(2), sp], dim=2) + t_embed[:, None, None, :]).permute(0, 2, 1, 3)
        pt = self.particle_transformer
        c = action_particle + t_embed.unsqueeze(1)
        if not pt.positional_bias:
            h = h + pt.pos_emb[:, None, :T, :]
        for blk in pt.blocks[:self.loop_start]:
            h = blk(h, c)
        outs = []
        for r in range(self.n_loops):
            for blk in pt.blocks[self.loop_start:self.loop_end]:
                h = blk(h, c)
            if self.collect_loops and r < self.n_loops - 1:
                outs.append(self._decode(h, c, bs, T))
        self._loop_outputs = outs
        return self._decode(h, c, bs, T)


class LoopedFlowMatching(ConditionalFlowMatching):
    """Deep supervision, Final+Mean weights: final loop weight 1, intermediate
    loops share weight 1 (each 1/(N-1)). Production loss form for every loop."""

    def loss(self, x, cond):
        m = self.model
        m.collect_loops = True
        try:
            loss, info, det = self._compute_flow_loss(x, cond, return_details=True)
            inter = list(m._loop_outputs)
        finally:
            m.collect_loops = False; m._loop_outputs = []
        if inter:
            mask = det["conditioning_mask"]
            w = self.loss_weight_matrix.to(x.device, x.dtype).unsqueeze(0) * mask.to(x.dtype)
            den = mask.sum().to(x.dtype)
            extra = 0.0
            for p in inter:
                e = (p - det["target_velocity"]).abs()
                if self.loss_type == "l2":
                    e = e.square()
                extra = extra + (e * w).sum() / den / len(inter)
            info["loss_final_loop"] = loss.detach()
            info["loss_inter_loops"] = extra.detach()
            loss = loss + extra
        return loss, info
