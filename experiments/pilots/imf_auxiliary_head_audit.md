# Auxiliary-head iMF source-to-code audit

Primary sources:

- Paper: https://arxiv.org/abs/2512.02012
- Official repository: https://github.com/Lyy-iiis/imeanflow
- JAX commit: `bf60cd7cb653f6628e59d48034b333c5eba445e2`

The implemented objective follows Eq. 12 and the appendix: an auxiliary
prediction `v_theta` supplies the JVP tangent, while separate adaptively
weighted squared losses regress the compound `V_theta` and `v_theta` to the
conditional velocity `noise - data`. The JVP result remains stop-gradient.

Architecture mapping: the official DiT shares early blocks and branches for
its last eight blocks. The trajectory adaptation shares the temporal U-Net
time/interval embeddings, encoder, and bottleneck, then branches into
independently initialized u/v decoders. This is an explicit U-Net adaptation,
not a claim of architectural identity with the released image model. The
auxiliary decoder is training-only; sampling calls only the canonical u path.

Exact parameter counts for the OGBench architecture are 63,282,904 for the
boundary model and 78,415,024 for the auxiliary model: 48,150,784 shared plus
15,132,120 in each decoder. Training parameters increase 23.91%; inference
parameters and model-call count remain equivalent to the boundary model.

The paired pilot fixes data-loader and optimization RNG independently of model
construction so the larger auxiliary branch cannot change minibatch or noise
streams. No simulator result participates in selection.
