# Registers audit — "Vision Transformers Need Registers" (Darcet et al., ICLR 2024)

Source: https://arxiv.org/abs/2309.16588 (camera-ready v2, 12 Apr 2024).
Facts the paper omits (init, positional embeddings) resolved from the official
DINOv2 implementation and labeled as such.

## 1. The pathology

Artifacts in `[CLS]`-to-patch attention maps: bright spurious spots in background
regions. The quantitative signature is **output token L2 norm** — artifact
patches have "roughly 10× higher norm at the output" and are "around 2%" of the
sequence. The norm distribution is **clearly bimodal**.

**Threshold: norm > 150**, hand-picked from the bimodal valley; 2.37% of tokens
exceed it in DINOv2 ViT-g/14. The paper is explicit: "This hand-picked cutoff
value can vary across models." It is **not** a percentile rule — a top-2% rule
applied to a unimodal distribution manufactures outliers that do not exist.

## 2. Where they appear and what they do

On patches **redundant with their neighbors** (cosine similarity to the 4
immediate neighbors measured right after the patch embedding) — typically uniform
background, concentrated near the feature-map border.

Double dissociation (Fig. 5b, Table 1), the load-bearing evidence:

| probe | normal | outlier |
|---|--:|--:|
| position prediction top-1 | 41.7 | 22.8 |
| pixel reconstruction L2 (↓) | 18.38 | 25.23 |
| image classification from ONE token (Aircraft) | 17.1 | **79.1** |

Outliers hold **less local** and **more global** information. Hypothesis: "large,
sufficiently trained models learn to recognize redundant tokens, and to use them
as places to store, process and retrieve global information."

## 3. Conditions of emergence — the critical part for transfer

| condition | finding |
|---|---|
| model size | only **≥ ViT-Large**; nothing in S/B |
| training length | "only appear after one third of training" |
| depth | emerge ~layer 15 of a 40-layer ViT |
| objective | present in DINOv2, OpenCLIP, DeiT-III; **absent in DINO v1 and MAE** |

MAE's absence is attributed to "only a local loss on the patch tokens, rather
than an objective involving global aggregation."

The authors concede the mechanism is **not fully understood**.

## 4. Register design (as specified)

- **K = 4** in all their experiments. Sweep over 0/1/2/4/8/16: "one register is
  sufficient to remove artefacts, using more leads to improved downstream
  performance." 4 is a pragmatic compromise, **not a tuned optimum**.
- Inserted after the patch embedding, between `[CLS]` and patches.
- **No positional embedding** — DINOv2 adds pos-embed *before* concatenating
  registers (code, not paper).
- **No supervision, no auxiliary loss**; "their output value is not used for any
  purpose." Discarded at output, train and test alike.
- Init `nn.init.normal_(std=1e-6)` — effectively near-zero (code, not paper).
- Cost: <2% FLOPs at K=4.

## 5. Benefits — read honestly

Headline wins are on **unsupervised object discovery** (LOST CorLoc, DINOv2
VOC07 35.3 → 55.4). Representation-quality gains are **small**: ImageNet +0.5,
ADE20k mIoU +1.3, NYUd rmse −0.012 for DINOv2; ~zero for DeiT-III; flat or
slightly worse for OpenCLIP. The paper's own framing is defensive — registers "do
not lose performance and sometimes even work better."

Mechanism confirmation (Appendix D): with 1 register, outlier patches vanish and
**the register itself probes at 71.1** on Aircraft (vs outlier patches' 73.3
before) — the global-information role transfers wholesale, while normal patches
are unchanged. **This absorption test is what separates the mechanism claim from
a capacity confound.**

## 6. Where registers did NOT help

- OpenCLIP object discovery got **worse** (38.8 → 37.1); artifacts were removed
  but LOST was robust to them anyway — outliers "live in the null space of the
  value projection layer."
- DeiT-III linear probing gained nothing.
- Register specialization is uncontrolled; register norms are oddly quantized and
  unexplained.
- Appendix A records a self-caught confound: a non-antialiased position-embedding
  interpolation produced a spurious striped pattern — an implementation bug
  masquerading as a phenomenon.

## 7. TRANSFER RISK TO EC-DIFFUSER — the prior should be NEGATIVE

The phenomenon is a **conjunction**, and the paper's own controls show it fails
when any conjunct is missing. Assessed against our architecture (see
`../architecture/ARCHITECTURE_AUDIT.md`):

| assumption | EC-Diffuser | verdict |
|---|---|---|
| long, spatially redundant token sequence with expendable background | 49 tokens/step (1 action + 48 DLP particles), each a meant-to-be-distinct object slot | **fails hardest** — 2% of 49 is 1 token |
| capacity ≥ ViT-Large | 60.6M params, 8 blocks, d=512 | **fails** — smaller than ViT-S/B, which showed nothing |
| very long training | policy fine-tunes are 2k–20k steps | **likely fails** |
| global-aggregation objective | diffusion loss is per-token (MAE-like, argues against); but the action token must aggregate globally (argues for) | **ambiguous** |
| every token is a consumed representation | token roles are heterogeneous; only some are read | **different in kind** |
| additive absolute position embeddings | `pos_emb` is over the **horizon axis only**; particle axis has none | **partly fails** |

**The most important observation:** EC-Diffuser already has a designated
global-aggregation slot — the action token at index 0, which attends over all
particles and feeds a dedicated decoder. That is functionally what a register is.
The plausible reading is that registers are **already effectively present**, so
there may be nothing to fix.

## 8. Methodological traps if we run the diagnostic anyway

1. **Never compare norms across token types.** Action vs DLP tokens differ in
   norm for boring role-related reasons. The diagnostic is only meaningful
   **within a homogeneous population** — across the 48 DLP particles, or one
   token type across many samples.
2. **Require visible bimodality; do not apply a percentile.** If the DLP norm
   histogram is unimodal, the answer is "no artifact" and that is the result.
3. **Diffusion time `t` is an axis with no ViT analogue.** Norms may vary with
   noise level; diagnostics must be run per-`t` or at matched `t`, or
   `t`-dependence will masquerade as outlier structure.
4. **Ported probes need redefinition** — "predict patch position" has no direct
   equivalent. The signature is the **double dissociation**, not either probe.
5. **If we add registers and metrics improve, the causal claim is unsupported**
   unless bimodality was demonstrated first *and* the absorption test passes.
   Otherwise it is just "a slightly bigger model helped."

**Bottom line:** the diagnostic battery is cheap and inference-only, worth running
as a **negative check**. Expect and be prepared to report "no artifact found" —
that is the predicted outcome, and it is publishable. Treat any positive finding
with suspicion until bimodality is visible and absorption is demonstrated.
