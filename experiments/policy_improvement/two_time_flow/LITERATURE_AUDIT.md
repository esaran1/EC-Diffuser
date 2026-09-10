# Literature Audit — three recent MeanFlow-family works

Read from primary sources only (arXiv full text / PDF, project pages, code repos).
Prior summaries were not used. Where a source is silent, "NOT STATED".

**Headline finding: the three papers address three *different* failure mechanisms.
None of them is the same phenomenon, and only one of the three even uses the word
"collapse" for a representation-space effect.** Merging them under "collapse"
would be a category error.

| | arXiv 2602.13810 | OMP | DM1 |
|---|---|---|---|
| failure | boundary-condition underdetermination | gradient starvation → velocity-magnitude collapse | representation collapse (encoder) |
| uses word "collapse"? | **no — 0 occurrences** | yes, only as "magnitude collapse" | yes, 20+, means embedding similarity |
| fix | auxiliary diagonal loss `L_IVC` | cosine `L_DA` + finite-difference DDE | InfoNCE/hinge/cov dispersive loss |
| failure measured empirically? | **no — theory only** | yes, but on synthetic toys | **no — asserted, never measured** |
| code | none found | none (dead "coming soon" links) | yes, `Guowei-Zou/dm1-release` |

---

## 1. arXiv:2602.13810 — MVP with Instantaneous Velocity Constraint (IVC)

**Identity.** "Mean Flow Policy with Instantaneous Velocity Constraint for One-step
Action Generation". Zhan, Tao, Wang, Wang, Li, Chen, Li, Tomizuka, Li (Tsinghua /
BAIR / HKU). Header reads "Published as a conference paper at ICLR 2026 (Oral, top
1%)", OpenReview `mIeKe74W43`. v1 14 Feb 2026, v2 8 Mar 2026. Method **MVP**, fix
**IVC**.

**Base method.** MeanFlow, in an offline-to-online RL actor-critic. Mean velocity
`u(a(t),t,r,s) := (1/(r-t)) ∫_t^r v(a(τ),τ,s) dτ` (Eq. 6); trains the MeanFlow
identity residual (Eq. 8). Best-of-N action selection by Q; no Q-gradient through
the policy.

**Failure — boundary-condition underdetermination.** The paper **never writes
"collapse"** (0 occurrences, verified; also 0 for "degenerate", "trivial solution",
"spectral"). Verbatim (§1): *"this ODE theoretically suffers from the problem of
multiple solutions due to a lack of explicit boundary conditions, that is, the
value at any boundary point is not enforced."* §3.2: *"any solution learned by
perfectly minimizing the L_MF loss belongs to a family of functions with a unknown
constant C(a,r) … This allows an arbitrary, persistent bias to exist in the
learned u_θ."*

Thm. 2: `Δu(a(t),t,r) = C(a,r)/(r−t)`. Thm. 3: `Δv = lim_{r→t+} C(a,r)/(r−t)`,
which diverges unless `C ≡ 0`.

**Mechanism.** One auxiliary loss on the diagonal (Eq. 15):

```
L_IVC(θ) = E_{t, a(t)} || u_θ(a(t), t, t) − v ||²₂ ,   v = a* − a(0)
L_policy = L_MF + λ · L_IVC ,   λ = 1.0 default
```

No stop-gradient, no JVP in this term, no architecture/sampling change.

**Evidence.** Robomimic + OGBench, 9 sparse-reward tasks. MVP 0.88±0.05 vs QC
0.86±0.05 — **+0.02, inside the reported SD**. The real evidence is the λ ablation:
cube-triple-task4 **0.30±0.21 (λ=0) → 0.52±0.11 (λ=1)**. **The failure mode itself
is never measured** — no measurement of `C(a,r)` or its 1/(r−t) scaling.

**Relevance to our MeanFlow failure: POSSIBLY RELEVANT — and directly testable.**
Our implementation is *boundary* iMF, which parameterizes `v(z,t) := u(z,t,t)`, so
the diagonal is exactly the object IVC constrains. If our boundary behaviour is
badly biased, that is this paper's mechanism. Tested in
`MEANFLOW_FAILURE_DIAGNOSTICS.md` §3.

**Relevance to two-time flow: LOW.** Two time variables, but they are the endpoints
of **one** interval on **one** shared axis. The state `s` is clean conditioning,
never noised. Zero hits for "token", "modality", "per-block", "noise level". The
only structural contact: IVC deliberately trains the `r=t` diagonal that the main
loss never reaches — a precedent for explicitly over-training a degenerate
sub-manifold of a two-time domain.

**Code.** None found (0 "github" occurrences; only URL is a citation). OpenReview
supplementary blocked by bot-check — **unverified**.

---

## 2. OMP — One-step MeanFlow Policy with Directional Alignment

**Identity.** "OMP: One-step MeanFlow Policy with Directional Alignment". Fang,
Huang, Zhao, Weng, Li, Ban (SJTU / Duke Kunshan). arXiv:2512.19347, v3 02 Jun 2026,
**"Accepted as poster of ICML-2026"**. Variants **OMP-JVP** and **OMP-DDE**.

**Base method.** MeanFlow 3D point-cloud manipulation policy (DP3/MP1 lineage),
NFE=1. `L = L_mse + λ_Disp·L_Disp + λ_DA·L_DA` (Eq. 13). **λ values NOT STATED
anywhere**, including appendices — a real reproducibility hole given L_DA is the
headline.

**Failure — gradient starvation → velocity-magnitude collapse.** Three named
pathologies (§4.2): spectral bias (Thm. B.1, `S_u(ω)` decays `O(1/ω²)`), **gradient
starvation** (the dominant claim), and JVP memory cost.

The starvation argument, with `u = ρn̂`, `u* = ρ*n̂*` (Eq. 9):

```
∂L_MSE/∂α = 2 ρ ρ* sin α
```

§4.2.2: *"The magnitude of the gradient responsible for correcting the direction
(α) is multiplicatively coupled to the target magnitude ρ*. … As ρ* → 0, the
angular gradient ∂L/∂α → 0. … The network minimizes loss by simply reducing its
output magnitude ρ to zero, rather than aligning n̂ with n̂*."*

"Collapse" here means **only** magnitude going to zero: Appendix B.2, *"the
optimization dynamics inherently favor magnitude collapse over directional
alignment"*. Not representation collapse, not mode collapse.

**Mechanism.** Directional Alignment (Eq. 12), normalized so `ρ*` cancels:

```
cos α = (v₀ · u) / (||v₀|| · ||u||) ,   v₀ = z_T − z₀
L_DA  = − log( (cos α + 1) / 2 )
```

Plus DDE (Eq. 14), a central finite difference replacing the JVP, ε = 0.005.

**Evidence.** 37 sim tasks (Adroit + Meta-World) + 3 real. Averages: MP1 78.9±2.1,
**OMP-JVP 82.3±1.6**. Gains concentrate on hard tasks (Very Hard 67.2 → 77.8).
Ablation: OMP-JVP − L_DA → 78.9, i.e. **exactly MP1** by construction. Failure
demos are **synthetic toys**, not robot experiments.

**Relevance to our MeanFlow failure: DIRECTLY PLAUSIBLE.** This is the one
mechanism whose signature we can check directly: if our checkpoint shrank output
magnitude without learning direction, we should see norm ratio ≪ 1 with low cosine,
and the effect should worsen at small target magnitude. Tested in
`MEANFLOW_FAILURE_DIAGNOSTICS.md` §2.

**Relevance to two-time flow: LOW as prior art, HIGH as a predicted hazard.**
Zero hits for per-token time, modality-specific noise, inverse/forward dynamics.
But Eq. 9 implies a real hazard for *our* design: in a two-time regime the
per-block target magnitude scales with that block's own time gap, so a block held
near-clean (state in inverse mode, action in world mode) has small `||v₀||` for
that block and sits exactly in the starved regime. **The paper never makes this
argument — it is our inference from their Eq. 9, flagged as such.** L_DA is a
cheap per-block-applicable countermeasure if we ever observe it.

**Code.** None. "Code (Coming Soon)" is a dead `javascript:void(0)` link; GitHub
search returns zero repos. §7 unverifiable.

---

## 3. DM1 — MeanFlow with Dispersive Regularization

**Identity.** "DM1: MeanFlow with Dispersive Regularization for 1-Step Robotic
Manipulation". Zou, Wang, Wu, Qian, Wang, Li (Sun Yat-sen). arXiv:2510.07865v1,
9 Oct 2025. Project page says "AAMAS 2026 Submission"; **the PDF states no venue**.
Tables call the method **"MF+Disp"**.

**Base method.** MeanFlow vision policy, ViT encoder, RoboMimic + real Franka.
Standard MeanFlow identity (Eqs. 4–10) with `sg(·)` on the target.

**Failure — representation collapse (encoder embeddings).** Abstract: *"existing
flow-based policies suffer from representation collapse, the inability to
distinguish similar visual representations"*. §1: *"tighter mathematical
constraints that cause distinct observations to map to overly similar embeddings"*.
Root cause claimed: *"the mean-velocity formulation does not explicitly encourage
diversity among learned representations."*

**Critical caveat: the collapse claim is asserted and never measured.** No t-SNE,
UMAP, cosine statistics, effective rank, or any embedding-diversity metric appears
in the paper. Fig. 1(b)/(c) are schematic. RQ2 ("Does dispersive regularization
prevent collapse?") is answered **only** by downstream success rate. The paper's
central diagnosis is unverified by its own evidence.

**Mechanism.** Auxiliary dispersive loss on three embedding streams (Eq. 16):

```
L_total = L_MF + α_disp · [ L_disp(H^(T)) + L_disp(H^(R)) + L_disp(H^(Cond)) ]
```

Four variants; InfoNCE-cosine (Eq. 12) works best:

```
L_InfoNCE-Cos = − E [ log( exp(hᵢᵀhⱼ /(||hᵢ|| ||hⱼ|| τ)) / Σ_{k≠i} exp(hᵢᵀh_k /(||hᵢ|| ||h_k|| τ)) ) ]
```

Time sampling (Appendix B.2): logit-normal `ξ~N(−0.4, 1)`, `t = max(t̃₁,t̃₂)`,
`r = min(t̃₁,t̃₂)`, and **`r = t` forced for ρ = 0.5 of samples** — identical to our
`boundary_probability=0.5`.

Nomenclature warning: the paper calls `H^(R)` the "noise embedding", but code shows
it is the embedding of the **second time scalar r**, not of the noise ε.

**Evidence.** RoboMimic lift/can/square/transport, 100 episodes/task. At α=0.1,
MF+Disp vs ReFlow/ShortCut: Lift 97.0, Can 80.5, Square 37.5, Transport 53.0.
MF → MF+Disp gains "10–20 percentage points" on the harder three tasks.
**Flag: the "1-Step" title is not what is evaluated** — the best configuration is
**5 steps**.

**Relevance to our MeanFlow failure: POSSIBLY RELEVANT.** Representation collapse
is a distinct hypothesis from OMP's magnitude collapse and is separately testable
(effective rank / pairwise similarity of our outputs). Tested in
`MEANFLOW_FAILURE_DIAGNOSTICS.md` §1.

**Relevance to two-time flow: LOW as prior art, MODERATE as a hazard signal.**
Zero hits for inverse/forward dynamics or world model; observation is clean
conditioning, never noised; `t` and `r` are per-sample scalars broadcast over the
whole action horizon — **no per-token time**. But two things transfer: (a) a
working precedent for embedding a *second* time scalar and concatenating it into
conditioning (`mlp_meanflow.py` builds separate `time_embedding` and
`r_embedding`); (b) a cautionary signal — DM1's authors found the `(r,t)`
conditioning embeddings degenerate enough to need an explicit diversity penalty.
Making the two times **independent** enlarges that conditioning space from an
ordered pair on a line to a 2-D square, so this risk plausibly grows. Worth
measuring in any two-time model, not pre-emptively regularizing.

**Code.** https://github.com/Guowei-Zou/dm1-release (MIT, branch `master`).
Dispersive losses in `model/loss/dispersive_loss.py`; DM1 itself is
`class MeanFlowDispersive` in `model/flow/shortcut_dispersive.py`.

**Four paper/code discrepancies found**, one serious: the dispersive loss is
computed on a **throwaway forward pass with fresh uniform `t, r`** that neither
follows the logit-normal scheme nor enforces `r ≤ t`, and passes clean `action`
where `z_t` belongs — so the regularized embeddings come from a different time
distribution than the flow loss trains on. Also: `_covariance_loss` uses means not
Frobenius sums with `σ_min` hardcoded to 1 and no `λ_cov`; the InfoNCE code uses
non-squared distance over `τ` rather than Eq. 11's squared distance over `2τ²`,
with an inline `# Fixed: added negative sign` comment; and the whole dispersive
block sits in a `try/except` that **silently degrades to vanilla MeanFlow** on any
exception.

---

## Cross-cutting conclusions

1. **"Collapse" is not one phenomenon.** Boundary underdetermination (2602.13810),
   magnitude collapse from gradient starvation (OMP), and encoder representation
   collapse (DM1) are three different failures with three different fixes. Our
   diagnostics test them separately.

2. **Two of the three never measure their own failure.** 2602.13810 argues purely
   theoretically; DM1 asserts representation collapse without a single diversity
   metric. Only OMP demonstrates its mechanism, and only on synthetic toys. We
   should therefore not import any of their regularizers on authority — §17 of the
   brief is right.

3. **None of the three prefigures two-time state/action flow.** All three keep a
   single shared noise level on the action block with clean observation
   conditioning. Zero hits across all three for per-token/per-block noise levels or
   for inverse-vs-forward dynamics in a denoising context. **The two-time direction
   is not anticipated by this literature**, and the MP1/OMP novelty constraint is
   not threatened by it.

4. **What does transfer** is a hazard list for our design, not a method: the
   starvation mechanism (OMP Eq. 9) predicts trouble for whichever block is held
   near-clean; DM1's experience predicts embedding degeneracy risk that grows when
   the time domain becomes 2-D; and both DM1 (ρ=0.5) and IVC deliberately
   over-sample the degenerate diagonal — a direct precedent for our training
   distribution Option B.

## Sources

- https://arxiv.org/abs/2602.13810 · https://arxiv.org/pdf/2602.13810v2
- https://kasumigaoka-utaha.github.io/OMP_web/ · https://arxiv.org/abs/2512.19347 · https://arxiv.org/html/2512.19347v3
- https://guowei-zou.github.io/dm1/ · https://arxiv.org/abs/2510.07865 · https://github.com/Guowei-Zou/dm1-release
