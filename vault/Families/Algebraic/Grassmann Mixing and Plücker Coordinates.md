#comparison #derivation

> Zhang (2025) replaces self-attention by features of the **2-planes spanned by pairs of token
> states**, written in **Plücker coordinates**. The algebra behind the layer is the Grassmannian
> $\mathrm{Gr}(2, r)$ and its Plücker embedding, whose coordinates are the same $2\times2$ minors that
> express independence in algebraic statistics. This note records what the paper shows and what
> it does not, and what Lenticulum can take from it: a deterministic polynomial factor, a known
> variety to test algebraic fitting on, and $\mathrm{Gr}(k, m)$ as a parameter space
> ([[L-SPEC-004 Algebraic Factors Toric and Grassmannian]]).

> Sources: Zhang, *Attention Is Not What You Need*, arXiv:2512.19428v1, 2025 (§3.2 for the layer, §6.4 for the limitations); Edelman, Arias & Smith, *The Geometry of Algorithms with Orthogonality Constraints*, SIAM J. Matrix Anal. Appl. 20(2), 1998; the facts about $\Lambda^2$ and the Plücker relations are standard multilinear algebra, derived below; the comparison with attention and the uses in Lenticulum are original to this vault; code: `lib/AlgebraicFactors.jl/src/plucker.jl`, `lib/AlgebraicFactors.jl/src/polyfit.jl`, `veronese.jl`, `lib/AlgebraicFactors.jl/test/runtests.jl`, `test/polynomial.jl` (forward map, moments, quadrics and real Klein nullspace fits).
>
> Bibliography: [[Bibliography#^zhang2025attention|Zhang 2025]] · [[Bibliography#^edelman1998geometry|Edelman et al. 1998]]
>
> Theory (CT-ML wiki): [Parametric Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Parametric-Lens)

## The layer (arXiv v1, §3.2)

Hidden states $h_t \in \mathbb R^d$ are reduced to $z_t = W_{\text{red}} h_t + b_{\text{red}} \in \mathbb R^r$
($d = 256$, $r = 32$ in the paper). For each offset $\Delta$ in a set $\mathcal W$ (e.g.
$\{1,2,4,8,12,16\}$) with $t + \Delta \le L$, the pair $(z_t, z_{t+\Delta})$ gives

$$
p^{(\Delta)}_{ij}(t) = z_{t,i}\, z_{t+\Delta,j} - z_{t,j}\, z_{t+\Delta,i}, \qquad 1 \le i < j \le r,
$$

normalised to $\hat p = p / \max(\lVert p\rVert, \varepsilon)$, mapped back by $g^{(\Delta)}_t = W_{\text{plü}}\hat p + b$,
averaged over the valid offsets, and gated into $h_t$:
$\tilde h_t = \alpha_t \odot h_t + (1-\alpha_t)\odot g_t$ with $\alpha_t = \sigma(W_{\text{gate}}[h_t; g_t] + b)$; then
LayerNorm and a feed-forward block. There is no softmax and no $L\times L$ score matrix; the cost
is linear in $L$ for fixed $r$ and $|\mathcal W|$.

## The algebra (derivation)

Write $p = z \wedge w \in \Lambda^2\mathbb R^r$, with coordinates $p_{ij}$ for $i<j$.

1. **Rank.** $p = 0$ iff $z$ and $w$ are linearly dependent, i.e. iff all $2\times2$ minors of the
   $r\times2$ matrix $[z\ w]$ vanish: the determinantal condition that expresses independence in
   [[Algebraic Statistics Bridge]]. By Lagrange's identity
   $\lVert p\rVert^2 = \lVert z\rVert^2\lVert w\rVert^2 - \langle z, w\rangle^2$, the squared area of the parallelogram.
2. **Only the plane matters.** $(az + bw)\wedge(cz + dw) = (ad - bc)\, z\wedge w$. So $p$ depends on the
   pair only through the plane it spans and its oriented area. For $p \ne 0$, $p/\lVert p\rVert$ depends
   only on the **oriented** plane ($\pm$ are its two orientations); the paper's
   $p/\max(\lVert p\rVert, \varepsilon)$ has this reading only when $\lVert p\rVert \ge \varepsilon$. This is the $GL_2$ gauge of
   [[The Parameter is a Grassmannian]], with $k = 2$.
3. **Linear maps.** $(Az)\wedge(Aw) = (\Lambda^2 A)(z\wedge w)$. A change of basis in the reduced space
   acts linearly on $p$. Away from the $\varepsilon$ floor, a conformal $A = cQ$ ($c > 0$, $Q$ orthogonal)
   gives $\hat p \mapsto (\Lambda^2 Q)\hat p$, which $W_{\text{plü}}$ can absorb; a general $A$ adds a direction-dependent
   denominator.
4. **The image is a variety.** The **Plücker relations** $p_{ij}p_{kl} - p_{ik}p_{jl} + p_{il}p_{jk} = 0$
   for $i<j<k<l$ cut out the affine cone of decomposable 2-vectors (zero included); removing zero
   and dividing by scalars gives $\mathrm{Gr}(2, r)$, of dimension $2(r-2)$. For $r = 4$ it is one quadric in $\mathbb P^5$ (the Klein
   quadric). For the paper's $r = 32$, the features lie on a cone of dimension $2(r - 2) + 1 = 61$ in $\mathbb R^{496}$.
5. **Compared with attention** (an interpretation). An attention score
   $\langle W_q h_t, W_k h_s\rangle$ is one bilinear form in the pair. The Plücker features are
   $\binom r2$ *antisymmetric* bilinear forms $z_t^\top(e_ie_j^\top - e_je_i^\top) z_s$, combined linearly
   afterwards. The pairing is fixed by the offsets, not chosen by a softmax over positions.

## What the paper shows, and what it does not

- **Results.** Wikitext-2 validation perplexity within about 10–15% of size-matched Transformers
  (13–18M parameters); on SNLI, a Grassmann head on DistilBERT at 0.8538 test accuracy against
  0.8511 for a Transformer head. Single runs as reported, without variance across seeds; small
  scale; slower than attention at the lengths tried. No code is linked.
- **No theorem.** Universal approximation is invoked informally; the claim that the features are
  more interpretable than attention is argued, not measured.
- **Causality, as written.** §3.2 pairs $t$ with $t + \Delta$ and calls this causal, "positions
  strictly to its right (future), consistent with left-to-right language modeling". Read
  literally, the state at $t$ then depends on the token at $t + 1$, which next-token prediction
  at $t$ must not see, and the paper describes no shift or mask that prevents it. Without code
  this cannot be settled. A reproduction should pair $t$ with $t - \Delta$, and the Wikitext-2
  numbers should not be relied on until one has.

## What Lenticulum can take

1. **A Plücker factor.** $(z, w) \mapsto z\wedge w$ is a deterministic polynomial map of degree 2,
   bilinear, with a closed-form Jacobian. Implemented in `AlgebraicFactors` as `plucker`,
   `plucker_jacobian` (columns `vcat(z,w)`), and forward-only `PluckerFactor(r)`.
   `plucker_pairs` specifies lexicographic coordinate order; `plucker_relations` and
   `plucker_residuals` supply the quadrics. Diracs and paired particles transport exactly;
   independent Gaussian inputs give exact moments and a moment-matched Gaussian message,
   subject to the explicit singular-covariance handling in [[plucker]]. The
   inverse polarity remains open: a nonzero bivector $z\wedge w$ determines the pair only up
   to $SL_2$ (a unit-normalised oriented one up to $GL_2^+$, a plane up to $GL_2$), so the inverse
   message is a belief on a fibre, not a point; how to represent it is open, and $p = 0$ needs
   separate treatment.
2. **A variety tested by fitting.** `Klein quadric fit_nullspace` fits all 28 affine
   degree-two features in ℝ⁶ to 400 Gaussian-pair z∧w points, Xoshiro(5106).
   Both monomial and Chebyshev tables give exactly one null direction at
   rtol=1e-12; after basis conversion the fitted coefficients agree up to sign
   with `plucker_relations(4)` (normalised dot-product tolerance 2e-14).
   The monomial singular-value gap is 1.91054415134e14, Chebyshev
   1.21729757389e14; [[polyfit]] records the full run. This settles §3(a)'s
   fitting expectation in [[L-SPEC-004 Algebraic Factors Toric and Grassmannian]].
   Unit-normalised points are expected to add ‖p‖²−1 as a second null direction;
   that case remains unverified by repository tests.
3. **$\mathrm{Gr}(k, m)$ as a parameter space**, for the algebraic factor
   ([[The Parameter is a Grassmannian]]): a Stiefel representative, the projected gradient
   $G - G\Theta^\top\Theta$, a QR retraction (Edelman, Arias & Smith 1998). L-SPEC-004 §2.
4. **A causal Grassmann mixing block** as an experiment for language models. It belongs in
   LenticulumLM, not in this library.

Related: [[The Parameter is a Grassmannian]], [[Algebraic Statistics Bridge]], [[Toric Models as Factors]], [[Algebraic Implicit Learners]]
