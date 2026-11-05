#design

> [[The Inferencer and the Optimizer]] attached two graphs to a factor graph, one over its
> messages and one over its parameters. This note adds the other two parts of a learning system,
> a **regulariser** and a **loss**, and asks for the exact formulation of all four together.
>
> The answer splits them in two:
> - **Inside the model:** the regulariser (prior factors on parameter and structure variables)
>   and the inferencer (an amortised belief). They are factors and beliefs of **one joint
>   distribution**, and one free energy scores them all.
> - **Outside it:** the optimiser and the loss. The optimiser is the *dynamics* that minimises the
>   free energy; the loss is the *choice* of the objective. Neither is a factor of the joint. Each
>   can be learned only in a nested problem, against a criterion it does not control.
>
> Three conditions make the parts compose: the optimiser's fixed points are the objective's
> stationary points; the inferencer is judged by the same free energy; a learned loss or
> regulariser is anchored by a normaliser or by held-out data. Without the third, learning the
> weight of a loss or of a penalty on the training objective drives it to zero (§4, derived).
>
> A design note (2026-11-05, Daniel's question, written up by Claude). Nothing here is implemented
> beyond what §7 lists as existing; nothing is measured.

> Sources: Cruttwell, Gavranović, Ghani, Wilson & Zanasi, *Categorical Foundations of Gradient-Based Learning*, ESOP 2022 (Def. 3.14 update lenses, Example 3.16 momentum); MacKay, *Bayesian Interpolation*, Neural Computation 4(3), 1992 (the evidence for hyperparameters); Mitchell & Beauchamp, *Bayesian Variable Selection in Linear Regression*, JASA 83(404), 1988 (spike-and-slab); Bissiri, Holmes & Walker, *A General Framework for Updating Belief Distributions*, JRSS B 78(5), 2016 (loss-based posteriors); Lorraine, Vicol & Duvenaud, *Optimizing Millions of Hyperparameters by Implicit Differentiation*, AISTATS 2020; Andrychowicz et al., *Learning to Learn by Gradient Descent by Gradient Descent*, NeurIPS 2016; Myers, *Double Categories of Open Dynamical Systems*, 2020. The decomposition and the derivations are original to this vault. code: `lib/Mycelium.jl/src/factors.jl` (`OptimiserFactor`, `LossFactor`, `PriorFactor`), `exposure.jl` (`expose`), `em.jl` (`fit_em`), `lib/LenticulumCore.jl/src/energy.jl` (scalarisations)
>
> Bibliography: [[Bibliography#^cruttwell2022gradient|Cruttwell et al. 2022]] · [[Bibliography#^mackay1992bayesian|MacKay 1992]] · [[Bibliography#^mitchell1988bayesian|Mitchell & Beauchamp 1988]] · [[Bibliography#^bissiri2016general|Bissiri et al. 2016]] · [[Bibliography#^lorraine2020hyperparameters|Lorraine et al. 2020]] · [[Bibliography#^andrychowicz2016learning|Andrychowicz et al. 2016]] · [[Bibliography#^myers2020double|Myers 2020]]
>
> Theory (CT-ML wiki): [Gradient Based Learning with Parametric Lenses](https://mathstruct.org/CategoryTheory-ML-Wiki/Gradient-Based-Learning-with-Parametric-Lenses) · [Parametric Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Parametric-Lens) · [Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy) · [Double Category](https://mathstruct.org/CategoryTheory-ML-Wiki/Double-Category)

## 1. Four roles

| role | acts on | inside the joint? | in Lenticulum today |
|---|---|---|---|
| **model** | data $y$, latents $z$ | yes: $p(y, z \mid \theta, G)$ | the base graph |
| **regulariser** | parameters $\theta$, structure $G$ | yes: $p(\theta \mid \lambda, G)\, p(G)\, p(\lambda)$ | `PriorFactor` on an exposed parameter (`expose`); BIC's penalty in the structure examples |
| **inferencer** | beliefs $b$ over $z$ (and $\theta$) | yes, as a belief: $q_\phi(z \mid y)$ | the message rules; `AmortisedInversion` per factor |
| **optimiser** | the update of $\theta$ | **no**: a dynamics on the objective | `OptimiserFactor` (`GradientDescent`, `Momentum`, `Nesterov`), `fit_em` |
| **loss** | the objective itself | **no**: the choice of the objective | `LossFactor`, the scalarisations (`WeightedSum`, `GradedScalarisation`) |

The first three are parts of one probabilistic model, the last two are how and what one minimises.
The "loss network" of the question is a learned version of the last row, and a learned optimiser a
learned version of the row above it.

## 2. Inside: one joint, one free energy

With parameters and structure as variables (exposed parameters, [[L-SPEC-003 Exposed Parameters and Energy Channels]]),
the joint is

$$
p(y, z, \theta, \lambda, G) = p(y, z \mid \theta, G)\; p(\theta \mid \lambda, G)\; p(G)\; p(\lambda),
$$

and one free energy over a belief $b(z, \theta, \lambda, G)$ scores everything:

$$
F[b] = \mathbb{E}_b\bigl[-\log p(y, z, \theta, \lambda, G)\bigr] - H(b) \;\ge\; -\log p(y).
$$

Its terms sort into the subgraphs. The **model's** factors give $\mathbb E_b[-\log p(y, z \mid \theta, G)]$,
the fit. The **regulariser's** factors give $\mathbb E_b[-\log p(\theta \mid \lambda, G) - \log p(G) - \log p(\lambda)]$,
the penalty. A belief over $\theta$ contributes its entropy, which is what makes the penalty a
complexity term and not only a shrinkage.

**A point for $\theta$.** When $\theta$ is fitted as a point (EM, gradient steps), its belief is a Dirac,
whose entropy is $-\infty$ for continuous $\theta$. The working objective is then

$$
F(b_z, \theta) = \mathbb{E}_{b_z}\bigl[-\log p(y, z \mid \theta, G)\bigr] - H(b_z) - \log p(\theta \mid \lambda, G),
$$

MAP learning: the regulariser is an added energy and no longer a complexity term. That is why
structures cannot be compared by $F(b_z, \theta)$ alone; the structure examples add BIC's penalty
($2F + p \log N$, `examples/structure_clusters.md`), the crude Laplace approximation of the missing
entropy.

**The regulariser subgraph.** It is the factors on $\theta$, $\lambda$ and $G$:
- **Priors on parameters:** a `PriorFactor` on an exposed parameter channel; weight decay is a
  Gaussian prior.
- **Gated inclusion of edges** (spike-and-slab, Mitchell & Beauchamp 1988): a binary gate
  $g_e \in \{0, 1\}$ per edge with $\theta_e = g_e w_e$, a prior $\pi_e$ on the gate and a slab on $w_e$.
  This separates *whether* an edge exists from *how large* it is, so a rarely used but large
  edge is not shrunk away the way a magnitude penalty shrinks it. That is the answer to "valuable
  but rare, or to regularise" in the language of the joint: the gate's posterior
  $b(g_e = 1)$ is the edge's value, and a structure edit is a hard decision on $g_e$.
- **Hyperpriors:** $\lambda$ as a variable with its own prior. Fitted as a point by the evidence
  (type-II maximum likelihood, MacKay 1992), $\lambda^\star = \arg\max_\lambda p(y \mid \lambda)$; with
  the free energy as the bound, $\lambda^\star \approx \arg\min_\lambda \min_b F[b; \lambda]$.

**The inferencer, inside.** An amortised belief $q_\phi(z \mid y)$ is one choice of $b$, so training $\phi$
on $F$ is the same objective, not a new one, and

$$
F[q_\phi] - \min_b F[b] \;\ge\; 0
$$

is the amortisation gap, measurable whenever the message rules reach the minimiser $b^\star$. When
$b^\star$ is the exact posterior, the gap is $\mathrm{KL}(q_\phi \,\|\, b^\star)$. Because $F$ is
bounded below by $-\log p(y)$, an inferencer cannot lower it by cheating: it can only approach the
evidence. This is what [[The Inferencer and the Optimizer]] §4 means by "one objective".

## 3. Outside: the optimiser

An optimiser is a map on parameters and its state,

$$
(s', \theta') = U_\psi\bigl(s, \theta, \nabla_\theta F\bigr),
$$

Cruttwell et al.'s update lens (Def. 3.14). It is not a factor of the joint: it says how to
*move* towards a minimum of $F$, not what $F$ is.

**Condition (O): the fixed points are the stationary points.** $(s, \theta)$ is a fixed point of $U_\psi$
exactly when $\nabla_\theta F = 0$ (with its state at rest). If not, the optimiser minimises something
other than $F$. For Mycelium's rules this holds, from their definitions in `factors.jl`:
- `GradientDescent(η)`: $\theta' = \theta - \eta \bar\theta$ is fixed iff $\bar\theta = 0$.
- `Momentum(η, γ)`: $s' = \gamma s - \eta \bar\theta$, $\theta' = \theta + s'$. Fixed requires $s' = 0$
  (from $\theta' = \theta$) and $s' = s$, so $s = 0$ and then $\eta\bar\theta = 0$.
- `Nesterov(η, γ)`: the same, with the gradient taken at $\theta + \gamma s = \theta$ at rest.

A **learned optimiser** (Andrychowicz et al. 2016) has parameters $\psi$ of its own. They cannot be
trained on $F$'s value at the current step, which every $\psi$ leaves unchanged. They are trained on
an outer criterion: $F$ (or a held-out $F$) after $K$ steps of $U_\psi$. Condition (O) is then not
automatic. A learned rule can have spurious fixed points or none, and it must be checked
(or built in, as "the update is zero when the gradient is").

## 4. Outside: the loss, and why a learned one must be anchored

The loss is the scalarisation $S$ that turns the graded energies into the number being minimised.
The free energy is one choice of $S$: the plain sum of fit, penalty and negative entropy. Others:
- **Weighted fit** (generalised or tempered Bayes, Bissiri et al. 2016): minimising
  $w\, \mathbb E_q[\ell(\theta)] + \mathrm{KL}(q \,\|\, \pi)$ over $q$ gives $q \propto \pi\, e^{-w\ell}$. The loss $\ell$ need
  not be a log-likelihood, and $w$ says how much to trust it against the prior.
- **Risk scalarisations** of an energy belief ([[RQ-007 Risk Scalarisations and Local Decomposition]]).
- **Contrastive losses**, which generate negatives and raise their energy: Lenticulum's
  `RatioFactor` ([[Why Learned Relations Do Not Collapse]]).

A **loss network** is a learned $S_\eta$, or a learned $\ell_\eta$. It cannot be trained by minimising the
loss it computes. The simplest cases show what happens, exactly:

- **The weight of the fit.** With $\ell \ge 0$, the minimum over $q$ of the tempered objective is
  $-\log Z(w)$, $Z(w) = \int \pi\, e^{-w\ell}$, and
  $\frac{d}{dw}\bigl(-\log Z(w)\bigr) = \mathbb E_{q_w}[\ell] \ge 0$.
  The minimum over $w$ is at $w = 0$: trust no data.
- **The weight of a penalty.** For an unnormalised penalty $\lambda R(\theta)$ with $R \ge 0$, the envelope
  theorem gives $\frac{d}{d\lambda} \min_\theta\bigl[\ell(\theta) + \lambda R(\theta)\bigr] = R(\theta^\star) \ge 0$:
  the minimum is at $\lambda = 0$, no regularisation.
- **The normalised version does not collapse.** As a Gaussian prior, weight decay is
  $-\log \mathcal N(\theta; 0, \lambda^{-1} I) = \tfrac{\lambda}{2}\lVert\theta\rVert^2 - \tfrac d2 \log \lambda + \text{const}$.
  The log-normaliser $-\tfrac d2 \log\lambda$ pushes back; the evidence has an interior optimum
  (MacKay 1992).

This is LP-022's collapse one level up
([[LP-022 Constraint factors are normalised in their residual, not their channels]]): a learned part
that is not normalised can lower the objective by giving up, instead of by fitting. The cures are
the same two:

**Condition (L): a learned loss or regulariser is anchored.** Either it is a normalised density
in what it scores (then the evidence may choose its parameters), or its parameters are chosen by
an outer criterion it does not control: a held-out free energy or likelihood, or a task metric.
A learned $S_\eta$ that is no density, such as the fit weight $w$, has only the second option.

## 5. The exact statement: a nested problem

$$
\min_{\psi,\, \eta,\, \lambda}\; \mathcal L_{\text{outer}}\bigl(\theta^\star, b^\star\bigr)
\quad \text{subject to} \quad
(\theta^\star, b^\star) = \text{a fixed point of } U_\psi \text{ and the inferencer on } S_\eta\bigl(E(b, \theta)\bigr) - H(b) + R_\lambda(\theta).
$$

- **Inner problem:** model, regulariser and inferencer in one objective, solved by the optimiser.
- **Outer problem:** the optimiser's $\psi$, the loss's $\eta$, the regulariser's $\lambda$, judged by
  $\mathcal L_{\text{outer}}$: the evidence (for normalised parts) or held-out data (for everything else).
- **Hypergradients.** At an inner minimum, the implicit function theorem gives (Lorraine et al. 2020)

$$
\frac{d\mathcal L_{\text{outer}}}{d\lambda}
= \frac{\partial \mathcal L_{\text{outer}}}{\partial \lambda}
- \frac{\partial \mathcal L_{\text{outer}}}{\partial \theta}\,
\bigl(\nabla^2_{\theta\theta} F\bigr)^{-1}\, \nabla^2_{\theta\lambda} F .
$$

  Lenticulum already differentiates through fixed points this way (the implicit learners'
  adjoints, the DEQ factors), so the machinery is the same.

The conditions of composition, in one place:

| | condition | if it fails |
|---|---|---|
| (O) | the optimiser's fixed points are $F$'s stationary points | it minimises something else |
| (I) | the inferencer is trained on the same $F$ (the gap $\ge 0$) | it can lower a different objective than the one that is reported |
| (L) | a learned loss or regulariser is normalised, or set by an outer criterion | it gives up: weights go to zero, penalties vanish |

When the inner problem has several minima (switches, mixtures), $\theta^\star$ is a choice of minimum
and the hypergradient is local to it. The outer problem then inherits the inner problem's
restarts and beams (as in the structure search).

## 6. Where structure search sits

Structure $G$ is a discrete variable of the joint, so structure search is inference over $G$ with
$\theta$ integrated out, approximately: the score is $\min_b F$ plus the complexity term the point
$\theta$ lacks (BIC today). Three ways to make it sharper, all from §2 and §5:
- **gates instead of edits:** a soft belief $b(g_e)$ per candidate edge, inferred with everything
  else; a structure edit becomes the rounding of a gate;
- **a Laplace evidence instead of BIC**, using the Hessian the hypergradients need anyway;
- **held-out scoring**, which also judges structures whose parts are not normalised densities.

And the outer problem's own structure, a beam of inner solutions, is what
`examples/structure_clusters.jl` already runs by hand.

## 7. What exists, and what would be built

| part | exists | missing |
|---|---|---|
| model, regulariser as factors | `PriorFactor`, `expose` (L-SPEC-003 §1) | gates per edge; hyperpriors with an M-step for $\lambda$ |
| inferencer | message rules; `AmortisedInversion` per factor | a graph-wide amortised belief; the gap reported |
| optimiser | `OptimiserFactor`, three rules satisfying (O); `fit_em` | a `train!` that interleaves inference and updates (`factors.md` §5); a check of (O) for a user's rule |
| loss | `LossFactor`, scalarisations | learned scalarisations with (L) enforced: a warning, as for LP-022, when a learned weight's own objective is used |
| outer problem | restarts, beams, refits in the examples | held-out scores; hypergradients through `fit_em` and the message fixed point |

The rows are in [[Feature Backlog]] ("Learning" and "Structure search and editing").

Related: [[The Inferencer and the Optimizer]], [[The Two-Part Diagram]], [[Everything is a Factor]],
[[Unusual Wirings]], [[L-SPEC-003 Exposed Parameters and Energy Channels]],
[[Why Learned Relations Do Not Collapse]], [[Energy-Based Learning]],
[[RQ-007 Risk Scalarisations and Local Decomposition]], [[L-SPEC-010 Structure Edits]]
