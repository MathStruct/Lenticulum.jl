# Composition of Gradients — Definition 29, Remark 30

> AutoBayes, Definition 29, Remark 30, and the closing discussion of §5.
> **This is the part that does *not* work perfectly, and knowing why matters.**

## Definition 29

Given $(\Theta, c) : X \multimap Y$ and $(\Phi, d) : Y \multimap Z$, their gradients
$\nabla_\theta F^c$ and $\nabla_\varphi F^d$ compose to form

$$\Bigl(\;\nabla_\varphi F^d\bigl(c(\theta)_*\pi,\, z;\, \varphi\bigr)
\;\;\Bigl|\;\;
\mathop{\mathbb{E}}_{y \sim d'_{c(\theta)_*\pi}(z;\varphi)}\bigl[\nabla_\theta F^c(\pi, y; \theta)\bigr]\;\Bigr)^{\!\top}$$

i.e. stack the downstream gradient with the *expected* upstream gradient, the expectation
taken under the downstream inversion. This is the direct differentiation of
[[Composition of Statistical Games|Theorem 23]].

## Where it fails to be functorial

> The question of functoriality then reduces to whether this expression is equal to
> $\nabla_{\varphi,\theta} F^{dc}$ when evaluated at the same points. Looking back at
> Theorem 23, we see that, in general, **it is not**: the $F^d$ term may depend on $\theta$
> (via the pushforward prior), and the $F^c$ term may depend on $\varphi$ (via $y$).
> However, this just means that the assignment of gradients is **lax**, which can be
> accounted for mechanistically by an implementation.

Unpack the two missing terms. Writing $F^{dc}(\pi,z;\theta,\varphi) = \mathbb{E}_{(y,b)\sim d'_{c(\theta)_*\pi}(z;\varphi)}[F^c(\pi,y;\theta)] + F^d(c(\theta)_*\pi, z;\varphi)$:

$$\nabla_\theta F^{dc} = \underbrace{\mathbb{E}_{(y,b)}\bigl[\nabla_\theta F^c\bigr]}_{\text{kept}}
+ \underbrace{\nabla_\theta \mathbb{E}_{(y,b)\sim d'_{c(\theta)_*\pi}}\bigl[F^c\bigr]}_{\substack{\text{\textbf{dropped}: } \theta \text{ moves the}\\ \text{sampling distribution}}}
+ \underbrace{\nabla_\theta F^d(c(\theta)_*\pi, z;\varphi)}_{\substack{\text{\textbf{dropped}: } \theta \text{ moves the}\\ \text{pushforward prior}}}$$

$$\nabla_\varphi F^{dc} = \underbrace{\nabla_\varphi F^d}_{\text{kept}}
+ \underbrace{\nabla_\varphi \mathbb{E}_{(y,b)\sim d'(\cdot;\varphi)}\bigl[F^c\bigr]}_{\substack{\text{\textbf{dropped}: } \varphi \text{ moves the}\\ \text{sampling distribution}}}$$

So Definition 29 is exactly the **block-diagonal** part of the true Jacobian:

$$\nabla_{(\theta,\varphi)} F^{dc}
= \begin{pmatrix}
\mathbb{E}[\nabla_\theta F^c] & 0 \\[2pt]
\ast & \nabla_\varphi F^d
\end{pmatrix}
+ \begin{pmatrix}
0 & \ast \\[2pt] 0 & 0
\end{pmatrix}$$

and the $\ast$s are what laxness names.

## This is a familiar bug

The dropped terms are the score-function / reparametrisation terms. Every VAE
implementation faces the same thing:

- $\nabla_\theta \mathbb{E}_{q_\theta}[f]$ needs either the **reparametrisation trick**
  (push $\theta$ inside via $y = g_\theta(\epsilon)$) or the **REINFORCE / score-function
  estimator** ($\mathbb{E}[f \nabla_\theta \log q_\theta]$).
- Dropping it entirely is exactly a **stop-gradient** on the sampling path.

So laxness is not an exotic categorical pathology. It is precisely the well-known question
"do you backprop through the sampler?", stated at the right level of generality. And the
paper's answer — "this can be accounted for mechanistically by an implementation" — is the
correct one: **an implementation should choose, per edge, which correction terms to apply,
and record the choice.**

> [!note] Concrete design rule for Lenticulum
> Every edge in the factor graph carries a `GradientCoupling` annotation:
> `Diagonal()` (drop the off-diagonal, i.e. stop-gradient),
> `Reparametrised()` (pathwise, when the inversion is reparametrisable),
> `ScoreFunction()` (REINFORCE, with optional baseline),
> `Exact()` (conjugate / analytic).
> These are the "different **semantics functors**" of the paper's last paragraph: the
> Laplace method, the delta rule, and sampling schemes each give a different compositional
> assignment of gradients to the same syntactic model.

## Remark 30 — the formal home

> Parameterized statistical games form a **monoidal bicategory**. One then couples
> parameter spaces to their **tangent bundles**, and allows for maps back into those
> bundles. This yields a **fibration** over parameterized statistical games, and the
> assignment of gradients is a **lax section** of this fibration.

Two things to extract:

1. "Couples parameter spaces to their tangent bundles, and allows maps back into those
   bundles" is the definition of a [[Parametric Lens|parameter pair $(P, P')$]] with
   $P' = T_pP$. AutoBayes and Cruttwell et al. converge here.
2. "Lax section" is the precise statement of "the gradient assignment is a *choice* that is
   only approximately compatible with composition". A *strict* section would be a
   functorial autodiff. We have a lax one, and that is the honest situation.

## Closing discussion — the algorithmic reality

The paper ends with the practical costs, which are Lenticulum's actual engineering agenda:

1. **Priors propagate by pushforward** (marginalisation), as expensive as exact inversion.
   Remedy: belief propagation / variational message passing — *and the paper states this
   fits into the framework*. This is Mycelium.jl.
2. **Information propagates backwards by sampling from / taking expectations under the
   posteriors.** Mathematically correct, computationally awkward, especially when
   optimising the posteriors' own parameters.
3. **Conjugacy** (Khan & Lin) simplifies both, but pushforward does not preserve conjugate
   families, so you need moment-matching projections back into the family — which means
   annotating games with predicates ("this wire carries a Gaussian"), compositionally but
   again **laxly**.

Related: [[Parameterized Statistical Game]], [[Composition of Statistical Games]], [[Learning Components as Parametric Lenses]]
