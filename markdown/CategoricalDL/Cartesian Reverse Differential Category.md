# Cartesian Reverse Differential Category (CRDC)

> Cruttwell et al., §2.4, Definition 2.6, Proposition 2.7.

## The problem it solves

[[Parametric Lens]] tells us what shape a learner has, but not where the backward map
$f^*$ comes from. For neural networks it is the reverse derivative. A CRDC is the
axiomatisation of "a category in which reverse derivatives exist and behave".

## The data

A CRDC is a Cartesian left-additive category $\mathcal{C}$ (you can add parallel maps, and
there is a zero map) equipped with an operator

$$R[-] \;:\; \mathcal{C}(A, B) \longrightarrow \mathcal{C}(A \times B, A)$$

subject to seven axioms (linearity in the second argument, the chain rule, symmetry of
second partial derivatives, and so on). For $\mathcal{C} = \mathbf{Smooth}$,

$$R[f](a, \bar b) \;=\; J_f(a)^\top \, \bar b$$

the transposed-Jacobian–vector product. **This is exactly a `pullback` / `vjp`**, which is
what Zygote, Enzyme and Mooncake compute.

## The bridge to lenses — Proposition 2.7

> If $\mathcal{C}$ is a CRDC, there is a functor $R : \mathcal{C} \to \mathbf{Lens}(\mathcal{C})$

sending $f : A \to B$ to the lens $(f, R[f]) : (A,A) \to (B,B)$.

**That one sentence is the whole load-bearing structure of gradient-based learning.**
Functoriality of $R$ says $R[g \circ f]$ is the lens composite of $R[f]$ and $R[g]$, which
unpacks to $R[g \circ f](a, \bar c) = R[f](a, R[g](f(a), \bar c))$ — the reverse chain
rule. Autodiff is *correct* precisely because $R$ is a functor.

Applying $\mathbf{Para}(-)$ to it gives

$$\mathbf{Para}(R) \;:\; \mathbf{Para}(\mathcal{C}) \longrightarrow \mathbf{Para}(\mathbf{Lens}(\mathcal{C}))$$

"a parametrised map, plus autodiff, is a parametric lens". A Lux layer is on the left; its
`Zygote.pullback` puts it on the right.

## Why Lenticulum cannot just use this

Because reverse differentiation is not the only functorial backward pass. AutoBayes' whole
point (Theorem 13, Theorem 23) is that **Bayesian inversion and free-energy accumulation
satisfy their own chain rules**, giving different functors into different lens-like
categories. Lenticulum needs both:

| backward pass | forward | backward | chain rule |
|---|---|---|---|
| autodiff | $f : A \to B$ | $R[f] : A \times B \to A$ | Prop. 2.7 |
| Bayesian inversion | $c : X \nrightarrow \llbracket c \rrbracket \times Y$ | $c^\dagger_\pi : Y \nrightarrow X \times \llbracket c \rrbracket$ | [[Bayesian Inversion]] Thm 13 |
| free energy | $l^c$ | $H^c$ | [[Composition of Statistical Games]] Thm 23 |

They stack: a Lenticulum factor's *inversion* is often itself a Lux network trained by
autodiff, so the CRDC layer sits *inside* the statistical-game layer.

Related: [[Lens]], [[Learning Components as Parametric Lenses]], [[Bayesian Inversion]]
