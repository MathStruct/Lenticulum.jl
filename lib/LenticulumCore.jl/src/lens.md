# lens.jl — implementation note

Implements: [[Bayesian Lens]] (Defs. 9, 10), [[Composition of Bayesian Lenses]] (Defs. 12, 15).

## The idea in one comparison

$$\text{autodiff:}\quad (g \circ f)^*(a, \bar c) = f^*\bigl(a,\ g^*(f(a), \bar c)\bigr)$$
$$\text{Bayes:}\quad\ \ (d \circ\!\!\!\bullet\; c)'_\pi = c'_\pi \circ\!\!\!\bullet\; d'_{c_*\pi}$$

**The prior is the linearisation point.** A `put` needs the cached forward value; an
`invert` needs the prior it was conditioned on. `BayesianLens(model, inversion)` is the pair.

## The inversion zoo

| type | realisation | family |
|---|---|---|
| `ExactInversion` | Bayes' law | conjugate |
| `AmortisedInversion(net)` | a Lux layer | VAE encoder |
| `SolverInversion(solver)` | root-finding | algebraic, DEQ, NeuralODE |
| `ProximalInversion(prox)` | denoising steps | RED-Diff, ProxDM |
| `TrivialInversion` | nothing to infer | priors, clamped channels |

All five inhabit the same slot. That is the point of [[Bayesian Lens|Definition 9]]: $c'$ is
*any* function of the right type, and its quality is measured by the free energy rather
than enforced by the type. The three solver-ish ones are the three families of
[[Implicit Learners]].

## Implementation difficulties

### 1. A factor has two independently parametrised halves

`AmortisedInversion` holds a Lux layer with its own `ps`. So does the forward model. This is
the structural reason a factor cannot be an `AbstractLuxLayer`: a Lux layer has one
direction and one parameter tree; a Bayesian lens has two directions and (often) two trees.

Practical consequence: the parameter tree of a factor is naturally
`(; model = ..., inversion = ...)`, and the two halves may want **different optimisers** —
in a VAE the encoder and decoder are usually trained together, but in wake-sleep or in
amortised inference with a fixed generative model they are not. The interface must not
assume a single `ps`.

### 2. `invert` must return $X \times \llbracket c \rrbracket$, not just $X$

Easy to get wrong, and it fails silently: dropping the latent part gives a belief of the
right *type* that breaks the chain rule, because the next factor upstream consumes exactly
that latent. Documented on the `invert` docstring; will need a test once a concrete
inversion exists.

### 3. `TensorLens` is lossy and there is no way around it

[[Composition of Bayesian Lenses|Remark 16]]: parallel composition of inversions can only
feed each branch the **marginal** of a joint prior, so $(c \otimes d)^\dagger \neq c^\dagger \otimes d^\dagger$
and the composite is mean-field. The gap is the mutual information between branches
([[Composition of Statistical Games|Remark 26]]).

This is not a defect to fix. It is the formal content of "mean-field VI is wrong, and by
this much". The honest options are (a) accept it and report the gap, or (b) refuse to factor
across correlated branches and keep a joint inversion. Both should be available at the graph
level. Currently `TensorLens` just holds the parts and the docstring warns; the gap is not
computed.

### 4. Exact inversions are only *almost surely* well defined

Footnote 3 of the paper: inversions may not be fully supported and are defined only up to
a.s. equality, so $(-)^\dagger$ is only a.s. a pseudofunctor. Numerically this is
"conditioning on a null set" — a zero denominator in Bayes' law. `ExactInversion` will need
a support check and a policy (throw? return `TrivialBelief`? widen with a floor?) before it
does anything. Unresolved.

### 5. `compose` argument order

`compose(c, d)` builds $d \diamond c$ — the arguments follow the **wiring** (first `c`, then
`d`) while the mathematical notation is written right-to-left. This will confuse anyone
reading the paper with the code open, so the docstring says it explicitly. The alternative,
matching the notation, would confuse everyone else.

Related: [[Bayesian Lens]], [[Composition of Bayesian Lenses]], [[Implicit Learners]], [[open_model]]
