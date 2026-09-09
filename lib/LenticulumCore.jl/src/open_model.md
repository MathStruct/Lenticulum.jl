# open_model.jl — implementation note

Implements: [[Open Model]] (AutoBayes Def. 1) and the belief types that flow along the
graph.

## The one idea

$$c : X \rightsquigarrow \llbracket c \rrbracket \times Y$$

The latent space $\llbracket c \rrbracket$ exists so that
[[Composition of Open Models|composition]] can *file away* the intermediate value instead of
integrating it out:

$$(q \circ\!\!\!\bullet\; p)(ds,dy,dt,dz \mid x) = q(dt,dz\mid y)\,p(ds,dy\mid x)$$

No integral. `OpenModelResult(observed, latent)` is the runtime witness of this: a forward
pass returns both, and the `latent` field is exactly an autodiff tape entry. Same trade of
memory for tractability, under [[Bayesian Inversion|the same chain rule]].

## Interface

| function | paper | note |
|---|---|---|
| `forward(m, x, ps, st)` | sample $c$ | returns `OpenModelResult` |
| `logdensity(m, x, a, y, ps, st)` | $\log p_c(a,y\mid x)$ | needed whenever the energy is a NLL |
| `pushforward(m, π, ps, st)` | $c_*\pi$ | **expensive** — see below |
| `latentspace` / `observedspace` / `unobservedspace` | $\llbracket c\rrbracket$ / $Y$ / $X$ | |
| `ispure(m)` | $\llbracket c\rrbracket \cong 1$ | **not preserved by composition** |

Beliefs: `DiracBelief` (a clamp / a cup), `SampleBelief` (particles), `TrivialBelief` (the
unit space).

## Implementation difficulties

### 1. `pushforward` is one of the two things that make this hard

The paper's own closing discussion says it: computing $c_*\pi$ is marginalisation, "similarly
expensive to computing exact inversions", and the recommended remedy is belief propagation
or variational message passing. That is `Mycelium.jl`, and it does not exist yet.

The consequence for this file is that `pushforward` must be allowed to lie. It returns an
`AbstractBelief` with no promise of exactness, and `isexact` (defaulting to `false`) is how
a caller finds out. An implementation that returns a moment-matched Gaussian is legitimate;
one that returns it while claiming `isexact() == true` is a bug.

### 2. Conjugate families are not preserved by pushforward

Also from the paper's closing discussion: conjugate models make everything cheap, but
pushing forward does not generally preserve the family, so you need moment-matching
projections back into it — and encoding "which family this wire carries" into the game data
"requires annotating the games with predicates, which can be done compositionally, but
laxly".

Nothing in this file does that yet. When it lands it will want a `family(belief)` trait and
a `project(belief, family)` operation, and the laxness of the annotation will need the same
treatment as everything else here: track it, report it, do not hide it.

### 3. `ispure` defaults to `false` and composition must not override it naively

Purity ($\llbracket c \rrbracket \cong 1$) is a genuine optimisation: a pure model needs no
latent bookkeeping. It is also **destroyed by composition** — composing two pure models
gives a model whose latent space is the intermediate space. Any future `ispure(::ComposedModel)`
must return `false` unconditionally, not `all(ispure, parts)`. Writing it the natural way
would be wrong, which is why it is called out here.

### 4. Belief representations are the real open question

`DiracBelief` / `SampleBelief` / `TrivialBelief` are placeholders that name the important
degenerate cases. What is missing is the exponential-family / natural-parameter
representation that [[Parameterized Statistical Game|Definition 27]] gestures at when it
mentions the Fisher information metric, and that Khan & Rue's Bayesian learning rule
requires. That representation is where the natural gradient becomes cheap, so it is not
optional in the long run — but designing it before there is one working factor would be
guessing.

### 5. `logdensity` on an unnormalised measure

[[Copiers Cups and Caps]] notes that cyclic models require unnormalised measures, so
`logdensity` may be a log-*potential* rather than a log-density, differing by an unknown
constant. For energy purposes this is fine (the constant is irrelevant to gradients), but
for anything that compares densities across models it is not. Currently unmarked. A
`isnormalised(m)` trait will be needed before any model-comparison feature.

Related: [[Open Model]], [[Composition of Open Models]], [[lens]], [[Bayesian Inversion]]
