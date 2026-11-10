#implementation

> `LuxFactor`: any `AbstractLuxLayer` as a factor. **One polarity.** This is the wrapper that
> works today on everything in the SciML ecosystem, and the demonstration of why that is not
> enough.

> Sources: code: `luxfactor.jl`
>
> Theory (CT-ML wiki): [Bayesian Inversion](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Inversion) · [Parametric Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Parametric-Lens) · [Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy) · [Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Lens)

## 1. It really does wrap anything

`DeepEquilibriumNetwork`, `NeuralODE`, `NeuralDSDE`, `Chain`, a Boltz backbone — all are
`AbstractLuxLayer`s, so all of them satisfy the only interface this wrapper uses. No SciML
dependency is required to wrap a SciML model, because `LuxCore` is the contract they all
already meet.

`postprocess` handles the one wrinkle: some of those layers return a solution object rather
than an array.

```julia
LuxFactor(NeuralODE(net, (0.0,1.0), Tsit5()), :x => :y;
          postprocess = sol -> Array(sol)[:, end])
LuxFactor(DeepEquilibriumNetwork(cell, NewtonRaphson()), :x => :z)
```

## 2. And it gets you one polarity

`isunidirectional(::LuxFactor) == true`, always. [[Lux as a Parametric Lens]] says why:

> A Lux layer *is* a lens, whereas a factor only *becomes* one once a direction is chosen.

A layer has already chosen. So `assemble` returns a lens with `TrivialInversion` — the same
inversion `lens.md` gives a prior, on the same grounds that *there is nothing to infer* — and
`factor_message` on the input channel throws with a message naming `DEQFactor` and
`NeuralODEFactor` as the alternatives.

What wrapping *does* buy: graph membership, parameter and state management through the
`LuxCore` interface, participation in the free-energy accounting, and a channel-named place
in a `FactorGraph`. That is not nothing. It is just not bidirectionality.

The test suite makes the comparison on one object — the same `LinearCell` wrapped both ways:

```julia
length(supported_polarities(LuxFactor(CELL, :in => :out))) == 1
length(supported_polarities(DEQFactor(CELL, (x = 2, z = 2)))) == 2
```

Same network, same parameter count, twice the directions. That pair of numbers is the shortest
statement of what this package is for.

## 3. Implementation difficulties

### 3.1 Zero energy is the compatible default; endpoint scoring is opt-in

The default wrapper delegates losses to other factors and reports zero energy.
`LuxFactor(layer, :x=>:y; dims=(nin,nout), residual=true)` instead gives pointwise
vector energy $y-\mathrm{layer}_\theta(x)$, scalarised by half its squared norm.
Its free energy scores incoming endpoint points with the same scalar energy and zero
point-belief entropy. An explicit output dimension is required. Such graph nodes need
absorbing/bidirectional edges to read endpoint beliefs.

This opt-in preserves existing downstream-loss graphs while allowing fixed-message
parameter learning. With zero energy, the graph's partial parameter derivative is zero:
training a downstream loss through an inference schedule is a different, unimplemented
operation. Users may differentiate `apply_layer` inside `parameter_gradient` directly.
The backend compatibility of a wrapped solver is the wrapped layer's responsibility.
ForwardDiff and Zygote are checked on a scalar Lux parameter in `test/parameter_ad.jl`.

### 3.2 `dims` are optional and unchecked

`LenticulumCore.Channel` accepts `nothing` for its space, so `LuxFactor` defaults both
dimensions to `nothing` and never validates the shapes it passes through. `validate(g)` will
therefore accept a graph wiring a 4-vector into a layer expecting 10, and the error surfaces
inside the user's model.

### 3.3 `postprocess` runs on the output only

There is no `preprocess`. A layer wanting a `NamedTuple` or a tuple input has to be adapted by
the caller before it reaches the factor. Asymmetric, and only because the output case (SciML
solution objects) is the one that actually comes up.

Related: [[deq]], [[neuralode]], [[DEQ as a Relation]], [[Lux as a Parametric Lens]],
[[The Equilibrium Family]]

## Game quantities and common checker (2026-10-26)

By default the conditional point game has `free_energy = 0` and `entropy = 0` in
`ScalarEnergySpace`, exactly like `energy` and `local_free_energy`: the wrapper delegates
loss accounting to other factors. `test/conformance.jl` runs the shared default lens path
at its sole polarity using the independent map `y = 2x`, and checks option-1 game/graph
agreement. The scalar-parameter test layer supplies its own LuxCore parameter count;
LuxCore's default parameter counting expects array parameters.

The game observation `y` must be a Dirac input belief; other observations throw
`ArgumentError`. No distributional inversion entropy is inferred from the layer API.
