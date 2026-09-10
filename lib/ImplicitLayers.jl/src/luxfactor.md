# luxfactor.jl — implementation note

> `LuxFactor`: any `AbstractLuxLayer` as a factor. **One polarity.** This is the wrapper that
> works today on everything in the SciML ecosystem, and the demonstration of why that is not
> enough.

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

### 3.1 The energy is zero, which is a placeholder rather than a fact

A function has no residual of its own — there is nothing to be "approximately satisfied". So
`energy` returns `0.0` and `local_free_energy` likewise, on the reasoning that a `LuxFactor`'s
contribution to a graph's loss comes from a downstream `LossFactor`
([[Everything is a Factor]]).

That is defensible and it is also how a factor silently contributes nothing to a free energy
that is supposed to total $-\log p(y)$. A `LuxFactor` in a graph makes the Bethe sum wrong in
a way no assertion catches.

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
