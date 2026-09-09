# LenticulumCore.jl — implementation note

Implementation of: [[Parameterized Statistical Game]] (AutoBayes Def. 27) as the Julia
analogue of `LuxCore.AbstractLuxLayer`.

Source: `LenticulumCore.jl` (module) + the six included files.

## What this package is

`LuxCore.jl : Lux.jl :: LenticulumCore.jl : Lenticulum.jl`. It defines abstract types and
an interface, nothing concrete beyond a handful of value types for the energy algebra. It
must stay light: no AD backend, no solvers, no distributions package.

$$\underbrace{\mathbf{Para}(\mathbf{Lens}(\mathcal{C}))}_{\texttt{LuxCore}}
\qquad\longrightarrow\qquad
\underbrace{\mathbf{Para}(\mathbf{StatGame})}_{\texttt{LenticulumCore}}$$

See [[Lux as a Parametric Lens]] and [[AutoBayes to Lenticulum]].

## File map

| file | paper | note |
|---|---|---|
| `abstract_types.jl` | Defs. 1, 9, 20, 27, 29 | the hierarchy — [[abstract_types]] |
| `channels.jl` | Def. 1's $X/\llbracket c\rrbracket/Y$ | ports and polarity — [[channels]] |
| `energy.jl` | Def. 20's $l^c$, adapted | the two energies — [[energy]] |
| `open_model.jl` | Defs. 1, 4, 6 | kernels with latent spaces — [[open_model]] |
| `lens.jl` | Defs. 9, 10, 12, 15 | model + inversion — [[lens]] |
| `statistical_game.jl` | Defs. 20, 22, 25, 27, 28 | the factor interface — [[statistical_game]] |

## Dependencies, and why they are exactly these

- **`LuxCore`** — we *extend* `initialparameters` / `initialstates` / `parameterlength` /
  `statelength` rather than defining parallel functions. A factor's parameter tree is built
  by the same machinery as a Lux layer's, so a factor may contain Lux layers and a Lux
  layer's `ps` may be nested inside a factor's without any adapter. This is the single most
  important interop decision in the package.
- **`Random`** — `initialparameters(rng, factor)`; parameters are *constructed*, not given.
- **`DispatchDoctor`** — type-stability checking, as LuxCore does. Currently imported but
  not applied; `@stable` should be turned on for `assemble` once concrete factors exist,
  since an assembled lens escaping into a type-unstable region would destroy the point of
  putting polarity in the type domain.

Deliberately **absent**: `Distributions`, `Functors`, `ChainRulesCore`, any AD backend, any
solver. Beliefs are an abstract type here; concrete representations belong downstream, the
same way LuxCore has no arrays.

## Status

Abstract types, the channel/polarity algebra, the energy algebra, and the composition
combinators are implemented and tested. `forward`, `invert`, `pushforward`, `logdensity`,
`assemble`, `energy`, `entropy`, `free_energy` are **interface stubs** (`function f end`) —
they exist to be extended, and there is no concrete factor yet. That is deliberate: the next
step should be one worked end-to-end example, not more abstraction.

`test/runtests.jl` covers the parts that have content: the polarity partition, the direct-sum
energy algebra, and — the claim worth checking numerically — that
`F_scalar == F_multi + chain_rule_defect` with the defect equal to $\tfrac12\operatorname{tr}\operatorname{Cov}$
for a squared-norm scalarisation, and exactly zero for a linear one.

## Implementation difficulties

### 1. Two name collisions with `Base`

`Channel` shadows `Base.Channel` (the concurrency primitive) and `precision` shadows
`Base.precision` (the floating-point one). Both are *domain* vocabulary we want to keep.

Resolutions taken:
- `Channel` is defined but **not exported**. Use `LenticulumCore.Channel` or import it
  explicitly. Defining it is legal inside the module; exporting it would ambush every user
  who also does `using Base`.
- `precision` was **renamed** to `channel_precision`. The test suite caught this — the
  unqualified call resolved to neither binding and errored with an ambiguity hint. Renaming
  is right here because, unlike `Channel`, `precision` is a verb-ish name with no strong
  claim to the concept.

### 2. `⊕` on unnamed operands has to invent names

`oplus(a, b)` for two ungraded energies wraps them as `(; left = a, right = b)`. That is
arbitrary, and iterating it gives `((a ⊕ b) ⊕ c)` the shape `(left = (left=…, right=…), right = …)`
rather than a flat three-way grading. Associativity therefore holds only up to
renaming — which is *honest* (composition of open models is likewise only a bicategory,
associative up to isomorphism) but is a nuisance in practice.

The fix, when the graph layer exists, is that names should come from the **factor graph**,
not from the composition operator: `E_G = ⊕_{f ∈ Factors(G)} E_f` should be built in one
step from a node-name-keyed `NamedTuple`, never by folding a binary `⊕`. Treat the binary
`oplus` as an implementation detail of tests.

### 3. Expectations force $E$ to be a vector space, not a monoid

[[Composition of Statistical Games|Definition 22]]'s entropy law averages $\mathbf{H}^c$
under the downstream inversion, so $E$ must admit barycentres. A commutative monoid — the
minimum needed for the paper's `+` — is not enough. Hence `GradedEnergy` carries `+`, `-`,
scalar `*` and `zero`, and hence the definition in [[Scalar and Multivariate Energy]]
demands a topological vector space with a cone rather than just a monoid.

Consequence: energies are values in a *cone* $K$, but the *loss* $\mathbf{F} = \mathbb{E}[\mathbf{l}] - \mathbf{H}$
can leave the cone. The paper types $F : \mathcal{P}X \times Y \to [0,\infty]$, which
silently assumes $H^c \le \mathbb{E}[l^c]$. We do not assume it; `free_energy` returns an
element of $E$, not of $K$. If a caller needs non-negativity it must be checked, not typed.

### 4. Eager versus deferred scalarisation is a correctness issue, not an optimisation

With a linear $\sigma$ you may accumulate the scalar loss factor by factor as you walk the
graph. With a convex $\sigma$ you may not: you would compute
$\mathbb{E}[\sigma(\mathbf{F})]$ where you asked for $\sigma(\mathbb{E}[\mathbf{F}])$, and
the two differ by [`jensen_gap`](#). This is why `islinear` is a trait on
`AbstractScalarisation` rather than a comment. Any future graph walker must branch on it.

### 5. Package layout

The `[workspace] projects = ["test"]` pattern (as Lux uses) means `test/` shares the parent
`Manifest.toml`. Running the tests is `julia --project=lib/LenticulumCore.jl/test`, and a
stray `test/Manifest.toml` will break it — delete it if one appears. Also: the package name
in `Project.toml` was `LentriculumCore` (typo); corrected to `LenticulumCore`, which is the
name the module and the UUID now agree on.

Related: [[Index]], [[AutoBayes to Lenticulum]], [[Scalar and Multivariate Energy]]
