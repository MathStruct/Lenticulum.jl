#implementation #derivation

> `ToricFactor` enumerates a finite integer-statistic exponential family. Sum-product
> sites and joint factor entropies are exact finite sums. Tree BP and its Bethe free energy
> are compared with independent enumeration in `../test/runtests.jl`.

> Sources: derivation from [[Toric Models as Factors]] and [[L-SPEC-004 Algebraic Factors Toric and Grassmannian]] §1.1–§1.5; code: `toric.jl`, `../test/runtests.jl`, `lib/Mycelium.jl/src/free_energy.jl`.
>
> Theory (CT-ML wiki): [Statistical Game](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game) · [Bayesian Inversion](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Inversion) · [Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy)

## Representation and ordering

For alphabet sizes `(s₁, …, sₙ)`, columns follow `CartesianIndices`: variable 1 varies
fastest. In a 2×3 table the columns are `(1,1), (2,1), (1,2), (2,2), (1,3), (2,3)`.
Channels carry their alphabet size as their space. Names must be distinct Symbols,
alphabets positive integers, and the product must fit in `Int`. The constructor copies
the design matrix and label vectors. A zero-row matrix represents a uniform potential.

`independence` returns the matrix for singleton generators. `hierarchical` returns one
indicator block per supplied generator, retaining redundant rows. Generators may name
channels or give their indices. Within a block, the first listed variable varies fastest;
reversing a generator permutes its rows. An empty generator gives a constant row and an
empty generator collection gives no rows. These constructors do not split a graph into
separate factors.

Default labels are `1:sᵢ`. Explicit labels are a tuple of vectors, unique within each
channel. Diracs and pointwise energy tuples use `isequal` to match actual labels; there
is no integer-as-index fallback. An unlabelled categorical input is positional. A labelled
one must have the same complete support, and can be reordered. Outputs use the factor's
canonical label order, with `labels = nothing` for default alphabets. Mycelium's generic
categorical `combine` still requires identical label order, so factors and priors sharing
a graph variable must use that same output support and order.

Natural parameters are a Float64 zero vector initially, stored as `ps.η`; the state is
empty and passes through unchanged. Redundant parameters are retained. Normalised models
agree iff `A' * (η - η′)` is constant. No gauge reduction, integer kernel or augmented
matrix is needed for these finite sums: they work even when the constant vector is absent
from the row space of `A`.

## Messages and conformance

For target `t`, the message is the normalised version of

$$
\mu_{f\to t}(x_t) = \sum_{x_{-t}} \exp(\eta^\top a_x)
                      \prod_{i\ne t}\mu_{i\to f}(x_i).
$$

Missing keys, `nothing`, `missing`, `TrivialBelief()` and `Latent()` have unit weight.
Mycelium supplies only informative channels in `inputs`; the other channels are latent.
All assignments with one `Unobserved()` and any combination of `Observed()`/`Latent()`
elsewhere are declared. The implementation excludes the target input and prior, even if
supplied. Uniform categorical inputs differ from unit weights only by a constant, so their
normalised messages agree. The factor's normalising constant also cancels in a message.

Computation uses shifted log weights and log-sum-exp, including `-Inf` for impossible
states. Common logits are removed before incoming evidence is added; normalised energies
use shifted log probabilities directly, avoiding cancellation against a large log partition. Zero joint support raises `ArgumentError`. The full state table is enumerated;
the cost includes computing `A'η` and visiting every state and channel. It is exponential
in arity. This is CPU enumeration, without GPU or AD guarantees.

The factor implements Mycelium's direct-message interface. It does not implement
`assemble`/`invert`: the core lens contract asks for a joint posterior including latent
channels, whereas a categorical site on one target has marginalised those channels.
The conformance checker's explicit `direct_message = true` path requires independently
computed site and pooled-posterior fixtures. It checks those values, excludes both the
target input and prior, and shares all parameter, channel, energy, game and graph checks
with the default path. Existing factors still take the unchanged lens checks. There is
no forward generative model or correlated joint latent posterior API.

## Factor and game free energy

Given variable-to-factor messages, form

$$
b_f(x) \propto \exp(\eta^\top a_x)\prod_i\mu_{i\to f}(x_i),\qquad
F_f = \sum_x b_f(x)E_f(x) + \sum_x b_f(x)\log b_f(x).
$$

`local_free_energy` returns this scalar and the unchanged state. It charges the energy
of this factor alone, without the negative log of the incoming messages. The energy is
`-η'aₓ`, plus the global log partition when `normalised = true`. The entropy is the
**joint** factor entropy, including correlations, with zero probabilities contributing zero.

Mycelium adds `sum((degree(v)-1)H(bᵥ))` separately. Thus the factor terms subtract each
factor entropy once, and the variable terms correct the repeated variable entropies.
The test graph has five variables, three toric factors and two priors, optionally an
observation; every marginal and the total Bethe energy are checked against the joint
enumeration. Prior edges are bidirectional so their local factor beliefs receive the
cavity messages too. Emitting-only prior edges would omit that information from their
free-energy computation, even though the prior's outgoing message remains valid: the
marginals stay exact but the Bethe free energy is wrong, silently. That is a Mycelium
problem, not a toric one, recorded as [[LP-014 Emitting edges drop the cavity message from the free energy]]
and by a `@test_broken` in the tests.

Game-level quantities require `normalised = true`. `π` can be a NamedTuple of channel
beliefs, and `y` a disjoint NamedTuple of beliefs or point labels. A single prior belief
is assigned to the sole channel omitted from `y`; `nothing` means no evidence. Other
unspecified channels are latent. Overlapping or unknown channel names throw rather than
silently overwrite evidence. `free_energy` uses the same joint factor belief as above;
`entropy` returns its positive Shannon entropy. With no evidence, free energy is zero.

The seven testsets in `../test/runtests.jl` check constructor failures, hand-written design
columns, pointwise energies, all evidence kinds for every target at arities two and three,
local/game quantities, tree inference and the common interface. The references compute
ordinary products and finite sums independently of the log-domain implementation.

No fitting, model-membership check, LP solver, Markov basis, manifold or polynomial
inference is implemented. Label matching and finite summation are exact; large tables
remain expensive.
