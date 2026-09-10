# [Vocabulary](@id vocabulary)

Six words. You need them to read the API; everything else can wait.

This page is deliberately shallow — it says what each word *does*, not why it is the right
word. The reasons are in the vault (`markdown/Index.md`).

## Factor

The unit of computation, and the analogue of a Lux layer. A factor is a **relation** among
named channels: some collection of values is either consistent with it or not.

Concretely, a factor is a struct subtyping `LenticulumCore.AbstractLenticulumFactor` that
knows its channels, which directions it can be run in, and how to compute a message. Its
parameters live in a separate `ps` object, exactly as in Lux:

```julia
f = GaussianFactor(2 => 3; noise = 0.5)      # the description
ps, st = LuxCore.setup(rng, f)               # an inhabitant
```

## Channel

A named port. `LenticulumCore.Channel(:x, 2)` is a channel called `:x` carrying a
2-dimensional value. A factor declares its channels; a graph connects them to variables.

Channels are how a factor avoids having "inputs" and "outputs" — it has ports, and their roles
are assigned later.

## Polarity

The assignment of roles to channels, **for one particular use** of a factor. Three roles:

| role | means | think |
|---|---|---|
| `Observed()` | clamped to a value you have | input |
| `Unobserved()` | what you are solving for | output |
| `Latent()` | internal; neither given nor asked for | scratch |

```julia
Polarity(; x = Observed(), y = Unobserved())   # "predict y from x"
Polarity(; x = Unobserved(), y = Observed())   # "infer x from y" — same factor
```

A factor declares which polarities it supports. `supported_polarities(f)` lists them, and
**the length of that list is the interesting number**: `1` means the factor is really just a
function; `n` means it is a relation you can solve any way round.

Each channel also carries a precision `ρ`, and `Observed()` defaults to `Inf` — a hard clamp.
A finite value is a soft one.

## Belief

What travels along an edge: a distribution over a channel's values. The concrete types:

| type | is | where it comes from |
|---|---|---|
| `DiracBelief` | a point | clamped data; any solver that returns a point |
| `GaussianBelief` | a Gaussian, in **canonical form** (`η`, `Λ`) | `Lenticulum` |
| `SampleBelief` | a weighted particle set | `Adversarial` generators |
| `TrivialBelief` | "no information" | the identity for pooling |

`GaussianBelief` stores information form rather than mean/covariance, which makes pooling two
beliefs into one simple addition and lets a belief be *improper* (`Λ` singular) — which is what
a likelihood that constrains only some directions looks like.

## Message

One belief travelling along one edge. A **factor → variable** message is computed by choosing
a polarity and inverting the factor:

```
target channel                              → Unobserved()
channels with an incoming message           → Observed()
everything else                             → Latent()
```

then `assemble(factor, polarity)` builds the machinery and the message falls out. The
important bit for correctness is the **exclusion principle**: the message a variable sends to
a factor pools everything *except* what that factor last said, or beliefs get compounded.
`Mycelium` handles this; you do not.

## Schedule

The order messages are computed in. This is the part with no counterpart in Lux, because a
graph with cycles has no topological order.

| schedule | for |
|---|---|
| `tree_schedule(g)` | trees — **two sweeps and it is exact** |
| `flooding_schedule(g)` | anything — iterate to convergence, no guarantees |
| `forward_backward_schedule(g)` | chains — this is the Kalman/RTS smoother |

`istree(g)` tells you which you are entitled to. If it is `true`, use `tree_schedule` and the
answer is exact. If it is `false` you are running loopy belief propagation, and
`ConvergenceReport` tells you whether the messages stopped moving — which is not the same as
the answer being right.

---

That is enough to use the library. The rest — why a factor is a "parameterized statistical
game", what the free energy measures, why the canonical form is load-bearing — is in the
vault.
