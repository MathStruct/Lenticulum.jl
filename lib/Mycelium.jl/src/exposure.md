#implementation

> `expose(f, names)` moves parameter blocks of a factor into channels, so a graph variable can
> hold a weight: priors on weights, weight tying and inference over weights become ordinary
> wiring and message passing. Generic when the blocks are points; exact beliefs on blocks need
> per-factor methods.

> Sources: code: `exposure.jl`; the linear-Gaussian methods in the top-level `src/exposure.jl`; tests in the top-level `test/exposure.jl`
>
> Theory (CT-ML wiki): [Bayesian Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Lens) · [Statistical Game](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game)

Implements §1 of [[L-SPEC-003 Exposed Parameters and Energy Channels]]. Theory:
[[Unusual Wirings]], [[Hierarchical Models]] §1.

## The wrapper

`ExposedFactor` holds the factor, the exposed parameter names, their channel names (a block
may be renamed, `:A => :w`) and the block sizes, read once from `initialparameters` at
construction. Its channels are `f`'s plus one per block, of length `length(block)`; the
channel carries `vec(block)`, column-major. Its `initialparameters` are `f`'s without the
exposed blocks, so graph-level parameter trees, freezing and optimisers see only what remains.
Energy spaces and scalarisations are forwarded.

A message or free energy goes one of two ways:

| exposed inputs | target | what runs |
|---|---|---|
| all `DiracBelief` | one of `f`'s channels | `f`'s own `factor_message` / `local_free_energy`, with the blocks substituted into `ps` (§1.1, every factor) |
| any belief, missing, or the target is a block | any | `parameter_message` / `parameter_free_energy` of `f`, which throw by default |

Polarities and messages handed to `f` are renamed to its view: own channels as they are,
exposed channels under their parameter names. `supports_polarity` accepts `f`'s polarities
with every block `Observed()`, and whatever `supports_parameter_polarity(f, ·)` accepts;
`supported_polarities` adds, for each block, the polarity with that block the target, when
supported.

`uses_cavity` is `true`: the substitution reads the incoming messages on the exposed edges.

`supported_polarities` lists every accepted assignment up to five channels in all; above that
it throws, as L-SPEC-001 permits, because each assignment is a separate `Polarity` type and
listing $3^n$ of them compiles that many (21.9 s for six channels, measured by Codex,
2026-10-30). `hasnopolarity` and `isunidirectional` are exact, by the lazy
`LenticulumCore.polarities` search stopping at the first (resp. second) accepted assignment;
a structural shortcut was wrong for custom hooks (Codex, 2026-10-30). A search that finds
none or exactly one visits all $3^n$ candidates, so these predicates are slow for factors with
many channels.

A factor with a channel named like an exposed block can be exposed, but only on the clamped
path: `parameter_message` sees channels and exposed parameters in one namespace, so the
belief path refuses it. A factor whose belief rules read further parameter names checks
those itself (`GaussianFactor` refuses channels named `:A` or `:b` there; `:noise` is fine).

## What it does not do

- **Approximate messages.** A factor without the three methods supports only points on its
  blocks; no Laplace or sampling fallback exists. That keeps every result exact or refused.
- **Game-level forwarding.** `assemble`, `energy`, `free_energy` and `entropy` are not
  defined for the wrapper; it is for graphs.
- **Nested blocks.** Only top-level array entries of `ps` can be exposed, so a Lux network's
  weights (nested `NamedTuple`s) cannot yet.
- **Gradients through exposure.** The graph gradient drivers (LP-002, LP-011) are untested on
  exposed factors.

## Difficulties

- A clamped weight whose edge is `Bidirectional()` still receives a message back in the
  tree's outward pass, and with an uncertain input that message is bilinear and throws. Wire
  a clamped block like data, `Absorbing()`.
- Leaf predictions add a factor whose output has no incoming message; the factor belief
  still integrates to one, so the tree's Bethe free energy stays $-\log p(y)$ (tested).
