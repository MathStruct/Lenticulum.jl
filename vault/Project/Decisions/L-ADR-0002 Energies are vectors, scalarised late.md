#design

> **Status: accepted (recorded 2026-10-22; the design of [[Scalar and Multivariate Energy]]).** Related: [[energy]], [[statistical_game]], [[L-ADR-0003 Pointwise energy and free energy are separate levels]].

## Context

AutoBayes gives a statistical game one non-negative scalar energy, and composites add them. Adding loses which factor contributed what, in which coordinates a factor is wrong, and the Jacobian of the residual. That rules out Gauss–Newton steps, the Fisher metric and the implicit function theorem, which Lenticulum's implicit learners need.

## Options

1. A scalar energy, as in the paper.
2. A vector energy in an energy space, with a scalarisation applied as late as possible.

## Decision

Option 2: energies live in energy spaces, compose by direct sum, and a scalarisation turns them into a number. It was a requested design constraint ([[Scalar and Multivariate Energy]], header).

## Consequences

- Linear scalarisations compose strictly; convex ones are lax, by a Jensen gap (§5 there), which `islinear` and `jensen_gap` expose.
- Reference input/output Jacobians now exist as `energy_jacobian` (central finite differences), with `gauss_newton_metric` for the pull-back (`lib/LenticulumCore.jl/src/energy.jl`). Backend AD and parameter Jacobians remain open: [[LP-002 The Jacobian side is not implemented]].
- Which scalarisation to use is a choice with consequences, listed in §8 there.

## Implementation follow-up (2026-10-26)

LP-002 now supplies optional DI input/parameter Jacobians, scalar energy/free-energy
parameter gradients with Lux tree structure, and parameter Gauss–Newton metrics.
Finite differences remain the default reference. The existing `energy_jacobian` API
is extended with `wrt=:ps` to retain input compatibility and share the pull-back code.
Mycelium has a separate fixed-belief parameter step; DEQ solved-state AD uses IFT rules,
and built-in NeuralODE Euler/RK4 differentiates the discrete map. ForwardDiff and
Zygote are checked in `LenticulumCore/test/parameter_ad.jl` and
`ImplicitLayers/test/parameter_ad.jl`. This realises the derivative payoff for those
paths; matrix-free DEQs, SciML adjoints and graph schedule derivatives remain future work.
