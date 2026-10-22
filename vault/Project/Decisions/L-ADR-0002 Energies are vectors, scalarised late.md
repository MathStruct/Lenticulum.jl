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
- The payoff, Jacobians of the vector energy, is not implemented yet: [[LP-002 The Jacobian side is not implemented]].
- Which scalarisation to use is a choice with consequences, listed in §8 there.
