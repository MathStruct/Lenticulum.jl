#design

> **Status: accepted (2026-10-22, Daniel; merged to master in be229615).** Related: [[Scalar and Multivariate Energy]] §7–§8, [[energy]], [GitHub issue #8](https://github.com/MathStruct/Lenticulum.jl/issues/8), [[L-ADR-0002 Energies are vectors, scalarised late]].

## Context

A factor's `energyspace` and `scalarisation` served both its pointwise energy (`energy`) and its free energy. For the residual factors they described only the free energy, graded `(fit, complexity, negentropy)`. So `scalar_energy` applied a graded scalarisation to a residual vector and threw a `MethodError` (issue #8, reported by docxology).

## Options

1. One space for both, as in AutoBayes: make the pointwise energy graded too, losing the residual vector.
2. Two levels, each with its own space and scalarisation, defaulting to one space.
3. Scalarisations as parameters chosen per call, like the `p` of a norm (Daniel's direction), and per polarity.

## Decision

Options 2 and 3 together. `pointwise_energyspace` and `pointwise_scalarisation` describe `energy` and default to `energyspace` and `scalarisation`. A residual factor's pointwise energy is the signed residual with `SquaredNorm(Q⁻¹)`. `scalar_energy` and `scalar_free_energy` take `σ` and `polarity` keywords. All four functions take an optional polarity, falling back to the polarity-free answer. Choices that fork the theory are documented, not fixed: [[Scalar and Multivariate Energy]] §8.

## Consequences

- At a point, the scalar pointwise energy equals the `fit` summand of the free energy; tested for four factors (`test/runtests.jl`, "pointwise energy and free energy: one contract").
- No factor depends on the polarity yet.
- The free energy of a factor is built from the *scalarised* pointwise energy, not from the vector.
