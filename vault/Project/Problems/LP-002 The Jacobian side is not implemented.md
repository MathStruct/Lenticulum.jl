#open-problem

> **Status: open (2026-10-22).** Related: [[energy]] §"Implementation difficulties" 6, [[Scalar and Multivariate Energy]] §6, [[L-ADR-0002 Energies are vectors, scalarised late]], [Fisher information metric (CT-ML wiki)](https://mathstruct.org/CategoryTheory-ML-Wiki/Fisher-Information-Metric).

**Problem.** The vector energy exists so that its Jacobian is available: Gauss–Newton steps, the Fisher metric of AutoBayes Definition 27, and the implicit function theorem. `energy.jl` provides the values and the algebra, but nothing extracts $D_\theta \mathbf F$. Until it does, the main payoff of [[L-ADR-0002 Energies are vectors, scalarised late]] is unrealised.

**Options.**
1. A Jacobian function in `Lenticulum.jl` proper, with one package extension per AD backend; `LenticulumCore` stays free of AD dependencies.
2. Per-factor hand-written Jacobians where a closed form exists (linear-Gaussian, constraint), AD elsewhere.

**Leaning.** Both: closed forms as tests, AD extensions as the general path. The compositional Fisher metric of [Karwowski, Nielsen & Staton](https://mathstruct.org/CategoryTheory-ML-Wiki/Information-Geometry-of-Probabilistic-Programs) is a reference for how the pieces should combine.

**Progress (`review-argon`, 2026-10-23).** `LenticulumCore` now provides `energy_jacobian(f, x, a, y, ps, st; wrt)` (finite-difference reference on the pointwise vector energy) and `gauss_newton_metric(f, x, a, y, ps, st; wrt)` ($J^\top M J$ for `SquaredNorm(M)`), keeping `LenticulumCore` free of AD dependencies. Parameter-space AD extensions ($D_\theta \mathbf F$) and closed-form factor overrides remain open.
