#open-problem

> **Status: partly resolved (2026-11-10).** The defects below are fixed on the `refactor` branch,
> each with a test. What was found and not changed is listed at the end. Related:
> [[LP-023 Findings of the external reviews of 2026-11-04]], [[Refactor 2026-11]].

> Sources: a review of `dev-danielb` at `28b1a315` by Claude (2026-11-09/10), which ran every
> suite and the scripts it cites; the full text is outside the repository
> (`Lenticulum.jl-Review-2026-11-09/`); code and tests as named in the tables.

**Problem.** The review read LenticulumCore and Mycelium in full, the Gaussian part of
Lenticulum, `implicit.jl` and `solve.jl`, searched the rest, and rederived the mathematics behind
the three paths ([[Refactor 2026-11]]). About half of the source was read; the three packages off the paths (ImplicitLayers, Adversarial, AlgebraicFactors)
were not reviewed.

## Fixed

| # | finding | what the code did | fix | test |
|---|---|---|---|---|
| 1 | a partly flat message into a `GaussianFactor` | a Kalman smoother with a two-dimensional state and a one-dimensional observation threw `belief_mean needs a proper belief`, or, when rounding let the Cholesky factorisation of the rank-deficient precision through, returned wrong marginals on a tree without an error (for $H = [0.1\ \ 0.3]$ the mean of $x_0$ was off by 0.049, the covariance by 0.019) | the backward message is declared improper when the coefficient is rank-deficient; a partly flat input goes through the flat-direction limit of the constraint factors (`_flat_message`); the same for `NonlinearFactor` | `lib/Lenticulum.jl/test/gaussian_oracle.jl`: 40 random trees with rectangular coefficients and the position-only smoother against the dense joint (means, covariances, free energy, gradient) |
| 2 | freeze paths | `_trainable_indices(ps, (:second,))` froze coordinates from 1 when `second` began with an empty entry (a Lux container whose first layer has no parameters): the wrong parameters were frozen and the named ones trained | the start of the range is accumulated while walking the path | LenticulumCore `test/runtests.jl`, against a brute-force mask on nested trees with empty entries |
| 3 | composition of two `GaussianFactor`s | the collapsed factor's noise $BQ_aB^\top + Q_b$ failed the constructor's exact `issymmetric` for 186 of 200 random parameter draws with a non-diagonal $Q_a$ | the constructor accepts symmetry up to rounding and uses the symmetric part | `gaussian_oracle.jl` |
| 4 | damping | `damp` of two Gaussian messages dropped the `improper` declaration | kept when both sides carry it (CPU and device methods) | Mycelium `test/runtests.jl` |
| 5 | `parameter_step` against `free_energy_gradient` | for exact or pathwise edges without a `schedule`, the second returned the envelope gradient (LP-021, option 3) and the first threw | one rule, `_uses_coupled_driver`; `ad` defaults to `nothing` in both | `gaussian_oracle.jl` |
| 6 | dimensions | a channel and its variable were never compared; a `DataFactor` of length 1 on a two-dimensional variable was broadcast | `validate` refuses a mismatch where both sides declare a dimension | `gaussian_oracle.jl` |
| 7 | element types | `scalarise` of a graded energy and `_energy_vector` promoted `Float32` to `Float64` | fixed in LenticulumCore. Mycelium's graph total still accumulates in `Float64` (open, below) | LenticulumCore `test/runtests.jl` |
| 8 | accelerated EM | `fit_em` caught every exception of an extrapolated step, interrupts and method errors included | only numerical failures are a rejected step | covered by `em.jl` |
| 9 | coupled gradient | `_graph_scalar` evaluated the entropy of variables whose counting number is 0, which the forward free energy skips on purpose | uses `_variable_term` | covered by `lp021.jl` |
| 10 | two clamps to one value | `combine(DiracBelief(0.1 + 0.2), DiracBelief(0.3))` threw "contradictory hard clamps" | equal up to rounding is equal | Mycelium `test/runtests.jl` |
| 11 | `Polarity` | `==` without `hash` | `hash` added | LenticulumCore `test/runtests.jl` |
| 12 | docstrings on the wrong object | `⊕`, `MessageStore`, `mean_field_message`, `variable_degree`; `infer!` documented three return values and has four | moved or corrected | Mycelium and LenticulumCore `test/runtests.jl` |
| 13 | `Unscented` | the default `α = 1e-3` placed the sigma points a thousandth of a standard deviation out | default `α = 1` ([[nonlinear]]) | the existing check on $y = x^2$ |
| 14 | uncalibrated beliefs | a `DiffusionFactor` with `message = :gaussian` or `:mixture` and the default `λ = 1` sends covariances off by `density_lambda` (about 84 at the default levels) | a warning, once | none |
| 15 | packaging | `DispatchDoctor` was a dependency of five packages and used by none; no package bounded `LuxCore` | removed; `[compat]` entries added | |

## Corrected in the notes

| note | what was wrong |
|---|---|
| [[Implicit Diffusion Learners]] §5 | the smoothing-bias formula holds for noise-free nodes; nodes with noise average the logarithm over their shifts and smooth a second time, so the default nodes have about twice the bias (measured: `lib/VariationalDiffusion.jl/examples/ring_bias.jl`) |
| [[Universal Approximation for Implicit Diffusion Learners]] | (R1) asked more than Lemma 1's constant controls; restated |
| [[Acausal Composition is a Hypergraph Category]] | `combine(b, b)` is not $\mu\circ\delta$; the special axiom holds for potentials, and what fails is representing a copied state by marginals |
| [[Channels and Polarity]] | compact closure lives among unnormalised kernels |
| [[Bethe Free Energy]] | rewritten: the node rules that came after it, and the condition under which a tree is exact |
| [[The Linear Gaussian Chain]] | "the smoother comes for free" needed the vector case |
| `README.md` | counts, the package table, and a statement that conditional sampling is not implemented (it is: `implicit_sample`) |

## Found and not changed

| finding | where | why not |
|---|---|---|
| the cost of a sweep and of the free energy grows quadratically with the number of parameter keys (7.7, 25.8, 102 ms for 601, 1 201, 2 401 factors) | `passing.jl` (`_setsubtree`), `free_energy.jl` | the backlog's integer-slot store. Partly fixed 2026-11-11: the main cost of the free energy was `_check_tied_state` (O(keys × factors)), not `_setsubtree`; it is linear now and an unchanged state is not written back, so the free energy is 1.5–2.3 times faster (more for larger graphs) and a sweep unchanged, both still quadratic through the lookups by name (`lib/Mycelium.jl/src/passing.md`, "The state tree") |
| `excluded_marginal` recombines the other messages for each edge: quadratic in a variable's degree | `messages.jl` | needs prefix products or division; not a correctness problem |
| the graph free energy accumulates in `Float64` whatever the element type | `free_energy.jl` | touches every energy path; wants the element-type test first |
| convergence of Gaussian messages is an absolute bound on natural parameters | `gaussian.jl` (Mycelium) | a change of behaviour for every schedule; `criterion = :marginals` is the way round today |
| `AndersonSchedule` writes per edge, which unblocks a plate's rows | `anderson.jl` | by reading; not run |
| a mixture into a Gaussian or constraint factor is collapsed to its moments, after which a tree is not exact | by design | stated in [[Bethe Free Energy]]; cluster nodes restore exactness |
| root-solver tolerances of ImplicitLayers are absolute and written for `Float64` | `lib/ImplicitLayers.jl/src/solve.jl` | the package is off the paths and not yet reviewed |

**Resolution.** The numbered defects are closed by their tests. The open rows are in the
[[Feature Backlog]] or belong to the shared solver layer proposed in [[Refactor 2026-11]].
