#overview #design

> The repository after the refactor of November 2026 (branch `refactor`): five packages built
> around three paths, the rest parked. This note says what is where, what is verified and by
> what, what is not, and how parked work comes back.

> Sources: original to this repository; the review that preceded it is [[LP-024 Findings of the review of 2026-11-09]]; code: every `Project.toml`, `.github/workflows/CI.yml`, `docs/make.jl`

## The three paths

| path | question it answers | packages | notes |
|---|---|---|---|
| **A. Gaussian factor graphs** | given a linear-Gaussian model (a chain, a tree, a loop) and data, what are the marginals, the evidence and its gradient? | `LenticulumCore`, `Mycelium`, `Lenticulum` (`gaussian.jl`, `nonlinear.jl`, `composition.jl`) | `vault/Gaussian Graphs/` |
| **B. Relation learners** | given a trained diffusion model of a joint space, what are the answers to a query in any direction, how sure are they, and how does one train through the query? | `VariationalDiffusion` | `vault/Relation Learners/` |
| **C. Structure and regimes** | which linear relations, switched by which discrete outcomes, explain the data, and how are they fitted? | `Lenticulum` (`constraint.jl`, `switching.jl`, `regime.jl`, `likelihood.jl`, `m_steps.jl`), `Mycelium` (`cluster.jl`, `plates.jl`, `overlay.jl`), `MyceliumLearning` (`em.jl`) | `vault/Structure and Regimes/` |

## Layout

```
lib/LenticulumCore.jl     what a factor is; belief types          (no dependency on the others)
lib/Mycelium.jl           inference                               ← LenticulumCore
lib/MyceliumLearning.jl   gradients, EM, the optimiser step       ← Mycelium
lib/Lenticulum.jl         the factors, their M-steps, examples    ← MyceliumLearning
lib/VariationalDiffusion.jl  diffusion models as relations        ← Mycelium
parked/                   ImplicitLayers, Adversarial, AlgebraicFactors, two tutorials, three docs pages
vault/                    Foundations, one folder per path, Design, Project, Research Questions, Parked
docs/  bench/             as before; bench has its own environment
```

What changed against `dev-danielb` at `28b1a315`:

- The top-level package moved from the repository root to `lib/Lenticulum.jl/`. The root has no
  `Project.toml`.
- Mycelium was split. Inference stayed; `free_energy_gradient` of a graph, `parameter_step`, the
  couplings, the graph adjoint, EM and `sample_gradient` are `MyceliumLearning`
  ([[MyceliumLearning]] has the boundary). Mycelium's suite went from 6 minutes to 41 seconds.
- `ImplicitLayers`, `Adversarial` and `AlgebraicFactors` are in `parked/` with their tests and
  notes; tutorials 05 and 07 (they use `ImplicitLayers`) and the three packages' documentation
  pages are with them. The vault notes on those families and on applications off the paths are
  in `vault/Parked/`.
- `meta/` and `poster/` are outside the repository.
- The shared test support (the factor conformance checker) is in
  `lib/LenticulumCore.jl/test/support/`.

## What is verified, and by what

Suites run on 2026-11-10 (Julia 1.13.1), after the restructure:

| package | result | time |
|---|---|---|
| LenticulumCore | pass | 1 min |
| Mycelium | pass | 41 s |
| MyceliumLearning | pass | 5.5 min |
| Lenticulum | pass | 14.5 min |
| VariationalDiffusion | pass | 6 min |

| claim | oracle | test |
|---|---|---|
| A: marginals, free energy and gradient on linear-Gaussian trees, rectangular and rank-deficient coefficients included | the dense joint solve | `lib/Lenticulum.jl/test/gaussian_oracle.jl` (40 random trees), `runtests.jl` ("the chain is exact at every length") |
| A: a smoother with partial observations | the dense joint solve | `gaussian_oracle.jl` |
| A: loopy Gaussian graphs: exact means, wrong variances | Weiss & Freeman | `runtests.jl` ("acausal and loopy") |
| A: frozen coordinates | a brute-force mask | LenticulumCore `runtests.jl` |
| B: the field is the gradient of the stated potential; the Jacobian is symmetric | a Gaussian mixture's closed-form score | VariationalDiffusion `runtests.jl` |
| B: the adjoint | central differences | VariationalDiffusion `runtests.jl` |
| B: the smoothing bias, noise-free nodes and nodes with noise | the two formulas of [[Implicit Diffusion Learners]] §5 | `lib/VariationalDiffusion.jl/examples/ring_bias.jl` (a script, not a test) |
| C: switching chains: marginals, `−log p`, gradient | enumeration of the switch sequences | `lib/Lenticulum.jl/test/cluster.jl`, `switching_constraint.jl` |
| C: mean-field and hybrid free energies | closed forms, Monte Carlo, enumeration | `mean_field_free_energy.jl` |
| C: EM does not increase the free energy with exact beliefs | monotonicity, known optima | `em.jl` |

The mathematics behind each row was rederived in the review; what it found is in
[[LP-024 Findings of the review of 2026-11-09]].

## What is not verified

- **About half of the source was read.** Read: LenticulumCore, Mycelium except `cluster_batch.jl`,
  the Gaussian, nonlinear, composition, constraint and switching factors, `m_steps.jl`,
  `implicit.jl`, `answers.jl`, `sampler.jl`, `schedule.jl`. Searched only: `likelihood.jl`,
  `regime.jl`, `noise.jl`, `exposure.jl`, and the other files of VariationalDiffusion.
- **The documentation build passes locally** (`julia --project=docs docs/make.jl`: the nine
  tutorials execute, the API pages and the vault build, no unresolved reference). It was not run
  in CI, and nothing was deployed.
- **The device (JLArrays) tests, the examples and the benchmarks were not run.**
- **The parked packages' suites were not run** after the move, and two of them were broken by it:
  `AlgebraicFactors`' tests included Lenticulum's source files by their old path, and
  `ImplicitLayers`' DifferentiationInterface extension extended `Mycelium._gradient_message`, which
  moved to MyceliumLearning (the extension did not load). Fixed on 2026-11-10: AlgebraicFactors
  takes Lenticulum and MyceliumLearning as test dependencies, ImplicitLayers depends on
  MyceliumLearning. All three suites pass (ImplicitLayers with and without the device files), and
  CI runs them in a separate job. A local `Manifest.toml` from before the move still points
  Lenticulum at the repository root (`path = "../.."`); delete it and instantiate again.
- Expectation propagation's free-energy claim, the likelihood factors' Laplace and EP sites, and
  the learned-noise priors were not rederived.

## Known limits of the core

These are true of the code as it stands and are the first candidates for work on the paths.

| limit | path | where recorded |
|---|---|---|
| a mixture into a Gaussian or constraint factor is collapsed to its moments; a tree is exact only with cluster nodes | A, C | [[Bethe Free Energy]] |
| cost grows quadratically with the number of separately parametrised factors | A | [[LP-024 Findings of the review of 2026-11-09]], [[Feature Backlog]] ("integer slot") |
| `excluded_marginal` is quadratic in a variable's degree outside plates | A, C | LP-024 |
| the graph free energy accumulates in `Float64` | A | LP-024 |
| Gaussian message convergence is an absolute bound on natural parameters | A | LP-024 |
| nodes with noise double the smoothing bias; every query returns a root, on the relation or not | B | [[Implicit Diffusion Learners]] §5, `lib/VariationalDiffusion.jl/examples/calibration.md` |
| Gaussian and mixture answers are calibrated only for `λ = density_lambda` (a warning otherwise) | B | [[implicit]] §5 |
| a nonlinear factor's free energy is its linear model's | A | [[nonlinear]] |
| constraint factors with every coefficient learned collapse on data | C | [[LP-022 Constraint factors are normalised in their residual, not their channels]] |
| BIC is used as the structure score for singular (mixture) models | C | [[Finding a Factor Graph]] |
| `Mycelium.reduce_message` for Gaussian mixtures is defined in Lenticulum (`beliefs.jl`, beside Runnalls' merge): Mycelium alone refuses to reduce one. Type piracy, declared in Lenticulum's `test/aqua.jl`; it belongs in Mycelium | A, C | this note |

## Next steps on the paths

In the order I would take them. Each is small enough to finish and has an oracle.

1. **A belief type for linear relations** (path A, C): the Gaussian-plus-flat decomposition of
   [[constraint]] §4.1 as a belief with exact `combine` and pushforward. It replaces
   `_split_moments`, `_flat_message` and the `improper` flag by one representation, and makes
   hard linear constraints exact.
2. **The parameter and state store by integer slot** (path A): removes the quadratic cost; the
   partial fix (skip `_setsubtree` when a factor's state is unchanged) is a few lines.
3. **A benchmark against RxInfer and a dense solve** on the chain and on a mixture model
   (paths A, C): `RxInfer` is already a test dependency of Lenticulum.
4. **Calibration remedies for the implicit learner** (path B): off-relation detection, and the
   comparison of `noisefree_nodes` with `field_nodes` on the calibration benchmark now that the
   bias of each is known.
5. **A query report** (path B): converged, stable, the roots found from other starts, the
   conditioning of the answer; the pieces exist in `ImplicitSolution`, `implicit_roots` and the
   pullback's conditioning check.
6. **A structure-search driver in the library** (path C): today every search is example code
   ([[Feature Backlog]], "Structure search and editing").
7. **Sorting the tests**: the graph-gradient tests on Gaussian factors are in Lenticulum's suite;
   Lenticulum's `runtests.jl` is 1 300 lines and VariationalDiffusion's 2 000.

## Moving things back

A parked package or note comes back when it is needed on a path, or when someone takes it up,
and it comes back **corrected**: reviewed, with an oracle test, on the current interfaces.

For a package (`parked/X.jl`):

1. `git mv parked/X.jl lib/X.jl`; in its `Project.toml` and `test/Project.toml` set the
   `[sources]` paths back to `../<Package>.jl`.
2. If it uses gradients, EM or `parameter_step`, add `MyceliumLearning` to its dependencies; the
   names that moved are listed in [[MyceliumLearning]].
3. Run its suite. Fix what the interface changes since it was parked broke.
4. Review it as the core was reviewed: read it, rederive what it computes, add an oracle test on
   random instances for each message path, record what is found as a problem.
5. Add it to the CI matrix, to `docs/make.jl` and `docs/Project.toml`, move its documentation
   page back from `parked/docs/` and its tutorials from `parked/tutorials/`.
6. Move its notes from `vault/Parked/` to the folder of the path it serves, revise them against
   the code, and update [[Map of Content]].

Known before starting:

| package | state when parked | first thing to fix |
|---|---|---|
| `ImplicitLayers` | suite passes on 2026-11-10 after the move (20 minutes; device files included) | root-solver tolerances are absolute and written for `Float64`; its solvers duplicate VariationalDiffusion's. Bring it back as, or after, a shared solver layer (root finding, the implicit-function adjoint, one tolerance rule, one report) |
| `AlgebraicFactors` | suite passes on 2026-11-10 after the move | not reviewed |
| `Adversarial` | suite passes on 2026-11-10 after the move | not reviewed; the least connected to the rest. `belief_mean` and `belief_cov` of LenticulumCore's `SampleBelief` are defined here (type piracy, declared in `test/aqua.jl`): move them to LenticulumCore |

For a note: move it, check every claim against a test or a script, remove what describes
history rather than the subject, and give it the `tests:` entry in its sources block
([[Start Here]]).

## Decisions taken in the refactor that are yours to revisit

- **`ImplicitLayers` is parked**, although tutorials 05 and 07 use its Picard solver. Keeping it
  would have kept a 20-minute suite and an unreviewed package in the core.
- **The split of Mycelium is inference against learning.** Cluster nodes, plates, overlays and
  structure edits stayed in Mycelium: they are inference. A further split (the belief algebra
  down into LenticulumCore, cluster nodes into their own package) is possible and was not done.
- **`Unscented` defaults to `α = 1`** and **`validate` refuses dimension mismatches**: both
  change behaviour for existing callers.
- **The branch is pushed** (`origin/refactor`). CI passed every job on `7de4488e` (five packages on
  Julia 1 and 1.12, two device jobs); the parked packages have their own CI job since 2026-11-10.
- **Versions.** The libraries are `0.0.1-dev`; Lenticulum keeps its calendar version
  (`20261015.1.0-proto`, as in `CITATION.cff`), which the refactor had changed and was restored.
