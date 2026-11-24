#design #engineering

> The bar every package of the repository is held to, written down (Q1). The bar has five criteria:
> types usable for dispatch, independence from the AD backend, independence from the array type,
> parallel execution, and profiling with benchmarks kept in CI. The note says which of them applies to
> which layer, how a kernel is written so that it meets all five, when we use an ecosystem package and
> when our own code, which check enforces each criterion and in which CI job, and how an exception is
> recorded. The reference is the SciML and JuliaHub packages (LinearSolve.jl, NonlinearSolve.jl,
> Krylov.jl, DifferentiationInterface.jl). Status: written 2026-11-24 from the request of that day.
> **Accepted by the maintainer on 2026-11-24**; the thresholds marked *to calibrate* are guesses until
> Q13 has measured the noise.

> Sources: original to this vault (design); [[Backlog Q Quality and Engineering]] (the baseline of
> 2026-11-24, `tools/type_audit.jl`); [[Shared Solver Layer]] "Why not Krylov.jl"; [[Parallelism and Compilation]]
> §2, §4, §8; [[The Type Discipline of a Factor Graph]]; code:
> `lib/ImplicitSolvers.jl/src/linsolve.jl`, `lib/MyceliumLearning.jl/src/gradient.jl`,
> `.github/workflows/CI.yml`, `Slow.yml`, `Weekly.yml`, `Examples.yml`, `tools/type_audit.jl`,
> `bench/structure_compile.jl`, `bench/krylov_allocations.jl`

> Theory (CT-ML wiki): —

## The five criteria

| | criterion | met when |
|---|---|---|
| C1 | **types for dispatch** | every field of a struct on a compute path has a concrete type or a type parameter; algorithm choices are types, not symbols; inner functions are type-stable for concrete inputs (JET finds no runtime dispatch in them) |
| C2 | **AD backend** | every differentiable entry point takes an `ADTypes` backend and is tested against finite differences with Zygote, ForwardDiff, Enzyme and Mooncake (Reactant where it applies); an implicit solve is differentiated by a rule, never through its iterations |
| C3 | **array type** | element type and array type come from the inputs; no scalar indexing on a compute path; tests in `Float32`, `Float64`, `BigFloat`, `ForwardDiff.Dual` and on JLArrays with `allowscalar(false)` |
| C4 | **parallel** | independent work runs on threads with a fixed reduction order (results independent of the thread count, bitwise or within a stated tolerance); every mutable shared state names what guards it and has a test from several tasks |
| C5 | **measured** | each workload of paths A, B and C is in a BenchmarkTools `SUITE` compared on every pull request (time, allocations, first-call latency); each kernel's allocations after warm-up are tested; each path has a profile script |

## Layers: what applies where

The criteria cost work, and not every layer needs all of them. The split follows the one
[[Parallelism and Compilation]] §5 argues for: structure is built dynamically, then a typed program runs
on it.

| layer | examples | C1 | C2 | C3 | C4 | C5 |
|---|---|---|---|---|---|---|
| **kernels** | factor messages and energies, M-steps, plate block kernels, `linsolve`, `solve_root`, `solve_columns`, message updates | all | all | all | thread-safe (no shared mutable state) | allocation budget, in the `SUITE` |
| **drivers** | `infer!`, the schedules, `free_energy`, `free_energy_gradient`, the graph adjoint, EM | after the function barrier | all | all | the parallel loops live here (Q10) | in the `SUITE`, profiled |
| **construction and editing** | `FactorGraph` building, structure edits, structure search moves | may be dynamic | — | the data it stores keep their array type (Adapt) | guarded if shared | first-call latency only |
| **examples, benchmarks, tutorials** | `examples/`, `bench/`, `docs/literate/` | — | — | — | — | honour `LENTICULUM_SMOKE` |

**The function barrier.** A driver may read a dynamic structure (a `Vector{Any}` cache keyed at run
time, a dictionary of plates) but hands a kernel concrete types: the dynamic lookup happens once per
call of the kernel, not inside it, and its cost is measured and written in the source note. Today
Mycelium's message store and plates are dynamic *inside* the sweep (30 abstract fields, the baseline);
that is the gap Q2 closes, and I2 (typed execution plans) is the form the barrier takes there.

**Off-path packages.** ImplicitLayers, AdversarialFactors and AlgebraicFactors (not yet reviewed, tested
weekly) are held to C1–C3 for code they share with the core and to nothing more until they are
reviewed. A core change must not lower what they already meet (ImplicitLayers tests all four AD
backends today).

## How a kernel is written

The conflict to decide: parts of Mycelium are written functional "for Zygote"
(`lib/Mycelium.jl/src/free_energy.jl`), and in-place kernels with workspaces are what C5's allocation
budgets and Q11's caches need. Zygote cannot differentiate mutation.

**Decision: SciML's split by size, with the adjoint by rule wherever a kernel mutates.**

1. **Small, fixed dimension** (a Gaussian message on a variable of dimension ≤ ~16, a scalar factor's
   energy): out-of-place, pure, allocation-light (tuples or small arrays; StaticArrays where it is
   measured to pay, none is a dependency today). Every AD backend traces these directly; no rule.
2. **Large or batched** (the plate block kernels, Krylov and Newton solves, a sweep's message buffers,
   EM statistics over data): an in-place primal `f!(out, ws, inputs…)` on a workspace from a cache
   (Q11), with an out-of-place wrapper for callers that do not hold a cache. The adjoint is a rule on the
   wrapper: a ChainRulesCore `rrule` (Zygote, and Mooncake through `@from_rrule`), an `EnzymeRules` rule
   where Enzyme would otherwise differentiate through the mutation badly, ForwardDiff on the dual element
   type through the same in-place code (C3 makes that work). LinearSolve and NonlinearSolve are the model.
3. **Implicit solves** (a fixed point, a linear system, a root) are always differentiated by the
   implicit-function theorem, by a rule, never by unrolling the iterations (Q6). The tangent correction
   of ImplicitSolvers (P38) is the forward-mode form of that rule.
4. **The transition.** Existing functional code stays until a row touches it (Q11, Q12, Q6); new large or
   batched kernels are written in the form of item 2. A functional kernel converted to in-place keeps its
   old form as the oracle of its tests until the conversion is reviewed.

**Algorithm choice by type.** An option that selects an algorithm is an algorithm object, not a
symbol: `linsolve(A, b, GMRES())`, not `linear_solver = :gmres`. Symbols already public
(`linear_solver = :dense` in `free_energy_gradient`, `KrylovSolve(method = :auto)`) stay as aliases that
construct the object, with a deprecation when a breaking release comes. Keyword defaults never pin
`Float64`: tolerances follow D2 of [[Shared Solver Layer]] (scaled by the problem's element type).

**AD backend as an argument.** Public functions take `ad::AbstractADType` (from ADTypes), never a package
name or a symbol; `ad = nothing` keeps meaning "reference differences" where it does today
(`free_energy_gradient`). Gradients are computed through DifferentiationInterface, so a new backend needs
no code of ours unless it needs a rule.

## Dependencies: ours or the ecosystem's

The lesson of P40: P14 replaced Krylov.jl by an in-house GMRES for consolidation alone, and no reason or
measurement was recorded; the in-house solver turned out to allocate where Krylov.jl's workspace does not.

1. **Default: use the SciML or JuliaHub package** where it meets the five criteria for our use
   (Krylov.jl, LinearSolve.jl, NonlinearSolve.jl, DifferentiationInterface.jl, ADTypes, Adapt.jl,
   OhMyThreads.jl, BenchmarkTools.jl, JET.jl, AllocCheck.jl, PrecompileTools.jl).
2. **Our own code needs a recorded reason and a measurement**, in the source note (`x.md`) under a heading
   "Why not <package>": what the package lacks for our use (a criterion it fails, a contract we need: D2's
   tolerance, the true residual, duals), and a comparison on our oracle tests (results, time,
   allocations, backends, devices). Q17 audits the existing reimplementations against this rule.
3. **An optional backend** goes in as a weak dependency and an extension, behind our contract (P41's
   `KrylovJL` is the model): our tolerance rule and report on top, the package's solver underneath, and
   both run on the same tests.
4. **A hard dependency** of a core package (LenticulumCore, Mycelium, MyceliumLearning, Lenticulum,
   ImplicitSolvers) needs the maintainer's approval, a `[compat]` entry, and a load-time measurement
   (`@time using`) recorded with the commit. AD packages stay weak dependencies everywhere (as today).

## Enforcement

Each criterion is checked by a test or a CI job. Rows marked *to add* name the backlog row that adds
them; until then the criterion is the standard but not a gate.

| criterion | check | where it runs | state |
|---|---|---|---|
| C1 | `tools/type_audit.jl` lists only documented exceptions | CI, per package | to add (Q2) |
| C1 | JET `@test_opt` on entry points and kernels with concrete types; ExplicitImports | CI, per package (a new JET test file) | to add (T3, Q3) |
| C1 | Aqua (ambiguities, unbound parameters, piracy, stale deps, compat) | every suite, `test/aqua.jl` | exists |
| C2 | backend matrix against finite differences (DifferentiationInterfaceTest) | every suite (a new AD-matrix test file) | to add (Q5); partial today (ImplicitLayers, VariationalDiffusion: all four) |
| C2 | no differentiation through solver loops (operator calls counted) | every suite | to add (Q6) |
| C3 | element-type tests (`Float32`, `BigFloat`, `Dual`) per factor type | every suite | to add (Q4) |
| C3 | JLArrays with `allowscalar(false)` | `CI.yml` `device` (Mycelium, MyceliumLearning); ImplicitSolvers' and VariationalDiffusion's in their suites | exists; factors to add (Q7) |
| C3 | hardware GPU | Buildkite or a self-hosted runner, weekly | to add (Q8) |
| C4 | thread-safety tests from several tasks; results equal at 1 and 8 threads | every suite, with `JULIA_NUM_THREADS=auto` in `CI.yml` | to add (Q9, Q10) |
| C5 | `benchmark/benchmarks.jl` per package compared with `dev` and `master` | a pull-request job (AirspeedVelocity.jl) | to add (Q13) |
| C5 | allocation budgets after warm-up | every suite | to add (Q12); three test files today |
| C5 | first-call latency (`bench/structure_compile.jl`, `jit_latency.jl`) | Q13's job | to add (Q13, Q15) |
| C5 | every `bench/` script runs small | `Examples.yml`, weekly | exists |

**Regression thresholds** (*to calibrate*: Q13 measures run-to-run noise on the CI runners first and
replaces these): a pull request is flagged when a benchmark's median time rises by more than 10 %
and by more than three times its measured noise, when allocations of a kernel with a budget rise at
all, or when first-call latency rises by more than 20 %. A flag asks for a reason in the pull
request; it does not block the merge until the thresholds are calibrated.

## Exceptions

An exception to a criterion is allowed with a reason, recorded in two places, as Aqua's are today:

- in the check's own file (`test/aqua.jl`, the JET and AD-matrix test files with a `@test_broken` with
  the reason in a comment, the type audit's exception list), so the check stays green and the exception
  is visible;
- in the source note beside the code, with the measured cost where one exists (a function barrier's
  dynamic dispatch, a backend that cannot differentiate a kernel).

An exception without a reason is a defect. A known limit is a `@test_broken`, not only a sentence in a
note (`AGENTS.md`).

## What this does not decide

- **Which Krylov backend is the default** (P40). This note only says how the decision is made
  (Dependencies, item 2).
- **How the graph is compiled ahead of time** (Q16, Sophia). C1–C3 make it possible; Q15 measures first.
- **The thresholds** above, until Q13 has measured the noise.
- **The minimum Julia version** stays 1.12 (`Weekly.yml`, `oldest`).

## Open questions

- Is the size split of "How a kernel is written" at the right place? Below which dimension an
  out-of-place kernel is faster than an in-place one with a workspace is not measured here; Q14's
  profiles and Q12's budgets will show it per kernel.
- Running every suite with `JULIA_NUM_THREADS=auto` on push changes the timings of `CI.yml`; not measured.
- Mooncake and Enzyme on the inference core (Mycelium's functional free energy) are untested (baseline);
  C2 may need more `@test_broken` there than anywhere else before Q6 lands.
