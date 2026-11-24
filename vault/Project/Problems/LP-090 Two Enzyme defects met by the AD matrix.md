#problem

> **Status: open (2026-11-24), upstream; reported as EnzymeAD/Enzyme.jl#4130 (defect 1) and #4131 (defect 2).** Two defects of Enzyme 0.13.221 (Julia 1.13.1),
> found through Q5's AD matrix, each with a minimal example that needs Enzyme alone. (1) Enzyme writes the
> adjoints of constant data into that data when the data reaches the differentiated code through a call
> dispatched at run time: the gradient is right, the caller's arrays are changed. It hits Mycelium's
> `ExposedFactor` (`local_free_energy` under Enzyme). (2) Enzyme loaded without a package image throws
> `UndefVarError: vaEltypes not defined in Enzyme` from a generated rule. This was the "order-dependent" row of
> [[LP-089 Findings of the review of Q5|LP-089]]; it does not depend on order. Related: [[Engineering Standard]]
> (C2), [[Backlog Q Quality and Engineering]] (Q6).

> Sources: Enzyme.jl 0.13.221, `src/Enzyme.jl` (lines 234 and 293), `src/rules/jitrules.jl` (line 1686); upstream
> issues EnzymeAD/Enzyme.jl#2868 and #3136 (the same family as defect 1, both closed 2026-06-08, before 0.13.221).
> code: `lib/Mycelium.jl/src/exposure.jl` (`_restrict`, 141), `lib/Mycelium.jl/test/ad_matrix.jl` (`MUTATES`),
> `lib/LenticulumCore.jl/test/support/ad_matrix.jl` (the fixture check).

## Defect 1: Enzyme writes into memory passed as `Const`

A closure over two arrays is passed to `autodiff` as a `Const` argument. The arrays reach the differentiated code
through one call dispatched at run time. The gradient with respect to `v` is right; the adjoints of the
"constant" arrays are added into the arrays.

```julia
using Enzyme                     # 0.13.221, Julia 1.13.1
energy(m, v) = (m.y[1] - m.x[1] - v[1])^2
apply(v, f) = f(v)
dynamic_closure(m) = v -> energy(Base.inferencebarrier(m), v)   # one dynamic call
static_closure(m) = v -> energy(m, v)

for make in (static_closure, dynamic_closure)
    m = (x = [0.8], y = [1.8])
    v = [0.3]; dv = zero(v)
    autodiff(Reverse, apply, Active, Duplicated(v, dv), Const(make(m)))
    println(make, ": gradient ", dv, ", m after ", m)
end
# static_closure:  gradient [-1.4], m after (x = [0.8], y = [1.8])
# dynamic_closure: gradient [-1.4], m after (x = [-0.6], y = [3.2])     <- x += ∂/∂x, y += ∂/∂y
```

What matters, from the variants tried (scripts in the session's scratch directory, not kept):

| variant | result |
|---|---|
| closure as a `Const` **argument**, dynamic call | right gradient, data changed |
| same, data in a wrapper struct, or through Mycelium's `_restrict` | same |
| closure as a `Const` **argument**, static call | right gradient, data unchanged |
| closure as the `Const` **function** (`Enzyme.gradient(Reverse, Const(f), v)`), static or dynamic | `EnzymeRuntimeActivityError` (correct: it refuses) |
| data in a global instead of a captured local | right gradient, data unchanged |
| `set_runtime_activity(Reverse)` (through DI) | right gradient, data changed |

DifferentiationInterface passes a `DI.Constant(f)` context exactly as a `Const` argument, so
`DI.gradient(apply, AutoEnzyme(), v, DI.Constant(f))` corrupts the data too. It is how Q5's helper calls Enzyme
(`apply_loss`), and how LenticulumCore's public AD helpers pass their fixtures.

**Where it hits us.** `local_free_energy` of an `ExposedFactor` (Mycelium) under Enzyme: the messages `x` and `y`
change from 0.8 and 1.8 to 1.71 and 0.97, and the gradient is right. The dynamic call is `_restrict(nt, ks)`
(`exposure.jl` 141), which builds a `NamedTuple` from a filtered generator, so its type is known only at run
time. No other factor-level row of the matrix changes its fixture under any backend (checked 2026-11-24, all five
matrices, both tiers). Graph rows (`free_energy`, `infer!`, the graph adjoint) are not checked: inference fills
the graph's caches and the store, so their fixtures change legitimately.

**In the matrix.** The helper now copies the fixture before each backend and checks it afterwards (`same`). A
backend that changes it fails, unless the row lists it in `mutates` (a `@test_broken`). Mycelium's `MUTATES` has
`local_free_energy/ExposedFactor => Enzyme`.

**On our side.** A type-stable `_restrict` (keys computed at compile time) would avoid the dynamic call, and with
it the corruption on this path. Not done: the defect is Enzyme's, and the matrix records it. Whether other
type-unstable paths of the inference core corrupt data under Enzyme is not known; the graph rows, where they
would show, are refused or broken under Enzyme today.

## Defect 2: `UndefVarError: vaEltypes` when Enzyme has no package image

`Enzyme.jl` includes `compiler.jl` (line 234, which includes `rules/jitrules.jl`) before it defines `vaEltypes`
(line 293). The generated function behind `runtime_iterate_rev` (`jitrules.jl` 1686) reads `Enzyme.vaEltypes`
while it builds its body. Since Julia 1.12, a generator runs in the world where its method was defined. When
Enzyme is loaded from source, file by file, that world predates the binding, and the generator throws. From a
package image, all of Enzyme arrives in one world and the binding exists.

```julia
using Enzyme                     # run with: julia --compiled-modules=no
apply(v, f) = f(v)
energy(t, v) = sum(b -> (b[1] - v[1])^2, t)
# A tuple of run-time length from a filtered generator: Enzyme differentiates the
# iteration with runtime_iterate_rev.
closure(m) = v -> energy(Tuple(a for a in Base.inferencebarrier(m) if length(a) == 1), v)
m = ([0.8], [1.8], [1.0, 2.0])
v = [0.3]; dv = zero(v)
autodiff(Reverse, apply, Active, Duplicated(v, dv), Const(closure(m)))
# with a package image:      dv == [-4.0]
# --compiled-modules=no:     UndefVarError: `vaEltypes` not defined in `Enzyme`
#   Suggestion: define the const at top-level before running function that uses it (stricter Julia v1.12+ rule).
#   Note: the binding state changed since the error occurred (was: 11, now: 0).
```

**Why it looked order-dependent.** Codex ran the matrices with `--compiled-modules=existing --check-bounds=yes`.
No `--check-bounds=yes` package image of Enzyme existed then, and `existing` does not build one, so Enzyme was
loaded from source. `Pkg.test` builds the image, so the suites never meet it: under `Pkg.test` both rows pass,
in any order. The `unstable` class that LP-089 added for them is removed.

## Gates after the helper change (2026-11-24)

`Pkg.test()` of ImplicitSolvers, LenticulumCore, Mycelium and MyceliumLearning (default tier): passed. All five
`ad_matrix.jl` files alone with `LENTICULUM_SLOW_TESTS=true` (both tiers, `--check-bounds=yes`, a scratch
environment with the five packages): passed, ImplicitSolvers 44 pass / 14 broken, LenticulumCore 56 / 8,
Mycelium 115 / 37, MyceliumLearning 7 / 9, Lenticulum 186 / 67. Lenticulum's whole suite was not rerun: only
its matrix file changed. Logs: `~/.cache/lenticulum/q5/main2/`, `mut2/`, `mut3/`.

## Upstream

Reported 2026-11-24 with the maintainer's approval: defect 1 as EnzymeAD/Enzyme.jl#4130, defect 2 as #4131
(both rerun that day on 0.13.221, the latest release; `src/Enzyme.jl` on Enzyme's `main` at e91ff369f6 still
defines `vaEltypes` after including `compiler.jl`). No earlier issue matched either. #2868 (a `Const` argument
changed by a gradient, with runtime activity and a copy into a `Duplicated` buffer) and #3136 (reverse mode
leaving an internal ODE buffer dirty) are the same family as defect 1 and were closed before 0.13.221; defect 1
reproduces without runtime activity. When #4130 is fixed, Mycelium's `MUTATES` row turns into an unexpected
pass and must be removed.

## Not checked

Other Enzyme versions; Julia 1.12 (only 1.13.1 was run); forward mode; whether defect 1 needs the closure to be a
`Const` argument of an outer function or also happens with other activities of the outer function.
