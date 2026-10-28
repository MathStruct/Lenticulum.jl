#implementation

> Opt-in first-order parameter adjoints of a converged BP message fixed point.
> A dense reference solve and an optional matrix-free GMRES solve share the same
> message equation, cuts and fixed-point checks. Ordinary inference is unchanged.

> Sources: original derivation from the message fixed-point equation; code:
> `graph_adjoint.jl`, `gradient_couplings.jl`, `ext/MyceliumKrylovExt.jl`; tests:
> `lib/Mycelium.jl/test/graph_adjoint.jl`, `graph_adjoint_devices.jl`.
>
> Theory (CT-ML wiki): [Statistical Game](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game)

## Equation and API (2026-10-28)

```julia
free_energy_gradient(store, g, ps, st; ad=AutoZygote(),
    schedule=flooding_schedule(g), adjoint=:implicit,
    maxsweeps=1000, tol=1e-10, damping=0.0,
    linear_solver=:dense, linear_tol=1e-10, linear_maxiter=1000)
```

`parameter_step` forwards the same keywords and retains its deterministic
three-tuple return. The input store is untouched. Reference inference starts
from empty messages at the look-ahead parameters; `nsweeps` only controls the
unrolled driver. The default remains `adjoint=:unrolled`, and all-diagonal
calls without the opt-in retain their original fixed-cavity derivative.

Let $m$ concatenate every legal directed message, $\theta$ the active parameters,
and $L(m,\theta)$ the scalarised local free energies plus Bethe variable corrections.
The map $T$ computes **all** candidate messages from the same input store.
It uses the ordinary factor and variable exclusions and damps only outgoing
factor messages, when an old message exists and the belief family supports damping.
The chosen reference schedule finds a root; it does not select a different
fixed-point equation. Sequential, tree, forward/backward, flooding and residual
reference schedules are accepted. Static lists must cover every legal message.

At a differentiable root $m^*=T(m^*,\theta)$,

$$
(I-D_mT)^\top\lambda=\nabla_mL,\qquad
\nabla_\theta L_{\mathrm{total}}=\partial_\theta L+(D_\theta T)^\top\lambda.
$$

The implementation differentiates $L(m^*,\theta)+\lambda^\top T(m^*,\theta)$
with $m^*$ and $\lambda$ held constant. It never differentiates convergence
tests or the reference iterations. This is a local derivative of the selected
root branch. Differentiability and invertibility of $I-D_mT$ are required.
A contraction is sufficient for local invertibility and iteration convergence;
convergence by itself does not prove either differentiability or invertibility.
No spectral-radius bound or uniqueness guarantee is computed.

## Boundaries and representations

Exact and Pathwise couplings are supported; Pathwise still requires Gaussian,
Dirac or uninformative messages. A diagonal/detached attachment replaces each
of its legal directed update rows by their constant reference coordinates. Hence
$\delta m_{\mathrm{cut}}=0$: paths through those rows disappear from the linear
system. Downstream columns can still read the fixed boundary values. Frozen or
intrinsically nonlearnable parameter coordinates are pinned, preserving input
message derivatives through their factors. The mixed system is the fixed-point
counterpart of the cut replay. It agrees with its limiting gradient when the
differentiated replay converges; invertibility alone does not ensure that limit.
This need not be the ordinary derivative of the converged primal objective
with no cuts.

The coordinate adapter supports CPU real floating Gaussian canonical parameters
(`Vector` information vector, dense `Matrix` or vector-backed `Diagonal` precision),
numeric scalar/CPU-array Diracs, and coordinate-free Trivial messages. Dense
precisions use all matrix entries; diagonal precisions use their diagonal only.
Message families, shapes, polarity choices and support must be locally stable.
There is no device coordinate adapter, implicit score estimator, state derivative,
higher-order rule, arbitrary outer-AD rule, or differentiation of graph structure.

Incoming state and every state returned by a message update must be
recursively empty. Stateful
factors would require either a larger fixed-point equation including state or a
specified fixed-state map; silently dropping state is not such a specification.
Reference differences (`ad=nothing`), DI ForwardDiff and Zygote are the tested
backend choices. Unsupported backends, messages, states and solver choices throw
`ArgumentError`. Krylov requires explicitly loading Krylov.jl; Mycelium has no
dependency on ImplicitLayers.

## Validation, cost and difficulties

The reference loop verifies convergence and then checks every ordinary synchronous
BP row before applying derivative cuts. This avoids trusting the ordinary one-sweep
TreeSchedule report for damped or manually assembled schedules. Both linear
solvers check the original adjoint residual against
`linear_tol * max(1,norm(rhs))`; singular dense systems, failed GMRES solves and
nonfinite results fail. These checks are numerical evidence, not a condition-number
estimate. Tight finite-difference linear tolerances can fail due to cancellation.

For $N$ message coordinates, dense assembly stores $O(N^2)$ entries and dense LU
costs $O(N^3)$. GMRES stores $O(\min(N,30)N)$ Krylov vectors and applies
$v\mapsto v-(D_mT)^\top v$ with a backend gradient; ForwardDiff obtains that
gradient with forward seeds, so choosing it does not give reverse-mode cost.
Reference inference, AD workspace and functional store copies add to these costs.
Each map evaluation currently copies edge-sized vectors while rebuilding stores,
so this implementation is a checked baseline rather than an allocation claim.
Its differentiation memory does not grow with the reference sweep count.

The full-suite ambiguity check caught an overly broad `mul!` method on the new
operator: it intersected LinearAlgebra's multiplication by an `AbstractQ`.
The operator now restricts both operands to vectors, which is the GMRES contract.
Its advertised element type follows the adjoint RHS, including promotion by CPU
scalar objective constants; message coordinates themselves are unchanged.

The Gaussian fixtures are independent unnormalised unary/pair potentials,
including a correlated two-dimensional dense-precision tree oracle. On a
tree, the Bethe value and gradient agree with the dense global Gaussian $-\log Z$
and converged exact replay. On the loop, references reconverge BP at each parameter
perturbation; the test measures convergence of the unrolled gradient to this
adjoint. Tests also cover cuts, freezing, damping, parameter steps, Diracs and
failure paths. Device replay uses diagonal Gaussian fixtures with scalar indexing
disabled; device implicit calls explicitly fail for both linear solvers. See
LP-002's dated batch C section for run measurements and verification results.
