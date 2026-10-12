#design #later

> What the backward pass should look like at each level where inference has a forward
> signature: the solver, the factor and the graph. **Short version:** a backward function takes
> the forward's arguments, the forward's *result* and a cotangent of that result, and returns
> cotangents shaped like the inputs. On top sit `ChainRulesCore.rrule`s, so that Zygote (and
> Enzyme, by import) differentiate through `infer!` without users calling the pullbacks. The
> solver level exists, with its `rrule` (§6, step 1); the factor and graph levels are a design.

> Sources: original to this vault (design); the ChainRules.jl `rrule` / `Tangent` / `NoTangent` conventions; Lux.jl's `(ps, st)` convention; Bai, Kolter & Koltun, *Deep Equilibrium Models*, NeurIPS 2019 (backward by the implicit function theorem); code: `implicit.jl` (`implicit_pullback`), `implicit_factor.jl` (`implicit_factor_pullback`), `solve.jl` / `deq.jl` (`ift_sensitivity`, `deq_sensitivity`), `passing.jl` (`factor_message`, `propagate!`, `infer!`), `lens.jl`
>
> Bibliography: [[Bibliography#^bai2019deq|Bai et al. 2019]]
>
> Theory: [[Inference Signatures]] · [[Backpropagation through Implicit Inference]] · [[Backpropagation by the Implicit Function Theorem]] · [[Inversions and Bayesian Lenses]]
>
> Theory (CT-ML wiki): [Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Lens) · [Reverse Derivative Category](https://mathstruct.org/CategoryTheory-ML-Wiki/Reverse-Derivative-Category) · [Bayesian Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Lens)

## 1. Why the two signatures are related

`lens.jl` opens with the analogy this note relies on:

$$
\text{autodiff: } (g\circ f)^*(a, \bar c) = f^*\bigl(a,\ g^*(f(a), \bar c)\bigr)
\qquad
\text{Bayes: } (d\circ c)'_\pi = c'_\pi \circ d'_{c_*\pi}
$$

Both are the `put` of a lens. Inference is conditioned on a **prior** $\pi$; backpropagation is
conditioned on the **forward point** $f(a)$. So a backward signature is the forward signature
with the prior replaced by the forward result, plus a cotangent of that result, and the
cotangents it returns compose by the same chain rule the messages do.

## 2. The forward signatures

| level | forward |
|---|---|
| solver | `implicit_infer(m, z₀, ρ, ps, st) -> (sol, st)` |
| factor | `factor_message(f, target, polarity, inputs, prior, ps, st) -> (belief, st)`, generically `assemble` then `invert(lens, π, y, ps, st)` |
| graph | `infer!(g, sched, ps, st) -> (marginals, report, st, store)` |

## 3. The backward signatures

### 3.1 Solver (exists)

```julia
implicit_pullback(m, sol, z₀, ρ, z̄, ps, st) -> (z₀ = z̄₀, ρ = ρ̄, ps = ps̄)
```

This is already the target shape:
- it takes the forward result `sol`, so nothing is solved twice;
- it returns a `NamedTuple` keyed like the differentiable inputs;
- it returns no cotangent for `st`;
- it throws at a non-converged state, because the implicit function theorem says nothing there
  ([[Backpropagation through Implicit Inference]] §6).

The equilibrium family has only forward sensitivities (`deq_sensitivity`, by finite
differences, a test-scale tool) and no parameter cotangent. Its pullback would have the same
signature, with the adjoint solve $(I-\partial_z g)^\top\lambda = \bar z$ in place of
$J_{FF}^\top\lambda = -\bar z_F$.

### 3.2 Factor

```julia
factor_pullback(f, target, polarity, inputs, prior, ps, st, out, out̄)
    -> (inputs = NamedTuple, prior = π̄, ps = ps̄)
```

- `out` is the forward message, or the solver state behind it (an `ImplicitSolution`), so the
  backward reuses the root instead of solving again.
- `out̄` is the cotangent of a **belief**. A `DiracBelief` has a vector cotangent. A
  `GaussianBelief` has `(η = η̄, Λ = Λ̄)`, in the natural parameters it is stored in: a ChainRules
  `Tangent{GaussianBelief}`. A mixture has one entry per component plus the weights.
- The returned `inputs` is keyed by channel and holds belief cotangents, the same types as
  `out̄`. The precisions of soft channels are part of those beliefs' cotangents (the $\Lambda$
  part), not a separate field.
- `prior` has a cotangent because $c'_\pi$ depends on $\pi$; for a `TrivialInversion` it is
  `NoTangent()`.

`implicit_factor_pullback(f, polarity, inputs, π, z̄, ps, st)` is the diffusion family's
version today, and it differs from this in three ways:
1. it re-solves instead of taking the forward result;
2. it accepts only point cotangents, so the `message = :gaussian` path has no backward pass. That
   one differentiates the Laplace covariance $(\operatorname{sym} J_{FF})^{-1}$, which needs
   third derivatives of the field (a Hessian-vector product of the Jacobian);
3. it returns `precisions` as a separate field, one summed value per channel.

Structural factors (`DataFactor`, `PriorFactor`, `RelayFactor`) have trivial pullbacks: the
identity on the belief, or `NoTangent()`.

### 3.3 Graph

```julia
infer_pullback(g, sched, ps, st, store, marginals̄) -> (ps = ps̄, data = …)
```

`marginals̄` is keyed by variable name, like `marginals`. `ps̄` has the layout of the graph's
`ps`, one entry per factor. `data` holds cotangents for observed values, if the caller wants
them. Two ways to compute it:

| | **unrolled** | **implicit** |
|---|---|---|
| what | reverse-mode AD through every `step!` of the schedule, using the factor pullbacks of §3.2 | the implicit function theorem on the converged messages $m^\star = F(m^\star,\theta)$ |
| exact for | trees (one sweep *is* the answer) | any converged fixed point |
| memory | grows with the number of sweeps | constant: one backward sweep of **adjoint messages** over the reversed schedule, iterated to its own fixed point |
| fails when | the unroll diverges | the forward did not converge, or $I - \partial_m F$ is singular |

`propagate!` already returns a `ConvergenceReport`; the implicit pullback should refuse when it
says `converged = false`, exactly as `implicit_pullback` does. The unrolled pullback can run
anyway, but it then differentiates the iteration, not the fixed point.

## 4. The closure form, for AD packages

Users should not have to call any of this. Each level gets a `ChainRulesCore.rrule` that returns
`(y, back)`, where `back` closes over the forward result and calls the explicit pullback:

```julia
function ChainRulesCore.rrule(::typeof(implicit_infer), m, z₀, ρ, ps, st; kwargs...)
    sol, st′ = implicit_infer(m, z₀, ρ, ps, st; kwargs...)
    function back((sol̄, _))
        b = implicit_pullback(m, sol, z₀, ρ, sol̄.z, ps, st)
        return (NoTangent(), NoTangent(), b.z₀, b.ρ, b.ps, NoTangent())
    end
    return (sol, st′), back
end
```

Then `Zygote.withgradient(ps -> loss(first(infer!(g, sched, ps, st))), ps)` works as it does for a
Lux model, and Enzyme can use the same rules through `Enzyme.@import_rrule`. The rules live in a
package extension on ChainRulesCore, which is light. The explicit functions stay the documented,
tested API, and every rule is checked against finite differences, as `implicit_pullback` is.

## 5. Conventions

- **No `st` cotangent.** State (warm starts, solver caches, optimiser moments) is not
  differentiated, as in Lux.
- **Hard clamps get `NoTangent()`**, not a zero array: an infinite precision is not a
  differentiable quantity, and `NoTangent` lets AD skip the branch.
- **Cotangents have the shape of the primal.** A belief's cotangent is a `Tangent` of the belief
  type; a factor's parameter cotangent has the layout of its `ps`; a graph's has the layout of
  the graph's `ps`.
- **Non-convergence throws** for implicit pullbacks. A caller who wants a gradient anyway asks
  for an unrolled or one-step surrogate explicitly.

## 6. What to build, in order

1. An `rrule` for `implicit_infer` around `implicit_pullback` — **built**, in the extension
   `VariationalDiffusionChainRulesCoreExt`. It passes the solver's `linsolve` on to the
   pullback, returns no cotangent for the report fields (residual, iterations, flags), and is
   tested against `implicit_pullback` and finite differences, including a Lux network trained
   through inference with `Zygote.gradient`.
2. `factor_pullback` as a generic function in `Mycelium`, with methods for the structural factors
   and the implicit diffusion factor (rewriting `implicit_factor_pullback` to take the forward
   result).
3. The unrolled graph pullback, which needs only 1 and 2 plus AD through `step!`. Enough for
   trees.
4. The implicit graph pullback (adjoint message passing), for loopy graphs.
5. The Gaussian-message backward (third derivatives), when a loss is placed on a covariance.

Related: [[Inference Signatures]], [[Backpropagation through Implicit Inference]],
[[The Inferencer and the Optimizer]], [[Everything is a Factor]], [[Lux as a Parametric Lens]],
[[RxInfer as a Backend]]
