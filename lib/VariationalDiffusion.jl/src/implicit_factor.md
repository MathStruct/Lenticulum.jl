#implementation

> `DiffusionFactor` with the deterministic implicit solver as its inversion:
> `DiffusionFactor(blocks, pred; prox = ImplicitProx(nodes))`. `invert`, `factor_message` and
> `local_free_energy` work unchanged; `implicit_solution` returns the solver's report and
> `implicit_factor_pullback` the backward pass, per channel.

> Sources: code: `implicit_factor.jl`, `factor.jl`, `implicit.jl`
>
> Theory (CT-ML wiki): [Bayesian Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Lens) · [Statistical Game](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game) · [Open Model](https://mathstruct.org/CategoryTheory-ML-Wiki/Open-Model)

## 1. How it is wired

- `DiffusionFactor`'s `prox` field may be a `REDDiff` or an `ImplicitProx`. `_run_prox(f, …)`
  dispatches on the **type of the field**, so `invert`, `factor_message` and `local_free_energy`
  in `factor.jl` needed no change beyond that one line.
- The problem is built from the factor exactly as RED-Diff's is: `assemble_state` lays the
  incoming beliefs into the state vector, `precision_vector` reads the polarity, and infinite
  precisions become `Inf` entries (hard clamps) of the solver's precision vector.
- The **prior is the warm start**, as in `ImplicitLayers`' DEQ factor: its point seeds the solver,
  and on a multivalued relation it selects the branch. Whether it also *anchors* the solution is
  the polarity's call: `default_precision(Unobserved()) == 1` is an anchor; precision `0` is a
  pure conditional.
- The `score` summand of the graded energy is the fixed-node quadrature of the weighted
  denoising loss, so the factor's free energy is **deterministic** (RED-Diff's is one
  Monte-Carlo sample).

## 2. The backward pass, per channel

`implicit_factor_pullback(f, polarity, inputs, π, z̄, ps, st)` takes the downstream cotangent
as a `NamedTuple` over channels and returns cotangents for each channel's incoming point
(`inputs`), for the predictor's parameters (`ps`), and for each channel's precision
(`precisions`). A channel has one precision for all its coordinates, so its gradient is the
sum over the block. A hard clamp's precision is infinite and has no gradient (zero is returned).

## 3. Implementation difficulties

1. **Dispatch on a value parameter.** `DiffusionFactor{names,D,P,C}` has a *value* parameter
   (`names`, a tuple of symbols). A method on `DiffusionFactor{<:Any,<:Any,<:Any,<:ImplicitProx}`
   never matches, because `<:Any` constrains to types; written with `where {N,D,P}` it matched
   in `isa` but lost on specificity against the generic method, because the bounds of `D` and
   `P` were dropped. Dispatching on `f.prox` instead sidesteps both.
2. **Include order.** `ImplicitProx` must exist before `DiffusionFactor`'s type bound is
   evaluated, so `analytic.jl` and `implicit.jl` are included before `factor.jl`, and the
   functions that need both live in this separate file.
3. **What is still not fixed.** The diffusion model's own prior is in every message, so a
   message is a posterior, not a likelihood ([[The Diffusion Factor]] §4); §5 divides out
   the anchor, which is the part that can be divided out.

## 4. How it is tested

The test set "DiffusionFactor with the implicit solver as its inversion": branch selection by
the prior, agreement with the bare solver, the message, a deterministic free energy, the
per-channel pullback against finite differences of `invert`, the dimension check, and the error
for a RED-Diff factor.

The test set "answers as beliefs" covers §5: the energy of a closed-form mixture against the
field, `laplace_belief` against `implicit_laplace`, both branches of a circle query as a mixture
whose weights recover a 3:1 reweighting of the data (odds ratio 3.0), and the factor's Gaussian
message, which times the anchor gives the posterior.

## 5. Gaussian messages

`ImplicitProx(nodes; λ, message = :gaussian)` makes the factor return a `GaussianBelief` on the
target channel instead of a `DiracBelief`:

| call | returns |
|---|---|
| `invert` | the Laplace posterior at the answer, marginalised onto the target ([[implicit]] §5) |
| `factor_message` | the same, with the target's own anchor divided out |

The solver's query multiplies in an anchor on the target, $\mathcal N(z_0, \rho^{-2})$, from the
incoming belief and the polarity's precision. If the message kept it, the variable would
count its own belief twice: once inside the message, once when it pools the message with
itself. In canonical form the division is a subtraction,

$$\Lambda_{\text{msg}} = \Lambda_{\text{post}} - \operatorname{diag}(\rho^2), \qquad \eta_{\text{msg}} = \eta_{\text{post}} - \rho^2 z_0,$$

exact for the Laplace approximation. Marginalising onto the target commutes with it, because the
anchor is diagonal and sits on the target's own coordinates. The result may be improper, a
likelihood that constrains only some directions, which `GaussianBelief` allows.

What cannot be divided out is the diffusion model's own prior: the message is the model's
*conditional* belief about the target given the inputs, not a likelihood
([[Open Problems in Implicit Diffusion Learning]] T10). Two more conditions:

- covariances are in data units only with `λ = density_lambda(schedule, nodes)`;
- the answer must be converged and stable; otherwise the call throws rather than sending a
  meaningless curvature.

Related: [[implicit]], [[factor]], [[The Diffusion Factor]], [[Inference Signatures]]
