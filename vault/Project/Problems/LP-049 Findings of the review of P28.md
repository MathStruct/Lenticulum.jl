#open-problem

> **Status: resolved (2026-11-17).** The review of P28 (a `NonlinearFactor` with no input on $x$ sends
> the forward message of its linearisation at `x0`) found no defect in seven probes. One observation
> went into the note: at an `x0` where the Jacobian loses rank (a critical point of $f$) the message is
> as confident as the linear model, which is wrong for $f$.
> Related: [[Feature Backlog]] (P28, P29), [[lib/Lenticulum.jl/src/nonlinear|nonlinear]] "No input on x".

> Sources: a review of the uncommitted P28 change by Mistral (`vibe --agent review`, Mistral Large 4,
> `--max-price 3`, $0 recorded), in the helper worktree `Lenticulum.jl-mistral` at `cd22f574` with the
> diff applied. It ran to its 100-turn limit. The brief, the answer, the probes and the outputs of
> probes 5–7 are in `../Lenticulum.jl-Review-2026-11-17/p28/`.
> code: `lib/Lenticulum.jl/src/nonlinear.jl`; test: `lib/Lenticulum.jl/test/p28_no_input.jl`.

**Problem.** The requirement given to the reviewer: with no information on $x$ (no entry, `nothing`,
`TrivialBelief()`, zero precision, everywhere flat, mixtures of those) the forward message is, with
`x0`, the flat-$x$ message of the linear model at `x0` ($\Lambda = U(U^\top QU)^{-1}U^\top$,
$\eta = \Lambda b$, $U$ a basis of $\operatorname{null}(A^\top)$), the same for both methods and equal
to `GaussianFactor`'s for an affine $f$; without `x0`, `TrivialBelief()`; the polarity $x$ `Latent()`,
$y$ the target, is listed only with `x0`; in a graph the factor informs $y$.

The answer file records probes 1–4; probes 5–7 are not in it. The maintainer ran them (outputs
`probe5.out`–`probe7.out`).

| # | finding | right? | what was done | test |
|---|---|---|---|---|
| 1 | Probes 1–3: the closed form for every input form, ten shapes, every rank, three unscented settings, exact and difference Jacobians; affine $f$ against `GaussianFactor` and through the lens (`likelihood_message`, `invert`); the closed form itself against the `BigFloat` limit | no defect (worst $2.2\cdot10^{-14}$); difference Jacobians with rank-deficient $A$ never claim more precision than the exact message (105 cases), the recorded P29 limit | — | — |
| 2 | Probe 4: a chain $x \to y \to z$ against a dense joint solve, tree and sequential schedules; `scalar_free_energy` equal to the `GaussianFactor` graph; the ForwardDiff free-energy gradient against differences; an observed $y$; a loopy graph | no defect | — | — |
| 3 | Probes 5–6: `_gradient_message` (86 cases); a plate of four `NonlinearFactor`s against the single graph (free energy equal, difference 0.0) | no defect | — | — |
| 4 | Probe 7: edge cases (constant $f$, scalar noise, square full rank, `iterations > 1`, backward with no input on $y$, `invert` with a trivial prior, the UKF regression); an improper nonzero input on $x$ forwards throws `ArgumentError` | no defect; the throw predates P28 and is in the note's "Does not do" | — | — |
| 5 | Probe 7: $f(x) = x^2$ at $x_0 = 0$ gives $\mathcal N(0, Q)$ on $y$ | right as a property, not a defect of the requirement: at a critical point the tangent plane is a point, so the linear model claims $y \approx f(x_0)$ | a sentence in the note ("Does not do") | — |

Not reviewed: Zygote through the new path; `MeanField()` nodes.
