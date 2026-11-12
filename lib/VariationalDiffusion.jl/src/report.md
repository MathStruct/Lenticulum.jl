#implementation

> Every query returns a root, and the root alone does not say how far it can be trusted.
> `query_report` answers the query and runs the checks the package already has on the answer:
> the solve's own flags, the roots reached from other starts, the conditioning of the
> free-coordinate Jacobian, the round trip and, given data, coverage. What failed is listed as
> flags. Feature P11.

> Sources: code: `report.jl`, `answers.jl` (`implicit_roots`, `round_trip`), `coverage.jl`, `implicit.jl` (`_adjoint_solve`'s singularity test), `krylov.jl` (`_lambda_min`); tests: `test/runtests.jl`, testset "query report: what an answer can be trusted for (P11)"
>
> Theory: [[Open Problems in Implicit Diffusion Learning]] (T1, T4, T5) · [[implicit]] §5 · [[coverage]] · [[calibration]] "Remedies"

## 1. What it reports

| check | how | flag |
|---|---|---|
| the solve | `ImplicitSolution.converged`, `.stable` | `:not_converged`, `:unstable` |
| other starts | `implicit_roots` with `nstarts` random starts around `z₀` (and `z₀` itself); `rank` is the answer's index among the distinct stable roots, 0 if absent | `:other_roots` when more than one root is found; `:not_lowest` when the relation has an energy and a root found has a query energy below the answer's by more than $\sqrt\epsilon$ relative |
| conditioning | dense: $\operatorname{cond}_1 J_{FF}$ and $\lambda_{\min}(\operatorname{sym}J_{FF})$; `KrylovSolve`: $\lambda_{\min}$ by Lanczos, no condition number | `:ill_conditioned` when $\operatorname{cond}_1 J_{FF}\,\epsilon \ge 10^{-2}$, ε of $J_{FF}$'s element type |
| on the relation | `round_trip`, for a converged stable answer only | `:round_trip` when the reversed query finds no stable point, or the distance exceeds `round_trip_tol` (none by default) |
| coverage | `coverage(reference, sol; alpha)` when a `CoverageReference` is passed | `:atypical` when `coverage` flags it |

`trusted(r)` is `isempty(r.flags)`.

## 2. Design decisions

1. **One record, no new checks.** Every check existed; what was missing is that nothing ran them
   together or said which failed. The report adds no heuristic of its own.
2. **The singularity test is the pullback's.** `:ill_conditioned` uses the rule by which
   `implicit_pullback` refuses an answer (`_near_singular`, now shared by both), so a report
   without that flag means the gradient through the answer exists numerically.
3. **A finite round-trip distance is judged only against a caller's tolerance.** No stable
   point on the way back is always flagged. The distance is in data units, and the calibration benchmark showed it fails also on true roots where the
   relation is parallel to the clamped coordinates (`calibration.md` "Remedies", point 3). A
   default tolerance would be a claim about the data's scale that the library cannot make.
4. **`:not_lowest` by energy, and only with an energy.** The query energies of the answer and the
   roots are compared, not ranks: the answer may be missing from the roots, and tied roots come in
   no particular order (LP-031). Without an energy, `implicit_roots` orders roots as found, and
   nothing says which is more plausible.
5. **`:other_roots` is a flag, not a failure of the answer.** A multivalued relation (a circle
   queried for $y$ given $x$) has several correct answers; the flag says that the answer is one of
   several, which a caller that wants one value must know (T1). `implicit_mixture` returns all.

## 3. What it does not do

- It does not find every root: `implicit_roots` has no guarantee, so the absence of
  `:other_roots` is not proof of uniqueness.
- No ridge residual (the rest of F3): membership in the relation is tested only by the round trip
  and coverage.
- No energy threshold, no annealing.
- The condition number is not computed matrix-free.

## 4. Difficulties

Mistral's review found four flags that did not mean what the docstring said
([[LP-031 Findings of the review of the query report]]): ε taken from the state instead of
$J_{FF}$ (so a Float32 state with Float64 precisions was flagged where the pullback works), rank used
for energy (twice), and an `Inf` round trip unflagged without a tolerance. The cost is the point to know: about `nstarts + 3` solves (the answer, the
starts, the round trip), plus one Jacobian block, plus the coverage scores.

## 5. How it is tested

Closed forms where they exist (`test/runtests.jl`, the testset named above):

- one Gaussian: one root, $J_{FF} = \kappa + \rho^2$, so $\operatorname{cond}_1 = 1$ and
  $\lambda_{\min} = \kappa + 4$ to $10^{-10}$; the Lanczos value agrees to $10^{-8}$; the round trip
  sends the clamp $x = 0.3$ to the slice mode $x = 0$, and the flag follows the tolerance;
- the ring: two roots for $y$ given $x$, flagged `:other_roots` and nothing else; the query
  $x = 1.3$ flagged `:round_trip`, and with a coverage reference `:atypical`, the on-ring answer
  not;
- two components of weights 0.8 and 0.2: from a start near the lighter one the answer has rank 2
  and `:not_lowest`, from the heavier one rank 1; with the answer missing from the roots (rank 0)
  still `:not_lowest`; with equal weights and the answer found second, not;
- a Float32 state with Float64 precisions at $\operatorname{cond}_1 \approx 6\cdot10^7$: no
  `:ill_conditioned`, and the pullback runs; a reversed query with no stable point: `:round_trip`
  without a tolerance;
- a given unconverged, unstable solution: both flags, no round trip; an unreachable tolerance
  gives `:not_converged`.

Not tested: `:ill_conditioned` on a real fold (only the shared rule is tested on numbers), a
non-conservative network, the $\sqrt\epsilon$ margin of `:not_lowest` with a Float32 network.
