#implementation #derivation

> `fit_mle` fits finite toric models by Newton or cyclic binary-feature IPF. It reports
> moment convergence separately from a sufficient existence certificate or a boundary
> heuristic. `inmodel` checks positive points; `binomial_residuals` evaluates supplied
> integer-kernel equations, including at zero probabilities.

> Sources: derivation from [[Toric Models as Factors]] (Birch's theorem and the relative-interior existence condition, Fienberg–Rinaldo Theorems 9 and 13 in the arXiv numbering cited there) and [[L-SPEC-004 Algebraic Factors Toric and Grassmannian]] §1.6–§1.7; the numerical choices below are original to this implementation; code: `fit.jl`, `../test/runtests.jl`.
>
> Theory (CT-ML wiki): [Statistical Game](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game)

## Moments, gauge and Newton

For counts $u$ with positive total $n$, set $q=u/n$ and $t=Aq$. The normalised
log-likelihood per observation, omitting the multinomial coefficient, is

$$
\ell(\eta)=t^\top\eta-\log\sum_x\exp(a_x^\top\eta),\qquad
\nabla\ell=t-Ap_\eta,\qquad D^2\ell=-\operatorname{Cov}_{p_\eta}(a_x).
$$

The implementation uses dense CPU Float64 calculations, with no new dependency.
Counts must be finite non-negative reals, in an array of size `f.alphabets` or a
vector of joint states. Arrays flatten column-major, as in [[toric]]. Scaling by the
largest count before normalisation avoids overflow of the total. Invalid counts,
zero total, unknown methods, non-positive/non-finite tolerances and negative or
non-integer iteration budgets throw `ArgumentError`. A budget of zero is allowed.

Write $B=(I-\mathbf1\mathbf1^\top/m)A^\top=U\Sigma V^\top$. Retain singular values
strictly above `max(size(B)...) * eps(Float64) * maximum(Σ)`. With $Q=U_r$, optimise
logits $Q\beta$ from $\beta=0$. Return

$$
\eta=V_r\Sigma_r^{-1}\beta.
$$

This is the **minimum Euclidean norm** representative: all representatives differ by
$\ker B$, and this one lies in its orthogonal complement. It removes both redundant
rows and constant-logit directions, even when $\mathbf1$ is absent from the row space
of $A$. It is a floating-point rank decision, not an exact integer rank computation.
The zero-row/constant models have no identifiable coordinates and fit uniformly.

Newton uses $Q^\top(q-p)$ and the covariance of the rows of $Q$. Eigenvalues in the
covariance solve are floored at `max(eps(Float64)*λmax, floatmin(Float64))` to handle
numerically singular near-boundary Hessians. Backtracking halves the step at most
60 times, with Armijo coefficient `1e-4` and likelihood roundoff allowance
`8eps(Float64)*max(1,abs(ℓ))`. A failed search or non-ascent direction stops with
whatever residual was reached; the flag still tests that residual. Probabilities and
likelihoods use shifted log-sum-exp, including during the line search.

## IPF variant and endpoints

`:ipf` accepts only designs with entries in `{0,1}`. It cycles over **individual rows**,
not generator blocks. For a nonconstant row with current mean $s$ and target
$0<t_r<1$, the exact coordinate maximum is

$$
\delta_r=\log\frac{t_r}{1-t_r}-\log\frac{s}{1-s},\qquad
\log p_x\leftarrow\log p_x+\delta_r A_{rx}-\log Z.
$$

Both masses in the current log odds are computed by log-sum-exp over the two
subsets, avoiding subtraction from a rounded mean of one. Constant rows are skipped.
This is cyclic exact coordinate ascent on the concave likelihood; when the ordinary
MLE exists, likelihood level sets are bounded after removing the gauge, and strict
concavity there makes the fitted distribution unique. Equivalently each step is an
information projection onto one binary moment constraint. Redundant rows do not
change the optimum. This argument does not promise a particular finite iteration
count or a numerical convergence rate.

At $t_r=0$ or $1$, an exact step has infinite log odds, incompatible with a finite
returned `η`. The implementation uses **-32 or +32** per visit instead. These steps
increase the likelihood toward its coordinate supremum and approach the endpoint
without introducing `Inf` parameters or `NaN`. They are finite approximations to the
exact endpoint update, not an exact projection onto a zero-support face. The stopping
test still uses the original, unclipped target moments. Binary targets are clamped to
`[0,1]` solely to remove summation roundoff. One iteration is a full sweep. IPF logits
are projected into the SVD chart on return to remove accumulated gauge roundoff.

## Convergence versus existence

The result is `(η, p, converged, iterations, residual, existence)`. `p` is always a
vector, recomputed from the returned finite `η` through the factor's logit and
normalisation functions. The residual is $\lVert Ap-t\rVert_\infty$, including that
last reconstruction's roundoff; `converged` is exactly `residual <= tol`. Defaults
are `tol=1e-10`, `maxiter=1000`. A small residual identifies an approximation to the
extended MLE; it does not certify a finite ordinary MLE.

Only **all counts strictly positive** gives `existence == :exists`. Proof: a convex
combination with positive weight on every column lies in the relative interior of
the column convex hull. Any supporting hyperplane containing that combination must
contain every column. The relative-interior existence criterion therefore applies.
The certificate is independent of whether the optimiser has converged.

Otherwise the heuristic reports `:suspected_boundary` when the Euclidean norm of
centred logits has grown from its zero initial value to at least **10**, and
**`m*minimum(p) <= 1e-8`**. Centred-logit growth is used instead of parameter norm so
that rescaling or duplicating features does not change this diagnostic. This tests
large growth and a small fitted cell, not monotonic divergence or a facial-set
certificate. It can misclassify an interior MLE with very small cells. All other
cases return `:unknown`; zero counts alone do not prove anything. There is no
non-existence status. In the empty-row independence tests, both algorithms satisfy
the moment tolerance and report `:suspected_boundary` simultaneously.

## Membership and supplied equations

`inmodel` first requires finite strictly positive probabilities, with normalisation
checked using `atol=1e-12, rtol=1e-8`. Subtract the mean of $\log p$ and project onto
$Q$. The **Euclidean 2-norm** of the least-squares residual must be at most `atol`
(default `1e-8`), in log probabilities. This is equivalent to projection onto the
row space of $\bar A=[\mathbf1^\top;A]$, so it works without an intercept in $A$.
Zero entries throw: the logarithmic criterion does not test boundary membership.

`binomial_residuals` accepts non-negative probabilities, with the same shape and
normalisation checks, and a collection of integer-vector moves. It checks both
$\sum_x u_x=0$ and $Au=0$ in **BigInt arithmetic**, avoiding fixed-width overflow.
It evaluates $p^{u^+}-p^{u^-}$ by log products, with empty products equal to one and
positive powers of zero equal to zero. Tiny products can underflow in Float64.
No kernel or Markov basis is computed. Vanishing on an arbitrary supplied set need
not certify membership, especially at the boundary.

## Checked and not checked

The `Toric fitting and membership` testset in `../test/runtests.jl` checks:

- positive 2×3 and 3×4 independence against empirical margin products, both methods,
  with cell error at most `3e-10`, moment tolerance `1e-10` and `:exists`;
- a positive 2×3×2 chain against $\hat p_{AB}\hat p_{BC}/\hat p_B$, both methods,
  with cell tolerance `5e-10`;
- a seeded random non-binary integer design with no intercept: moment matching,
  normalised likelihood gradient at the fit, and the analytic gradient at a separate
  random parameter against central finite differences (`h=1e-5`, tolerance `2e-9`);
- redundant rows and an added intercept: identical fits, independent pseudoinverse
  checks of the minimum-norm gauge, and reproduction through normalised energies;
- empty-row independence and its reversed table (targets zero and one), both methods,
  against the extended MLE with cell tolerance `3e-10`, finite parameters, no NaN,
  and the documented boundary status; exhausted budgets and uncertified interior fits;
- input failures, uniform/zero-row/one-state models and overflowing unscaled count totals;
- positive membership against all independence minors on seeded random model and
  generic positive tables, boundary binomials, exact kernel validation, and an
  independent least-squares check of the membership tolerance.

These are small-table checks, not a general solver guarantee. No LP existence test,
exact boundary support recovery, Markov-basis computation, GPU/AD contract, standard
errors, Grassmannian geometry or later specification sections are implemented.
