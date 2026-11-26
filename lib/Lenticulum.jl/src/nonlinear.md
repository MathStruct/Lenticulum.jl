#implementation

> `NonlinearFactor`: $p(y\mid x) = \mathcal N(y; f(x), Q)$ for a **hand-written** $f$, with
> Gaussian messages by linearisation (the extended Kalman filter) or the unscented transform
> (the unscented Kalman filter). The factor that known physics needs: a measurement model or
> known dynamics, next to a learned relation in the same graph.

> Sources: code: `nonlinear.jl`, `gaussian.jl`; Julier & Uhlmann, *A New Extension of the Kalman Filter to Nonlinear Systems*, 1997; Wan & van der Merwe, *The Unscented Kalman Filter for Nonlinear Estimation*, 2000; Bell & Cathey, *The iterated Kalman filter update as a Gauss–Newton method*, IEEE TAC 1993; García-Fernández, Svensson & Särkkä, *Iterated Posterior Linearization Smoother*, IEEE TAC 2017; the RxInfer delta-node approximations ([[Sharing Types with RxInfer]] §4)
>
> Bibliography: [[Bibliography#^julier1997ukf|Julier & Uhlmann 1997]] · [[Bibliography#^wan2000ukf|Wan & van der Merwe 2000]] · [[Bibliography#^bell1993iekf|Bell & Cathey 1993]] · [[Bibliography#^garciafernandez2017iterated|García-Fernández et al. 2017]]
>
> Theory: [[Belief Algebra]] · [[RxInfer as a Backend]] · [[Open Problems in Implicit Diffusion Learning]]

## Graph free-energy edge directions

`NonlinearFactor` declares `Mycelium.uses_cavity == true`: it linearises at the incoming
belief, then passes those messages to the Gaussian local free energy.
Graph free-energy evaluation rejects an attached `Emitting()` edge with `ArgumentError`;
use `Bidirectional()` if that channel must also send messages. Missing messages on permitted
edges retain their existing local behaviour. See [[free_energy]] and
[[LP-014 Emitting edges drop the cavity message from the free energy]].

## 1. One local linear model, then GaussianFactor's algebra

Every message goes through a local model $y \approx Ax + b + \nu$, $\nu \sim \mathcal N(0, Q + \Omega)$,
around an expansion belief $\mathcal N(\bar x, P)$ on $x$ (`linearize`):

| method | $A$ | $b$ | $\Omega$ |
|---|---|---|---|
| `Linearization()` | $\partial f(\bar x)$ | $f(\bar x) - A\bar x$ | 0 |
| `Unscented()` | $C_{xy}^\top P^{-1}$ on the $2n+1$ sigma points | $\bar y - A\bar x$ | $S_{yy} - APA^\top$ |

The unscented row is statistical linear regression: the line that best explains $f$ over the
belief, plus the spread it does not explain. After that the messages are `GaussianFactor`'s for
$(A, b, Q + \Omega)$: forwards $\mathcal N(Am + b, ASA^\top + \Omega + Q)$, backwards the likelihood
$\mathcal N^{-1}(A^\top R^{-1}(m_y - b), A^\top R^{-1}A)$ with $R = Q + \Omega + S_y$.

## 2. Where to linearise

- **Forwards** around the incoming belief on $x$. A point input is exact: $\mathcal N(f(x), Q)$.
- **Backwards** around the **cavity**, the `prior` argument of `factor_message` (the product of
  the other messages into $x$), which Mycelium already passes. It is used only as the expansion
  point, never multiplied in, so nothing is counted twice; and as a schedule sweeps again the
  cavity sharpens and the linearisation improves (the expectation-propagation pattern). With no
  cavity the message is uninformative, unless the factor was given an `x0`. This makes the
  message depend on the schedule: a `TreeSchedule` runs one sweep and may reach the factor before
  the cavity has arrived, so use `SequentialSchedule(tasks(tree_schedule(g)))`, which sweeps until
  the messages stop changing (three sweeps for prior, measurement and data).
- **`invert`** relinearises the posterior `iterations` times: with `Linearization` that is the
  iterated EKF, i.e. Gauss–Newton on the MAP; with `Unscented` it is iterated posterior
  linearisation. On $x \sim \mathcal N(1, 0.5^2)$, $y = x^3 + \mathcal N(0, 0.1^2)$, $y = 2$, thirty
  iterations reach the MAP to $10^{-5}$ and the covariance is the Gauss–Newton curvature there; one
  EKF step is visibly off.

## 3. Design decisions

- **No AD dependency.** The Jacobian is the user's `jacobian(x)` or central differences, as the
  diffusion package's fallback is; pass `x -> ForwardDiff.jacobian(f, x)` for an exact one.
  Both are generic in the element type: at a dual expansion point (ForwardDiff) the Jacobian is a
  matrix of duals, the finite-difference step taken from the values (P21; `gaussian.md`, "Dual
  inputs on the flat path"; `test/forms_crossing.jl`).
- **Written here, not borrowed.** ReactiveMP has these approximations, but importing it would
  make `Lenticulum` depend on RxInfer's engine for ~150 lines ([[Sharing Types with RxInfer]] §4).
- **Not learnable.** `f` is fixed and `ps` is empty. A parametrised `f(x, θ)` with gradients is a
  separate step.
- **The free energy is the linear model's.** `local_free_energy` builds the `GaussianFactor` of the
  local model around the incoming belief on $x$; exact for affine $f$, the usual EKF/UKF evidence
  approximation otherwise.

## 4. How it is tested

`test/runtests.jl`: for an affine $f$ both methods give `GaussianFactor`'s messages and free
energy exactly, in both directions; for $y = x^2$ the unscented forward message has the exact
mean $m^2 + s^2$ and variance $4m^2s^2 + 2s^4$, the linearised one neither; the finite-difference
Jacobian matches a supplied one; iterated inversion reaches the MAP; and in a graph
(prior, $x^3$ measurement, data) repeated sweeps converge to one EKF update.

## 5. Missing

Learnable parameters in $f$; Gauss–Hermite and cubature rules; CVI; and multi-input functions
$y = f(x_1, \dots, x_k)$ over several channels (today one input channel, which may be a stacked
vector).

## 6. Iterated messages

With `iterations > 1`, `likelihood_message` runs the posterior relinearisation loop and returns the
final Gaussian site; `invert` is that site combined with $\pi$ once, and the graph message
(`factor_message`) uses the same routine. So the BP message is the site of the posterior `invert`
returns (L-ADR-0006). Tested for both methods with $y = x^2$, four iterations, and in the conformance
checker.

Related: [[constraint]], [[likelihood]] (non-Gaussian observations), [[Belief Algebra]], [[Sharing Types with RxInfer]]

## LP-018 covariance validation (2026-10-29)

As for Gaussian factors, check noise shape against `nout`, then symmetry, then positive
definiteness. This prevents a one-dimensional metric on a two-dimensional residual and
prevents `Symmetric(Q)` from accepting an asymmetric covariance by reading one triangle.
Shape and symmetry failures are `ArgumentError`s with clear messages. Evidence:
`test/lp018.jl`, “LP-018.7 residual covariance validation”, for both constraint and nonlinear constructors.

## Review of 2026-11-10

**The default spread of `Unscented` is `α = 1`.** The sigma points lie at
$\bar x \pm \alpha\sqrt{n + \kappa}$ standard deviations. Wan & van der Merwe's $\alpha = 10^{-3}$,
the former default, puts them a thousandth of a standard deviation from the mean, with a centre
weight of about $-10^6$: the regression then sees $f$ only where it is locally quadratic, so the
linear model and its residual covariance $\Omega$ carry nothing of the nonlinearity at the scale
of the belief, and the weights cost about six digits to cancellation. With $\alpha = 1$,
$\kappa = 0$, $\beta = 2$ the centre weight of the mean is 0, that of the covariance is 2, and
$\Omega$ is positive semidefinite without the clamp. The suite's check ($y = x^2$:
$\mathbb E y = m^2 + s^2$, $\operatorname{Var} y = 4m^2s^2 + 2s^4$) holds for both settings.

**The free energy is the linear model's.** `local_free_energy` returns the free energy of the
local linear model with noise $Q + \Omega$, including $\tfrac12\log\det 2\pi(Q + \Omega)$: the
evidence of the approximating model, as an extended or unscented Kalman filter reports it. It is
neither a bound on nor an estimate of $\mathbb E_b[-\log\mathcal N(y; f(x), Q)] - H(b)$. It is
linearised at the incoming message on $x$, while a backward message with `iterations > 1` is
linearised at the posterior; with more than one iteration the messages and the free energy
therefore describe two linear models, and the free energy is not stationary at the fixed point
the messages reach.

**Partly flat inputs.** A partly flat message on `y` (a rank-deficient likelihood from further
down the graph) goes through the linear model's relation and the flat-direction limit, as for
`GaussianFactor` ([[lib/Lenticulum.jl/src/gaussian|gaussian]], "Improper inputs"); the backward message is declared
improper when the linear model's coefficient is rank-deficient.

## No input on x (P28, 2026-11-17)

**Theory.** With nothing known about $x$, $y = f(x) + \varepsilon$ still says that $y$ lies near
the image of $f$, which for $n_{\text{in}} < n_{\text{out}}$ is a proper submanifold. That message is
not Gaussian. Its local linear version at a point $x_0$ is the message of the linear model
$y \approx Ax + b + \varepsilon$, $A = \partial f/\partial x\,(x_0)$, $b = f(x_0) - Ax_0$, with $x$
flat: $y$ flat along $\operatorname{range}(A)$ and Gaussian off it, the closed form of
[[lib/Lenticulum.jl/src/gaussian|gaussian]] "No input on x". It describes the tangent plane at
$f(x_0)$, so it is good only where $f$ stays close to that plane.

**Implementation.** A flat $x$ gives no point to linearise at, so the factor's `x0` is used, the
point already used for a backward message with no cavity. `_nl_no_input_forward` linearises there
(the unscented transform of a point is the Jacobian, so both methods give the same message) and
sends `GaussianFactor`'s flat-path message for $(A, b, Q)$. Without `x0` the message stays
`TrivialBelief()`. The polarity $x$ `Latent()`, $y$ the target, which Mycelium needs to call the
factor when $x$ has no message (P3), is listed by `supported_polarities` only when `x0` is set, so
a factor without `x0` behaves as before. Every input form of P3's "no information" is handled (no
message, `TrivialBelief()`, zero precision, an everywhere-flat extended Gaussian, mixtures of these).

**Measured** (`test/p28_no_input.jl`): $f(x) = \tanh(Wx + c)$ with random $W$ of random rank,
$n_{\text{in}}, n_{\text{out}} \in 1{:}4$, 60 factors, both methods, the seven empty input forms:
with an exact Jacobian every message equals the closed form for the hand-written Jacobian at $x_0$
to $10^{-8}$, and `TrivialBelief()` exactly when $\operatorname{null}(A^\top)$ is empty; with central
differences and $W$ of full rank, to $10^{-5}$. Without `x0` every message is `TrivialBelief()`.
An affine $f$ gives `GaussianFactor`'s message to $10^{-10}$, and in a graph ($x$ with no other
factor, a prior on $y$, 20 instances, $n_{\text{in}} < n_{\text{out}}$) the marginal of $y$ under
`infer!` matches the dense posterior with $x \sim \mathcal N(0, \tau I)$, $\tau = 10^{30}$ in
`BigFloat`, to $10^{-8}$.

**Difficulty: central differences and rank.** When $f$ depends on fewer directions than $x$ has
($W$ rank-deficient), the difference Jacobian carries an error of about $10^{-10}$, far above the
rank test's threshold ($\max(n_{\text{in}}, n_{\text{out}})\,\varepsilon\,s_1$), so it had full
numerical rank and the message lost directions of $\operatorname{null}(A^\top)$ (21 of the test's
draws). Since P29 the difference Jacobian is cut to the rank its accuracy supports (next section);
the test now requires the exact message in those draws too.

## The rank of a difference Jacobian (P29, 2026-11-17)

**Problem.** Every rank test of the messages (`_rank_deficient`, the flat path's null space of
$A^\top$) is relative to $\varepsilon$. A Jacobian from central differences is accurate to about
$10^{-10}$ relative, so when $\operatorname{rank}(\partial f/\partial x) = r < \min(n_{\text{in}}, n_{\text{out}})$
its $\min - r$ smallest singular values are of that size instead of zero. The forward message with no
input on $x$ then loses directions (conservative), and a backward message is not declared improper:
it is sent as a `GaussianBelief` whose precision has eigenvalues of order $10^{-20}$, which a later
factor inverts.

**What is done** (`_rank_clean` in `nonlinear.jl`). Each entry of $J$ gets an error bound
$B_{ji} = R_{ji} + \tfrac13 \lvert J(h) - J(2h)\rvert_{ji}$: the rounding of $f$'s values over the step
and of the shifted point, $R_{ji} = \varepsilon\bigl((\lvert f^+_j\rvert + \lvert f^-_j\rvert)/2h_i + \lvert x_i\rvert\lvert J_{ji}\rvert/h_i\bigr)$,
and the truncation error by Richardson's estimate. The rank is decided where every entry's bound
is at most 1: with diagonal powers of two $D_r$, $D_c$ such that $B \le r c^\top$,

$$
\tilde J = D_r^{-1} J D_c^{-1}, \qquad \lVert \operatorname{error}(\tilde J)\rVert_2 \le \sqrt{n_{\text{out}} n_{\text{in}}},
$$

and the singular values of $\tilde J$ below $\kappa\sqrt{n_{\text{out}} n_{\text{in}}}$, $\kappa = 10$, are cut,
by the oblique projection $P = D_c^{-1}\tilde V_r \tilde V_r^\top D_c$ on the right: $D_r^{-1}(JP)D_c^{-1}$ is
$\tilde J$'s truncated SVD, and $JP$ has rank exactly $r$ for the $\varepsilon$-relative tests downstream.
A first version used one global bound ($\max\lvert f\rvert$, the smallest step); the review found that
rows of very different size ($[10^{-3}x_1,\ 10^7 + x_2]$) or steps of very different size ($x = [0, 10^6]$)
then hid a singular value of $10^{-3}$ that its own entries resolve to $10^{-12}$.

With dual numbers (ForwardDiff) $P$, computed from the values, is a constant, and
$(J - \operatorname{value}(J))(I - P)$, whose value is exactly zero, is added: it gives back the derivative
of $J$ in the cut directions. Without it, where the rank changes at the point ($f(x) = x^2$ at
$x_1 = 0$) the derivative of $J$ lost its entries in those directions (review). A non-finite $J$ is
returned as it is.

$J(2h)$ costs $2 n_{\text{in}}$ more evaluations of $f$ and is computed only in a gray zone: a singular
value of $J$ below $10^{-3} s_1$, or of the rounding-scaled $\tilde J$ below $10^3\sqrt{mn}$; a
well-conditioned $J$ costs nothing extra.

Measured by `bench/p29_difference_rank.jl` (20 000 random $f = g(Wx + c)$, $g$ one of $\tanh$, affine,
$\sin +$ quadratic, $W$ of random rank scaled by $10^{\pm 2}$; 2026-11-17): a bound on rounding alone
is not enough, the raw error exceeds it by more than 94 times in 1% of the draws and by up to
$3 \cdot 10^4$, from $h^2 f'''$, which also leaves the true range of $J$. With the Richardson term the
cleaned rank equals the exact one in 19 663 draws, is never above it, and is below it in 337, where
exact singular values of at most $1.7 \cdot 10^{-8}$ (18.7 rounding bounds) were cut, mostly a
saturated coordinate of $\tanh$. Cutting makes a message more confident (one more direction of
$\operatorname{null}(A^\top)$); it is bounded by the error estimate, not removed. $\kappa$ was not tuned.

**Tested** (`test/p29_rank_tolerance.jl`): on 400 draws of three maps, every instance whose singular
values are resolvable (none in $(10^{-12} s_1, 10^{-5} s_1)$, none below $10^{-6}$) has the exact
rank and agrees with the exact Jacobian to $10^{-5}$; a backward message for $W$ of rank below
$n_{\text{in}}$ is an `ExtendedGaussian` flat along $\operatorname{null}(A)$ and equal to the message with
the exact Jacobian; the number of evaluations of $f$ ($2n_{\text{in}}$, or $4n_{\text{in}}$ in the gray
zone); ForwardDiff through the forward message against central differences, $W$ of rank 1; the
three findings of the review ([[LP-050 Findings of the review of P29]]); and P28's no-input test
without its `@test_broken`. On the code before P29 the first testset fails in 393 of 2344 checks.

**Does not do.** A Jacobian lying wholly below the difference error (a saturated $\tanh$) is cut to
zero, as if $f$ were constant there; differences cannot resolve it, pass `jacobian`. A singular value
near the error bound is kept or cut by $\kappa$, not by $f$; the bound is per entry, so a
singular value that mixes entries of very different accuracy is judged by the coarse ones. The unscented
transform with $S \ne 0$ fits its $A$ by regression and is not cut. Zygote through `_jacobian` is
unchanged (it builds $J$ by mutation). User-supplied Jacobians are used as given.

**Does not do.** Linearise anywhere but `x0` (a flat $x$ has no mean; the cavity of $y$ is not
used to choose a point). At an `x0` where the Jacobian loses rank the message is as confident as the
linear model: for $f(x) = x^2$ at $x_0 = 0$ it is $\mathcal N(f(x_0), Q)$, although $y$ ranges over
$[0, \infty)$ ([[LP-049 Findings of the review of P28]], probe 7); choose `x0` away from critical points. A partly flat input on $x$ forwards (an extended Gaussian with some flat
directions) still has no forward path: `_moments` refuses it, as before. Loopy schedules and plates
with this polarity are not tested.

## The tree shortcut and the cavity (GH-15, 2026-11-17)

The backward message linearises at the target's cavity, so one tree sweep is not necessarily a
fixed point even under BP; `NonlinearFactor` now declares `Mycelium.message_uses_cavity(::NonlinearFactor) = true`,
which keeps such trees off the one-sweep shortcut of `propagate!` (Mycelium `passing.md`). Test:
`test/gh15_nonlinear_tree.jl`.


## Element type

Computation follows Julia promotion of the parameters, noise, incoming messages and data it reads;
data and parameters are not converted to force a result type. Constants and numerical
hyperparameters are cast at use.

User Jacobians and difference Jacobians retain the types of the values they read. The difference
rule keeps the existing Float64 step; other float types use a cube-root-epsilon step to avoid
subtracting nearly identical function values. Rank-error buffers use the primal type. Sigma-point
tuning fields remain Float64, but enter arithmetic cast to the expansion computation type;
covariance decompositions and zero residual covariances preserve their types. The user function and
Jacobian decide their own output types. This does not add support for differentiating eigen
decompositions that were unsupported before.

Checks: `lib/Lenticulum.jl/test/p31_element_type.jl`. The implementation is not a claim of coverage
for every mixed precision, derivative, or device configuration.

## Element types (Q4)

Finite-difference buffers use the expansion point as their array anchor. The error
scale uses primal epsilon, with the historical Float64 difference step retained. The
unscented weights take the type of the expansion moments. In one dimension the spectral
step can use the scalar variance directly. The residual covariance warning retains its
relative `1e-10` threshold, evaluated in the covariance type. The stored tuning fields
remain Float64 configuration values and are converted at application.

## Device arrays (Q7)

Adapt moves the covariance and stored expansion point. The unscented transform builds its
weights on the backend of its expansion point and obtains its spectral data through the core
host helper. A user-supplied map or Jacobian provides device-compatible operations for its own
computations.

## Q20 device constructors

The direct constructor retains device covariance storage and uses the storage-aware positive-definiteness check.
