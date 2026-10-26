#implementation #derivation

> Optional homotopy continuation enumerates isolated roots of square polynomial
> targets. Regular real roots carry local Gaussian posterior components; entropy
> is reported as bounds. The single-branch algorithm remains the default.

> Sources: Breiding & Timme, *HomotopyContinuation.jl: A Package for Homotopy Continuation in Julia* (2018); Breiding, Rose & Timme, *Certifying Zeros of Polynomial Systems Using Interval Arithmetic* (2023), Theorem 3.6 and Lemma 7.1 (Krawczyk certification and reality); original implementation of [[L-SPEC-004 Algebraic Factors Toric and Grassmannian]] §4; code: `homotopy.jl`, `polynomial.jl`, `../ext/AlgebraicFactorsHomotopyContinuationExt.jl`, `../test/homotopy.jl`.
>
> Bibliography: [[Bibliography#^breiding2018homotopycontinuation|Breiding & Timme 2018]] · [[Bibliography#^breiding2023certifying|Breiding et al. 2023]]
>
> Theory (CT-ML wiki): [Bayesian Inversion](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Inversion) · [Statistical Game](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game)

## Interface and packaging

`HomotopyBranches(;real_tol=1e-8,singular_tol=1e-8,certify=true)` is a core method
object, passed as `PolynomialFactor(...;branches=HomotopyBranches())`.
`branches=:single` retains §3(b), including non-square Gaussian sites and its
`iterations` option. The multi-branch mode ignores `iterations` and uses roots.
`using HomotopyContinuation` activates `AlgebraicFactorsHomotopyContinuationExt`.
The public entry point and factor messages throw a clear load instruction when
HC has not been loaded; the test checks this before loading HC in the same process.
HC is a weak dependency with compat `2.22.4` (Julia caret semantics: ≥2.22.4,<3).
Version 2.22.4 installed, solved y²−1 and certified both solutions on Julia 1.13.1.
Tests depend on HC; the docs environment loads it. The existing CI package matrix
runs the extension tests without an additional job.

`polynomial_branches(f,target,inputs,prior,ps;method=HomotopyBranches())` works
regardless of the factor's chosen message algorithm. It returns distinct finite
root records and raw path records, plus posterior weights, components, likelihood
and posterior mixtures. Finite roots include complex and singular candidates;
zero weight means excluded. Path records include at-infinity, failed, singular
and real flags and HC's return code. A per-root certificate is `true`/`false`,
or `nothing` when certification was not requested. A certificate proves an
isolated nonsingular zero near that candidate; it does not prove completeness.
HC's `certify` uses interval arithmetic/Krawczyk, **not Smale's α-theory**.

The extension constructs the polynomial system directly from Θ, the exponent
table and the observed means. Chebyshev products use the T recurrence, including
homogeneous-index tables whose lower terms leave the original monomial span.
There is no approximate interpolation. HC treats homogeneous systems projectively;
the extension translates these before solving and maps candidates back, retaining
affine roots at the origin (the circle double root). Total-degree continuation, seed UInt32(4004),
and serial path tracking make the test runs reproducible. No cross-call parameter
homotopy, warm-start cache, witness sets or positive-dimensional solve is supplied.
Even a square system can have a positive-dimensional zero set: k=q alone does
not guarantee isolated roots. These systems are outside this API's contract.

A candidate is real if max |Im(rootᵢ)| ≤ real_tol. Target singularity means
HC's singular flag or σmin(Jᵤ) ≤ singular_tol max(1,σmax(Jᵤ)); the absolute part
of this numerical rule depends on row scaling. Invariance checks concern regular
systems comfortably away from this threshold. Singular roots retain diagnostics
but receive no Gaussian component or weight. Complex roots have complex det Jᵤ
and `A=nothing`, since real covariance and Gaussian sites are not defined there.
For real roots the full [[polynomial]] covariance is reused:

$$
A_j=J_j\Sigma_fJ_j^\top+\sum_o J_{o,j}C_oJ_{o,j}^\top+\lambda\Theta\Theta^\top.
$$

Observed channels must be Dirac or proper Gaussian; Gaussian observed uncertainty
is propagated at its mean, not integrated through a nonlinear relation. The
target prior must be proper Gaussian or trivial. A trivial prior is a constant
density relative to target **Lebesgue measure**. Dirac and other improper target
priors are rejected. A singular residual covariance at a regular root throws by
the same SPD rule as §3(b). Singular target roots need no SPD covariance to be
reported, because no site is formed for them.

## Branch masses and likelihood messages

For a regular real root aⱼ let Sⱼ=Jᵤⱼ⁻¹AⱼJᵤⱼ⁻ᵀ. Changing variables in the
normalised residual Gaussian gives

$$
\mathcal N(J_{u,j}(u-a_j);0,A_j)
=\frac{\mathcal N(u;a_j,S_j)}{|\det J_{u,j}|}.
$$

Consequently the extrinsic likelihood mixture has **normalised Gaussian** sites
N(aⱼ,Sⱼ) and masses proportional to 1/|det Jᵤⱼ|. Mycelium already supports their
products and Gaussian overlaps. Combining with π=N(μ,P) gives components
π×site, normalised, and weights

$$
w_j\propto\frac{\mathcal N(a_j;\mu,P+S_j)}{|\det J_{u,j}|}.
$$

For a trivial prior the overlap is one and posterior and likelihood coincide.
Log absolute determinants and log overlaps are normalised by log-sum-exp; weights
are not computed by multiplying densities. The inspection records include det Jᵤ
for convenience. Putting these **posterior** masses into the likelihood would
apply the overlap twice on combination; this corrects the ambiguous "same
masses" phrasing of §4.3. No LenticulumCore or Mycelium changes are required.

Row gauges multiply each determinant by the same |det B|, so normalised weights
are invariant. Σ and observed noise transform A as BABᵀ and sites are unchanged.
Undamped polynomial basis conversions also preserve these quantities. Damping
λΘΘᵀ remains basis dependent, as in §3: basis-invariance tests use λ=0.

The small-noise limit is wⱼ ∝ π(aⱼ)/|det Jᵤⱼ|. This belongs to a **normalised
residual density**. For an unnormalised energy exp(−r²/(2A(u))) with varying A,
the one-dimensional limiting mass instead includes √A(aⱼ). A common constant
A makes the two conventions coincide; the quadrature tests deliberately use
Σ_f=0, λ=δ/||Θ||², hence A=δ, for both circle and cubic. No claim is made that
the branch approximation equals the exact nonlinear posterior at finite noise.

## Entropy and empty fibres

`branch_entropy_bounds(m)` returns H(w) and

$$
\sum_jw_jH_j\le H(m)\le\sum_jw_jH_j+H(w).
$$

The identity H(X)=H(X|Z)+I(X;Z), with 0≤I(X;Z)≤H(Z), gives these bounds.
H(w) is discrete label entropy; Hⱼ is differential entropy of the Gaussian
posterior component. No exact Gaussian-mixture entropy or game quantity is
implemented. Zero weights contribute zero to H(w).

No regular real root means `nothing` mixtures and empty weights in inspection.
`factor_message`, `likelihood_message` and `invert` throw, including when all
real roots are singular; the error directs the caller to inspect diagnostics.
They never silently fall back or construct an empty MixtureBelief. Failed and
at-infinity paths remain inspectable even when other paths yield a mixture.
Inference warns when it excludes failed/infinite paths or singular real roots
while retaining regular real components.

## Original homotopy implementation run (Julia 1.13.1, HC 2.22.4)

`julia --project=lib/AlgebraicFactors.jl -e 'using Pkg; @time Pkg.test()'`
passed **14099 checks with one existing LP-014 broken check**. The added file
accounts for 95 passes: 3 extension-absence checks, 74 branch checks, 12
small-noise checks and 6 entropy checks. The original suite took 86.377808 s
before this work; the complete suite took 158.768452 s, a measured increase of
72.390644 s (single runs, compilation included). The new file reports
69.771567 s including HC loading, first solve/certification compilation and
checks. Installing/precompiling HC in a fresh temporary project took
28.003742 s separately; that is not included as test execution cost.

The branch checks cover the circle ±0.8 at x=0.6, equal weights for a symmetric
prior, the finite-noise closed-form asymmetric Gaussian ratio, branch posterior
moments, lens/graph likelihood equality and pooling equality. They inspect no
real roots at x=1.2 and the unweighted singular double root at x=1. The cubic
checks x=0,0.2,0.6 and ±2/(3√3)±1e-4. Product conics u₁²=1+x, u₂²=4−x at
x=0.2 have four known roots; a nontrivial row gauge preserves sites and posterior
weights under an asymmetric correlated Gaussian prior. Cubic monomial/Chebyshev
weights agree; a homogeneous-index Chebyshev polynomial exercises its lower
terms directly. The system u₁²=1, u₁u₂=1 checks two finite roots and actual
at-infinity paths. Failed-path flags are exposed but no intentionally failing
tracker run is asserted. All simple regular real candidates are certified.

The deterministic quadrature tests use π=N(0.3,0.7), 300001 grid points on
[-3,3] (spacing 2e-5), circle x=0.6 and cubic x=0.2. Constant residual noise
A=δ is obtained by Σ_f=0, λ=δ/||Θ||². Circle branches split at zero; cubic
branches split at the stationary points ±1/√3. Reported errors are absolute
posterior mean error, maximum branch-mass error, and maximum difference between
the mixture masses and normalised Gemini masses π(aⱼ)/|det Jᵤⱼ|:

| system | δ | mean error | branch-mass error | Gemini-limit error |
|---|---:|---:|---:|---:|
| circle | 1e-3 | 4.67148291e-4 | 1.40822625e-4 | 8.52017435e-5 |
| circle | 1e-4 | 4.65648455e-5 | 1.39949378e-5 | 8.52396810e-6 |
| circle | 1e-5 | 4.65499330e-6 | 1.39862575e-6 | 8.52438035e-7 |
| cubic | 1e-3 | 4.15460487e-3 | 2.09406120e-3 | 1.84690639e-4 |
| cubic | 1e-4 | 3.94375172e-4 | 1.95869635e-4 | 1.84771026e-5 |
| cubic | 1e-5 | 3.92542250e-5 | 1.94695081e-5 | 1.84778490e-6 |

Each sequence decreases; each final error is below the test tolerance 5e-5.
These are fixed examples, not a general convergence-rate theorem.

Entropy checks use the symmetric flat-prior circle mixture (means ±0.8),
Xoshiro(4404), 100000 samples per case, first branch variance 1 (overlap), then
0.01 (separation), with the same RNG continued. H(w)=log 2 in both cases.

| branch variance | lower bound | MC entropy | MC standard error | upper bound |
|---|---:|---:|---:|---:|
| 1 | 1.41893853 | 1.66221378 | 0.00204723 | 2.11208571 |
| 0.01 | −0.88364656 | −0.18939155 | 0.00224370 | −0.19049938 |

Both estimates satisfy the bounds with a five-standard-error allowance. The
separated estimate agrees with the upper bound within that allowance; the
overlapping estimate is at least 0.1 below it. Finite Monte Carlo estimates
can fall just outside an exact bound. These checks concern mixture differential
entropy, not implemented statistical-game entropy or free energy.

Branch masses use the public `Mycelium.combine_with_overlap(site, πprior)`
contract for Gaussian priors. Trivial priors contribute zero log-overlap;
this extension still requires a proper Gaussian or trivial target prior.
