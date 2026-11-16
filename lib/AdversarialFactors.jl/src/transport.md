#implementation #derivation

> Exact affine Gaussian transport entropy, and separately labelled nonlinear estimates.
> All continuous entropies use Lebesgue coordinates (LP-006); samples alone supply no density.

> Sources: original change-of-variables calculation, checked in `../test/transport.jl`; code: `transport.jl`, `generator.jl`.
>
> Theory (CT-ML wiki): [Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy)

## Laws and API (2026-10-27, LP-009)

For a global differentiable bijection between equal-dimensional Euclidean spaces,
change of variables in $-\int p\log p$ gives

$$
H(g_*p)=H(p)+\mathbb E_p[\log|\det J_g|].
$$

`AffineGenerator(A,b)` is a Lux layer with parameters `(A,b)`. The GeneratorFactor
must use the identity input adapter. For a proper Gaussian latent belief,
`pushforward_entropy` checks square nonsingular finite parameters and computes
$H(p)+\log|\det A|$. `pushforward` returns the exact Gaussian with covariance
$A\Sigma A^\top$ and mean $A\mu+b$. Rectangular and singular maps throw for these
Lebesgue-density APIs, although their existing point/particle transport remains available.

The distributional game uses `y` as the latent Gaussian and requires absent/trivial
output prior `π`. It has zero map energy, entropy $H(g_*p)$ and free energy $-H(g_*p)$.
An incoming output belief would condition the output and needs another law; it throws.
`local_free_energy` uses the same law for Gaussian input messages. These are explicit
unconditional forward games, not a GAN likelihood or an inverse generator. The earlier
conditional Dirac game retains its zero pair and its point convention.

`InvertibleGenerator(net,logabsdet)` explicitly asserts a global bijection, including
any GeneratorFactor input adapter. The caller supplies its exact log absolute Jacobian
determinant. A local determinant test cannot prove global invertibility. For a known
proper Gaussian source, `estimate_pushforward_entropy(...;rng,nsamples)` adds its exact
entropy to a Monte Carlo Jacobian expectation and returns
`EntropyEstimate(value,standard_error,nsamples)`. It never enters `entropy` or
`free_energy` as an exact value. Finite log determinants are required at every draw.

The Jacobian expectation under a Dirac or a finite supported law could be a finite exact
sum. However those sources are singular relative to Lebesgue: adding that sum to Shannon
entropy would not compute differential entropy. These APIs reject them. No unlabelled
quadrature, finite-difference Jacobian certification or entropy from particles is supplied.
The estimate's standard error describes sampling error, not an incorrect bijection assertion
or Jacobian callback.

The built-in NoiseSource sampler has the known law $N(0,I_d)$, so its unconditional
`entropy(f,TrivialBelief(),nothing,ps,st)` is $d(1+\log(2\pi))/2$ and free energy its
negative, independent of particle count and without resampling. Graph evaluation with no
incoming information uses the same constant. Custom samplers supply no law and keep
throwing for unconditional game entropy. Their existing fixed-Dirac game requires one
sample. Graph contributions from particle messages retain their previous zero convention;
they are not complete differential-entropy games.

## Checks and limits

`../test/transport.jl` compares affine entropy and pushforward against the closed-form
Gaussian law, including a negative determinant. The common checker covers the Gaussian
forward generator game with no output conditioning. Singular/rectangular maps, improper
sources, particle and Dirac differential entropies, and output conditioning are rejected.
The nonlinear oracle is the global bijection
$g(z_1,z_2)=(z_1,\exp(cz_1)z_2)$, whose log determinant is $cz_1$; the estimated
expectation and standard error are checked against the Gaussian closed forms.
NoiseSource's exact standard-normal entropy and rejection of custom unknown laws are
checked independently of sample generation. Its stochastic lens is not a deterministic
fixture for the common checker; its fixed-draw conformance remains in `conformance.jl`.
