#implementation

> `DiffusionFactor`: the point where [[ImplicitREDDiff]]'s selection matrices stop being
> notation and become a `LenticulumCore.Polarity`.

> Sources: code: `factor.jl`, `gaussian.jl`
>
> Theory (CT-ML wiki): [Bayesian Inversion](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Inversion) · [Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy)

## Graph free-energy edge directions

`DiffusionFactor` declares `Mycelium.uses_cavity == true` for every prox backend: its local
free energy reads messages to choose a polarity, assemble evidence and run the prox.
Graph free-energy evaluation rejects an attached `Emitting()` edge with `ArgumentError`;
use `Bidirectional()` if that channel must also send messages. Missing messages on permitted
edges retain their existing local behaviour. See [[free_energy]] and
[[LP-014 Emitting edges drop the cavity message from the free energy]].

## 1. P is derived from the polarity, not configured

The note asks for diagonal selection matrices with
$P_{in}+P_{out}+P_{latent} = \mathrm{Id}$ and $P = \rho_{in}P_{in}+\rho_{out}P_{out}+\rho_{latent}P_{latent}$.
All of that already exists in `LenticulumCore.channels`:

| `ImplicitREDDiff.md` | `LenticulumCore` |
|---|---|
| $P_{in}$ | the channels with `Observed()` |
| $P_{out}$ | the channels with `Unobserved()` |
| $P_{latent}$ | the channels with `Latent()` |
| $P_{in}+P_{out}+P_{latent}=\mathrm{Id}$ | `ispartition`, guaranteed by the `NamedTuple` |
| $\rho_{in},\rho_{out},\rho_{latent}$ | `channel_precision(p, name)` |
| $\rho_{in}=\infty$ | `default_precision(Observed()) == Inf` |

So `precision_vector(f, p)` is a *reading* of the polarity, and the factor has no
configuration of its own for it. The channels tile one $\mathbb{R}^n$ in declaration order,
and `blockranges` returns index ranges — a diagonal 0/1 matrix is a wasteful way to write a
range.

Note which way round the default falls: **the hard clamp is the default and the soft clamp is
what you opt into.** That is the right way round — an observation is evidence unless you say
otherwise — and it is `LenticulumCore`'s choice, inherited for free.

## 2. One network for the joint state

`predictor` is a `NoisePredictor` over the whole $\mathbb{R}^n$, not one per channel. That is
what makes this a **relation** rather than a bundle of conditionals: the diffusion model knows
the joint distribution of all channels, so any subset can be inpainted from any other. A
per-channel model could not do that, and would be an explicit learner wearing a costume.

## 3. Complete lists and lazy traversal

`supports_polarity` accepts every Observed/Unobserved/Latent assignment over the channels
with at least one `Unobserved()` channel. For $n$ channels there are $3^n - 2^n$ such
assignments: subtract the assignments that use only Observed and Latent.

`supported_polarities` lists all of them through four channels (at most 65 entries).
The original one-target/all-other-observed entries come first, in channel order. The
remainder follows `polarities(f)`: roles cycle Observed, Unobserved, Latent, with the
first channel changing fastest. Above four channels the list throws `ArgumentError`
pointing to `supports_polarity` and the lazy `polarities` iterator. The latter avoids
materialising the exponential set, but a complete traversal still takes exponential time
and is type-unstable because assignments have distinct types.

`hasnopolarity(f)` is always false; `isunidirectional(f)` is true exactly at arity one.
These predicates do not enumerate. Other factors retain the core's list-based defaults.
A caller that needs a complete eager list must stay within the cap. Mycelium's
`resolve_polarity` constructs a per-target assignment from available messages and
`check_legal` tests it with the predicate; these operations do not need a list.

`invert` and `likelihood_message` return a joint belief on all unobserved blocks,
concatenated in channel order. Point messages select those coordinates from the prox
solution. Gaussian and mixture messages marginalise the same full-state construction
onto those coordinates; latent blocks are omitted. `factor_message(f, target, …)` still
returns one target block. A joint prior point is split in the same order; a shorter point
retains the existing shared-block convention when it fits every target. Incoming points
on individual channels take precedence.

For the implicit solver, likelihood messages divide out the diagonal anchor on the
selected target coordinates. The diffusion prior remains. A Gaussian may be improper;
a mixture component made improper by division throws because its normalized weight
would be undefined. See [[implicit_factor]] §5 and §7.

## 4. The `energy` signature does not fit, again

`LenticulumCore.energy(factor, x, a, y, ps, st)` takes AutoBayes' $X\times\llbracket
c\rrbracket\times Y$ split. This factor needs the full state `x`, the reference $x_0$, and
**the polarity** (because $P$ depends on it), so the method here reads `a` as the polarity —
using the latent-space slot to carry something that is not a latent value.

That is ugly and it is the *same* mismatch `constraint.md` §3 records from the acausal side:
the core's `energy` signature presumes a causal factor whose $X$/$Y$ split is fixed before
`energy` is called. Two independent factor families have now hit it, which is the point at
which it stops being a quirk and becomes an interface bug. **Recorded, not fixed** — the
signature is used by `Mycelium` and `Lenticulum` and changing it for two callers is a
`LenticulumCore` decision.

## 5. Implementation difficulties

### 5.1 The message is a posterior, not a likelihood

Every other factor in this project returns a *likelihood* from `factor_message`, with the
prior divided out, because a variable of degree $d$ would otherwise count the prior $d$ times
(`gaussian.jl`'s `invert` docstring is explicit about this).

**A diffusion factor cannot divide its prior out.** The prior is a neural network; there is no
subtraction available. So this message double-counts whenever the target variable has degree
> 1.

> [!warning] Correct at degree 1, approximate otherwise
> The usual inverse-problem setting — one diffusion prior, one measurement — is degree 1 and
> is fine. Put two diffusion factors on one variable and each will re-assert its own prior.
> Nothing detects this.

### 5.2 A Dirac message dominates everything it meets

`invert` returns a `DiracBelief`, faithfully to RED-Diff's $\sigma\to0$ family. But
`Mycelium.combine` gives a Dirac **absolute precedence** over any other belief (it is the
$\Lambda\to\infty$ limit). So a diffusion factor in a graph does not negotiate with its
neighbours — it overrides them, and two diffusion factors disagreeing on one variable throw
a "contradictory hard clamps" error rather than averaging.

The fix is the paper's own general case: keep $\sigma > 0$ and return a `GaussianBelief` with
finite precision. RED-Diff's Section 3 derives it; the experiments drop it. Doing so here
would need a variance update in the inner loop and is the most valuable missing piece in this
file.

### 5.3 The free energy is Monte-Carlo noisy and its entropy term is not an entropy

`local_free_energy` re-runs the prox and evaluates `(clamp, score)` at the solution. Two
consequences:

- it is **stochastic** — calling it twice gives different numbers. Every other factor's free
  energy is a closed form. Anything in `Mycelium` that compares free energies across
  iterations (a convergence check, a line search) will see noise.
- the `score` summand plays the role of $-H(b_c)$ in the Bethe sum, but it is *not* the
  entropy of $b_c$: $b_c$ is a Dirac and its differential entropy is $-\infty$. The
  substitution follows the vault's first reading of the score-matching term as the entropy,
  because it depends on the learned distribution rather than the data point; that reading is
  revised in [[The Implicit Diffusion Factor as a Statistical Game]] §2 (it is the prior game's
  energy). Either way it is **not** the Bethe formula's $H$.

So the counting correction of [[Bethe Free Energy]] does not apply to this factor in the way
it applies to `GaussianFactor`, and mixing the two in one graph produces a total that is not
$-\log p(y)$ for anything.

### 5.4 Running the prox inside `local_free_energy` is expensive and re-does work

The inversion has usually just been computed by `factor_message`; `local_free_energy` runs it
again because the message store keeps beliefs, not the internal state the prox converged to.
Caching the solution on `st` is the obvious fix and is not done.

### 5.5 Missing points use a zero reference

A channel with no incoming or prior point contributes zeros to $x_0$. With nonzero
precision this anchors the channel to zero, including the default unobserved/latent
precision of one. With zero precision the reference is ignored. Supply a point or set
precision zero when a zero anchor is not intended.

Related: [[schedule]], [[predictor]], [[reddiff]], [[The Diffusion Factor]],
[[Channels and Polarity]], [[Implicit Learners]], [[Bethe Free Energy]]

## LP-008/009 scope (2026-10-26)

No Definition 20 game methods are added. Option 1 cannot supply an honest entropy from
the current `(clamp, score)` graph objective: `score` is a denoising surrogate, not the
entropy of the inferred belief. The graph method also reconstructs a polarity from missing
channels and runs with a trivial prior, losing an explicitly requested game's precision
and prior split. Fixed quadrature makes a value reproducible; it does not repair either
semantic gap. REDDiff additionally resamples the objective on every call.

`test/conformance.jl` includes the shared default lens checker with an explicit `game_gap`.
It covers all five accepted binary assignments and the unary point polarity. For an
isotropic unit Gaussian, one zero-noise node at time `t` gives the exact linear field
`g(z) = c z`, with `c = λ σ(t)^2`. Its potential is `c ||z||²/2`, so the site has
precision `c I`. Multiplying by an `N(0,I)` anchor gives the joint posterior
`N(0, I/(1+c))`. The binary joint fixture checks both coordinates and the full covariance.
The original single-target fixtures also check nonzero anchor means. Separate assertions
check clamp and score coordinates against elementary formulas.

"Diffusion polarity enumeration and cap (LP-017)" checks complete lists at arities two,
three and four, equality with lazy traversal, deterministic leading entries and structural
answers above the cap. "Joint diffusion messages (LP-017)" checks noncontiguous unequal
blocks for point/Gaussian/mixture messages, joint-prior cotangents, anchor pooling and
correlated Gaussian covariance. These resolve [[LP-017 DiffusionFactor omits supported polarities]].
They do not certify arbitrary priors or stochastic REDDiff/learned ProxDM paths.

## A separate normalised bound (2026-10-27, LP-009)

[[bound]] supplies `free_energy_upper_bound` for a fully specified finite VP reverse chain,
including terminal law, decoder variance and all normalisers. It returns a distinct
`FreeEnergyUpperBound`; arbitrary networks can use an explicitly different Monte Carlo
`FreeEnergyBoundEstimate`. This does not turn `(clamp,score)` into a negative ELBO. Earlier
language identifying the score summand with entropy is only a surrogate analogy, not a
Definition 20 entropy law. Exact game `entropy` and `free_energy` now throw explanatory
errors directing callers to the bound APIs. Existing graph surrogate behavior and its
common conformance exemption are retained. See `../test/bound.jl` for analytic Gaussian
likelihood equality, upper-bound and score-convergence checks.
