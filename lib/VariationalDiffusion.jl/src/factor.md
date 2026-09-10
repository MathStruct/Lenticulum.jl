# factor.jl — implementation note

> `DiffusionFactor`: the point where [[ImplicitREDDiff]]'s selection matrices stop being
> notation and become a `LenticulumCore.Polarity`.

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

## 3. `supported_polarities` under-reports on purpose

`supports_polarity` accepts *any* assignment over the channel set with at least one
`Unobserved()` channel — including `Latent()` ones — because a joint diffusion prior really
can answer all of them. `supported_polarities` enumerates only the $n$ "one unobserved, rest
observed" cases.

The gap is deliberate: a scheduler needs a **listable** set, and the full set has $3^n - 2^n$
elements. The predicate is the truth; the enumeration is a usable subset. `channels.md` notes
that the two exist precisely so they can differ, and this is the first factor where they do.

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
  substitution follows [[Implicit Learners]] §"Diffusion", which reads the score-matching term
  as the entropy because it depends on the learned distribution rather than the data point.
  It is a defensible reading and it is **not** the Bethe formula's $H$.

So the counting correction of [[Bethe Free Energy]] does not apply to this factor in the way
it applies to `GaussianFactor`, and mixing the two in one graph produces a total that is not
$-\log p(y)$ for anything.

### 5.4 Running the prox inside `local_free_energy` is expensive and re-does work

The inversion has usually just been computed by `factor_message`; `local_free_energy` runs it
again because the message store keeps beliefs, not the internal state the prox converged to.
Caching the solution on `st` is the obvious fix and is not done.

### 5.5 `assemble_state` silently zero-fills

A channel with no incoming message contributes zeros to $x_0$. That is safe *only* because
such a channel's precision is zero and the entry is multiplied out — but the invariant is
implicit, and a future polarity that gives a message-less channel nonzero $\rho$ would read
the zeros as data.

Related: [[schedule]], [[predictor]], [[reddiff]], [[The Diffusion Factor]],
[[Channels and Polarity]], [[Implicit Learners]], [[Bethe Free Energy]]
