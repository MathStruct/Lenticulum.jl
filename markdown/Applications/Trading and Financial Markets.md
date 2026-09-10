# Trading and Financial Markets

> The domain where **the residual is the product**. Everywhere else a nonzero energy is a
> diagnostic; here it is the trade signal.

## The relations

No-arbitrage conditions are exact relations among observable prices, with no input and no
output:

| relation | form |
|---|---|
| put–call parity | ``C - P = S - K e^{-rT}`` |
| triangular FX | ``r_{A\to B}\, r_{B\to C}\, r_{C\to A} = 1`` (log-linear) |
| index vs constituents | ``I = \sum_i w_i S_i`` |
| cash-and-carry | ``F = S e^{(r-q)T}`` |
| cross-listing / ADR | ``P_{\text{local}} = P_{\text{foreign}} \times \mathrm{FX}`` |

Several are **linear** — or linear in logs — which puts them squarely in the fragment where
`LinearConstraintFactor` is exact. None designates an output: parity does not say which of the
four legs is "the answer", and in practice you infer whichever one you cannot observe cleanly.

## What is observed

Quotes from multiple venues, at different times, with bid–ask spreads, varying liquidity and
missing instruments. **Asynchrony is not a nuisance here, it is the microstructure** — two
venues quoting the same relation at slightly different timestamps is where a large part of the
signal lives.

That is a strong case for [[Time as a Base]]: a price is a trajectory queried at a time, not a
value attached to a tick.

## What would be learned

- A **volatility surface** — fitted, not derived, and feeding the option legs.
- An **illiquid or exotic instrument's** pricing model, where no closed form applies.
- A **microstructure noise model** — how much of an observed deviation is spread versus signal.

Mixed with exact parity relations in one graph, which is the grey-box case: the relations you
trust constrain the components you fitted.

## What a residual means

**A mispricing.** The graded energy of [[Scalar and Multivariate Energy]] gives per-relation
attribution: not "something is off by 3 basis points" but *which* parity is violated and by how
much, with the uncertainty of the estimate attached.

And the uncertainty is the operationally important half. A deviation of two basis points means
nothing if the posterior width is five; the same deviation is a trade if the width is a quarter.
Point-estimate arbitrage screens cannot make that distinction and are famous for firing on
noise.

## One model, several questions

- **Which leg is stale?** Clamp the liquid legs, infer the illiquid one, compare with its quote.
- **What is the implied state?** Clamp all quotes, read the latent (implied vol, implied
  dividend, implied borrow).
- **What if this leg moved?** Clamp a hypothetical and propagate.

Same relations, different clamps.

## What would be hard

- **Returns are heavy-tailed.** The Gaussian fragment is where every exactness result in this
  project lives, and financial innovations are decidedly not Gaussian. This is
  [[messages]] §1's gap in its most commercially painful form: you want particle or
  heavy-tailed beliefs and `combine` does not support them.
- **The relations hold only approximately.** Transaction costs, borrow costs, settlement and
  short constraints all widen parity into a band. That is expressible — a band is a soft
  constraint with a suitable noise model — but it means the "exact relation" framing is already
  an idealisation.
- **Regime change.** The learned components are non-stationary, and nothing in the framework
  tracks that a fitted factor has gone stale.
- **Latency.** If the residual is the signal, the inference has to run at market speed. See
  [[Parallelism and Compilation]] — the current implementation is measured at quadratic in the
  number of factors.

Related: [[Motivating Examples]], [[Scalar and Multivariate Energy]], [[Time as a Base]],
[[messages]], [[Parallelism and Compilation]]
