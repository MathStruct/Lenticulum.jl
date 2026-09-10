# Bethe Free Energy

> The generalisation of [[Composition of Statistical Games|AutoBayes Theorem 23]] from a
> **chain** to a **graph** — and the strongest independent confirmation of the paper's central
> claim that I know of.

## The problem

Theorem 23 gives the chain rule for a *sequence* of factors:

$$F^{dc}(\pi, z) \;=\; \mathbb{E}_{(y,b)\sim d'_{c_*\pi}(z)}\bigl[F^c(\pi,y)\bigr] \;+\; F^d(c_*\pi, z)$$

A graph is not a sequence. What replaces this?

## The answer

$$\boxed{\;U \;=\; \sum_f U_f, \qquad H \;=\; \sum_f H_f \;-\; \sum_v (d_v - 1)\, H_v\;}$$

where $d_v$ is the degree of variable $v$. Equivalently, with **counting numbers**
$\kappa_f = 1$ and $\kappa_v = 1 - d_v$:

$$F_{\text{Bethe}} \;=\; \sum_f \kappa_f F_f \;+\; \sum_v \kappa_v F_v$$

This is the classical Bethe approximation from statistical physics, and the reason it belongs
here is what it says:

> **Energies just add. Entropies need a correction.**

That is *exactly* the asymmetry [[Variational Free Energy|Proposition 18]] identifies and
[[Composition of Statistical Games|Definition 22]] encodes, appearing again one level up. The
paper says energies add and entropies chain; the graph version says energies add and entropies
need a degree correction. **Two independent derivations of the same asymmetry.**

It is also the reason `free_energy.jl` exists rather than a one-line
`sum(local_free_energy, factors)`. That one-liner is the energy part, and it is right; the
entropy part is what a naive implementation gets wrong.

## Why the correction: over-counting

A variable of degree $d_v$ is mentioned by $d_v$ factors. Each factor's entropy $H_f$ includes
that variable's uncertainty, so it has been counted $d_v$ times when it should be counted once.
Subtract $d_v - 1$ copies.

- degree 1: $\kappa_v = 0$. Nothing to correct — only one factor mentions it.
- degree 2: $\kappa_v = -1$. Counted twice; subtract one.
- degree $d$: subtract $d-1$.

## The Euler characteristic

$$\sum_f 1 \;+\; \sum_v (1 - d_v) \;=\; |F| + |V| - |E| \;=\; \chi(g)$$

since $\sum_v d_v = |E|$.

> **The total counting number is the Euler characteristic of the factor graph.** It is `1`
> exactly when the graph is a connected tree, and `1 - L` when it has `L` independent loops.

So a single integer says how badly the bookkeeping can be wrong: on a tree the counting is
exact, and the deficit from `1` is the number of loops the approximation has to pretend are not
there. The test suite asserts `total_counting_number(g) == euler_characteristic(g)`, because it
is the cheapest possible check that the graph and the accounting have not drifted apart.

## Exactness, and what replaces it

- **On a tree**: Bethe is exact. Run a [[Schedules|`TreeSchedule`]] and the number is the true
  free energy.
- **On a loopy graph**: it is an approximation, and a specific one —
  **fixed points of loopy belief propagation are stationary points of the Bethe free energy**
  (Yedidia, Freeman & Weiss).

That last result is worth internalising, because it collapses two things that look separate:
*running BP* and *minimising the Bethe free energy* are the same activity. If your inference
loop and your reported objective ever seem to disagree, they are not two facts — one of them is
computed wrong.

## Two decompositions, one number

`Mycelium` offers both:

| | `bethe_free_energy` | `chain_free_energy` |
|---|---|---|
| shape | $\sum_f$ over factors $+$ counting correction over variables | Theorem 23's recursion folded along an order |
| order-dependent? | **no** | **yes** |
| on a tree | exact | exact |
| grading | `(factors = …, variables = …)` | nested per composition step |

They are two decompositions of the same quantity. They agree when every belief is a
`DiracBelief` — deterministic inference, where all expectations are evaluations and all
variable entropies vanish — which is exactly the regime the test suite checks against a
hand-computed value.

Off that regime, **prefer Bethe for reporting** (it has no arbitrary order in it) and the chain
form for reasoning about a specific message path.

> [!warning] Theorem 23 needs a "downstream", and a graph does not have one
> The recursion averages the upstream loss under the *downstream* inversion. On a chain
> "downstream" is unambiguous. On a graph it is a property of the **message schedule**, not of
> the wiring — so two schedules give two different decompositions of the same total. This is a
> genuine gap in transporting the paper's Theorem 23 to graphs, and Bethe is the standard way
> around it.

## The grading survives

`factor_free_energies` returns a `GradedEnergy` keyed by factor name — i.e.

$$E_G \;=\; \bigoplus_f E_f$$

from [[Scalar and Multivariate Energy]], realised. The composite energy of a graph **is** the
per-factor breakdown; per-factor loss attribution is not a logging feature added afterwards.

And because each factor carries its own `scalarisation`, a residual factor can use a squared
norm while a likelihood factor uses the identity, in the same graph — with
[[Scalar and Multivariate Energy]] §5 saying exactly when that composition is strict and when
it is lax.

> [!note] At zero temperature this reduces to LeCun's energy
> The entropy term carries a factor of $T$. Send $T\to0$ and it vanishes — even though a point
> mass has $H = -\infty$ — leaving $F = \sum_c E_c$, the energy of a non-probabilistic factor
> graph.
>
> That is why the `DiracBelief`-valued factors in `lib/` appear to "have no entropy": they run
> at $T=0$ inside a form that is implicitly $T=1$. The mismatch is real; the missing entropy is
> not the right description of it. See [[Energy-Based Factor Graphs]] §3.2.

Related: [[Composition of Statistical Games]], [[Scalar and Multivariate Energy]],
[[Loopy Message Passing]], [[free_energy]], [[Energy-Based Factor Graphs]]
