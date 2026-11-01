# E3: when composition beats a flat mixture

Run `julia -t auto --project=. examples/composition_e3.jl` (about 4 minutes on 8 threads); the test is
`test/composition_e3.jl`. Experiment E3 of the design note
[`Minimal Stochastic Factors`](../vault/Design/Minimal%20Stochastic%20Factors.md) (§8, §10).

## Question

By Proposition 7, $L$ binary switching factors give at most $2^L$ Gaussian components. Composition
therefore buys parameter efficiency, not a larger class, and only when the data's components have
the structure that the composition imposes. Which structure is that, and how much does it buy?

## Setup

There are three data sets in two dimensions, each with an exact density. Every mode has the same
covariance (variances 0.12 and 0.09, correlation 0.38).

- **`grid3`, `grid4`** have a product structure: $2^L$ modes at $c + \sum_l k_l \delta_l$, with independent
  bits $k_l$. The offsets are $(2,0)$, $(0,2)$, $(4,1)$ and, for `grid4`, also $(1,4)$. The modes are
  8 and 16, at least 2.0 and 1.41 apart.
- **`ring`** has no product structure: 8 equal modes on a circle of radius 3. Eight sums of three
  offsets are the vertices of a projected cube, and at most 6 of them lie on the convex hull, so an
  octagon is not of that form.

There are three models, all with exact likelihoods.

- **additive** ($3L + 5$ parameters). It adds $L$ binary switching factors: $z = c + \sum_l u_l + \varepsilon$,
  where $u_l \in \{0, \delta_l\}$ is chosen with weight $\sigma(\pm\rho_l)$ and $\varepsilon \sim \mathcal N(0, \Sigma)$.
- **rank-one, shared $z$** ($5L + 5$ effective parameters). This is the note's graph reading (§4). A
  Gaussian prior on $z$ is multiplied by $L$ rank-one switching constraints
  $\mathcal N(a_l^\top z; b_{l, k_l}, q_l)$. The bits are independent, so the density is a product of experts:
  $p(z) = \mathcal N(z; m, P) \prod_l \sum_j \pi_{lj} \mathcal N(a_l^\top z; b_{lj}, q_l) / \bar Z$. The coupling
  normaliser $\bar Z = \sum_k \pi_k \mathcal N(b_k; A m, A P A^\top + Q)$ runs over all $2^L$ configuration
  *tuples* $k = (k_1, \dots, k_L)$, with $\pi_k = \prod_l \pi_{l, k_l}$, $b_k = (b_{1, k_1}, \dots, b_{L, k_L})$, $A$ the
  stacked $a_l^\top$ and $Q = \operatorname{diag}(q_l)$.
  The density integrates to one to $10^{-15}$ on a grid, which checks $\bar Z$.
- **flat** ($6K - 1$ parameters): a $K$-component mixture with full covariances, fitted by EM
  with 5 restarts.

The two composed models are fitted by exact maximum likelihood (Adam with ForwardDiff, 1 500
iterations) from 16 random starts, and the best training likelihood is kept. Each setting has
150 or 2 000 training points and 2 000 held-out points.

## Measured (2026-10-31)

The table shows the held-out mean $\log p$ minus the truth's, which is the gap in nats; smaller is better.
The truth scores −2.509 on `grid3`, −3.134 on `grid4` and −2.821 on `ring`.

| data | model | parameters | gap, $n = 150$ | gap, $n = 2000$ |
|---|---|---|---|---|
| `grid3` | **additive, $L = 3$** | **14** | **0.040** | **0.003** |
| | rank-one, shared $z$ | 20 | 0.652 | 0.591 |
| | flat, $K = 4$ | 23 | 0.550 | 0.356 |
| | flat, $K = 8$ | 47 | 0.355 | 0.011 |
| `grid4` | **additive, $L = 4$** | **17** | **0.043** | **0.005** |
| | rank-one, shared $z$ | 25 | 0.669 | 0.632 |
| | flat, $K = 8$ | 47 | 0.476 | 0.425 |
| | flat, $K = 16$ | 95 | 0.790 | 0.063 |
| `ring` | additive, $L = 3$ | 14 | 0.924 | 0.894 |
| | rank-one, shared $z$ | 20 | 0.815 | 0.801 |
| | flat, $K = 4$ | 23 | 0.593 | 0.527 |
| | **flat, $K = 8$** | **47** | **0.341** | **0.014** |

Flat $K = 2$, omitted from the table, is worst everywhere: its gap is 0.73 to 1.38. The composed
fits reached the best training likelihood from 1 to 5 of their 16 starts, so the random-start
landscape has many local optima.

## Fitting: Adam against exact EM

Under the additive wiring, the posterior over the $2^L$ configurations is exact, and the means
$W\varphi_k$, with $\varphi_k = [1; k]$ and $W = [c\; \delta_1 \dots \delta_L]$, are linear in the parameters. So
EM is exact and its M-step is closed form: a weighted least squares for $W$, then the shared
covariance and the bit probabilities (`em_additive`). `compare_fitting` gives EM the same random
starts as Adam, plus 240 more. The "EM, the same 16" time is pro-rated from the 256-start run.

| data | $n$ | method | starts at the best | held-out gap | wall time |
|---|---|---|---|---|---|
| `grid3` | 150 | Adam, 1 500 iterations | 4/16 | 0.040 | 1.6 s |
| | | EM, the same starts | 0/16 | 0.706 | 0.1 s |
| | | EM | 27/256 | 0.040 | 1.3 s |
| `grid3` | 2 000 | Adam | 5/16 | 0.003 | 11.0 s |
| | | EM, the same starts | 4/16 | 0.003 | 0.1 s |
| | | EM | 37/256 | 0.003 | 2.3 s |
| `grid4` | 150 | Adam | 2/16 | 0.043 | 1.8 s |
| | | EM, the same starts | 0/16 | 0.480 | 0.0 s |
| | | EM | 16/256 | 0.043 | 0.5 s |
| `grid4` | 2 000 | Adam | 3/16 | 0.005 | 26.9 s |
| | | EM, the same starts | 1/16 | 0.005 | 0.4 s |
| | | EM | 23/256 | 0.005 | 6.5 s |

- **EM is monotone and converges fast.** The log-likelihood never decreased in any run. The
  median start converged in 48 to 114 iterations.
- **Per start, EM gets stuck more often than Adam.** It reaches the best optimum from 6 to 14 % of
  starts, against 12 to 31 % for Adam.
- **EM is still the better fitter here.** A start costs 1 to 2 % of an Adam start. At $n = 2000$,
  256 EM starts (2.3 and 6.5 s) reached the same optimum as 16 Adam starts (11.0 and 26.9 s). At
  $n = 150$ the two took about the same time. How large the advantage is depends on the success
  probability one wants and on that cost ratio, as Mistral's review notes: at the measured per-start
  success rates, a 1 % chance of missing the optimum needs 13–37 Adam starts against 31–75 EM
  starts. At 1–2 % of the cost per start, the EM starts cost about as much as one or two Adam starts. With
  EM, random restarts are a workable answer to the local optima.
- **Annealing did not help.** In exploratory runs that were not kept, deterministic annealing of
  the E-step (Ueda and Nakano) collapsed to the symmetric fixed point where all $\delta_l$ are
  equal, and perturbing the offsets during annealing did not reach the optimum either.

## What it shows

1. **Composition wins on product structure, and the advantage is in sample size.** The additive
   model fits both grids to within 0.005 nats with 14 and 17 parameters. With 150 points its gap
   is 0.04, against at best 0.36 and 0.48 for a flat mixture. A 16-component flat mixture needs
   95 parameters and 2 000 points to get within 0.06. With 150 points it overfits (0.79) and does
   worse than 8 components.
2. **It loses without that structure.** On the ring, three switches still give 8 components, but
   their means are forced onto a projected cube. The gap stays at 0.89 nats at $n = 2000$, worse
   than a flat mixture with 4 components. Composition is a prior on the arrangement of the
   components, and it pays exactly when the prior is right.
3. **How the factors meet matters as much as how many there are.** Rank-one constraints on a shared
   $z$ fail on the grids too, by about 0.6 nats. Constraints on one variable *multiply*: a mode sits
   where all $L$ strips $a_l^\top z \approx b_{l, k_l}$ cross, and the normaliser $\bar Z$ suppresses the
   configurations whose strips miss each other. Additive composition *adds*: each switch shifts the
   whole distribution. That gives the Minkowski-sum grid, and it needs the extra variables $u_l$
   that the $z = \sum_l u_l$ wiring introduces. So the expressive wiring is a sum node fed by
   switches, not several constraints on one node. This is the overlap penalty that §10 anticipated,
   now measured.

   It is a dimensional fact, as two reviews of the second round pointed out. All $L$ strips cross
   exactly for every configuration only if every $b_k$ lies in the column space of $A$ shifted by
   $Am$. The $2^L$ points $b_k$ span an affine space whose dimension is the number of switches with
   $b_{l0} \ne b_{l1}$, and the column space has dimension at most $n$. So with more than $n$ informative
   switches in $\mathbb R^n$ (here $L = 3, 4 > n = 2$), some configurations must miss, and $\bar Z$ suppresses
   them: the grids are outside the rank-one class, which is misspecified rather than hard to fit.
   With $L \le n$, rank-one constraints give the $2^L$ grid as well. That case was not run.

## Caveats

- These are synthetic data whose structure exactly matches one model class.
- Each setting has one seed and one draw of the data.
- The composed models are fitted from random starts, and many starts end in local optima. The
  table reports the best of 16, chosen on the training set. Exact EM reaches the same optima.
- The models are fitted by direct maximum likelihood, not by message passing in a Mycelium graph.
