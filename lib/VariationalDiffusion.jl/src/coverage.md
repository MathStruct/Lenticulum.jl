#implementation

> Inference always answers, also where the model has never seen data. `coverage` says how
> typical an answer is of the data, by three scores turned into p-values against reference
> points: a flag for open problem T5, not a certificate.

> Sources: code: `coverage.jl`; Bates et al., *Testing for Outliers with Conformal p-values*, Ann. Statist. 2023 (the p-values); Sun et al., *Out-of-Distribution Detection with Deep Nearest Neighbors*, ICML 2022 (distance); Graham et al., *Denoising Diffusion Models for Out-of-Distribution Detection*, CVPRW 2023 (reconstruction); Liu et al., *Energy-based Out-of-distribution Detection*, NeurIPS 2020 (energy)
>
> Bibliography: [[Bibliography#^bates2023conformal|Bates et al. 2023]] · [[Bibliography#^sun2022knn|Sun et al. 2022]] · [[Bibliography#^graham2023ddpmood|Graham et al. 2023]] · [[Bibliography#^liu2020energyood|Liu et al. 2020]]
>
> Theory: [[Open Problems in Implicit Diffusion Learning]] (T4, T5, I9) · [[Implicit Diffusion Learners]] · [[energy_network]]

## 1. The problem

A query far from the training data still has a root: the network extrapolates, and
`implicit_infer` converges to something. Nothing in the solver's report (residual, stability)
distinguishes it from a root the data supports. Two cases matter:

- the **query** is off the relation (T4): a clamp the data never paired with any completion,
  such as $x = 1.3$ on the unit circle;
- the **relation** is unsupported where the answer lands (T5): the network was never trained
  there, so the root is the network's guess.

Both leave the answer in a region of low data density, and that is what the scores measure.

## 2. Three scores

For an answer $z$, with $K$ fixed noise draws $\xi_k$ at a small level $t$:

| score | definition | needs |
|---|---|---|
| distance | $\min_j \lVert (z - x_j) / s\rVert$ over the data $x_j$, scaled per coordinate by $s$ | the data |
| reconstruction | $\tfrac1K\sum_k\lVert\hat x_0(\alpha_t z + \sigma_t\xi_k) - z\rVert^2$, Tweedie's denoiser | any predictor |
| energy | $\tfrac1K\sum_k E(\alpha_t z + \sigma_t\xi_k, t)$, $E = -\log p_t$ | an energy network, a closed-form mixture or a KDE |

- **Distance** is model-free: it asks the data directly. Its cost is a pass over the data per
  query, and in high dimensions raw distances lose contrast, so `scale` (or, later, a learned
  embedding) matters.
- **Reconstruction** reads the model. On the data the denoiser barely moves a point; off it,
  it pulls the point towards the data, and the squared displacement is large. It is the
  only model-based score an ε-network has.
- **Energy** is the model's density, where one exists. It does not see what the density
  smooths over (at $t = 0.05$ the ring's inside, $r = 0.8$, has moderate energy).

The noise draws are fixed when the reference is built and reused for every query (common
random numbers), so scores are deterministic and comparable.

## 3. From scores to p-values

A score has no natural threshold; its distribution on in-distribution points supplies one.
`coverage_reference` scores $N_c$ calibration points and `coverage` returns, for each score $s$,

$$
p = \frac{1 + \#\{j : s_j \ge s\}}{1 + N_c},
$$

the conformal p-value of Bates et al.: if the answer is exchangeable with the calibration
points, $p$ is uniform and $\Pr(p \le \alpha) \le \alpha$. The combined `p` is the smallest
p-value times the number of scores (Bonferroni), which keeps that guarantee for the flag "any
score is atypical".

The calibration points should be held out. Taken from the training data, a fitted network
reconstructs them too well and every answer looks slightly atypical; `coverage_reference`
then at least leaves each point out of its own nearest-neighbour search.

**The guarantee is approximate for answers.** A root of the smoothed relation is not a draw from
the data distribution: it sits at a mode, slightly off the data (T2). Modes are more typical
than draws for the density scores, and the bias moves them a little in distance. On the
ring the effect is small (§4); in general read `p` as a ranking, with $p$ at its floor
$1/(1 + N_c)$ meaning "beyond every reference point".

## 4. Validation

Unit circle, a 48-component ring with width 0.05 (the test suite's `RING`), 2000 data points,
500 held-out calibration points, $t = 0.05$, $K = 8$:

| point | p (distance, reconstruction, energy) | combined p |
|---|---|---|
| 400 fresh data points | flagged at $\alpha = 0.05$: 5.3%, 5.0%, 4.3% | 4.3% |
| origin | 0.002, 0.002, 0.002 | 0.006 |
| $r = 1.3$ | 0.002, 0.002, 0.002 | 0.006 |
| $r = 0.8$ (inside the ring) | 0.002, 0.002, 0.24 | 0.006 |
| answer to $x = 0.6$: $z = (0.6, 0.772)$ | 0.86, 0.99, 0.16 | 0.49 |
| answer to $x = 1.3$: $z = (1.3, 0.011)$ | 0.002, 0.002, 0.002 | 0.006 |

The query $x = 1.3$ converges ([[implicit]]: inference answers with a point near the ridge)
and is flagged by all three scores. The noise level matters: at $t = 0.2$ the denoiser and the
density are smoothed past the ring's hole, and the origin passes both model scores (only the
distance flags it); at $t = 0.01$ every score flags $r = 1.15$. Choose $t$ below the scale of
the detail that should count as "off the data".

An energy network built from the mixture's own $-\log p_t$ reproduces the mixture's energy
scores to 1e-10 and its reconstruction scores to 1e-8; batched and unbatched calls agree to 1e-12.

## 5. Use

```julia
ref = coverage_reference(m, data, ps, st; calibration = heldout, t = 0.05)
sol, _ = implicit_infer(m, z₀, ρ, ps, st)
c = coverage(ref, sol)          # (distance, reconstruction, energy, p_distance, …, p)
c.p < 0.05 && @warn "answer outside the data" c
```

The reference stores the parameters; rebuild it after training. `coverage(ref, Z)` scores the
columns of a matrix, such as the output of [[sampler|implicit_sample]].

For a classification answer, `coverage(ref, z; unassigned = a.unassigned)` carries the mass of
`categorical_answer` that lies between the class codes, and `flagged` is true when `p < alpha`
(default 0.05) or at least half the answer is unassigned: a root between two codes is flagged
rather than read as a class ([[implicit]] §10).

## 6. Not done

- **Scores of the free coordinates only.** A clamp is given, not inferred; a score of the free
  block given the clamp would separate "the query is unusual" (T4) from "the answer is"
  (T5). The current scores mix both.
- **High dimensions.** Raw nearest-neighbour distance and pixel-space reconstruction are weak
  on images; an embedding (the network's features) and multi-level reconstruction (Graham et
  al. use many noise levels) are the known remedies. Not tried; MNIST is the test
  ([[MNIST in Both Directions]]).
- **Coverage as a belief.** A factor could widen its message where coverage is low, rather
  than only flag it.

Related: [[implicit]], [[energy_network]], [[sampler]], [[krylov]]
