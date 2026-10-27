#design #example #algorithm

> Explore the observations, propose a model structure, fit its factors, then compare
> structural revisions. A Gaussian worked example measures both the usefulness and the
> failure of the Bethe criterion: entropy bookkeeping is not a parameter-count penalty.

> Sources: Singh, Mémoli & Carlsson, *Topological Methods for the Analysis of High Dimensional Data Sets and 3D Object Recognition* (2007), §§2.1–2.2 (Mapper construction, no numbered theorem); Edelsbrunner, Letscher & Zomorodian, *Topological Persistence and Simplification* (2002), §3 (persistent homology definition); Kalisch & Bühlmann, *Estimating High-Dimensional Directed Acyclic Graphs with the PC-Algorithm* (2007), §2.2, Theorems 1–2; Chickering, *Optimal Structure Identification With Greedy Search* (2002), §3, Lemmas 9–10; Zheng et al., *DAGs with NO TEARS: Continuous Optimization for Structure Learning* (2018), Theorem 1; Friedman, Hastie & Tibshirani, *Sparse inverse covariance estimation with the graphical lasso* (2008), §2, eq. (2.1); Boyd et al., *Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers* (2011), §6.5, eq. (6.5); Yedidia, Freeman & Weiss, *Constructing Free-Energy Approximations and Generalized Belief Propagation Algorithms* (2005), Theorem 2; Watanabe, *A Widely Applicable Bayesian Information Criterion* (2013), Theorem 2 and Lemma 3. Pipeline and Gaussian derivation original to this vault; code: `examples/gaussian_structure_search.jl`, `test/structure_search.jl`, `src/gaussian.jl`, `lib/Mycelium.jl/src/free_energy.jl`.
>
> Bibliography: [[Bibliography#^singh2007mapper|Singh et al. 2007]] · [[Bibliography#^edelsbrunner2002persistence|Edelsbrunner et al. 2002]] · [[Bibliography#^kalisch2007pc|Kalisch & Bühlmann 2007]] · [[Bibliography#^chickering2002ges|Chickering 2002]] · [[Bibliography#^zheng2018notears|Zheng et al. 2018]] · [[Bibliography#^friedman2008glasso|Friedman et al. 2008]] · [[Bibliography#^boyd2011admm|Boyd et al. 2011]] · [[Bibliography#^yedidia2005bethe|Yedidia et al. 2005]] · [[Bibliography#^watanabe2013wbic|Watanabe 2013]]
>
> Theory (CT-ML wiki): [Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy) · [Statistical Game](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game)

## Explore → propose → fit → revise

1. **Explore observations.** Mapper clusters inverse images of a cover under a chosen
   filter, then records intersections (Singh et al. §§2.1–2.2). Persistent homology tracks
   homology classes through a filtration (Edelsbrunner et al. §3). These may suggest regimes
   or coordinates worth modelling. A Mapper vertex represents a region of observations;
   it does not represent a random variable or one of our factors. Neither construction
   supplies a conditional-independence graph. Using them to select regimes here is a
   proposed workflow, not an implemented or measured TDA stage.
2. **Propose variables and relations.** Physical equations or a probabilistic program may
   supply these directly ([[ModelingToolkit as an Acausal Relation]], [[RxInfer as a Backend]]).
   With data, PC tests conditional independences; GES searches DAG equivalence classes;
   NOTEARS uses a smooth acyclicity constraint (Theorem 1); graphical lasso estimates zeros
   of Gaussian precision. The first three concern DAG models, whereas Gaussian precision
   zeros describe an undirected conditional-independence graph. Their outputs are not
   interchangeable causal claims. PC's consistency requires its assumptions (including
   faithfulness); GES Lemmas 9–10 assume DAG-perfect data and a consistent score. None of
   those guarantees applies automatically to the search below.
3. **Fit a specified graph.** Build variables and factors, estimate parameters and calibrate
   beliefs. A proposed relation needs a likelihood/noise model and normalisation, not just
   endpoints. The example fits Gaussian precisions by Newton's method, represents them as
   Lenticulum factors, and runs Mycelium BP to score each proposal.
4. **Revise and refit.** Compare a graph with and without one factor, using the same variables,
   observations, base measures and parameter convention. Refit both alternatives and rerun
   inference; a local energy term alone does not account for changed beliefs. See
   [[Dynamic Graphs]] §4, [[Bethe Free Energy]] and [[Loopy Message Passing]]. Our example
   rebuilds immutable graphs and performs forward additions only; deletion and warm starts
   are future work.

Yedidia et al. Theorem 2 relates positive BP fixed points and interior stationary points of
the constrained Bethe free energy in their discrete setting. It is not a convergence or
global-minimum theorem. In our continuous Gaussian example, exact dense Gaussian integration
independently checks forest scores, and the script measures the error on loopy proposals.

## What score is being compared?

For factor beliefs $b_a$ and variable beliefs $b_i$, the sign convention is

$$
F_B=\sum_a\bigl(\mathbb E_{b_a}[-\log\psi_a]-H(b_a)\bigr)
+\sum_i(d_i-1)H(b_i)\approx-\log Z.
$$

The correction **adds back** entropy already subtracted in the factor terms. The earlier
minus sign in [[Dynamic Graphs]] §4 was wrong and is corrected with this example.
With fitted parameters $\widehat\theta_G$, the unnormalised potential $\Psi_G$ has
$U_G(x)=-\log\Psi_G(x)$. A normalised data score needs both pieces:

$$
\ell_B(G)=\sum_{r=1}^n U_G(x^{(r)})-nF_B(G),\qquad
C_B(G)=\ell_B(G)+\frac{d+|E_G|}{2}\log n.
$$

**Proof of the sign.** $p_G=\Psi_G/Z_G$, so $-\log p_G=U_G+\log Z_G=U_G-F_G$.
Consequently comparing just the unclamped $F_B$ is insufficient. Multiplying a factor by
a constant shifts $U$ and $F$ equally; their difference is unchanged. Fully clamping all
variables without a normaliser would instead only score the unnormalised data energy.

The model has known zero mean, $d$ diagonal precision parameters and one off-diagonal
parameter per edge; thus $d+|E_G|$ counts **distribution** parameters. Pairwise potentials
and unary precisions share this parameterisation, rather than having independently learned
noise/weight parameters. The last term is BIC on negative-log-likelihood scale (half the
usual $-2\log$ scale). Watanabe's Theorem 2 and Lemma 3 recover that leading parameter
penalty in regular realisable models; [[Factor Graphs and Schemes]] explains the singular case.
Here BIC is added explicitly: parameters are fitted, not integrated out.

The Gaussian factor's `complexity` field includes its noise normaliser. The Bethe entropy
terms correct dependence bookkeeping over states. Neither counts fitted parameters. Without
a penalty, exact maximum likelihood cannot prefer a smaller nested Gaussian model: its
feasible parameter set is contained in the larger one, so its minimum NLL is at least as
large. This proves the overfitting incentive for the exact score. The approximate Bethe
score can additionally favour or reject loops because of normaliser error.

## Runnable Gaussian example

Run from the repository root:

```sh
julia --project=. examples/gaussian_structure_search.jl
```

It lives in the top-level package's `examples/` because it needs both `GaussianFactor` /
`GaussianPrior` from Lenticulum and the graph/free-energy machinery from Mycelium. Keeping
it outside `src/` makes its deliberately small dense solvers an inspectable experiment,
without introducing a public structure-learning API or an optional optimisation dependency.
Its companion `examples/gaussian_structure_search.md` records implementation limits.

The fixture uses `Xoshiro(1010)`, $d=8$, $n=10000$, known zero mean and precision $K$ with
$K_{ii}=1$, $K_{i,i+1}=K_{i+1,i}=-0.12$, and all other entries zero. Samples are
$X=U^{-1}W$ for $K=U^\top U$ and independent standard-normal $W$, hence have covariance
$K^{-1}$ (multiply the covariance matrices to check). Set $S=XX^\top/n$.

For each allowed edge set, minimise $\operatorname{tr}(SK)-\log\det K$ subject to the
specified zeros. The gradient in a symmetric basis $B_r$ is
$\operatorname{tr}((S-K^{-1})B_r)$ and its Hessian is
$\operatorname{tr}(K^{-1}B_rK^{-1}B_s)$, obtained by differentiating the inverse. Newton
steps use positive-definite Armijo backtracking and an explicit convergence check.

For strictly diagonally dominant fitted $K$, let $w_{ij}=|K_{ij}|$ and
$q_i=K_{ii}-\sum_{j\ne i}w_{ij}>0$. Unary factors have precision $q_i$ and each pair has
potential $\mathcal N(x_j;-\operatorname{sign}(K_{ij})x_i,1/w_{ij})$. Expanding their
squares yields joint quadratic $x^\top Kx/2$: each pair supplies diagonals $w_{ij}$ and
cross coefficient $K_{ij}$. All Gaussian normaliser constants are retained in both $U$ and
$F_B$. Candidates need not be forests; failed fitting, failed BP or failed diagonal
dominance throws explicitly rather than silently changing the search space.

**Measured run, 2026-10-27, Julia 1.13.1.** The script prints every accepted proposal,
the unclamped $F_B$, $\ell_B$, $C_B$, and $\ell_B-\ell_{\mathrm{exact}}$. Rounded values:

| step | added pair | $\ell_B$ | $C_B$ | unclamped $F_B$ |
|---|---|---:|---:|---:|
| 0 | — | 114497.379175 | 114534.220536 | 0.000000 |
| 1 | 4–5 | 114399.743494 | 114441.190026 | 2.051884 |
| 2 | 5–6 | 114309.830021 | 114355.881723 | 4.130217 |
| 3 | 2–3 | 114226.776635 | 114277.433507 | 6.209710 |
| 4 | 1–2 | 114146.165546 | 114201.427588 | 8.301310 |
| 5 | 7–8 | 114077.855904 | 114137.723117 | 10.416950 |
| 6 | 3–4 | 114012.960771 | 114077.433154 | 12.555092 |
| 7 | 6–7 | 113964.436648 | 114033.514201 | 14.746063 |
| 8 | 3–5 | 113959.115468 | 114032.798191 | 17.656991 |

At step 7 the graph is exactly the true chain. Step 8 adds a false edge: its Bethe NLL
improvement is 5.321180, exceeding the per-edge penalty $\log(10000)/2=4.605170$.
However, 3.181914 of that improvement is a normaliser error; exact refitted BIC rejects
the edge. Thus even this modest example detects a failure of the Bethe approximation.

| procedure | edges | precision | recall | final NLL / criterion |
|---|---:|---:|---:|---|
| Bethe + BIC | 8 | 0.875000 | 1.000000 | $\ell_B=113959.115468$, $C_B=114032.798191$ |
| Bethe, no penalty | 23 | 0.304348 | 1.000000 | $\ell_B=113951.781503$ |
| graphical lasso path + refitted exact BIC | 7 | 1.000000 | 1.000000 | $C=114033.514201$ |

The unpenalised run accepts 16 false edges and stops before the complete graph; this stop
is not evidence for a complexity penalty, since its score is approximate. The full trajectory
is printed by the same script. At its endpoint the Bethe-minus-exact NLL is $-4.218661$.

The comparator solves the off-diagonal graphical-lasso objective
$\operatorname{tr}(SK)-\log\det K+\lambda\sum_{i\ne j}|K_{ij}|$. It implements Boyd et al.
§6.5's ADMM eigenvalue/soft-threshold updates, leaving the diagonal unpenalised (the
variant distinguished in Friedman et al. §2). Thirty log-spaced $\lambda$ values between
0.3 and 0.003 propose supports; each support is refitted and chosen by **exact** BIC,
without true labels. The selected $\lambda$ is 0.09871033686938005, ADMM residual
$7.22\times10^{-10}$. The separate exact score makes this a useful comparator, not a claim
that both procedures optimise the same approximate criterion. It also avoids penalisation
bias in the reported comparator fit. Tests check the KKT equations and the $\lambda=0$
solution against $S^{-1}$.

The script also reruns penalised search at coupling 0.32 with the same seed and sample
size. It returns ten edges (false pairs 1–3, 3–5, 6–8), precision 0.7 and recall 1.0;
the final Bethe-minus-exact NLL is $-52.079546$. Increasing the penalty alone is not a
general remedy for approximation error.

The fast regression test uses a separate six-variable chain, `Xoshiro(1011)`, coupling
0.12 and 20000 samples: it recovers all five true edges with no false edge. Every accepted
intermediate forest's BP score agrees with dense Gaussian integration within $10^{-5}$
on total NLL scale. A separate loop fixture must exhibit nonzero normaliser error. This
is a checked fixture, not a consistency theorem or a claim that more samples always cure
Bethe bias. There is no benchmark here of PC, GES, NOTEARS or TDA.

## What remains an expectation

The fourth route in [[LP-010 Where the graph of a problem comes from]]—learn one joint
relation, then split it into a product of smaller factors—is still an **expectation**.
This script starts with Gaussian pairwise candidates and never learns or splits a joint
diffusion relation. Dynamic graph mutation, learned arities and Moonkale visual editing
are also outside the experiment. For accurate loopy model selection, a practical next
experiment is to check candidate normalisers with exact integration where available or
with richer regions; no general correction is claimed here.

Related: [[Dynamic Graphs]], [[Bethe Free Energy]], [[Loopy Message Passing]],
[[The Linear Gaussian Chain]], [[Factor Graphs and Schemes]].
