#open-problem

> The honest ledger of the implicit diffusion learner, in two parts. **Theoretical problems**
> are open questions or intrinsic limits: nobody knows the answer, or there is provably no free
> lunch. **Missing implementation** is work whose design is known and that simply has not been
> done. Each entry links to where it was observed or measured.

> Sources: original to this vault (design and analysis), collecting results from the notes and tutorials linked below; Du et al., *Reduce, Reuse, Recycle*, ICML 2023 (composition at $t > 0$); Chung et al., *Diffusion Posterior Sampling*, ICLR 2023 (conditional sampling)
>
> Bibliography: [[Bibliography#^du2023reduce|Du et al. 2023]] · [[Bibliography#^chung2023dps|Chung et al. 2023]]
>
> Theory (CT-ML wiki): [Statistical Game](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game) · [Bayesian Inversion](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Inversion)

The companion ledger for the algebraic family is [[Open Problems in Algebraic Implicit Learning]];
several entries below are the same problem in a different family.

## Part I. Theoretical problems

### T1. Finding all answers — **open; multi-start search implemented**

A query on a multivalued relation has several stable roots (the circle's $y \approx \pm 0.8$, the
robot arm's elbow up and down). Point inference returns the root whose basin contains the
start. There is no guarantee that a set of starts finds every stable root, and no way to know
when all have been found. Sampling the conditional distribution would find them with the
right frequencies, but then the answer is a distribution, and how to turn a learned relation
into well-calibrated branch probabilities is itself open. `implicit_roots` now searches from
several starts and returns every distinct stable answer it reaches ([[implicit]] §5), which finds
both circle branches; completeness remains unguaranteed. Observed: [[Implicit Diffusion Learners]]
§6; tutorials *A relation without training* and *Robot arm*.

### T2. Bias against conditioning — **intrinsic**

The relation recovered is the ridge of a *smoothed* density, displaced inwards by about
$(s^2 + \sigma_t^2/\alpha_t^2)/(2R)$ for a ring of radius $R$. Smaller noise levels shrink the
bias and make the field stiff and ill-conditioned ($\varepsilon^\ast \sim 1/\sigma_t$ near the
data). There is no setting with neither; the open part is a principled choice of the levels for
a given relation and data density. Measured: [[Implicit Diffusion Learners]] §5 (radius 0.98 at
the default levels, 0 at RED-Diff's range).

### T3. Branch points — **intrinsic**

Where branches meet (the circle at $x = \pm 1$), smoothing merges them and creates saddles; the
solver lands on them and reports `stable = false`. This is the smoothed form of the
discriminant ([[Branches and the Discriminant]]; [[Open Problems in Algebraic Implicit Learning]]
§7). Near a branch point the answer is ill-conditioned for every method, not only this one.
Observed: tutorial *Train a small diffusion model* (unstable answers near $x = \pm 0.9$).

### T4. What a query off the relation should return — **open (semantics)**

Clamping $x = 1.05$, just off the circle, returns the ridge point on that line with zero
residual: the clamped problem *has* a root. Whether the right answer is that point, a refusal,
or a point with a flag is a modelling decision without an agreed answer. Energy-parametrised
models at least provide the number to decide with ($U$ at the answer, [[energy_network]] §3);
ε-networks provide nothing comparable.

### T5. Coverage and out-of-distribution queries — **open**

The relation is only learned where data were seen. Outside, the field is extrapolation, and
answers degrade silently (robot arm: misses up to 0.18 at targets needing angles outside the
training range). Where data are sparse *along* the relation, the ridge can thin out or break.
There is no measure of how well a learned relation covers a region, and no guarantee that it
is connected where the true one is. Measured: tutorial *Robot arm* (in- vs out-of-distribution
statistics).

### T6. A non-conservative field defines no energy — **intrinsic to ε-networks; resolved by energy networks at a cost**

A network that outputs $\varepsilon$ directly is not the gradient of anything (2.6% Jacobian
asymmetry on the circle). Then the relation has no energy, "stable" uses the symmetric part
of the Jacobian as a heuristic, and two answers cannot be compared. The energy
parametrisation removes the problem by construction ([[energy_network]]); what remains open is
whether the extra cost (about 3× in training, second derivatives everywhere) is always worth
it, and how much the asymmetry of a well-trained ε-network matters in practice.

### T7. Training through inference shapes only the visited branches — **intrinsic to the bilevel objective**

Backpropagating a task loss through inference moves the branches inference actually lands
on. When a parabola was learned from a circle this way, the old upper arc survived as another
branch. Density training (the game's own gradient) shapes everything but ignores the task.
How to combine the two with guarantees is open. Observed: [[Backpropagation through Implicit Inference]]
§7; [[The Implicit Diffusion Factor as a Statistical Game]] §4.

### T8. Hyperparameters without theory — **partly open**

The field nodes (levels, samples), the weighting λ and the scale of the precisions decide
what relation is learned and how strongly evidence counts. λ is derivable for Gaussian data
(`calibrate_lambda`, [[RED-Diff as a Statistical Game]] §4); in general it is not. Precision
interacts with the field's scale: on the trained circle, a precision of 1 already pulled an
answer most of the way from the relation to the anchor.

### T9. Composing diffusion factors at $t > 0$ — **known obstruction; mild for root finding**

Noising does not commute with products, $q_t * (p_1 p_2) \ne (q_t * p_1)(q_t * p_2)$, so adding
the scores of two diffusion factors is exact only at $t = 0$. Remedies (MCMC corrections at each
level) exist but are approximate. See [[Language Models]] §7. For root finding (intersections
of relations, `ProductRelation`) the effect is a smoothing bias of the size a single relation
already has: circle ∩ ellipse is found within 0.029 of the exact points
([[Composing Diffusion Factors]] §4).

### T10. Messages are posteriors, not likelihoods — **open design problem**

A diffusion factor carries its own prior, which cannot be divided out of its inversion, so it
sends posteriors where a factor graph expects likelihoods, and neighbours double-count. See
[[The Diffusion Factor]] §4.

### T11. No guarantees — **open**

Nothing guarantees that the solver converges on a learned field, that a stable root exists for
a given query, or that the relation is identifiable from data (which relations produce the same
smoothed ridges?). In flat regions far from data, descent stalls and the solver reports it
([[Implicit Diffusion Learners]] §6).

## Part II. Missing implementation

### I1. All branches: mixture beliefs and conditional sampling — **mostly done**

Built: multi-start inference returning the distinct stable roots (`implicit_roots`), their
Laplace covariances (`implicit_laplace`), and `implicit_mixture`, which assembles them into a
`MixtureBelief` ([[Mixture Belief]]) with Laplace mass estimates as branch weights,
$w_k \propto e^{-E(z_k)}\det(\Sigma_k)^{1/2}$. That needs an energy, which energy networks and
closed-form mixtures (hence kernel estimates) have; on a circle whose upper half carries three
times the data, the weights recover the 3:1 ratio. Missing: weights for plain ε-networks (no
energy, so equal weights), and a conditional sampler (DPS-like, or ProxDM's sampler with the
clamp) as an independent check of branch frequencies.

### I2. Uncertainty: a Gaussian answer — **done**

`implicit_laplace` returns the Laplace covariance $(\operatorname{sym} J_{FF})^{-1}$ at an answer,
`density_lambda` chooses the weighting under which it is in data units ([[implicit]] §5), and
`laplace_belief` returns it as a `GaussianBelief`. Still missing: using its entropy in the
statistical game ([[The Implicit Diffusion Factor as a Statistical Game]] §6).

### I3. Gaussian messages from learned factors — **done**

All belief types now live in `LenticulumCore` ([[Belief Algebra]] §6), and
`ImplicitProx(…; message = :gaussian)` makes a diffusion factor send the Laplace Gaussian, with
the target's anchor divided out ([[implicit_factor]] §5). The model's own prior stays in the
message (T10).

### I4. Automatic recovery from unstable answers — **done**

`implicit_roots` restarts from perturbed points and reports only converged, stable answers
([[implicit]] §5).

### I5. Cost: batching and matrix-free solves

Each field evaluation calls the network once per node (128 times by default), sequentially;
the Jacobian costs one derivative pass per coordinate, and the linear algebra is dense,
$O(n^3)$. Known fixes: evaluate all nodes in one batched call (also the right shape for
Reactant), and replace dense Newton by Newton–Krylov with Jacobian-vector products. Required
before anything beyond a handful of dimensions ([[backends]] §5).

### I6. Energy networks on every backend

Second derivatives work with Zygote and ForwardDiff-over-Zygote; Enzyme's forward-over-reverse
fails on Lux layers, and Reactant is not wired for energy models ([[energy_network]] §4, §7).

### I7. ProxDM, completed

No adjoint for `prox_infer`, no ProxDM inversion for `DiffusionFactor`, no `ad` field on
`ProxNetwork`, and the sampler is first order (spread 0.29 against 0.30) ([[proxdm]] §5).

### I8. Composition — **partly done**

Products of relations on the same variables are built (`ProductRelation`, [[product]]): all
query functions and the adjoint work on them. Not built: MCMC correction steps (T9's remedy for
sampling), and products over different variable sets with a separator term.

### I9. Coverage diagnostics

A practical flag for T5: the energy at the answer (energy networks), the distance to the
nearest training data, or a density estimate on $Z$. Any of them would mark out-of-distribution
answers. Not built.

### I10. Real data and higher dimensions

Everything validated is two- to four-dimensional and synthetic or closed-form. A real-data
benchmark, and a problem with tens of coordinates, are the obvious next tests; I5 comes first.

### I11. Nonlinear factors around the learner

Latent states with nonlinear dynamics (the particle tutorial's noisy positions, SLAM) need
nonlinear Gaussian factors and Gauss–Newton in the graph, which Lenticulum does not have yet.

## Summary

| | problem | kind |
|---|---|---|
| T1 | finding all answers | open; multi-start search implemented |
| T2 | smoothing bias vs conditioning | intrinsic |
| T3 | branch points | intrinsic |
| T4 | queries off the relation | open (semantics) |
| T5 | coverage, out of distribution | open |
| T6 | non-conservative ε-networks | intrinsic; resolved by energy networks at a cost |
| T7 | training shapes only visited branches | intrinsic to the bilevel objective |
| T8 | hyperparameters | partly open |
| T9 | composition at $t > 0$ | known obstruction; mild for root finding |
| T10 | posterior messages | open design problem |
| T11 | convergence, existence, identifiability | open |
| I1 | mixture beliefs, conditional sampling | mostly done: `implicit_mixture` with energy-based weights; no sampler |
| I2 | Laplace (Gaussian) answers | done (`laplace_belief`) |
| I3 | Gaussian messages from `lib/` | done (`message = :gaussian`) |
| I4 | restarts after unstable solves | done |
| I5 | batched nodes, Newton–Krylov | missing, needed for scale |
| I6 | energy networks on Enzyme and Reactant | missing |
| I7 | ProxDM adjoint and factor | missing |
| I8 | composition | products on shared variables done; corrections and separators missing |
| I9 | coverage diagnostics | missing |
| I10 | real data, higher dimensions | missing |
| I11 | nonlinear factors | missing (Lenticulum-wide) |

Related: [[Implicit Diffusion Learners]], [[Inference Signatures]], [[Backpropagation through Implicit Inference]],
[[The Diffusion Factor]], [[The Implicit Diffusion Factor as a Statistical Game]], [[energy_network]],
[[proxdm]], [[backends]], [[Belief Algebra]], [[Open Problems in Algebraic Implicit Learning]]
