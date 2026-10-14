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
the default levels, 0 at RED-Diff's range). With noise blocks the clamped coordinates can be held
clean, so only the free ones are smoothed: on the circle the radius goes from 0.978 to 0.993 at
the default levels and from 0.64 to 0.78 at levels up to 0.2 ([[implicit]] §9). The bias of the
free coordinates remains.

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

### T5. Coverage and out-of-distribution queries — **open; per-answer flag implemented**

The relation is only learned where data were seen. Outside, the field is extrapolation, and
answers degrade silently (robot arm: misses up to 0.18 at targets needing angles outside the
training range). Where data are sparse *along* the relation, the ridge can thin out or break.
`coverage` (I9) now flags a single answer that lies outside the data, but there is still no
measure of how well a learned relation covers a region, and no guarantee that it is connected
where the true one is. Measured: tutorial *Robot arm* (in- vs out-of-distribution
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

### T9. Composing diffusion factors at $t > 0$ — **known obstruction; mild for root finding, corrected for sampling where log-densities exist**

Noising does not commute with products, $q_t * (p_1 p_2) \ne (q_t * p_1)(q_t * p_2)$, so adding
the scores of two diffusion factors is exact only at $t = 0$. Remedies (MCMC corrections at each
level) exist but are approximate. See [[Language Models]] §7. For root finding (intersections
of relations, `ProductRelation`) the effect is a smoothing bias of the size a single relation
already has: circle ∩ ellipse is found within 0.029 of the exact points
([[Composing Diffusion Factors]] §4). For sampling, corrector steps and SMC reweighting are
built and measured (I8, [[product]] §5): with reweighting the mode masses of a product are right
within sampling error on the test problems. Reweighting needs log-densities, which a direct
ε-network does not have.

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

### I1. All branches: mixture beliefs and conditional sampling — **done**

Built: multi-start inference returning the distinct stable roots (`implicit_roots`), their
Laplace covariances (`implicit_laplace`), and `implicit_mixture`, which assembles them into a
`MixtureBelief` ([[Mixture Belief]]) with Laplace mass estimates as branch weights,
$w_k \propto e^{-E(z_k)}\det(\Sigma_k)^{1/2}$. That needs an energy, which energy networks and
closed-form mixtures (hence kernel estimates) have; on a circle whose upper half carries three
times the data, the weights recover the 3:1 ratio. The conditional sampler exists too:
`implicit_sample` runs the reverse diffusion with the hard-clamped coordinates replaced at every
step (with RePaint's resampling), and on that circle draws the upper branch with frequency 0.75,
the exact conditional's ([[sampler]]). Still missing: weights for plain ε-networks (no energy,
so equal weights; the sampler's frequencies are the substitute), and sampling under soft
evidence, which needs guidance.

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

### I5. Cost: batching and matrix-free solves — **done**

Each field evaluation calls the network once per node (128 times by default), sequentially;
the Jacobian costs one derivative pass per coordinate, and the linear algebra is dense,
$O(n^3)$. Known fixes: evaluate all nodes in one batched call (also the right shape for
Reactant), and replace dense Newton by Newton–Krylov with Jacobian-vector products. Required
before anything beyond a handful of dimensions ([[backends]] §5).

Done: every Jacobian is now computed only on the free coordinates (`prior_jacobian(…; coords)`,
at a cost that grows with their number), the clamped part of the adjoint is one VJP
(`prior_vjp`), and `ImplicitDiffusion(…; batched = true)` evaluates one noise level per network
call. A query with 10 of 794 coordinates free went from about 26 s per solver iteration to
0.04 s ([[implicit]] §6). For when *many* coordinates are free (generating an image),
`linsolve = KrylovSolve()` makes every solve matrix-free: GMRES on Jacobian-vector products for
the Newton step, on vector-Jacobian products for the adjoint, Lanczos for the stability test,
conjugate gradients and stochastic Lanczos quadrature for the Laplace approximation. With 784 of
794 coordinates free, `implicit_infer` takes 1.1 s where one dense block took 33 s ([[krylov]]).
Not built: many queries at once (a batch of test images as one solve).

### I6. Energy networks on every backend — **done (Enzyme on CPU blocked upstream)**

Second derivatives work with Zygote, ForwardDiff-over-Zygote, Mooncake forward-over-reverse and
Reactant. Reactant has its own five `_energy_*` hooks: Enzyme forward mode over a reverse
gradient inside `Reactant.@compile`, exact against finite differences, with inference, the
energy and the adjoint tested through them ([[energy_network]] §7). Enzyme's forward-over-reverse
on plain Julia code still fails on CPU for two upstream reasons (a missing LuxLib forward rule
for `fused_dense!`, and Enzyme.jl #3368 on Julia 1.12); Reactant does not need it. On a GPU
there is also the route without either: with `device = (to = gpu_device(), back = cpu_device())`
and ForwardDiff over Zygote, every derivative runs on CUDA and agrees with the CPU to about 1e-7
([[energy_network]] §8).

### I7. ProxDM, completed — **done**

Done: the adjoint of `prox_infer` (`prox_pullback`, also its `rrule`; dense or GMRES on prox
VJPs), exact derivatives of `MixtureProx`, and an `ad` field on `ProxNetwork`; checked against
finite differences of the whole solve ([[proxdm]] §5); and `ProxDM` as a `DiffusionFactor`
inversion, with two parameter trees ([[proxdm]] §6); and a second-order sampler, the
trapezoidal scheme, with one prox per step ([[proxdm]] §2); and the experiment of
backpropagating at a higher noise level than inference used: no benefit on the circle task,
little cost up to $\sigma(t_b) \approx 6\sigma(t_f)$ ([[proxdm]] §8). Open: the same experiment
where conditioning matters, with a learned prox and many free coordinates
([[Backpropagation through Implicit Inference]] §9).

### I8. Composition — **done**

Products of relations are built (`ProductRelation`, [[product]]): on the same variables, and on
different but overlapping ones (`relation => coords`) with a separator of weight $-1$, so that
$p_A(x,y)p_B(y,z)/p_S(y)$ does not count the shared marginal twice (Laplace weights 0.80 / 0.20
with the separator, 0.94 / 0.06 without, for data weights 0.8 / 0.2). All query functions and
the adjoint work on them. Sampling a product has the corrections T9 asks for: Langevin or MALA
steps at every noise level fix the shape of the modes, and sequential Monte Carlo reweighting
fixes the mass between them (circle ∩ line: upper share 0.744 ± 0.019 against 0.751 exact, 0.655
for summed scores alone; [[product]] §5). Open: deriving the separator from a factor instead
of training it, and SMC in high dimensions.

### I9. Coverage diagnostics — **done**

`coverage` flags answers outside the data for T5 (and clamps off the relation, T4): the
distance to the nearest data point, Tweedie's reconstruction error at a small noise level (any
predictor) and $-\log p_t$ (energy networks, mixtures, KDEs), each a conformal p-value against
held-out points. On the circle it flags about 5% of fresh data at $\alpha = 0.05$ and the
answer to $x = 1.3$ at $p = 0.006$ ([[coverage]]). Open: scores of the free block given the
clamp, which would separate T4 from T5, and embeddings for high dimensions.

### I10. Real data and higher dimensions — **done on MNIST: clean digits and 96.8% from one energy network**

[[MNIST in Both Directions]] §6–9. One energy network on (image, label), 794 coordinates, trained on
a GPU. Image → label by root finding: 96.1% on 9000 test images for the first MLP energy, equal to
a diffusion classifier with the same network (95.8%) and below a softmax MLP (97.7%). With noise
blocks (I12) the same network also answers label → image, but an MLP energy's samples are mostly
noise, and a convolutional energy alone (§8) gives digit-like but noisy samples.

§9 found the cause at high noise, not at low noise as first thought: the samples' speckles are laid
down above t ≈ 0.3, where a small ε error is a large image error. Two changes remove them:
- a **reference energy**, the exact energy of a diagonal Gaussian fitted to the data, to which the
  network adds the rest;
- **training on the condition queries use**: `block_times(…; clean)` holds one block at $t_{\min}$ in
  a share of the columns (uniform times give a clean label in about 1% of them), and that block
  leaves the loss.

With both, the convolutional energy's samples are clean digits (76.5% recognised with guidance 1),
and the same network classifies 96.8% of the tuning images (one start, 291 ms per query), above the
MLP energies (95.0%).

Open:
- the reference energy with an MLP: its samples collapse to one spurious image;
- the reference as a package option (it is in the example's network, its derivatives by AD);
- the full test set, and how far a larger convolutional energy goes.

### I11. Nonlinear factors around the learner — **done**

Latent states with nonlinear dynamics (the particle tutorial's noisy positions, SLAM) need
nonlinear Gaussian factors and Gauss–Newton in the graph, which Lenticulum does not have yet:
a factor from a **hand-written** function $y = f(x)$, with Gaussian messages by linearisation or
the unscented transform (RxInfer's delta nodes, [[RxInfer as a Backend]] §2). The learned
factors are nonlinear too, but their nonlinearity comes from data.

Built: `NonlinearFactor(f, nin => nout; method = Linearization() | Unscented())` in `Lenticulum`,
with backward messages linearised at the cavity and `invert` relinearising at the posterior
(Gauss–Newton, the iterated EKF) ([[nonlinear]]). Missing: parameters inside $f$, several input
channels, and a SLAM or particle example that uses it.

### I12. Both directions of one relation — **done (found on MNIST)**

Training by denoising with one noise level shared by all coordinates learns a joint score in
which a small block (the label) can be ignored by a large one (the image):
$\nabla_x\log p(x, y) = \nabla_x\log p(x) + \nabla_x\log p(y\mid x)$, and the second term is small.
Then "label → image" failed while "image → label" worked. Built:
- **a noise level per block of coordinates** (UniDiffuser): `NoisePredictor(…; blocks)`, time as one
  value per block in the predictor's input, $\alpha$ and $\sigma$ per coordinate, exact for the
  closed-form mixture and energy networks, also on a device ([[predictor]] §6);
- **training on independent block times** (`block_times`, with `clean` for the held condition)
  with **loss weights per coordinate** (`denoising_gradient(…; weights)`, also per entry);
- **block levels in the field** (`ImplicitDiffusion(…; levels)`, `conditional_levels`): clamped
  blocks held clean, so the field is the conditional score ([[implicit]] §9);
- **held clamps and guidance in the sampler** (`clamp_level`, `guidance`, [[sampler]] §5);
- **the test**: a judge model scores label → image samples (MNIST §7).

Result: on MNIST, 85.5% of label-clamped samples are recognised as their label, against 10.5%
for the same network sampled the old way. Guidance lowered it (74%). Remaining: image quality
(I10), and a cost of one to two points in classification for the block-trained network. Both
were fixed on a convolutional energy by a reference energy and by training with one block held
clean in a share of the columns (`block_times(…; clean)`): clean digits, 96.8% (MNIST §9).

### I13. Answers of a single root — **done**

On MNIST every classification query had one stable root, so the Laplace mixture had one
component and its class probabilities were 0 or 1. Built:
- **probabilities from a root's position**: `categorical_answer(…; temperature, cov_scale)`, the
  posterior of the code under $\mathcal N(\mu, \kappa\Sigma + \tau^2 I)$, with τ and κ fitted on held-out
  answers (`calibrate_temperature`). On MNIST: calibration error 0.009, NLL 0.244 against 1.459 for
  the hard reading ([[MNIST in Both Directions]] §7);
- **unassigned mass as a coverage flag**: `coverage(…; unassigned)` sets `flagged`;
- **tolerances that follow the arithmetic**: `default_tolerance(m, z, ps)`;
- **a trust-region solver** (`method = :trust_region`, the default for relations with an energy):
  a step that must lower the query's energy, exact or by Steihaug CG. On the circle 160 of 160
  starts converge against 122; on MNIST 19 of 20 modes with 784 free pixels against none, in a
  third of the time ([[implicit]] §11);
- **continuation in the noise level** (`implicit_infer(…; continuation)`): with Newton-or-descent
  3–4 of 10 MNIST modes, at five to nine times the cost; superseded by the trust region.

  `implicit_infer_many` has it too, query by query identical to the single-query solver;
  on easy batched queries (MNIST classification) it costs 2–20% more than Newton-or-descent.

- **a cost model** (`pass_costs`, `root_cost`, `classifier_cost`, `break_even_classes`): network
  passes weighted by their measured cost per column. On MNIST it predicts the ratio of root
  finding to the diffusion classifier within 10% (absolute times 1.3–1.6× low). With one-hot
  labels root finding is never cheaper; with a $d$-dimensional label code it is above about
  $10d + 4$ classes ([[implicit]] §12).

## Summary

| | problem | kind |
|---|---|---|
| T1 | finding all answers | open; multi-start search implemented |
| T2 | smoothing bias vs conditioning | intrinsic |
| T3 | branch points | intrinsic |
| T4 | queries off the relation | open (semantics) |
| T5 | coverage, out of distribution | open; per-answer flag implemented (`coverage`) |
| T6 | non-conservative ε-networks | intrinsic; resolved by energy networks at a cost |
| T7 | training shapes only visited branches | intrinsic to the bilevel objective |
| T8 | hyperparameters | partly open |
| T9 | composition at $t > 0$ | known obstruction; mild for root finding; corrected for sampling by SMC where log-densities exist |
| T10 | posterior messages | open design problem |
| T11 | convergence, existence, identifiability | open |
| I1 | mixture beliefs, conditional sampling | done: `implicit_mixture` with energy-based weights; `implicit_sample` (hard clamps) |
| I2 | Laplace (Gaussian) answers | done (`laplace_belief`) |
| I3 | Gaussian messages from `lib/` | done (`message = :gaussian`) |
| I4 | restarts after unstable solves | done |
| I5 | batched nodes, Newton–Krylov | done: free-coordinate blocks, batched nodes, `KrylovSolve` |
| I6 | energy networks on Enzyme and Reactant | done: Zygote, Mooncake, Reactant (nested Enzyme, compiled), CUDA through `device`; Enzyme on plain CPU code blocked upstream |
| I7 | ProxDM adjoint, factor, sampler | done: adjoint, factor inversion (`ProxDM`), trapezoidal sampler; higher-noise backprop tried (no benefit on the circle) |
| I8 | composition | done: products on shared and overlapping variables (separators), MCMC correctors and SMC reweighting for sampling |
| I9 | coverage diagnostics | done: `coverage` (distance, reconstruction, energy as conformal p-values) |
| I10 | real data, higher dimensions | done on MNIST: a conv energy with a Gaussian reference energy and clean-block training gives clean digits (76.5% recognised) and classifies 96.8% with the same network; open: the reference with an MLP, as a package option |
| I11 | nonlinear factors | done: `NonlinearFactor` (EKF, UKF, iterated) |
| I12 | both directions of one relation | done: noise blocks, block times, loss weights, held clamps, guidance (MNIST: 85.5% against 10.5%) |
| I13 | answers of a single root | done: calibrated code posterior (ECE 0.009), unassigned flag, precision-aware tolerances, trust-region solver (MNIST modes 19/20 against 0/20), cost model (ratio within 10%) |

Related: [[Implicit Diffusion Learners]], [[Inference Signatures]], [[Backpropagation through Implicit Inference]],
[[The Diffusion Factor]], [[The Implicit Diffusion Factor as a Statistical Game]], [[energy_network]],
[[proxdm]], [[backends]], [[Belief Algebra]], [[Open Problems in Algebraic Implicit Learning]]
