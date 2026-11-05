#open-problem

> **Status: partly resolved (2026-11-04).** The concrete defects that the code confirms are fixed, each with a regression test. The architectural and theory proposals are recorded here and not started. Related: [[LP-018 Defects found in the review of the older packages]], [[LP-019 Defects found in the review of the core packages]], [[LP-022 Constraint factors are normalised in their residual, not their channels]], [[passing]], [[lens]].

> Sources: two source-level reviews that Daniel supplied (ChatGPT, of `master` at `3074717`, 2026-11-03; Gemini, undated), neither of which ran code; code: `lib/Mycelium.jl/src/passing.jl`, `messages.jl`, `graph.jl`, `cluster.jl`, `lib/LenticulumCore.jl/src/lens.jl`, `open_model.jl`, `src/switching.jl`, `lib/VariationalDiffusion.jl/src/implicit.jl`, `answers.jl`, `lib/ImplicitLayers.jl/src/deq.jl`, `neuralode.jl`; tests: listed per item

**Problem.** Two external reviews listed defects, risks and proposals. Neither reviewer executed
Julia, and both say so. Each finding was checked against the code on `dev-danielb` (newer than the
reviewed commit) before anything was changed.

## Confirmed and fixed

| # | finding (reviewer) | what the code did | fix | test |
|---|---|---|---|---|
| 1 | `likelihood_message` defaults to `invert` (ChatGPT P1) | the default applied to every lens, so a lens whose `invert` returns a posterior would count the cavity twice in belief propagation. Every lens in the repository that relied on the default returns a point or particle and does not multiply in $\pi$ (TrivialInversion lenses, DEQ, neural ODE), so no current result was affected | the default now applies only to lenses declaring `invert_is_extrinsic(lens) = true` (TrivialInversion lenses, DEQ, neural ODE); other lenses without `likelihood_message` throw | LenticulumCore `test/runtests.jl`, "likelihood_message defaults to invert only for extrinsic lenses" |
| 2 | a skipped EP update can be stored by flooding (ChatGPT P1) | confirmed, and wider than reported. `FloodingSchedule` stored `TrivialBelief()` on an edge that had no message. `ResidualSchedule` scored the skipped task at residual Inf, selected it and could not commit it, so it could spend its whole budget on one task | `_step!` reports a skipped update; flooding leaves the edge as it is, and the residual schedule never selects such a task | `test/mixture_rules.jl`, "a skipped update is not stored…" |
| 3 | a Dirac combined with a contradicting categorical returns the point (ChatGPT P2) | confirmed: `combine(DiracBelief(2), CategoricalBelief([1, 0]))` returned the point | the point's probability under the categorical is checked; zero is an `ArgumentError` | Mycelium `test/runtests.jl` |
| 4 | switching factors ignore categorical labels (ChatGPT P2) | confirmed: the outcome probabilities were read by position | labels other than `1:K` in order are refused where switching factors, cluster nodes and tables read outcomes by position (`_check_positional`) | `test/switching_constraint.jl`, "…by position (labels 1:K only)" |
| 5 | `SampleBelief` accepts no samples and all-zero weights (ChatGPT) | confirmed | both refused at construction | LenticulumCore `test/runtests.jl` |
| 6 | parameter preparation does needless work per message (ChatGPT P1) | confirmed: `_factor_parameters` built the offset tree of every parameter on each message even when nothing was frozen | fast path for a learnable, unfrozen node. Not timed | the existing suites (the frozen paths are unchanged) |
| 7 | the dense implicit pullback at a singular Jacobian (Gemini 2.3) | `Jff' \ b` without a check: at a fold or branch point this gives Inf/NaN or a `SingularException` | refused with an `ArgumentError` when the Jacobian is singular to working precision (cond₁·eps ≥ 10⁻²; host copy, skipped for AD number types); no silent Tikhonov damping, which would change the gradient | VariationalDiffusion `test/review_singular.jl` |
| 8 | `implicit_laplace` on a non-conservative field (ChatGPT 4.1, Gemini 2.2) | the docstring said the symmetric part is used, but the result was still called a Laplace covariance | the docstring now says it is a local-curvature measure then, and a warning fires when $\lVert J - J^\top\rVert > 2 \cdot 10^{-6} \lVert J\rVert$ | `test/review_singular.jl` |

## Checked and not changed

- **Missing parameter keys resolve to empty trees** (ChatGPT P2). True (`_subtree`), and data
  factors rely on it, so the lookup is unchanged. Since 2026-11-04 `validate_parameters(g, ps)`
  checks a tree against the nodes' initial parameters (missing keys, names, array sizes, frozen
  paths; unread keys warn); `test/validate_parameters.jl`. It is not called by inference itself.
- **Root deduplication with a fixed tolerance** (Gemini 2.1). `implicit_roots` uses a relative
  `unique_tol` (default $10^{-3}$), a keyword. Scaling it by the local conditioning, as proposed, is
  plausible but not measured; no failing case is known.
- **Bethe, EP and surrogate free energies should be named apart** (ChatGPT 4.3). The EP free
  energy has its own stationarity test (`test/mixture_rules.jl`). A uniform report of which
  objective a number is remains open.
- **Fallback to fixed-message gradients** (ChatGPT 4.2). That is the documented fallback for
  factors without `factor_belief` (LP-020), and it warns. Exposing the gradient semantics in the
  result is a proposal.
- **Implicit-relation calibration** (ChatGPT 4.4, Gemini §5). Stable zeros of a smoothed score
  are modes of $p_t$, not points of the data manifold; `lib/VariationalDiffusion.jl/examples/circle_mlp.jl`
  was the one manifold example. Since 2026-11-05 `lib/VariationalDiffusion.jl/examples/calibration.jl`
  measures it on the circle, the lemniscate and unbalanced rings (`calibration.md`). Every query
  outside a relation's range returns a root, the lemniscate's recall is at best 0.52, and a 10% ring
  is never found. The proposed
  metrics (distance to the manifold, spurious roots, branch recall on a lemniscate, a pitchfork,
  a redundant arm, unbalanced mixtures) and remedies (annealing in $t$, ridge residuals, normal-bundle
  parameterisation) are research items, not defects.

## Proposals, not started

- Typed, compiled execution plans grouped by factor type (both reviews); runtime symbol lookup
  in graph-wide parameter trees (ChatGPT P1). [[Parallelism and Compilation]] already records
  the latter.
- A dependency-aware priority queue for `ResidualSchedule` (each sweep now probes every task,
  $O(T^2)$ evaluations).
- Trait types instead of `Bool` traits, a promotion lattice for `combine`, `Val` channel names
  (Gemini §1): plausible; no measured cost.
- Pre-allocated message buffers (Gemini §3); batched multi-start roots; matrix-free Newton–Krylov
  as the default above a dimension threshold.
- A lazy product-of-potentials belief; incremental (iSAM2-like) inference; heteroscedastic and
  heavy-tailed likelihoods; a second Julia version and device tests in CI (ChatGPT §5–6).
