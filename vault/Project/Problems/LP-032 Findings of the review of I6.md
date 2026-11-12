#open-problem

> **Status: resolved (2026-11-12).** One defect, reproduced by a test that failed before the fix and
> passes after it: a batched plate of cluster nodes read a categorical input with reordered labels
> on any replica but the first, which the per-replica path refuses. It predates I6 (the shape that
> decides batching compared the log-probabilities, not the labels). Three other probes, and a
> second review by GLM-5.3 with five probes, found the batched path equal to the per-replica one.
> Related: [[Feature Backlog]] (I6), [[L-SPEC-009 Cluster Nodes]], [[LP-031 Findings of the review of the query report]].

> Sources: a review of the uncommitted I6 change by Mistral (`vibe --agent review`), run with Julia
> in the helper worktree `Lenticulum.jl-mistral` (2026-11-12). It stopped at its turn limit three
> times without writing an answer (`../Lenticulum.jl-Review-2026-11-12/I6-from-mistral*.md`); its
> probe scripts, `scratch/i6/p1_multi_discrete.jl`, `p3_labels.jl`, `p4_fallback.jl` and
> `p5_graph.jl` in that worktree, were run by the maintainer's session and are summarised here
> (logs in `~/.cache/lenticulum/i6/probe_*.log`).
> code: `lib/Mycelium.jl/src/cluster_batch.jl` (`_input_shape`); test: `lib/Lenticulum.jl/test/regime_plate.jl`,
> "a mislabeled categorical input on a later replica is refused".

**Problem.** Tied cluster nodes over a plate are batched when their inputs have the same shape;
the batched methods must then give what the per-replica methods give, refusals included. I6
extended the batching to factors whose coefficients and noise depend on their outcome.

| # | finding | right? | what was done | test |
|---|---|---|---|---|
| F1 | `_input_shape` recorded a categorical input as `(:cat, logp)`. Replica 1's joint is built by `_cluster_joint`, which refuses labels other than `1:K` (`_check_positional`); the other replicas reuse it, so a replica > 1 with labels `(2, 1)` was batched and read positionally, while `factor_message` and `factor_belief` on that replica throw | right: reproduced (`p3_labels.jl`; the regression test failed 3 of 3 before the fix) | the labels are part of the shape, so such replicas fall back to the per-replica path, which refuses them | the testset named above |
| P1 | a plate of clusters holding a regime factor, a switching factor with a fixed form and a point channel, and a table over two discrete variables (several assignments per outcome group): messages, beliefs, energies, Zygote and ForwardDiff gradients, EM statistics | no defect: batched equals per-replica to $2 \cdot 10^{-14}$ | — | `p1_multi_discrete.jl` (scratch) |
| P4 | inputs of different shape (a point against a Gaussian, different categorical inputs, different observed switch) are not batched and fall back exactly; identical shapes are batched and equal | no defect | — | `p4_fallback.jl` (scratch) |
| P5 | P1's model as a graph, plated against unplated: free energy, gradient, EM statistics batched and not | no defect: differences $\le 2 \cdot 10^{-14}$ | — | `p5_graph.jl` (scratch) |

A second review, by GLM-5.3 (`vibe --model glm-5-3`, same worktree, after the F1 fix, without
access to Mistral's probes), found no defect. It wrote its answer probe by probe
(`../Lenticulum.jl-Review-2026-11-12/I6-from-glm.md`) and stopped at its turn limit before its
summary; its last probe was run by the maintainer's session. Its scripts are in `scratch/i6-glm/`.

| # | probe | result |
|---|---|---|
| G1 | a cluster row with two varying factors on different switches (one with learned noise, one with fixed), Gaussian priors outside with per-replica means: messages to three channels, beliefs, energies, Zygote and ForwardDiff gradients, `moment_statistics` batched and not, `em_step` | equal to $10^{-9}$ |
| G2 | flat directions under per-outcome forms, an observed discrete state equal across replicas, a categorical input with a $-\infty$ entry | equal; batched in all three |
| G2b | flat directions where the point matters ($m = 2$, $A_x$ of rank 1, four replicas): energies, messages, pooled statistics | equal |
| G3 | $K = 1$ (a varying form with one outcome), positional labels on every replica, two varying factors on the same switch (classes keyed by $(k, k)$) | equal (`probe3_corners.jl`, run by this session) |

**Leaning / resolution.** F1 fixed with its regression test. The probes P1, P4 and P5 were not copied
into the test suite; P1's mix of factors is a case `regime_plate.jl` does not cover. The review did
not report on Zygote through `belief_free_energies` beyond P1, nor on device arrays (none are used
by the cluster code).
