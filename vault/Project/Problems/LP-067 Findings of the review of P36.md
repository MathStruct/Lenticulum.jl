#open-problem

> **Status: resolved (2026-11-19).** The review of P36 (tests that make each package's own suite reach
> its own files) found three checks missing from the new `OverlayFactor` test, each shown by a mutation
> of `overlay.jl` that left the suite green; a new testset closes all three and fails under each mutation.
> Found by the session, outside the review: the main checkout's untracked Manifests predated the newest
> package extensions, so Mycelium's local suite never loaded its ForwardDiff extensions.
> Related: [[Feature Backlog]] (P36), [[Backlog Done 2026-11]], [[LP-065 Findings of the review of P15]].

> Sources: the tests written by Codex (GPT, one run in `../Lenticulum.jl-codex`, branch `codex/p36`), reviewed
> by Mistral Large (`vibe`, review agent, \$0, mutation probes); the extra testset and the Manifest finding
> by the maintainer's session (Claude). Briefs, diff, answer, probes: `../Lenticulum.jl-Review-2026-11-19/p36/`.
> code: `lib/Mycelium.jl/test/p36_overlay_ad.jl`, `lib/Mycelium.jl/src/overlay.jl`.

**Problem.** The requirement given to the reviewer: each new test checks its file against a closed form
or an independent reference, and would fail if the code were broken. The reviewer tested that by
mutation (a copy of the function under test, broken, then the new test file included).

| # | finding | right? | what was done |
|---|---|---|---|
| 1 | joint outcomes of two or more switch channels: their order (column-major, as the docstring says) is never checked; a row-major `_overlay_columns` passes, because every overlay in the test has one switch | right (its probe `revcols`) | new testset: a member with two switches (2 × 3) and an overlay of it with a switched member, every outcome against the model written by hand; under `revcols` it fails (BoundsError) |
| 2 | the members' log-weights are summed, but only one member had nonzero log-weights, so keeping the first member's alone passes | right (`logwfirst`) | the same testset: both members carry weights; 14 checks fail under `logwfirst` |
| 3 | the message to a continuous channel is never asked for; returning `TrivialBelief()` there passes | right (`conttriv`) | the same testset: the message to `y` under soft switch messages, a 12-component mixture, compared as a set of (mean, variance, weight) against enumeration (product of the selected Gaussians, weight times the product's normaliser); fails under `conttriv` |

Every other mutation the reviewer tried was caught (7 on `exposure.jl`, 7 on the overlay, 3 on the cluster
rule, 2 on the flat extension, 4 on `sample_gradient.jl`, 7 on `statistical_game.jl`, 6 on the interop
extension), and it recomputed each closed-form expected value by hand without finding one wrong. Its probes:
`mistral/probe_*.jl`; the session's runs of the three mutations against the new testset: `mut_probe_*.jl`.

Seen on the way: a mixture message from a cluster lists its components in the cluster's own outcome order
(the last switch fastest), not in the overlay's joint-outcome order. Nothing documents either order for a
message, so the test compares the mixture as a set.

**Stale Manifests (session).** Rerun locally, `p36_overlay_ad.jl` threw at `Base.get_extension(Mycelium,
:MyceliumForwardDiffExt)`: `nothing`, even with ForwardDiff and Zygote loaded. The main checkout's
`lib/Mycelium.jl/Manifest.toml` (untracked, written 2026-11-10) listed only the extensions of that day, and
Julia loads a package's extensions from its Manifest entry; the ForwardDiff extensions came on 2026-11-17.
After `Pkg.resolve()` in each package, six of the eight Manifests gained extension entries (AlgebraicFactors,
ImplicitLayers, Lenticulum, Mycelium, MyceliumLearning, VariationalDiffusion). So until now the local suites
ran without those extensions (Codex's fresh worktree and CI were not affected), and the 0 % of Mycelium's
ForwardDiff extensions in the first coverage report came from this as well as from missing tests. After
adding a weak dependency or an extension, run `Pkg.resolve()` in the package (the old Manifests are in
`~/.cache/lenticulum/p36/gates/*-Manifest.before.toml`).

Not checked: the new CI job `reactant` on GitHub; the device tests.
