#open-problem

> **Status: resolved (2026-11-18).** The review of P32 (Mycelium's cluster joints, MyceliumLearning's
> EM statistics and the Gaussian overlap in the element type of their inputs) found one defect, also
> found by the maintainer's session before the review answered: the thresholds that decide flat
> directions of a cluster joint were Float64's, so a Float32 joint misread rounding as a definite
> direction. Fixed (`_flat_rtol`, `_primal_eps` in `cluster.jl`). The reviewer's third point (Float32
> EM diverging from Float64) was a degenerate test problem, not a defect.
> Related: [[Feature Backlog]] (P32, P33), [[LP-060 Findings of the review of P31]],
> [[lib/Mycelium.jl/src/cluster|cluster]] "Element type".

> Sources: implemented by Codex (GPT, `codex exec -C ../Lenticulum.jl-codex`, branch `codex/p32`, one
> run); reviewed by Mistral Large 4 (`vibe --agent review` in `../Lenticulum.jl-mistral`, \$0, stopped
> at its 100-turn limit with an answer file). Brief, probes and answer:
> `../Lenticulum.jl-Review-2026-11-18/p32/`; the session's probes and logs: `~/.cache/lenticulum/p32/`.
> code: `lib/Mycelium.jl/src/cluster.jl`, `cluster_batch.jl`, `gaussian.jl`, `messages.jl`,
> `lib/MyceliumLearning.jl/src/em.jl`; tests: `lib/Mycelium.jl/test/element_type.jl`,
> `lib/MyceliumLearning.jl/test/element_type.jl`, `lib/Lenticulum.jl/test/p31_element_type.jl`,
> `lib/Lenticulum.jl/test/cluster_flat.jl`.

**Problem.** The requirement given to the reviewer: with Float32 forms, parameters and messages a
cluster node (joint, messages, marginals, frozen beliefs, free energy), the Gaussian overlap and
mixture products, and `MomentStats`/`em_step`/`fit_em` compute in Float32; promotion decides, nothing
is converted; Float64 results bit for bit unchanged; the flat/indefinite tests (`_psd_factor`,
`_flat_factor`, `_definite`, `_check_range`) must not misclassify a Float32 joint that Float64
classifies correctly.

| # | finding | right? | what was done | test |
|---|---|---|---|---|
| M1 | `_check_range` used `sqrt(eps(Float64))` cast to Float32: a Float32 cluster with a flat direction (`u + w` in `cluster_flat.jl`'s graph) was refused as divergent; Float64 matched the closed form | right (also the session's own probe `~/.cache/lenticulum/p32/flat32-probe.jl`, case 1) | tolerance `sqrt(_primal_eps(T))`, the precision of the joint's numbers (a dual's value type) | `cluster_flat.jl`, "Flat directions in Float32 (P32 review)" |
| M2 | `_psd_factor`/`_flat_factor`/`_definite` used `_FLAT_RTOL = 1e-10`: a rank-2 Float32 regime joint kept its rounding eigenvalue (5e-7·λmax) as a direction; the message to `:k` was off by 2e-2 (0.31 with `W` scaled by 1e3), and a tilt along the null space was not refused | right (session probe, cases 2–3) | `_flat_rtol(T) = 1e-10 · (eps(T)/eps(Float64))^(5/8)` (exactly 1e-10 for Float64, ≈ 2.9e-5 for Float32); indefinite test `sqrt(eps(T))`. After the fix the Float32 errors are 3e-8–9e-7 against the closed forms | same testset (graph, `W` scaled by 1, 100, 1000, tilts) |
| M3 | Float32 `fit_em` on a clustered regime plate diverges from Float64, its free energy rises | not a defect of P32: on the reviewer's 12-point problem each regime fits its points exactly, a noise variance collapses (log-diagonal −5.5) and Float64's own history is non-monotone (−16.2 → −13.8); one `em_step` from the same start agrees (probe 6 on the fixed code: all checks pass, b to 1e-6). With 60 points the Float32 and Float64 histories agree to 7e-7 over 15 iterations | none | `p31_element_type.jl`, "P32: Float32 EM on a regime plate tracks Float64" (6 iterations, `rtol = 1e-5`) |
| M4 | probe 6 (a): a Float32 categorical prior on a cluster with an `Int` table gives a Float64 free energy | not a defect: the integer table gives Float64 messages (integers float to Float64 by design); with a Float32 table everything is Float32 (`~/.cache/lenticulum/p32/pa32.jl`) | none | — |
| M5 | probe 2: Float64 results of regime clusters, tables, a flat cluster, mixture products, `em_step` and `fit_em` (plain, accelerated, tied plate) and ForwardDiff through a joint, printed with `repr` under the old code (`cb289789`) and the new | no defect: identical (`out_old.txt`, `out_new.txt`); the session repeated the comparison on the final code with Codex's script and the flat-direction probe: identical | — | `probe2_f64_same.jl` |

The threshold scaling (power 5/8, so that Float64 keeps its 1e-10) is a choice, not derived: a Float32
eigenvalue below ≈ 2.9e-5 of the largest is now treated as flat, so a Float32 joint with condition
number above ≈ 3.5e4 loses that direction. Recorded in `cluster.md`.

Not reviewed by a second model.

The reviewer's probes 1, 3 and 5, rerun on the fixed code (logs
`~/.cache/lenticulum/p32/probe{1_f32_types,3_thresholds,5_ad}-fixed.log`), still report failures; none
is a defect:

| # | probe failure | right? |
|---|---|---|
| M6 | probe 1: the cluster's message to `:a` over an `Int` table, with no inputs, is Float64 | no: integers float to Float64 by design (as M4) |
| M7 | probe 1: the store message into `:a` is the prior (0.6, 0.4), not table × prior | no: the probe indexes `s.to_variable` by the variable's index and so reads the prior's message |
| M8 | probe 1 stops with `KeyError :x` | probe bug: `x` is interior to the cluster. The session's repair (`~/.cache/lenticulum/p32/nearsing.jl`): prior variances 1 and 1e-4 give the closed form to 5e-8 in Float32; 1 and 1e-6 are refused as divergent in Float32, the documented consequence of the Float32 flat threshold (`cluster.md`, "Thresholds") |
| M9 | probe 3: joints with δ = 1e-5, 1e-6 are flat in Float32 and definite in Float64 | by design (condition above ≈ 3.5e4 in Float32; `cluster.md`) |
| M10 | probe 3: a tilt "in the range" is refused, in Float32 and Float64 | no: the tilt lies in regime 1's row space only; regime 2's flat direction rightly refuses it |
| M11 | probe 5: a dual message to `:k` throws the flat-direction error | no: it throws in Float64 too; the probe's incoming message tilts `x` along a direction neither regime determines |
