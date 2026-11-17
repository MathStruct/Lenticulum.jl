#open-problem

> **Status: resolved (2026-11-17).** The review of P27 (a Gaussian mixture times an extended
> Gaussian: the overlap rule, expectation propagation with an extended cavity, mixture inputs to the
> relation factors) found one missing check, now fixed with a regression test. A second report
> (wrong `logZ` far from the origin) came from the reviewer's own reference and did not reproduce.
> Related: [[Feature Backlog]] (P27), [[lib/Mycelium.jl/src/gaussian|gaussian]] (Mycelium) "Mixtures
> meeting extended Gaussians".

> Sources: a review of the uncommitted P27 change by Mistral (`vibe --agent review`, Mistral Large 4,
> `--max-price 3`, $0 against the budget), in the helper worktree `Lenticulum.jl-mistral` at
> `7d1014f8` with the diff applied. The brief, the answer, the probes and the maintainer's script
> (`maintainer_f1_translation.jl`) are in `../Lenticulum.jl-Review-2026-11-17/p27/`.
> code: `lib/Mycelium.jl/src/gaussian.jl`; tests: `lib/Mycelium.jl/test/overlap.jl`,
> `lib/Lenticulum.jl/test/mixture_extended.jl`.

**Problem.** The requirement given to the reviewer: `combine_with_overlap(c, d)` of a Gaussian and an
extended Gaussian returns the product and $\log\int_S c\,p_d\,d\sigma$ (up to a constant of $d$ alone
when $d$ is flat somewhere), the same in either order; `ep_project`/`ep_divide` of extended Gaussians
give a site whose product with the cavity is the projection; an indefinite Gaussian times an
extended one is the product of potentials or an `ArgumentError`; the relation factors collapse
mixture inputs; on a tree with the switching factor a leaf the marginals are exact (BP) or the
projection of the exact ones (EP).

The run kept its answer file, stopped at its turn limit (100 turns) with five probes, and recorded
four of them; the fifth (`probe5.jl`) the maintainer ran.

| # | finding | right? | what was done | test |
|---|---|---|---|---|
| 1 | Probes 1, 3, 4: `logZ` against Monte Carlo (10 cases), against an independent closed form checked by grid quadrature (80 cases: proper and improper $c$, flat and hard $d$), mixture weights $w_i Z_i$ (100 cases) | no defect | — | `test/overlap.jl` covers the same against its closed form |
| 2 | F1: `ep_divide(p, cavity)` checks that the hard *directions* agree, not the hard *values*; with the same $H$ but $H^\top\mu_p \ne H^\top\mu_c$ it returns a site whose product with the cavity has the cavity's values, not $p$'s | **right** as a missing check (`maintainer_f1_translation.jl` returns a site). Not reachable through `ep_message`, whose projection keeps the cavity's hard values. The reviewer's own example (`probe2b.jl`) had equal values and a correct site | `ep_divide` returns `nothing` there, as for different directions | "P27: expectation propagation with an extended cavity", the line after `nindef > 0` |
| 3 | Probe 5: `logZ` of a proper $c$ wrong by about $10^{-4}$ when both means are near $10^6$ (19 of 20 cases) | **not reproduced**: the reference evaluates $-\tfrac12\mu^\top\Lambda\mu$ at $10^{12}$ and its values are rounded to multiples of $2^{-10}$; translating both sides by $10^6$ changes the rule's `logZ` by at most $1.5\cdot10^{-9}$ (40 cases) | kept as a test | "P27: combine_with_overlap …", the translation by 1e6 |
| 4 | Probe 3: an improper $c$ with a flat $d$ whose summed precision is singular gives `logZ = 0` | the documented convention of the Gaussian pair (no finite value exists); not a defect | — | — |
