# Noise-free nodes against nodes with noise on the calibration benchmark (backlog P10). Recorded in
# `examples/nodes_calibration.md`.
#
# Run: julia -t auto --project=bench lib/VariationalDiffusion.jl/examples/nodes_calibration.jl
#
# The relations, models, queries and metrics are those of `calibration.jl` (`calibration_setup.jl`):
# the unit circle, the lemniscate of Gerono and two circles (0.9 / 0.1), each learned by the same
# MLP with the same seeds. Each model is queried with two kinds of nodes over the same noise levels:
#   - `field_nodes(Xoshiro(1), 2; samples = 8, levels)`: 8 antithetic pairs per level, 16 network
#     evaluations per level (as in `calibration.jl`; `field_nodes`' own default is `samples = 4`);
#   - `noisefree_nodes(2; levels)`: ε = 0, one evaluation per level.
# and three level ranges: low (t ≤ 0.01), default (t ≤ 0.05) and mid (t ≤ 0.1). At the default levels
# a second draw of the field nodes (`Xoshiro(2)`) shows how much the field's own randomness moves the
# numbers. Reported for all roots and for the roots the ridge residual keeps (the best filter of F3),
# with the wall time of the query grid.
# Smoke: LENTICULUM_SMOKE=true uses 1 training step, 2 queries per relation, 2 starts and 2 levels.
include(joinpath(@__DIR__, "calibration_setup.jl"))

const NODE_LEVELS = (("low, t ≤ 0.01", range(0.001, 0.01; length = SMOKE ? 2 : 8)),
                     ("default, t ≤ 0.05", range(0.002, 0.05; length = SMOKE ? 2 : 8)),
                     ("mid, t ≤ 0.1", range(0.004, 0.1; length = SMOKE ? 2 : 8)))
const NODE_KINDS = (("noise-free", levels -> noisefree_nodes(2; levels)),
                    ("with noise", levels -> field_nodes(Xoshiro(1), 2; samples = SMOKE ? 2 : 8, levels)),
                    ("with noise, seed 2", levels -> field_nodes(Xoshiro(2), 2; samples = SMOKE ? 2 : 8, levels)))
const NODE_FILTERS = (("all roots", r -> true), ("ridge", r -> r.ridge))

for (k, rel) in enumerate(RELATIONS)
    t0 = time()
    ps, st, loss = train(Xoshiro(100 + k), rel.sample)
    ps = f64(ps)
    @printf "\n%s: trained in %.0f s, final denoising loss %.4f\n" rel.name time() - t0 loss
    pred = NoisePredictor(mlp, sched; input, ad = AutoZygote())
    for (llabel, levels) in NODE_LEVELS, (nlabel, make) in NODE_KINDS
        nlabel == "with noise, seed 2" && !startswith(llabel, "default") && continue
        m = ImplicitDiffusion(pred, make(levels))
        t1 = time()
        grid = query_grid(m, ps, st, rel, nothing; remedies = false, spread = rel.name == "two circles (0.9 / 0.1)" ? 2.5 : 1.5)
        @printf "  %s, %s (%d nodes, %.0f s)\n" llabel nlabel length(m.nodes.t) time() - t1
        for (fname, keep) in NODE_FILTERS
            r = calibrate(rel, grid, keep)
            @printf "    %-9s recall %.3f  spurious %.3f (%d roots)  off-domain %d/%d  dist %.4f  radial %+.4f%s\n" fname r.recall r.spurious r.roots r.offdomain r.offqueries r.dist r.radial (isnan(r.minority) ? "" : @sprintf("  minority recall %.3f", r.minority))
        end
    end
end
