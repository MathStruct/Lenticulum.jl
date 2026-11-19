# Calibration of implicit relations: how well the stable roots of a trained noise predictor's
# field recover a known relation. Recorded in `examples/calibration.md`.
#
# Run: julia -t auto --project=bench lib/VariationalDiffusion.jl/examples/calibration.jl        (on all threads; the queries take most of the time)
#
# Three relations in R², each learned by the 3-layer MLP of `circle_mlp.jl` (denoising score
# matching, 6000 Adam steps, data noise 0.02):
#   - the unit circle;
#   - the lemniscate of Gerono, y² = x² − x⁴ (two branches meeting at the origin);
#   - two concentric circles of radius 1 and 2 with weights 0.9 and 0.1 (a minority branch).
# Queries clamp x on a grid and ask for every stable y (`implicit_roots`, 33 starts). Against the
# relation's branches y(x), with tolerance τ = 0.1 (five data-noise widths):
#   - branch recall: the fraction of true branches matched by a root within τ;
#   - spurious roots: the fraction of roots farther than τ from every branch at their x;
#   - off-domain queries: queries with no branch at their x that still return a root;
#   - precision distance: mean |y_root − nearest branch| over the roots matched to a branch;
#   - for circles, the radial bias of matched roots, mean (|z| − r).
# Each model is queried with three choices of the field's noise levels (no retraining): low
# (t ≤ 0.01), the default (`field_nodes`, t ≤ 0.05) and high (t ≤ 0.2).
# Smoke: LENTICULUM_SMOKE=true uses 1 training step, 2 queries/relation, 2 starts and reduced reference data; all filters and levels.
include(joinpath(@__DIR__, "calibration_setup.jl"))

const LEVELS = (("low, t ≤ 0.01", range(0.001, 0.01; length = SMOKE ? 2 : 8), ()),
                ("default, t ≤ 0.05", range(0.002, 0.05; length = SMOKE ? 2 : 8), ()),
                ("high, t ≤ 0.2", range(0.01, 0.2; length = SMOKE ? 2 : 8), ()),
                ("low, annealed", range(0.001, 0.01; length = SMOKE ? 2 : 8), (16.0, 4.0)))   # t ≤ 0.16, then 0.04, then 0.01
const FILTERS = (("all roots", r -> true), ("round trip", r -> r.trip), ("ridge", r -> r.ridge),
                 ("coverage", r -> r.cov), ("both", r -> r.trip && r.cov), ("ridge+cov", r -> r.ridge && r.cov))

for (k, rel) in enumerate(RELATIONS)
    t0 = time()
    ps, st, loss = train(Xoshiro(100 + k), rel.sample)
    ps = f64(ps)
    @printf "\n%s: trained in %.0f s, final denoising loss %.4f\n" rel.name time() - t0 loss
    pred = NoisePredictor(mlp, sched; input, ad = AutoZygote())
    rng = Xoshiro(200 + k)                                  # coverage: held-out data, not the training draws
    data = rel.sample(rng, SMOKE ? 4 : 2000) .+ 0.02 .* randn(rng, 2, SMOKE ? 4 : 2000)
    cal = rel.sample(rng, SMOKE ? 2 : 300) .+ 0.02 .* randn(rng, 2, SMOKE ? 2 : 300)
    for (label, levels, continuation) in LEVELS
        t1 = time()
        m = ImplicitDiffusion(pred, field_nodes(Xoshiro(1), 2; samples = SMOKE ? 2 : 8, levels))
        ref = coverage_reference(m, data, ps, st; t = 0.01, calibration = cal)
        grid = query_grid(m, ps, st, rel, ref; spread = rel.name == "two circles (0.9 / 0.1)" ? 2.5 : 1.5, continuation)
        @printf "  %s (%.0f s)\n" label time() - t1
        for (fname, keep) in FILTERS
            r = calibrate(rel, grid, keep)
            @printf "    %-11s recall %.3f  spurious %.3f (%d roots)  off-domain %d/%d  dist %.4f  radial %+.4f%s\n" fname r.recall r.spurious r.roots r.offdomain r.offqueries r.dist r.radial (isnan(r.minority) ? "" : @sprintf("  minority recall %.3f", r.minority))
        end
    end
end
