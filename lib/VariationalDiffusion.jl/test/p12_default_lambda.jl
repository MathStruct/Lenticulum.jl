# P12: an ImplicitProx left without λ gets density_lambda(schedule, nodes) for Gaussian and
# mixture messages (calibrated covariances) and 1 for point messages (answers unchanged), chosen
# by the DiffusionFactor from its predictor's schedule. An explicit λ is kept.
@testset "P12: density_lambda is the default λ of non-point messages" begin
    nodes = field_nodes(Xoshiro(1), 2; samples = 8)
    λd = density_lambda(SCHED, nodes)
    @test !(0.5 <= λd <= 2)                       # the old default 1 was off by more than 2
    @test ImplicitProx(nodes).λ === nothing
    @test ImplicitProx(nodes; λ = 3).λ === 3.0

    mk(; kw...) = DiffusionFactor((x = 1, y = 1), RING; prox = ImplicitProx(nodes; kw...))
    @test mk(message = :point).prox.λ === 1.0
    @test mk(message = :gaussian).prox.λ === λd
    @test mk(message = :mixture).prox.λ === λd
    @test mk(message = :gaussian, λ = 1.0).prox.λ === 1.0   # explicit kept, even far from λd
    @test mk(message = :point, λ = λd).prox.λ === λd

    inputs = (x = DiracBelief([0.6]),)
    q = Polarity((x = Observed(), y = Unobserved()), (x = Inf, y = 0.5))
    # the default gives exactly what an explicit density_lambda gives, without a warning
    for msg in (:gaussian, :mixture)
        fdef, fexp = mk(message = msg), mk(message = msg, λ = λd)
        a = @test_logs Mycelium.factor_message(fdef, :y, q, inputs, DiracBelief([0.5]), RPS, RST)
        b = Mycelium.factor_message(fexp, :y, q, inputs, DiracBelief([0.5]), RPS, RST)
        if msg === :gaussian
            @test a[1].Λ == b[1].Λ && a[1].η == b[1].η
        else
            @test length(a[1].components) == length(b[1].components)
            @test all(c.Λ == d.Λ && c.η == d.η for (c, d) in zip(a[1].components, b[1].components))
        end
        @test first(Mycelium.local_free_energy(fdef, inputs, RPS, RST)) ==
              first(Mycelium.local_free_energy(fexp, inputs, RPS, RST))
    end
    # an explicit λ far from density_lambda still warns
    fbad = mk(message = :gaussian, λ = 1.0)
    @test_logs (:warn, r"density_lambda") match_mode = :any Mycelium.factor_message(
        fbad, :y, q, inputs, DiracBelief([0.5]), RPS, RST)
    # point messages: the default is λ = 1, the answer of before
    fp, f1 = mk(), mk(λ = 1.0)
    @test first(Mycelium.factor_message(fp, :y, q, inputs, DiracBelief([0.5]), RPS, RST)).value ==
          first(Mycelium.factor_message(f1, :y, q, inputs, DiracBelief([0.5]), RPS, RST)).value

    # the Gaussian message at the default is calibrated. Oracle: data N(0, V₀) and a perfectly
    # trained ε (AnalyticEps), so the field is linear and the prior term's curvature in y is
    # λ Σ_k w_k σ_k²/(α_k² V₀ + σ_k²) exactly; at λ = density_lambda that is ≈ 1/V₀ (data units),
    # at the old default λ = 1 it was smaller by the factor λd.
    gn = field_nodes(Xoshiro(2), 2; samples = 16)
    λg = density_lambda(SCHED, gn)
    fg = DiffusionFactor((x = 1, y = 1), PRED; prox = ImplicitProx(gn; message = :gaussian, tol = 1e-12))
    @test fg.prox.λ === λg
    q0 = Polarity((x = Observed(), y = Unobserved()), (x = Inf, y = 0.0))
    post = first(LenticulumCore.invert(first(LenticulumCore.assemble(fg, q0, PS, ST)), DiracBelief([0.0]),
                                       (x = DiracBelief([0.6]),), PS, ST))
    curv = λg * sum(gn.w[k] * sigma(SCHED, t)^2 / marginal_variance(SCHED, t, V₀) for (k, t) in enumerate(gn.t))
    @test post.Λ[1, 1] ≈ curv rtol = 1e-8
    @test post.Λ[1, 1] ≈ 1 / V₀ rtol = 0.05

    # Float32 nodes give a Float32 λ
    n32 = FieldNodes(Float32.(nodes.t), Float32.(nodes.ε), Float32.(nodes.w))
    @test DiffusionFactor((x = 1, y = 1), RING; prox = ImplicitProx(n32; message = :gaussian)).prox.λ isa Float32

    # the resolved λ is inferred concretely (LP-063 F1: it was Union{Nothing,T})
    for msg in (:point, :gaussian, :mixture), λ in (nothing, 2.0), nd in (nodes, n32)
        @test (@inferred VariationalDiffusion._resolve_lambda(ImplicitProx(nd; λ, message = msg), SCHED)) isa eltype(nd.w)
    end

    # a DiffusionFactor built field by field, with λ left unset, resolves it where it is used
    fraw = DiffusionFactor((x = 1, y = 1), RING, ImplicitProx(nodes; message = :gaussian))
    @test fraw.prox.λ === nothing
    a = Mycelium.factor_message(fraw, :y, q, inputs, DiracBelief([0.5]), RPS, RST)
    b = Mycelium.factor_message(mk(message = :gaussian), :y, q, inputs, DiracBelief([0.5]), RPS, RST)
    @test a[1].Λ == b[1].Λ && a[1].η == b[1].η
    @test first(Mycelium.local_free_energy(fraw, inputs, RPS, RST)) ==
          first(Mycelium.local_free_energy(mk(message = :gaussian), inputs, RPS, RST))
end
