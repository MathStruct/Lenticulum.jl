using AlgebraicFactors, LenticulumCore, Mycelium, LinearAlgebra, LuxCore, Random, Test
using ADTypes, DifferentiationInterface, ForwardDiff, Zygote

# LP-020 for the algebraic factors: `ToricFactor` and the degree-one `PolynomialFactor` hold their
# joint belief fixed (`factor_belief`, `belief_free_energy`), so the graph parameter gradient is
# the envelope gradient of the Bethe free energy. On a tree at the fixed point that is the
# gradient of −log Z, checked against central differences with inference re-run at each point.

function af_converged_free_energy(g, θ, ps, st)
    p = LenticulumCore._unflatten(θ, ps)
    _, _, st2, s = infer!(g, tree_schedule(g), p, st)
    return first(scalar_free_energy(s, g, p, st2))
end
function af_converged_gradient(g, ps, st; h = 1e-6)
    θ = LenticulumCore._flatten(ps)
    return [(af_converged_free_energy(g, θ .+ h .* (1:length(θ) .== i), ps, st) -
             af_converged_free_energy(g, θ .- h .* (1:length(θ) .== i), ps, st)) / 2h for i in eachindex(θ)]
end
function af_graph_gradient(g, ps, st; ad = nothing)
    _, _, st2, s = infer!(g, tree_schedule(g), ps, st)
    return LenticulumCore._flatten(free_energy_gradient(s, g, ps, st2; ad))
end
# The fixed-message derivative the graph gradient used before these hooks, for contrast.
function af_message_gradient(g, ps, st)
    _, _, st2, s = infer!(g, tree_schedule(g), ps, st)
    parts = map(Tuple(g.factors)) do f
        p = ps[f.name]
        LenticulumCore.islearnable(f.factor) || return LenticulumCore._flatten(p)
        msgs = messages_into(s, g, f.id)
        LenticulumCore._flatten(LenticulumCore._parameter_gradient(nothing,
            MyceliumLearning._local_scalar, p, f.factor, msgs, st2[f.name]; ε = 1e-6))
    end
    return reduce(vcat, parts)
end
function af_check_frozen(g, ps, st; atol = 1e-12)
    _, _, st2, s = infer!(g, tree_schedule(g), ps, st)
    n = 0
    for f in g.factors
        msgs, p = messages_into(s, g, f.id), ps[f.name]
        b = factor_belief(f.factor, msgs, p, st2[f.name])
        b === nothing && continue
        F = first(local_free_energy(f.factor, msgs, p, st2[f.name]))
        Fb = first(belief_free_energy(f.factor, b, p, st2[f.name]))
        if F isa GradedEnergy
            @test collect(values(Fb.parts)) ≈ collect(values(F.parts)) atol = atol
        else
            @test Fb ≈ F atol = atol
        end
        n += 1
    end
    return n
end

@testset "LP-020: algebraic factors hold their joint belief fixed" begin
    @testset "toric factors on a tree, categorical messages" begin
        # a--ab--b--bcd--d--de--e, with c on bcd; categorical priors on a and e.
        design(sz) = independence(Tuple(Symbol(:v, i) for i in eachindex(sz)), sz)
        sizes = (2, 3, 2, 2, 3); names = (:a, :b, :c, :d, :e)
        subsets = ((1, 2), (2, 3, 4), (4, 5)); fnames = (:ab, :bcd, :de)
        pa, pe = [0.25, 0.75], [0.2, 0.5, 0.3]
        for normalised in (false, true), clamp in (false, true)
            b = GraphBuilder()
            for name in names; variable!(b, name); end
            for (fname, subset) in zip(fnames, subsets)
                ns = Tuple(names[i] for i in subset); sz = Tuple(sizes[i] for i in subset)
                # Interaction rows on top of the independence design, so η is not a pure gauge.
                A = vcat(design(sz), [k % 2 for k in 1:prod(sz)]', [k % 3 == 0 for k in 1:prod(sz)]')
                factor!(b, fname, ToricFactor(ns, sz, A; normalised))
                for name in ns; connect!(b, fname, name, name; direction = Bidirectional()); end
            end
            factor!(b, :pa, PriorFactor(:a, CategoricalBelief(pa), x -> -log(pa[x])))
            factor!(b, :pe, PriorFactor(:e, CategoricalBelief(pe), x -> -log(pe[x])))
            connect!(b, :pa, :a, :a; direction = Bidirectional())
            connect!(b, :pe, :e, :e; direction = Bidirectional())
            if clamp
                factor!(b, :obs, DataFactor(:c, 2))
                connect!(b, :obs, :c, :c; direction = Emitting())
            end
            g = build(b)
            ps0 = LuxCore.initialparameters(Xoshiro(0), g)
            ps = merge(ps0, NamedTuple{fnames}(ntuple(
                k -> (η = 0.4 .* randn(Xoshiro(k), length(ps0[fnames[k]].η)),), 3)))
            st = LuxCore.initialstates(Xoshiro(0), g)
            @test af_check_frozen(g, ps, st) == 3
            reference = af_converged_gradient(g, ps, st)
            for ad in (nothing, AutoForwardDiff(), AutoZygote())
                @test af_graph_gradient(g, ps, st; ad) ≈ reference atol = 1e-6
            end
            # The messages into every factor are categorical, not points, so holding them fixed
            # is a different, wrong derivative.
            @test norm(af_message_gradient(g, ps, st) - reference) > 1e-2
        end
    end

    @testset "degree-one polynomial factor, Gaussian prior and observed output" begin
        # y | x ~ N((A x + b)/2, Qr/4) as in game_exact.jl; x ~ N(0.2, 0.7); y = 0.8 observed.
        A, bias, μ, V = 1.3, 0.4, 0.2, 0.7
        for basis in (:monomial, :chebyshev), λ in (0.0, 0.2)
            f = PolynomialFactor((x = 1, y = 1), ExponentTable(2, 1; basis), 1;
                Σ = [0.03 0.01; 0.01 0.05], λ)
            b = GraphBuilder()
            variable!(b, :x, 1); variable!(b, :y, 1)
            factor!(b, :px, PriorFactor(:x, Gaussian(μ, V), x -> (only(x) - μ)^2 / (2V) + log(2π * V) / 2))
            factor!(b, :poly, f)
            factor!(b, :data, DataFactor(:y, [0.8]))
            connect!(b, :px, :x, :x; direction = Bidirectional())
            connect!(b, :poly, :x, :x; direction = Bidirectional())
            connect!(b, :poly, :y, :y; direction = Absorbing())
            connect!(b, :data, :y, :y; direction = Emitting())
            g = build(b)
            ps = merge(LuxCore.initialparameters(Xoshiro(0), g), (poly = (Θ = [-bias -A 2.0],),))
            st = LuxCore.initialstates(Xoshiro(0), g)
            @test af_check_frozen(g, ps, st) == 1
            # The tree's free energy is −log p(y = 0.8), the anchor for the gradient check.
            Qr = only([-A 2.0] * f.Σ * [-A 2.0]') + λ * sum(abs2, ps.poly.Θ)
            py = (A * μ + bias) / 2, A^2 * V / 4 + Qr / 4
            @test af_converged_free_energy(g, LenticulumCore._flatten(ps), ps, st) ≈
                  (0.8 - py[1])^2 / (2py[2]) + log(2π * py[2]) / 2 atol = 1e-10
            reference = af_converged_gradient(g, ps, st)
            for ad in (nothing, AutoForwardDiff(), AutoZygote())
                @test af_graph_gradient(g, ps, st; ad) ≈ reference atol = 1e-6
            end
            @test norm(af_message_gradient(g, ps, st) - reference) > 1e-2
        end
    end
end
