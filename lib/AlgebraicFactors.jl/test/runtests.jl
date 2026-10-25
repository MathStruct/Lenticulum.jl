using AlgebraicFactors
using LenticulumCore
using Mycelium
using LinearAlgebra
using LuxCore
using Random
using Test

# Use the common checker with independent direct-message site/posterior fixtures.
include(joinpath(@__DIR__, "../../../test/factor_conformance.jl"))

# Independent enumeration oracles: explicit integer-state loops and ordinary products.
# Moderate parameters keep these probability-domain references well conditioned.
states(sizes) = collect(Iterators.product((1:n for n in sizes)...))[:]
function oracle_weights(A, η, sizes, incoming; skip = 0)
    xs = states(sizes)
    w = [exp(sum(η[r] * A[r, k] for r in axes(A, 1))) *
         prod(i == skip ? 1.0 : incoming[i][x[i]] for i in eachindex(sizes))
         for (k, x) in enumerate(xs)]
    return xs, w
end
function oracle_message(A, η, sizes, incoming, t)
    xs, w = oracle_weights(A, η, sizes, incoming; skip = t)
    out = [sum(w[k] for (k, x) in enumerate(xs) if x[t] == j) for j in 1:sizes[t]]
    return out / sum(out)
end
function oracle_statistics(A, η, sizes, incoming; normalised = false)
    xs, w = oracle_weights(A, η, sizes, incoming)
    q = w / sum(w)
    logits = [sum(η[r] * A[r, k] for r in axes(A, 1)) for k in eachindex(xs)]
    offset = normalised ? log(sum(exp.(logits))) : 0.0
    U = sum(q[k] * (offset - logits[k]) for k in eachindex(q))
    H = -sum(p == 0 ? 0.0 : p * log(p) for p in q)
    return U - H, H
end
design(sizes) = [mod(3r + 5k + r*k, 7) - 3 for r in 1:4, k in 1:prod(sizes)]
polarity(names, t, observed) = Polarity(NamedTuple{names}(map(eachindex(names)) do i
    i == t ? Unobserved() : i in observed ? Observed() : Latent()
end |> Tuple))

@testset "AlgebraicFactors" begin
@testset "construction and design matrices" begin
    names, sizes = (:x, :y), (2, 3)
    expected = [1 0 1 0 1 0;
                0 1 0 1 0 1;
                1 1 0 0 0 0;
                0 0 1 1 0 0;
                0 0 0 0 1 1]
    @test independence(names, sizes) == expected
    @test hierarchical(names, sizes, ((:x,), (:y,))) == expected
    @test hierarchical(names, sizes, ((1,), (2,))) == expected
    @test independence((:a, :b, :c), (2, 3, 2)) ==
          hierarchical((:a, :b, :c), (2, 3, 2), ((:a,), (:b,), (:c,)))
    @test hierarchical(names, sizes, ((:x, :y),)) == Matrix{Int}(I, 6, 6)
    # Reversing a generator changes its row order, never the joint column order.
    @test hierarchical(names, sizes, ((:y, :x),)) ==
          Matrix{Int}(I, 6, 6)[:, [1, 4, 2, 5, 3, 6]]
    @test hierarchical(names, sizes, ((),)) == ones(Int, 1, 6)
    @test size(hierarchical(names, sizes, ())) == (0, 6)
    f = ToricFactor(names, sizes, expected)
    @test Tuple(channelname(c) for c in LenticulumCore.channels(f)) == names
    @test Tuple(channelspace(c) for c in LenticulumCore.channels(f)) == sizes
    @test islearnable(f)
    @test LuxCore.initialparameters(Xoshiro(1), f) == (η = zeros(5),)
    @test LuxCore.initialstates(Xoshiro(1), f) == NamedTuple()
    @test LuxCore.parameterlength(f) == 5
    @test LuxCore.statelength(f) == 0
    @test energyspace(f) == ScalarEnergySpace()
    for (ns, sz, A) in (((), (), zeros(Int, 0, 1)),
                          ((:x, :x), sizes, expected), (("x", :y), sizes, expected),
                          (names, (2,), expected), (names, (0, 3), expected),
                          (names, (-2, 3), expected), (names, (2.0, 3), expected),
                          (names, (true, 6), expected), (names, sizes, zeros(Int, 2, 5)),
                          (names, (typemax(Int), 2), expected))
        @test_throws ArgumentError ToricFactor(ns, sz, A)
    end
    for labels in (([1, 1], [1, 2, 3]), ([1], [1, 2, 3]), ([1, 2],),
                   [[1, 2], [1, 2, 3]], ((1, 2), [1, 2, 3]))
        @test_throws ArgumentError ToricFactor(names, sizes, expected; labels)
    end
    for gens in (((:z,),), ((3,),), ((0,),), ((:x, :x),), (:x,))
        @test_throws ArgumentError hierarchical(names, sizes, gens)
    end
    # Defensive copies: constructor arguments cannot silently alter the factor.
    A = copy(expected); labels = ([10, 20], [:a, :b, :c])
    g = ToricFactor(names, sizes, A; labels)
    A[1, 1] = 99; labels[1][1] = 99
    @test g.A == expected
    @test g.labels[1] == [10, 20]
end

@testset "pointwise energy and column order" begin
    names, sizes = (:x, :y), (2, 3)
    A = [2 -1 0 3 1 -2; 0 1 2 -1 3 1]
    η = [0.4, -0.7]; ps = (; η); st = (token = 7,)
    labels = ([20, 10], [:red, :green, :blue])
    logits = [sum(η[r] * A[r, k] for r in 1:2) for k in 1:6]
    for normalised in (false, true), custom in (false, true)
        f = ToricFactor(names, sizes, A; normalised, labels = custom ? labels : nothing)
        for j in 1:3, i in 1:2
            x = custom ? (labels[1][i], labels[2][j]) : (i, j)
            expected = -logits[i + 2*(j-1)] + (normalised ? log(sum(exp.(logits))) : 0)
            E, next = energy(f, x, nothing, nothing, ps, st)
            @test E ≈ expected
            @test next === st
            @test first(energy(f, x, ps, st)) ≈ expected
        end
        @test_throws ArgumentError energy(f, (1,), ps, st)
        @test_throws ArgumentError energy(f, (99, 1), ps, st)
        @test_throws ArgumentError energy(f, custom ? (20, :red) : (1, 1), (η = [0.2],), st)
        @test_throws ArgumentError energy(f, custom ? (20, :red) : (1, 1), (η = [Inf, 0.0],), st)
    end
    # Gauge directions can shift every logit without changing a normalised model.
    f = ToricFactor(names, sizes, vcat(ones(Int, 1, 6), A); normalised = true)
    @test first(energy(f, (2, 3), (η = [4.0; η],), st)) ≈
          first(energy(f, (2, 3), (η = [0.0; η],), st))
end

@testset "messages: all targets, evidence kinds, and polarities" begin
    η = [0.3, -0.2, 0.15, 0.4]; ps = (; η); st = (token = 8,)
    for sizes in ((2, 3), (2, 3, 2)), custom in (false, true), normalised in (false, true)
        n = length(sizes); names = Tuple((:a, :b, :c)[1:n]); A = design(sizes)
        labs = Tuple([[10i + j for j in 1:sizes[i]] for i in 1:n])
        f = ToricFactor(names, sizes, A; normalised, labels = custom ? labs : nothing)
        @test length(supported_polarities(f)) == n * 2^(n - 1)
        @test all(p -> supports_polarity(f, p), supported_polarities(f))
        @test !supports_polarity(f, Polarity(NamedTuple{names}(ntuple(_ -> Observed(), n))))
        @test !supports_polarity(f, Polarity(NamedTuple{names}(ntuple(_ -> Unobserved(), n))))
        for t in 1:n, kinds in Iterators.product(ntuple(_ -> 1:7, n-1)...)
            others = filter(!=(t), 1:n)
            beliefs = Dict{Symbol,Any}(); weights = [ones(s) for s in sizes]
            observed = Int[]
            for (i, kind) in zip(others, kinds)
                if kind == 1
                    # Omitted latent.
                elseif kind <= 5
                    beliefs[names[i]] = (nothing, TrivialBelief(), Latent(), missing)[kind-1]
                elseif kind == 6
                    beliefs[names[i]] = DiracBelief(custom ? labs[i][2] : 2)
                    weights[i] = [j == 2 ? 1.0 : 0.0 for j in 1:sizes[i]]
                    push!(observed, i)
                else
                    w = Float64.(1:sizes[i]); w ./= sum(w)
                    # Reverse labelled supports to test matching rather than positions.
                    beliefs[names[i]] = custom ? CategoricalBelief(reverse(w); labels = reverse(labs[i])) :
                                                                CategoricalBelief(w)
                    weights[i] = w; push!(observed, i)
                end
            end
            inputs = (; (name => beliefs[name] for name in names if haskey(beliefs, name))...)
            p = polarity(names, t, observed)
            site, next = factor_message(f, names[t], p, inputs, DiracBelief(:ignored), ps, st)
            @test probabilities(site) ≈ oracle_message(A, η, sizes, weights, t)
            @test site.labels == (custom ? labs[t] : nothing)
            @test next === st
            # Even an invalid target input must be excluded, as must any target prior.
            contaminated = merge(inputs, NamedTuple{(names[t],)}((DiracBelief(:ignored),)))
            again, _ = factor_message(f, names[t], p, contaminated, TrivialBelief(), ps, st)
            @test probabilities(again) ≈ probabilities(site)
        end
        for t in 1:n
            p = polarity(names, t, ())
            latent, _ = factor_message(f, names[t], p, (;), nothing, ps, st)
            inputs = NamedTuple{names}(map(i -> CategoricalBelief(ones(sizes[i])), 1:n) |> Tuple)
            uniform, _ = factor_message(f, names[t], p, inputs, nothing, ps, st)
            @test probabilities(uniform) ≈ probabilities(latent)
        end
    end
    f = ToricFactor((:a, :b), (2, 2), [1000 -1000 999 -999])
    site, _ = factor_message(f, :a, Polarity(a = Unobserved(), b = Latent()), (;), nothing,
                             (η = [1.0],), st)
    @test all(isfinite, probabilities(site))
    @test probabilities(site) ≈ [1.0, 0.0]
    # A huge common logit must preserve finite incoming evidence and log partition
    # differences, even when adding log(K) to that logit would round back to it.
    huge = ToricFactor((:a, :b), (2, 2), ones(Int, 1, 4); normalised = true)
    hugeps = (η = [1e300],)
    @test first(energy(huge, (1, 2), hugeps, st)) ≈ log(4)
    @test first(free_energy(huge, nothing, nothing, hugeps, st)) ≈ 0 atol = 1e-12
    evidence = (a = CategoricalBelief([0.2, 0.8]),)
    @test first(local_free_energy(huge, evidence, hugeps, st)) ≈
          log(2) + sum(p * log(p) for p in [0.2, 0.8])
    g = ToricFactor((:a, :b), (2, 2), zeros(Int, 0, 4))
    @test probabilities(first(factor_message(g, :a, Polarity(a = Unobserved(), b = Latent()),
                                            (;), nothing, (η = Float64[],), st))) ≈ [0.5, 0.5]
    p = Polarity(a = Unobserved(), b = Observed())
    @test_throws ArgumentError factor_message(g, :b, p, (;), nothing, (η = Float64[],), st)
    for bad in (DiracBelief(9), CategoricalBelief([1, 2, 3]),
                CategoricalBelief([1, 2]; labels = [1, 1]),
                CategoricalBelief([1, 2]; labels = [1, 9]), 2)
        @test_throws ArgumentError factor_message(g, :a, p, (b = bad,), nothing, (η = Float64[],), st)
    end
    # Label 1 must not silently select the first entry of a custom alphabet.
    g = ToricFactor((:a, :b), (2, 2), zeros(Int, 0, 4); labels = ([10, 20], [:x, :y]))
    @test_throws ArgumentError factor_message(g, :b, Polarity(a = Observed(), b = Unobserved()),
                                             (a = DiracBelief(1),), nothing, (η = Float64[],), st)
end

@testset "local and game free energy and entropy" begin
    names, sizes = (:a, :b, :c), (2, 3, 2); A = design(sizes)
    η = [0.3, -0.2, 0.15, 0.4]; ps = (; η); st = (token = 9,)
    for normalised in (false, true)
        f = ToricFactor(names, sizes, A; normalised)
        for a in (nothing, CategoricalBelief([0.4, 0.6]), DiracBelief(2)),
            b in (TrivialBelief(), CategoricalBelief([0, 0.3, 0.7]), DiracBelief(1)),
            c in (nothing, CategoricalBelief([0.8, 0.2]), DiracBelief(1))
            msgs = (; a, b, c)
            weights = map(zip(sizes, (a, b, c))) do (n, bel)
                bel isa CategoricalBelief ? probabilities(bel) : bel isa DiracBelief ?
                    [j == bel.value ? 1.0 : 0.0 for j in 1:n] : ones(n)
            end
            expected, H = oracle_statistics(A, η, sizes, weights; normalised)
            F, next = local_free_energy(f, msgs, ps, st)
            @test F ≈ expected atol = 1e-12
            @test next === st
            if normalised
                @test first(free_energy(f, (a = a, c = c), (b = b,), ps, st)) ≈ expected atol = 1e-12
                @test first(LenticulumCore.entropy(f, (a = a, c = c), (b = b,), ps, st)) ≈ H atol = 1e-12
            end
        end
        if normalised
            @test first(free_energy(f, nothing, nothing, ps, st)) ≈ 0 atol = 1e-12
            @test first(free_energy(f, CategoricalBelief([0.4, 0.6]), (b = 1, c = 2), ps, st)) ≈
                  first(local_free_energy(f, (a = CategoricalBelief([0.4, 0.6]),
                                             b = DiracBelief(1), c = DiracBelief(2)), ps, st))
            @test_throws ArgumentError free_energy(f, CategoricalBelief([1, 1]), (;), ps, st)
            @test_throws ArgumentError free_energy(f, (a = nothing,), (a = 1,), ps, st)
            @test_throws ArgumentError free_energy(f, nothing, (z = 1,), ps, st)
        else
            @test_throws ArgumentError free_energy(f, nothing, (;), ps, st)
            @test_throws ArgumentError LenticulumCore.entropy(f, nothing, (;), ps, st)
        end
    end
    f = ToricFactor(names, sizes, A; labels = ([:x, :y], [3, 2, 1], [true, false]), normalised = true)
    incoming = ([0.2, 0.8], [0.0, 1.0, 0.0], ones(2))
    expected, H = oracle_statistics(A, η, sizes, incoming; normalised = true)
    π = (a = CategoricalBelief([0.8, 0.2]; labels = [:y, :x]),)
    @test first(free_energy(f, π, (b = 2,), ps, st)) ≈ expected
    @test first(LenticulumCore.entropy(f, π, (b = 2,), ps, st)) ≈ H
end

@testset "tree BP and Bethe partition function" begin
    # Five variables, three toric factors, and two priors; optionally clamp c.
    # Incidence graph: a--f1--b--f2--d--f3--e, with c attached to f2.
    sizes = (2, 3, 2, 2, 3); names = (:a, :b, :c, :d, :e)
    subsets = ((1, 2), (2, 3, 4), (4, 5)); fnames = (:ab, :bcd, :de)
    As = map(s -> design(Tuple(sizes[i] for i in s)), subsets)
    ηs = ([0.3, -0.2, 0.15, 0.4], [-0.1, 0.5, 0.2, -0.3], [0.6, 0.1, -0.2, 0.25])
    pa, pe = [0.25, 0.75], [0.2, 0.5, 0.3]
    for normalised in (false, true), clamp in (false, true)
        b = GraphBuilder()
        for name in names; variable!(b, name); end
        for (fname, subset, A) in zip(fnames, subsets, As)
            ns = Tuple(names[i] for i in subset); sz = Tuple(sizes[i] for i in subset)
            factor!(b, fname, ToricFactor(ns, sz, A; normalised))
            for name in ns; connect!(b, fname, name, name; direction = Bidirectional()); end
        end
        factor!(b, :pa, PriorFactor(:a, CategoricalBelief(pa), x -> -log(pa[x])))
        factor!(b, :pe, PriorFactor(:e, CategoricalBelief(pe), x -> -log(pe[x])))
        # Priors need cavity messages too, to form their posterior factor beliefs for F.
        connect!(b, :pa, :a, :a; direction = Bidirectional())
        connect!(b, :pe, :e, :e; direction = Bidirectional())
        if clamp
            factor!(b, :obs, DataFactor(:c, 2))
            connect!(b, :obs, :c, :c; direction = Emitting())
        end
        g = build(b)
        @test istree(g)
        @test validate(g) === g
        ps = NamedTuple{fnames}(map(η -> (; η), ηs))
        st = LuxCore.initialstates(Xoshiro(42), g)
        xs = states(sizes)
        w = map(xs) do x
            value = pa[x[1]] * pe[x[5]] * (clamp && x[3] != 2 ? 0.0 : 1.0)
            for (subset, A, η) in zip(subsets, As, ηs)
                # Compute column strides explicitly, independently of factor indexing.
                col = 1; stride = 1
                for i in subset
                    col += (x[i] - 1) * stride; stride *= sizes[i]
                end
                logits = [sum(η[r] * A[r, k] for r in 1:4) for k in axes(A, 2)]
                value *= exp(logits[col]) / (normalised ? sum(exp.(logits)) : 1.0)
            end
            value
        end
        Z = sum(w)
        for root in (:a, :b, :e)
            marginals, report, next, store = infer!(g, tree_schedule(g; root), ps, st)
            @test report.converged
            for (i, name) in enumerate(names)
                expected = [sum(w[k] for (k, x) in enumerate(xs) if x[i] == j)/Z for j in 1:sizes[i]]
                bel = marginals[name]
                actual = bel isa DiracBelief ? [j == bel.value ? 1.0 : 0.0 for j in 1:sizes[i]] : probabilities(bel)
                @test actual ≈ expected atol = 1e-12
            end
            F, _ = scalar_free_energy(store, g, ps, next)
            @test F ≈ -log(Z) atol = 1e-12
            B, _ = bethe_free_energy(store, g, ps, next)
            @test sum(B[:factors][n] for n in keys(B[:factors])) +
                  sum(B[:variables][n] for n in keys(B[:variables])) ≈ -log(Z) atol = 1e-12
        end
    end
end

@testset "LP-014: a prior on an Emitting edge" begin
    # Marginals stay exact, but the prior's free-energy term gets no cavity message.
    pa = [0.25, 0.75]; A = design((2, 3)); η = [0.3, -0.2, 0.15, 0.4]
    w = [pa[i] * exp(sum(η .* A[:, i + 2*(j-1)])) for i in 1:2, j in 1:3]
    Z = sum(w)
    for direction in (Bidirectional(), Emitting())
        b = GraphBuilder()
        variable!(b, :a); variable!(b, :b)
        factor!(b, :ab, ToricFactor((:a, :b), (2, 3), A))
        connect!(b, :ab, :a, :a; direction = Bidirectional())
        connect!(b, :ab, :b, :b; direction = Bidirectional())
        factor!(b, :pa, PriorFactor(:a, CategoricalBelief(pa), x -> -log(pa[x])))
        connect!(b, :pa, :a, :a; direction)
        g = build(b)
        ps = (ab = (; η),); st = LuxCore.initialstates(Xoshiro(1), g)
        marginals, _, next, store = infer!(g, tree_schedule(g; root = :a), ps, st)
        @test probabilities(marginals[:b]) ≈ vec(sum(w; dims = 1)) / Z
        F, _ = scalar_free_energy(store, g, ps, next)
        if direction isa Bidirectional
            @test F ≈ -log(Z)
        else
            @test_broken F ≈ -log(Z)
        end
    end
end

@testset "direct-message factor conformance" begin
    for sizes in ((2,), (2, 3), (2, 3, 2))
        n = length(sizes); names = Tuple((:a, :b, :c)[1:n]); A = design(sizes)
        labs = Tuple([Symbol("v$(i)_$(j)") for j in 1:sizes[i]] for i in 1:n)
        f = ToricFactor(names, sizes, A; normalised = true, labels = labs)
        ps = (η = [0.3, -0.2, 0.15, 0.4],); st = NamedTuple()
        for p in supported_polarities(f)
            t = findfirst(isequal(only(unobserved_channels(p))), names)
            inputs = NamedTuple{observed_channels(p)}(Tuple(DiracBelief(labs[i][1])
                     for i in 1:n if names[i] in observed_channels(p)))
            π = CategoricalBelief(Float64.(1:sizes[t]); labels = labs[t])
            prior = NamedTuple{(names[t],)}((π,)); msgs = merge(prior, inputs)
            incoming = [names[i] in observed_channels(p) ? [j == 1 ? 1.0 : 0.0 for j in 1:sizes[i]] :
                        i == t ? probabilities(π) : ones(sizes[i]) for i in 1:n]
            site = oracle_message(A, ps.η, sizes, incoming, t)
            posterior = site .* probabilities(π); posterior ./= sum(posterior)
            check_factor_conformance(f, p; π, inputs, ps, st,
                pointwise = (Tuple(first.(labs)), nothing, nothing),
                game_prior = prior, game_y = inputs, msgs, direct_message = true,
                expected_site = CategoricalBelief(site; labels = labs[t]),
                expected_posterior = CategoricalBelief(posterior; labels = labs[t]))
        end
    end
end
end
