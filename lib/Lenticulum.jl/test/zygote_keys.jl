# Zygote's cotangent of the whole graph parameter tree at fixed messages (I16).
using Test, Lenticulum, LenticulumCore, Mycelium, LinearAlgebra, LuxCore, Random
using ForwardDiff, Zygote

i16_energy(s, g, st) = p -> first(LenticulumCore.scalar_free_energy(s, g, p, st))

function i16_store(g, ps, st)
    s = MessageStore(g)
    _, st = sweep!(s, g, SequentialSchedule(tasks(tree_schedule(g))), ps, st)
    return s, st
end

# Compare in the coordinates of the input tree, including absent cotangents at any level.
i16_cotangent(p, ::Nothing) = zero(LenticulumCore._flatten(p))
i16_cotangent(p::NamedTuple, dp::Nothing) = zero(LenticulumCore._flatten(p))
function i16_cotangent(p::NamedTuple, dp)
    return reduce(vcat, (i16_cotangent(p[k], get(dp, k, nothing)) for k in keys(p)); init = Float64[])
end
i16_cotangent(p, dp) = LenticulumCore._flatten(dp)

function i16_forward(F, ps)
    θ = LenticulumCore._flatten(ps)
    dθ = ForwardDiff.gradient(v -> F(LenticulumCore._unflatten(v, ps)), θ)
    return LenticulumCore._unflatten(dθ, ps)
end

function i16_compare(ps, actual, reference)
    for k in keys(ps)
        dp = actual === nothing ? nothing : get(actual, k, nothing)
        @test i16_cotangent(ps[k], dp) ≈ LenticulumCore._flatten(reference[k]) rtol = 1e-8 atol = 1e-10
    end
end

function i16_random_graph(trial; tied = true, frozen = false)
    rng = Xoshiro(1600 + trial)
    n = 6
    dims = rand(rng, 1:3, n)
    dims[2:3] .= dims[1] # equal shapes for the first two transitions
    b = GraphBuilder()
    for i in 1:n
        variable!(b, Symbol(:x, i), dims[i])
    end
    factor!(b, :prior, GaussianPrior(:x, randn(rng, dims[1]), Matrix(1.0I, dims[1], dims[1])))
    connect!(b, :prior, :x, :x1)
    params = Dict{Symbol,Any}()
    shared = (A = 0.5 .* randn(rng, dims[1], dims[1]), b = randn(rng, dims[1]))
    for i in 2:n
        parent = isodd(trial) ? i - 1 : (i <= 3 ? 1 : rand(rng, 1:(i - 1)))
        f, h, d, y = Symbol(:f, i), Symbol(:h, i), Symbol(:d, i), Symbol(:y, i)
        key = tied && i <= 3 ? :shared : f
        freeze = frozen && i == 4 ? :all : ()
        factor!(b, f, GaussianFactor(dims[parent] => dims[i]; noise = 0.5 + rand(rng)); params = key, freeze)
        connect!(b, f, :x, Symbol(:x, parent))
        connect!(b, f, :y, Symbol(:x, i))
        params[key] = i <= 3 ? shared : (A = 0.5 .* randn(rng, dims[i], dims[parent]), b = randn(rng, dims[i]))

        variable!(b, y, dims[i])
        factor!(b, h, GaussianFactor(dims[i] => dims[i]; noise = 0.5 + rand(rng)))
        connect!(b, h, :x, Symbol(:x, i))
        connect!(b, h, :y, y)
        params[h] = (A = Matrix(1.0I, dims[i], dims[i]) + 0.2 .* randn(rng, dims[i], dims[i]), b = randn(rng, dims[i]))
        factor!(b, d, DataFactor(:v, randn(rng, dims[i])))
        connect!(b, d, :v, y; direction = Emitting())
    end
    g = build(b)
    ks = parameter_keys(g)
    ps = NamedTuple{ks}(Tuple(get(params, k, NamedTuple()) for k in ks))
    return g, ps, LuxCore.initialstates(rng, g)
end

# A learnable factor with a default parameter, so a missing key exercises the empty-tree fallback.
struct I16OptionalFactor <: LenticulumCore.AbstractLenticulumFactor end
LenticulumCore.channels(::I16OptionalFactor) = (LenticulumCore.Channel(:x, 1),)
LenticulumCore.supports_polarity(::I16OptionalFactor, p::LenticulumCore.Polarity) =
    keys(p) == (:x,) && p[:x] isa LenticulumCore.Unobserved
LenticulumCore.islearnable(::I16OptionalFactor) = true
LuxCore.initialparameters(::AbstractRNG, ::I16OptionalFactor) = (weight = 2.0,)
Mycelium.uses_cavity(::I16OptionalFactor) = true
Mycelium.factor_message(::I16OptionalFactor, ::Symbol, polarity, msgs, old, ps, st) =
    (GaussianBelief([0.0], ones(1, 1)), st)
Mycelium.local_free_energy(::I16OptionalFactor, msgs, ps, st) = (get(ps, :weight, 1.0)^2, st)

# The position-velocity chain from bench/kalman_chain.jl.
function i16_kalman_chain(T; nx = 2, H = [1.0 0.0; 0.0 1.0], A = [1.0 1.0; 0.0 1.0])
    b = GraphBuilder()
    for t in 0:T
        variable!(b, Symbol(:x, t), nx)
    end
    factor!(b, :prior, GaussianPrior(:x, zeros(nx), Matrix(1.0I, nx, nx)))
    connect!(b, :prior, :x, :x0)
    for t in 1:T
        factor!(b, Symbol(:f, t), GaussianFactor(nx => nx; noise = 0.1, channels = (:a, :b)))
        connect!(b, Symbol(:f, t), :a, Symbol(:x, t - 1))
        connect!(b, Symbol(:f, t), :b, Symbol(:x, t))
        variable!(b, Symbol(:y, t), size(H, 1))
        factor!(b, Symbol(:h, t), GaussianFactor(nx => size(H, 1); noise = 0.5, channels = (:a, :b)))
        connect!(b, Symbol(:h, t), :a, Symbol(:x, t))
        connect!(b, Symbol(:h, t), :b, Symbol(:y, t))
        factor!(b, Symbol(:d, t), DataFactor(:v, fill(0.3t, size(H, 1))))
        connect!(b, Symbol(:d, t), :v, Symbol(:y, t); direction = Emitting())
    end
    g = build(b)
    rng = Xoshiro(1)
    ps = LuxCore.initialparameters(rng, g)
    st = LuxCore.initialstates(rng, g)
    kv = Dict{Symbol,Any}(pairs(ps))
    for t in 1:T
        kv[Symbol(:f, t)] = (A = A, b = zeros(nx))
        kv[Symbol(:h, t)] = (A = H, b = zeros(size(H, 1)))
    end
    ks = keys(ps)
    return g, NamedTuple{ks}(Tuple(kv[k] for k in ks)), st
end

function i16_gradient_bytes(T)
    g, ps, st = i16_kalman_chain(T)
    @test nfactors(g) == length(keys(ps)) == 3T + 1
    s, st = i16_store(g, ps, st)
    F = i16_energy(s, g, st)
    Zygote.gradient(F, ps)
    return @allocated Zygote.gradient(F, ps)
end

@testset "I16: fixed-message gradient agrees with ForwardDiff" begin
    for trial in 1:20
        g, ps, st = i16_random_graph(trial)
        s, st = i16_store(g, ps, st)
        F = i16_energy(s, g, st)
        dp = only(Zygote.gradient(F, ps))
        i16_compare(ps, dp, i16_forward(F, ps))
        if trial == 1
            θ = LenticulumCore._flatten(ps)
            v = normalize(randn(Xoshiro(16), length(θ)))
            h = 1e-5
            hi = F(LenticulumCore._unflatten(θ + h * v, ps))
            lo = F(LenticulumCore._unflatten(θ - h * v, ps))
            @test dot(i16_cotangent(ps, dp), v) ≈ (hi - lo) / (2h) rtol = 1e-8 atol = 1e-8
        end
    end
end

@testset "I16: tied keys sum the factors' cotangents" begin
    for trial in 1:20
        g, ps, st = i16_random_graph(trial)
        untied, separate, separate_st = i16_random_graph(trial; tied = false)
        s, st = i16_store(g, ps, st)
        F = i16_energy(s, g, st)
        # The untied graph has identical factor and edge order, so it reads the same store.
        G = i16_energy(s, untied, separate_st)
        @test F(ps) ≈ G(separate) rtol = 1e-12
        independent = i16_forward(G, separate)
        df2 = LenticulumCore._flatten(independent.f2)
        df3 = LenticulumCore._flatten(independent.f3)
        @test norm(df2) > 1e-8 && norm(df3) > 1e-8
        actual = only(Zygote.gradient(F, ps))
        @test i16_cotangent(ps.shared, actual.shared) ≈ df2 + df3 rtol = 1e-8 atol = 1e-10
    end
end

@testset "I16: extra, empty, and frozen keys have zero cotangents" begin
    for trial in 1:20
        g, ps, st = i16_random_graph(trial; frozen = true)
        s, st = i16_store(g, ps, st)
        F = i16_energy(s, g, st)
        with_extra = merge(ps, (unused = (A = randn(Xoshiro(trial), 2, 2),),))
        @test F(with_extra) ≈ F(ps) rtol = 1e-12
        @test !isempty(LenticulumCore._flatten(ps.f4))
        dp = only(Zygote.gradient(F, with_extra))
        @test dp isa NamedTuple
        for k in (:unused, :prior, :d2, :d3, :d4, :d5, :d6, :f4)
            p, cotangent = with_extra[k], get(dp, k, nothing)
            if isempty(p)
                @test cotangent === nothing || isempty(cotangent)
            else
                @test all(iszero, i16_cotangent(p, cotangent))
            end
        end
        @test norm(i16_cotangent(ps.shared, dp.shared)) > 1e-8
    end
end

@testset "I16: shuffled parameter keys preserve field names and gradients" begin
    for trial in 1:20
        g, ps, st = i16_random_graph(trial)
        s, st = i16_store(g, ps, st)
        F = i16_energy(s, g, st)
        ks = Tuple(shuffle(Xoshiro(1700), collect(parameter_keys(g))))
        @test ks != parameter_keys(g)
        shuffled = NamedTuple{ks}(map(k -> ps[k], ks))
        @test F(shuffled) ≈ F(ps) rtol = 1e-12
        dp = only(Zygote.gradient(F, shuffled))
        @test dp isa NamedTuple && keys(dp) == keys(shuffled)
        i16_compare(shuffled, dp, i16_forward(F, ps))
        canonical = only(Zygote.gradient(F, ps))
        for k in ks
            @test i16_cotangent(ps[k], get(dp, k, nothing)) ≈
                  i16_cotangent(ps[k], get(canonical, k, nothing)) rtol = 1e-8 atol = 1e-10
        end
    end
end

@testset "I16: a missing learnable key uses the empty parameter tree" begin
    b = GraphBuilder()
    variable!(b, :x, 1)
    for k in (:present, :omitted)
        factor!(b, k, I16OptionalFactor())
        connect!(b, k, :x, :x)
    end
    g = build(b)
    rng = Xoshiro(16)
    ps = LuxCore.initialparameters(rng, g)
    s, st = i16_store(g, ps, LuxCore.initialstates(rng, g))
    F = i16_energy(s, g, st)
    missing = (present = ps.present,)
    explicit = merge(missing, (omitted = NamedTuple(),))
    value, back = Zygote.pullback(F, missing)
    @test value == F(missing) == F(explicit)
    @test F(ps) - value ≈ ps.omitted.weight^2 - 1
    @test only(back(one(value))).present.weight ≈ 2 * ps.present.weight
end

@testset "I16: one factor and an all-empty parameter tree" begin
    for learned in (true, false)
        b = GraphBuilder()
        variable!(b, :x, 1)
        prior = learned ? I16OptionalFactor() : GaussianPrior(:x, [0.3], fill(2.0, 1, 1))
        factor!(b, :prior, prior)
        connect!(b, :prior, :x, :x)
        if !learned
            factor!(b, :data, DataFactor(:v, [1.0]))
            connect!(b, :data, :v, :x; direction = Emitting())
        end
        g = build(b)
        rng = Xoshiro(16)
        ps = LuxCore.initialparameters(rng, g)
        s, st = i16_store(g, ps, LuxCore.initialstates(rng, g))
        F = i16_energy(s, g, st)
        @test isfinite(F(ps))
        dp = only(Zygote.gradient(F, ps))
        if learned
            @test nfactors(g) == 1
            i16_compare(ps, dp, i16_forward(F, ps))
            @test dp.prior.weight ≈ 2 * ps.prior.weight
        else
            @test all(isempty, values(ps))
            @test all(iszero, i16_cotangent(ps, dp))
        end
    end
end

@testset "I16: fixed-message gradients on plate block rows" begin
    rng = Xoshiro(1607)
    for trial in 1:4
        D, N = rand(rng, 1:3), rand(rng, 2:7)
        A = randn(rng, D, D)
        Σ = A * A' + I
        b = GraphBuilder()
        plate!(b, :obs, N) do p, i
            variable!(p, :x, D)
            factor!(p, :prior, GaussianPrior(:x, randn(rng, D), Σ; learn_noise = true))
            factor!(p, :data, DataFactor(:x, randn(rng, D)))
            connect!(p, :prior, :x, :x)
            connect!(p, :data, :x, :x; direction = Emitting())
        end
        g = build(b)
        ps = LuxCore.initialparameters(rng, g)
        st = LuxCore.initialstates(rng, g)
        s = MessageStore(g)
        _, st = propagate!(s, g, plate_schedule(g), ps, st; maxsweeps = 3)
        @test any(!isnothing, s.to_variable.blocks) || any(!isnothing, s.to_factor.blocks)
        F = i16_energy(s, g, st)
        i16_compare(ps, only(Zygote.gradient(F, ps)), i16_forward(F, ps))
    end
end

@testset "I16: Kalman-chain gradient allocations grow less than eightfold" begin
    small = i16_gradient_bytes(32)
    large = i16_gradient_bytes(128)
    @info "I16 Kalman-chain Zygote allocations" small large ratio = large / small
    @test large < 8 * small
end

# --- Regression tests of the review of I16 (Codex; LP-073) --------------------------------------

# A belief with a scalar payload, and a factor whose energy is the payload squared.
struct I16Payload{T} <: LenticulumCore.AbstractBelief
    x::T
end
Mycelium.combine(a::I16Payload, b::I16Payload) = I16Payload(a.x + b.x)
Mycelium.variable_entropy(b::I16Payload) = b.x^2
i16_payload(b::I16Payload) = b.x
i16_payload(b::DiracBelief) = only(b.value)
struct I16EdgeEnergy <: LenticulumCore.AbstractLenticulumFactor end
LenticulumCore.channels(::I16EdgeEnergy) = (LenticulumCore.Channel(:x, 1),)
LenticulumCore.supports_polarity(::I16EdgeEnergy, p::LenticulumCore.Polarity) = keys(p) == (:x,) && p[:x] isa Unobserved
Mycelium.local_free_energy(::I16EdgeEnergy, msgs, ps, st) = (i16_payload(msgs.x)^2, st)
Mycelium.mean_field_energy(::I16EdgeEnergy, msgs, ps, st) = (i16_payload(msgs.x)^2, st)

@testset "I16 review: a pin set inside Zygote keeps its gradient (finding 1)" begin
    b = GraphBuilder()
    variable!(b, :x, 1)
    factor!(b, :mf, I16EdgeEnergy(); rule = MeanField())
    connect!(b, :mf, :x, :x)
    g = build(b)
    function F(p)
        s = MessageStore([nothing], [nothing], 0)
        pin!(s, g, :x, [p])
        return first(LenticulumCore.scalar_free_energy(s, g, (;), (;)))
    end
    @test F(7.0) == 49.0
    @test only(Zygote.gradient(F, 7.0)) == 14.0                 # was `nothing`
end

# A factor whose energy is `coefficient * ps.p` (or `ps.energy` when active), and a belief of fixed entropy.
struct I16Linear <: LenticulumCore.AbstractLenticulumFactor
    coefficient::Float32
end
LenticulumCore.islearnable(::I16Linear) = true
LenticulumCore.channels(::I16Linear) = (LenticulumCore.Channel(:x, 1),)
LenticulumCore.supports_polarity(::I16Linear, p::LenticulumCore.Polarity) = keys(p) == (:x,) && p[:x] isa Unobserved
Mycelium.local_free_energy(f::I16Linear, msgs, ps, st) = (f.coefficient * ps.p, st)
struct I16FixedEntropy <: LenticulumCore.AbstractBelief
    value::Float32
end
Mycelium.variable_entropy(b::I16FixedEntropy) = b.value

@testset "I16 review: the terms are summed from the left, as before (finding 2)" begin
    # Variable terms: the factor subtotal 2^25, then the variable terms -2^25 (x, degree 2) and 1 (y):
    # from the left 2^25 - 2^25 + 1 = 1; grouping the variable terms first loses the 1 in Float32.
    b = GraphBuilder()
    variable!(b, :x, 1); variable!(b, :y, 1)
    for (name, v, c) in ((:a, :x, 1f0), (:b, :x, 0f0), (:c, :y, 0f0), (:d, :y, 0f0))
        factor!(b, name, I16Linear(c)); connect!(b, name, :x, v)
    end
    g = build(b)
    s = MessageStore(g)
    large = Float32(2)^25
    for (v, h) in ((:x, -large), (:y, 1f0))
        e = first(Mycelium.edges_of_variable(g, Mycelium.variable(g, v).id))
        s.to_variable[e] = Mycelium.Message(I16FixedEntropy(h), 0)
    end
    pz = (b = (p = 0f0,), c = (p = 0f0,), d = (p = 0f0,))
    F(p) = first(LenticulumCore.scalar_free_energy(s, g, merge((a = (p = p,),), pz), (;)))
    @test F(large) === 1f0
    # Factor terms, more than 1 024 of them (where `sum` goes pairwise), tied: 2^25 p, …, -2^25 p, p.
    n = 1026
    cs = zeros(Float32, n); cs[[1, n - 1, n]] = [large, -large, 1f0]
    b = GraphBuilder()
    for (i, c) in enumerate(cs)
        name = Symbol(:f, i)
        variable!(b, name, 1); factor!(b, name, I16Linear(c); params = :shared); connect!(b, name, :x, name)
    end
    g = build(b)
    s = MessageStore(g)
    G(p) = first(LenticulumCore.scalar_free_energy(s, g, (shared = (p = p,),), (;)))
    @test G(1f0) === 1f0
    @test ForwardDiff.derivative(G, 0f0) === 1f0
    @test only(Zygote.gradient(G, 0f0)) == 1f0
end

# A factor reading a weight from its state.
struct I16StateEnergy <: LenticulumCore.AbstractLenticulumFactor end
LenticulumCore.islearnable(::I16StateEnergy) = true
LenticulumCore.channels(::I16StateEnergy) = (LenticulumCore.Channel(:x, 1),)
LenticulumCore.supports_polarity(::I16StateEnergy, p::LenticulumCore.Polarity) = keys(p) == (:x,) && p[:x] isa Unobserved
Mycelium.local_free_energy(::I16StateEnergy, msgs, ps, st) = (ps.p * get(st, :weight, 2.0), st)

@testset "I16 review: a state with one field per factor stays linear (finding 3)" begin
    function backward_bytes(n)
        b = GraphBuilder()
        names = ntuple(i -> Symbol(:f, i), n)
        for name in names
            variable!(b, name, 1); factor!(b, name, I16StateEnergy()); connect!(b, name, :x, name)
        end
        g = build(b)
        ps = NamedTuple{names}(ntuple(_ -> (p = 1.0,), n))
        st = NamedTuple{names}(ntuple(_ -> (weight = 2.0,), n))
        s = MessageStore(g)
        F(p) = first(LenticulumCore.scalar_free_energy(s, g, p, st))
        y, back = Zygote.pullback(F, ps)
        @test y == 2n
        @test all(p -> p.p == 2.0, only(back(1.0)))
        return minimum(_ -> (back = last(Zygote.pullback(F, ps)); @allocated back(1.0)), 1:3)
    end
    small, large = backward_bytes(64), backward_bytes(256)
    @info "I16 stateful backward allocations" small large ratio = large / small
    @test large < 8 * small                                       # was 20.8 (8.18 MB → 51.23 MB, 256 → 512)
end

@testset "I16 review: the rule of `_subtrees` takes every form of cotangent (finding 5)" begin
    CRC = Base.get_extension(Mycelium, :MyceliumChainRulesCoreExt).ChainRulesCore
    ps = (shared = (a = 1.0, b = 1.0),)
    nodes = [(params = :shared,), (params = :shared,), (params = :missing,)]
    _, back = CRC.rrule(Mycelium._subtrees, ps, nodes)
    for z in (CRC.ZeroTangent(), CRC.NoTangent(), nothing, CRC.Thunk(() -> nothing))
        @test back(z)[2] isa CRC.AbstractZero
    end
    sh(Δ) = CRC.backing(back(Δ)[2]).shared
    @test sh(CRC.Thunk(() -> Any[(a = 2.0, b = CRC.NoTangent()), CRC.Thunk(() -> (a = nothing, b = 3.0)), CRC.ZeroTangent()])) ==
          (a = 2.0, b = 3.0)
    @test sh(Any[(a = 2.0,), (b = 3.0,), CRC.NoTangent()]) == (a = 2.0, b = 3.0)
    @test sh(Any[CRC.Tangent{typeof(ps.shared)}(; a = 2.0), (a = 3.0,), CRC.NoTangent()]) == (a = 5.0,)
end
