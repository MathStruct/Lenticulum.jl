module SplitOracle
using Lenticulum, LenticulumCore, Mycelium, LinearAlgebra, Random, LuxCore, Test
using LenticulumCore: DiracBelief, TrivialBelief, Observed, Unobserved, Latent, MixtureBelief

spd(rng, n) = (M = randn(rng, n, n); Matrix(Symmetric(M * M' + 0.5I)))
prior(rng, n) = (μ = randn(rng, n), Σ = spd(rng, n))
ports(d, pieces) = merge((z = 1:d,), pieces)
selection(d, indices) = Matrix{Float64}(I, d, d)[indices, :]

function random_pieces(rng, d, k; full_cover = false)
    names = (:x, :y, :w)[1:k]
    order = randperm(rng, d)
    if full_cover
        # Collectively proper with only piece priors, even when each piece is a projection.
        return NamedTuple{names}(ntuple(i -> order[i:k:end], k))
    end
    # Reserve one uncovered coordinate and share one coordinate among all pieces.
    return NamedTuple{names}(ntuple(k) do _
        indices = vcat(order[1], shuffle(rng, order[2:(d - 1)])[1:rand(rng, 1:(d - 2))])
        shuffle!(rng, indices)
    end)
end

function graph(relation, indices, priors; point = nothing)
    gb = GraphBuilder()
    factor!(gb, :split, relation)
    for (v, ix) in pairs(indices)
        variable!(gb, v, length(ix))
        connect!(gb, :split, v, v)
        if haskey(priors, v)
            p = priors[v]
            name = Symbol(:prior_, v)
            factor!(gb, name, GaussianPrior(:out, p.μ, p.Σ))
            connect!(gb, name, :out, v)
        end
    end
    if point !== nothing
        factor!(gb, :data, DataFactor(:out, point.value))
        connect!(gb, :data, :out, point.channel; direction = Emitting())
    end
    return validate(build(gb))
end

function run_tree(g)
    ps = LuxCore.initialparameters(Xoshiro(0), g)
    st = LuxCore.initialstates(Xoshiro(0), g)
    marg, rep, _, s = infer!(g, tree_schedule(g), ps, st)
    @test rep.converged
    return marg, first(LenticulumCore.scalar_free_energy(s, g, ps, st))
end

# Multiply the normalised Gaussian densities N(S_v z; μ_v, Σ_v) in dense canonical form.
# Their negative log product is ½z'Λz − η'z + c. Integrate by completing the square.
function dense(indices, priors; point = nothing)
    d = length(indices.z)
    Λ = zeros(d, d)
    η = zeros(d)
    c = 0.0
    for (v, p) in pairs(priors)
        S = selection(d, indices[v])
        R = cholesky(Symmetric(p.Σ))
        h = R \ p.μ
        Λ += S' * (R \ S)
        η += S' * h
        c += (dot(p.μ, h) + logdet(R) + length(p.μ) * log(2π)) / 2
    end
    P = cholesky(Symmetric(Λ))
    μ = P \ η
    Σ = Matrix(inv(P))
    nlogZ = c - dot(η, μ) / 2 + (logdet(P) - d * log(2π)) / 2
    point === nothing && return μ, Σ, nlogZ

    # The extra evidence is the density of the observed piece under the soft posterior.
    # Substitute the observed coordinates into the quadratic to get the conditional
    # moments. This also gives exactly zero covariance on the fixed coordinates.
    fixed = indices[point.channel]
    free = setdiff(1:d, fixed)
    C = cholesky(Symmetric(Σ[fixed, fixed]))
    residual = point.value - μ[fixed]
    nlogZ += (dot(residual, C \ residual) + logdet(C) + length(fixed) * log(2π)) / 2
    Pfree = cholesky(Symmetric(Λ[free, free]))
    μ[free] = Pfree \ (η[free] - Λ[free, fixed] * point.value)
    μ[fixed] = point.value
    fill!(Σ, 0)
    Σ[free, free] = inv(Pfree)
    return μ, Σ, nlogZ
end

function check_message(actual, expected)
    @test actual isa ExtendedGaussian
    if actual isa ExtendedGaussian
        @test actual.μ ≈ expected.μ atol = 1e-7 rtol = 1e-9
        @test actual.Σ ≈ expected.Σ atol = 1e-7 rtol = 1e-9
        @test actual.D * actual.D' ≈ expected.D * expected.D' atol = 1e-7 rtol = 1e-9
    end
end

function message(f, target, inputs)
    names = Tuple(channelname(c) for c in LenticulumCore.channels(f))
    roles = map(names) do v
        v === target && return Unobserved()
        return get(inputs, v, nothing) === nothing ? Latent() : Observed()
    end
    p = LenticulumCore.Polarity(NamedTuple{names}(roles))
    return first(Mycelium.factor_message(f, target, p, inputs, nothing, (;), (;)))
end

function local_scalar(f, inputs)
    F, _ = Mycelium.local_free_energy(f, inputs, (;), (;))
    return LenticulumCore.scalarise(LenticulumCore.scalarisation(f), F)
end

# Write the incoming potentials in z coordinates as exp(−½‖A*z − b‖²)
# subject to H*z = h. Solve on the null space of H by dense least squares.
function dense_message(indices, target, inputs)
    d = length(indices.z)
    tol = 1e-10
    A, H = zeros(0, d), zeros(0, d)
    b, h = Float64[], Float64[]
    for (v, input) in pairs(inputs)
        e = extended(input, length(indices[v]))
        S = selection(d, indices[v])
        hard = nullspace(e.Σ + e.D * e.D'; atol = tol)
        E = eigen(Symmetric(e.Σ))
        finite = E.values .> tol
        W = Diagonal(inv.(sqrt.(E.values[finite]))) * E.vectors[:, finite]'
        A = vcat(A, W * S)
        b = vcat(b, W * e.μ)
        H = vcat(H, hard' * S)
        h = vcat(h, hard' * e.μ)
    end
    z0 = pinv(H; atol = tol) * h
    N = nullspace(H; atol = tol)
    AN = A * N
    L = N * pinv(AN; atol = tol)
    μ = z0 + L * (b - A * z0)
    D = N * nullspace(AN; atol = tol)

    # Restrict the affine support, covariance and flat subspace to the target.
    # The canonical representative removes the projected flat component.
    ix = indices[target]
    U = nullspace(D[ix, :]'; atol = tol)
    P = U * U'
    finite_root = P * L[ix, :]
    return (μ = P * μ[ix], Σ = finite_root * finite_root', D = nullspace(U'; atol = tol))
end

function random_input(rng, point, kind)
    n = length(point)
    kind === :dirac && return DiracBelief(point)
    kind === :proper && return Gaussian(randn(rng, n), spd(rng, n))
    kind === :trivial && return TrivialBelief()
    kind === :nothing && return nothing
    B = Matrix(qr(randn(rng, n, n)).Q)
    if kind === :improper
        V = B[:, 1:(n - 1)]
        Λ = V * spd(rng, n - 1) * V'
        return GaussianBelief(Λ * randn(rng, n), (Λ + Λ') / 2, true)
    end
    # One flat direction, a hard direction through the shared point, and finite variance
    # in the remaining directions. All exact observations in a trial are consistent.
    V = B[:, 3:end]
    Σ = V * spd(rng, n - 2) * V'
    return ExtendedGaussian(point, (Σ + Σ') / 2, B[:, 1:1])
end

end # module SplitOracle

@testset "SplitFactor" begin
    SO = SplitOracle

    @testset "Random trees against dense Gaussian marginals and evidence" begin
        rng = Xoshiro(201)
        npermuted = noverlap = nuncovered = npieces_only = npoint = 0
        for trial in 1:40
            @testset "tree $trial" begin
                d = rand(rng, 4:6)
                k = mod1(trial, 3)
                pieces_only = trial % 4 == 0
                pieces = SO.random_pieces(rng, d, k; full_cover = pieces_only)
                indices = SO.ports(d, pieces)
                union_indices = union(values(pieces)...)
                npermuted += any(!issorted(ix) for ix in values(pieces))
                noverlap += length(union_indices) < sum(length, values(pieces))
                nuncovered += length(union_indices) < d
                npieces_only += pieces_only

                # Clamp one piece in half the trees. When pieces overlap, its fixed
                # coordinates must also be fixed in the other pieces' marginals.
                has_point = trial % 4 in (1, 2)
                point = has_point ? (channel = :x, value = randn(rng, length(pieces.x))) : nothing
                npoint += has_point
                names = Tuple(v for v in keys(indices) if !(pieces_only && v === :z) && !(has_point && v === :x))
                priors = NamedTuple{names}(map(v -> SO.prior(rng, length(indices[v])), names))
                f = pieces_only ? SplitFactor(:z, pieces) : SplitFactor(:z, pieces; dim = d)
                μ, Σ, nlogZ = SO.dense(indices, priors; point)
                marg, F = SO.run_tree(SO.graph(f, indices, priors; point))
                for (v, ix) in pairs(indices)
                    e = extended(marg[v], length(ix))
                    @test size(e.D, 2) == 0
                    @test e.μ ≈ μ[ix] atol = 1e-7 rtol = 1e-9
                    @test e.Σ ≈ Σ[ix, ix] atol = 1e-7 rtol = 1e-9
                    if has_point
                        @test size(hard_directions(e), 2) == length(intersect(ix, pieces.x))
                    end
                end
                @test F ≈ nlogZ atol = 1e-7 rtol = 1e-9

                # Local entropy is on z's support. Each observed coordinate is fixed
                # once, regardless of how many pieces repeat it.
                inputs = map(p -> Gaussian(p.μ, p.Σ), priors)
                free = has_point ? setdiff(1:d, pieces.x) : (1:d)
                if has_point
                    inputs = merge(inputs, (x = DiracBelief(point.value),))
                end
                negentropy = -(length(free) * (1 + log(2π)) + logdet(Symmetric(Σ[free, free]))) / 2
                @test SO.local_scalar(f, inputs) ≈ negentropy atol = 1e-7 rtol = 1e-9
            end
        end
        @test npermuted >= 5
        @test noverlap >= 5
        @test nuncovered >= 5
        @test npieces_only >= 5
        @test npoint >= 5
    end

    @testset "Extrinsic messages against a dense least-squares oracle" begin
        rng = Xoshiro(202)
        d = 6
        pieces = (x = [4, 2, 1], y = 2:4, w = [5, 2])
        indices = SO.ports(d, pieces)
        f = SplitFactor(:z, pieces; dim = d)
        kinds = (:proper, :dirac, :improper, :extended, :trivial, :nothing, :omitted)
        for target in keys(indices)
            SO.check_message(SO.message(f, target, (;)), extended(TrivialBelief(), length(indices[target])))
        end
        for trial in 1:56, target in keys(indices)
            @testset "messages $trial to $target" begin
                z = randn(rng, d)
                others = Tuple(v for v in keys(indices) if v !== target)
                # Cycle one input through every kind in each direction; randomise the rest.
                input_kinds = (kinds[mod1(trial, length(kinds))], rand(rng, kinds), rand(rng, kinds))
                names = Tuple(v for (v, kind) in zip(others, input_kinds) if kind !== :omitted)
                vals = Tuple(SO.random_input(rng, z[indices[v]], kind)
                             for (v, kind) in zip(others, input_kinds) if kind !== :omitted)
                inputs = NamedTuple{names}(vals)
                SO.check_message(SO.message(f, target, inputs), SO.dense_message(indices, target, inputs))
            end
        end

        # A moment-collapsed mixture on another piece, with no message on z.
        # The shared coordinate has mixture mean 1 and variance 5/2.
        # The target's first coordinate is absent from x, so it stays flat.
        mix = MixtureBelief([Gaussian([0.0, 0.0, 0.0], Matrix{Float64}(I, 3, 3)),
                             Gaussian([0.0, 2.0, 0.0], 2.0 * Matrix{Float64}(I, 3, 3))], [0.5, 0.5])
        expected = ExtendedGaussian([0.0, 1.0], [0.0 0.0; 0.0 2.5], reshape([1.0, 0.0], 2, 1))
        SO.check_message(SO.message(f, :w, (x = mix,)), expected)
    end

    @testset "Relay equivalence in both directions and on a tree" begin
        rng = Xoshiro(203)
        d = 3
        pieces = (x = 1:d,)
        indices = SO.ports(d, pieces)
        split, relay = SplitFactor(:z, pieces), RelayFactor(:z, :x)
        for (target, source) in ((:x, :z), (:z, :x))
            for kind in (:proper, :dirac, :improper, :extended, :trivial, :nothing, :omitted)
                inputs = kind === :omitted ? (;) :
                    NamedTuple{(source,)}((SO.random_input(rng, randn(rng, d), kind),))
                expected, _ = Mycelium.factor_message(relay, target, nothing, inputs, nothing, (;), (;))
                SO.check_message(SO.message(split, target, inputs), extended(expected, d))
            end
        end
        priors = (z = SO.prior(rng, d), x = SO.prior(rng, d))
        _, Fsplit = SO.run_tree(SO.graph(split, indices, priors))
        _, Frelay = SO.run_tree(SO.graph(relay, indices, priors))
        @test Fsplit ≈ Frelay atol = 1e-7 rtol = 1e-9
    end

    @testset "Constructor and inconsistent or improper beliefs" begin
        @test_throws ArgumentError SplitFactor(:z, (;))
        @test_throws ArgumentError SplitFactor(:z, (;); dim = 2)
        @test_throws ArgumentError SplitFactor(:z, (x = 1:0,))
        @test_throws ArgumentError SplitFactor(:z, (x = 1:2, y = Int[]))
        @test_throws ArgumentError SplitFactor(:z, (x = [0, 2],))
        @test_throws ArgumentError SplitFactor(:z, (x = [-1, 2],))
        @test_throws ArgumentError SplitFactor(:z, (x = [2, 1, 2],))
        @test_throws ArgumentError SplitFactor(:z, (z = 1:2,))
        @test_throws ArgumentError SplitFactor(:z, (x = 1:1,); dim = 0)
        @test_throws ArgumentError SplitFactor(:z, (x = 1:1,); dim = -1)
        @test_throws ArgumentError SplitFactor(:z, (x = [3, 1],); dim = 2)

        f = SplitFactor(:z, (x = [2, 1], y = 2:3); dim = 4)
        inconsistent = (x = DiracBelief([0.0, 1.0]), y = DiracBelief([1.0, 2.0]))
        @test_throws ArgumentError SO.message(f, :z, inconsistent)
        @test_throws ArgumentError SO.local_scalar(f, inconsistent)
        soft = (x = Gaussian(zeros(2), Matrix{Float64}(I, 2, 2)),
                y = Gaussian(ones(2), Matrix{Float64}(I, 2, 2)))
        @test_throws ArgumentError SO.local_scalar(f, soft)  # z[4] is uncovered
    end

    @testset "Dimensions, polarities and factor conformance" begin
        isdefined(@__MODULE__, :check_factor_conformance) ||
            include(joinpath(@__DIR__, "../../LenticulumCore.jl/test/support/factor_conformance.jl"))
        f = SplitFactor(:z, (x = [3, 1], y = 2:3))
        chans = LenticulumCore.channels(f)
        names = Tuple(channelname(c) for c in chans)
        @test length(chans) == 3
        @test Dict(channelname(c) => channelspace(c) for c in chans) == Dict(:z => 3, :x => 2, :y => 2)
        allowed = supported_polarities(f)
        @test length(allowed) == 12
        roles = (Observed(), Unobserved(), Latent())
        for assignment in Iterators.product(roles, roles, roles)
            p = LenticulumCore.Polarity(NamedTuple{names}(assignment))
            valid = count(r -> r isa Unobserved, assignment) == 1
            @test supports_polarity(f, p) == valid
            @test (p in allowed) == valid
        end

        rng = Xoshiro(204)
        ps, st = LuxCore.initialparameters(rng, f), LuxCore.initialstates(rng, f)
        @test ps === (;)
        @test st === (;)
        @test !LenticulumCore.islearnable(f)
        @test Mycelium.exact_messages(f) === true
        @test coupling_precision(f, ps) === :hard
        @test Mycelium.uses_cavity(f) === true

        p = LenticulumCore.Polarity(NamedTuple{names}(map(n -> n === :z ? Unobserved() : Observed(), names)))
        inputs = (x = DiracBelief([0.75, -1.0]), y = DiracBelief([0.25, 0.75]))
        π = Gaussian(zeros(3), Matrix{Float64}(I, 3, 3))
        expected = ExtendedGaussian([-1.0, 0.25, 0.75], zeros(3, 3))
        # The free energy at x clamped and y without a message: z₁, z₃ fixed, z₂ ~ N(0, 1) from π,
        # so −H(q) = −(1 + log 2π)/2. (Two points on overlapping pieces, as `inputs`, have an infinite
        # free energy, δ(0), and throw, as HardConstraintFactor does: review of U2, finding 8.)
        msgs = (x = inputs.x, z = π)
        check_factor_conformance(f, p; inputs, π, ps, st, msgs, direct_message = true,
            expected_site = expected, expected_posterior = expected,
            # graded as HardConstraintFactor: ½ log det(G̃G̃ᵀ) = log √2 (z₂ is in y), and the entropy on R
            # carries the same log √2, so the sum is −(1 + log 2π)/2
            expected_free_energy = LenticulumCore.GradedEnergy((fit = false, complexity = log(2) / 2, negentropy = -(1 + log(2pi)) / 2 - log(2) / 2)),
            pointwise_gap = "an exact relation has no finite pointwise energy",
            game_gap = "the specification defines the graph free energy")
        @test SO.local_scalar(f, msgs) ≈ -(1 + log(2pi)) / 2 atol = 1e-12
        @test_throws ArgumentError SO.local_scalar(f, merge(inputs, (z = π,)))
    end

    @testset "Float32 messages and graph marginals" begin
        pieces = (x = [3, 1], y = 2:3)
        indices = SO.ports(4, pieces)
        f = SplitFactor(:z, pieces; dim = 4)
        priors = (z = (μ = Float32[0.1, 0.2, 0.3, 0.4], Σ = Matrix{Float32}(I, 4, 4)),
                  x = (μ = Float32[0.3, 0.5], Σ = Matrix{Float32}(I, 2, 2)),
                  y = (μ = Float32[-0.1, 0.2], Σ = Matrix{Float32}(I, 2, 2)))
        input = map(p -> Gaussian(p.μ, p.Σ), priors)
        for target in keys(indices)
            others = Tuple(v for v in keys(indices) if v !== target)
            msg = SO.message(f, target, NamedTuple{others}(map(v -> input[v], others)))
            @test msg isa ExtendedGaussian{Float32}
            @test size(msg.D, 2) == (target === :z ? 1 : 0)
        end
        marg, _ = SO.run_tree(SO.graph(f, indices, priors))
        for (v, ix) in pairs(indices)
            @test eltype(belief_mean(marg[v])) === Float32
            @test eltype(belief_cov(marg[v])) === Float32
            @test extended(marg[v], length(ix)) isa ExtendedGaussian{Float32}
        end
    end
end

# Regression tests of the review of U2 (Codex, 2026-11-20; LP-069), from the reviewer's probes.
module SplitReviewProbes
using Lenticulum, LenticulumCore, Mycelium, LinearAlgebra, LuxCore, Random
using LenticulumCore: Unobserved, Observed, Latent

# A normalised prior on the line z = (T, T), T ~ N(0, 1): an extended Gaussian with a hard
# direction oblique to the coordinates (finding 8).
struct LinePrior <: LenticulumCore.AbstractLenticulumFactor end
const line = ExtendedGaussian(zeros(2), ones(2, 2))
const axis = [1.0, 1.0] / sqrt(2)
LenticulumCore.channels(::LinePrior) = (LenticulumCore.Channel(:out, 2),)
LenticulumCore.supports_polarity(::LinePrior, p::LenticulumCore.Polarity) = keys(p) == (:out,) && p[:out] isa Unobserved
LenticulumCore.islearnable(::LinePrior) = false
LuxCore.initialparameters(::AbstractRNG, ::LinePrior) = (;)
LuxCore.initialstates(::AbstractRNG, ::LinePrior) = (;)
Mycelium.factor_message(::LinePrior, ::Symbol, _, _, _, ps, st) = (line, st)
Mycelium.exact_messages(::LinePrior) = true
Mycelium.uses_cavity(::LinePrior) = true
# −log of the prior's density on the line in arc length s = √2 T (variance 2), minus the entropy
function Mycelium.local_free_energy(::LinePrior, msgs, ps, st)
    q = Mycelium.combine(line, get(msgs, :out, nothing))
    U = log(4π) / 2 + (dot(axis, q.μ)^2 + dot(axis, q.Σ * axis)) / 4
    return U - Mycelium.variable_entropy(q), st
end

function free_energy(g)
    rng = Xoshiro(0)
    ps, st = LuxCore.initialparameters(rng, g), LuxCore.initialstates(rng, g)
    marg, rep, _, s = infer!(g, tree_schedule(g), ps, st)
    return rep.converged, first(LenticulumCore.scalar_free_energy(s, g, ps, st)), marg
end
end # module SplitReviewProbes

@testset "SplitFactor: the findings of its review (LP-069)" begin
    RP = SplitReviewProbes
    msg(f, target, inputs) = first(Mycelium.factor_message(f, target, nothing, inputs, nothing, (;), (;)))

    # F2: a true variance of 1e-16 against a whole of scale 1 is kept, not cut to a point
    v = [1e-8, 1.0]
    m = msg(SplitFactor(:z, (x = [1],); dim = 2), :x, (z = ExtendedGaussian(zeros(2), v * v'),))
    @test m.Σ[1, 1] ≈ 1e-16 rtol = 1e-6
    @test size(hard_directions(m), 2) == 0

    # F3: two proper Float32 pieces with variances 1 and 1e6 give a proper Float32 whole, and the
    # left-out input of the whole does not set the type
    f = SplitFactor(:z, (x = [1], y = [2]))
    gx, gy = Gaussian(Float32[0], fill(1.0f0, 1, 1)), Gaussian(Float32[0], fill(1.0f6, 1, 1))
    for inputs in ((x = gx, y = gy), (x = gx, y = gy, z = DiracBelief([0.0, 0.0])))
        m = msg(f, :z, inputs)
        @test m isa ExtendedGaussian{Float32}
        @test size(m.D, 2) == 0 && m.Σ ≈ Diagonal(Float32[1, 1e6])
    end
    function two_priors(var)
        gb = GraphBuilder()
        factor!(gb, :split, f)
        variable!(gb, :z, 2); variable!(gb, :x, 1); variable!(gb, :y, 1)
        for n in (:z, :x, :y); connect!(gb, :split, n, n); end
        for (n, v) in ((:x, 1.0f0), (:y, var))
            factor!(gb, Symbol(:prior_, n), GaussianPrior(:out, Float32[0], fill(v, 1, 1)))
            connect!(gb, Symbol(:prior_, n), :out, n)
        end
        return RP.free_energy(validate(build(gb)))
    end
    conv, F, _ = two_priors(1.0f4)
    @test conv && abs(F) < 1e-4             # two normalised priors: Z = 1
    # A known limit of LenticulumCore, not of the factor: a Float32 extended Gaussian takes a
    # variance below 32·eps(Float32) ≈ 3.8e-6 of the largest for a hard direction, so the entropy of
    # q = diag(1, 1e6) drops the unit direction and F is (1 + log 2π)/2 instead of 0.
    conv, F, _ = two_priors(1.0f6)
    @test conv
    @test_broken abs(F) < 1e-4

    # F5: a channel named by the empty symbol takes part in the free energy
    for f5 in (SplitFactor(Symbol(""), (x = [1],)), SplitFactor(:z, (var"" = [1],)))
        name = f5.whole === Symbol("") ? Symbol("") : :z
        F5, _ = Mycelium.local_free_energy(f5, NamedTuple{(name,)}((Gaussian([0.0], fill(1.0, 1, 1)),)), (;), (;))
        @test LenticulumCore.scalarise(LenticulumCore.scalarisation(f5), F5) ≈ -(1 + log(2π)) / 2
    end

    # F7: every constructor checks, and a polarity names only the three roles
    @test_throws ArgumentError SplitFactor(:z, (z = [1],), 1)
    @test_throws ArgumentError SplitFactor(:z, (x = [1, 1],), 2)
    @test_throws ArgumentError SplitFactor(:z, (x = [1],), -1)
    @test_throws ArgumentError SplitFactor(:z, (x = [true],))
    @test_throws ArgumentError SplitFactor(:z, (x = [1],); dim = true)
    @test_throws ArgumentError SplitFactor(:z, (x = [big(typemax(Int)) + 1],); dim = 2)
    p = LenticulumCore.Polarity((z = Unobserved(), x = nothing), (z = 1.0, x = 1.0))
    @test !LenticulumCore.supports_polarity(SplitFactor(:z, (x = [1],)), p)

    # F8: a prior on the line z = (T, T), x = z[1] clamped at 0: −log Z = −log N(0; 0, 1). The term
    # −H(q) gave log √2 more; the relation's term (HardConstraintFactor's) has the Jacobian
    gb = GraphBuilder()
    factor!(gb, :prior, RP.LinePrior())
    factor!(gb, :split, SplitFactor(:z, (x = [1],); dim = 2))
    factor!(gb, :data, DataFactor(:out, [0.0]))
    variable!(gb, :z, 2); variable!(gb, :x, 1)
    connect!(gb, :prior, :out, :z); connect!(gb, :split, :z, :z); connect!(gb, :split, :x, :x)
    connect!(gb, :data, :out, :x; direction = Emitting())
    conv, F, _ = RP.free_energy(validate(build(gb)))
    @test conv && F ≈ log(2π) / 2
end
