# P29: a NonlinearFactor's difference Jacobian has the rank its accuracy supports (nonlinear.md,
# "The rank of a difference Jacobian"). Singular values below the difference error are set to
# zero, so the rank tests of the messages see the directions f does not depend on.
#
# Oracles: the exact Jacobian of f(x) = g(W x + c) with W of rank r, written out by hand; its rank
# (the eps-relative test of the messages) against that of the cleaned difference Jacobian; the
# backward message's flat directions against null(W); and the number of evaluations of f.

using ForwardDiff

module P29Rank
using LinearAlgebra, Random

rk(A) = (s = svdvals(A); isempty(s) || s[1] == 0 ? 0 : count(>(maximum(size(A)) * eps() * s[1]), s))

function random_W(rng, nout, nin, r)
    r == min(nout, nin) && return randn(rng, nout, nin)
    return randn(rng, nout, r) * randn(rng, r, nin)
end

# three smooth maps of z = W x + c and their Jacobians
maps(W, c) = (
    (x -> tanh.(W * x .+ c), x -> Diagonal(1 .- tanh.(W * x .+ c) .^ 2) * W),
    (x -> 5 .* (W * x .+ c), x -> 5 .* W),
    (x -> sin.(W * x .+ c) .+ 0.1 .* (W * x .+ c) .^ 2, x -> Diagonal(cos.(W * x .+ c) .+ 0.2 .* (W * x .+ c)) * W),
)
end # module

@testset "P29: the rank of a difference Jacobian" begin
    PR = P29Rank
    rng = Xoshiro(2901)
    n = 0; nsame = 0; ndef = 0
    for trial in 1:400
        nin, nout = rand(rng, 1:4), rand(rng, 1:4)
        r = rand(rng, 1:min(nin, nout))
        W, c, x0 = PR.random_W(rng, nout, nin, r) .* 10.0^rand(rng, -1:1), randn(rng, nout), randn(rng, nin)
        for (fx, jx) in PR.maps(W, c)
            Aex = jx(x0)
            s = svdvals(Aex)
            # only instances whose singular values are all far from the difference error, about
            # 1e-9 here (|f| ≤ 1 for tanh): a saturated tanh is the known limit below
            any(σ -> 1e-12 * s[1] < σ < 1e-5 * s[1], s) && continue
            any(<(1e-6), s[1:PR.rk(Aex)]) && continue
            f = NonlinearFactor(fx, nin => nout; noise = 0.3)
            J = Lenticulum._jacobian(f, x0)
            n += 1; ndef += PR.rk(Aex) < nin
            nsame += PR.rk(J) == PR.rk(Aex)
            @test PR.rk(J) == PR.rk(Aex)
            @test J ≈ Aex atol = 1e-5 * (1 + opnorm(Aex))
        end
    end
    @test n > 600 && ndef > 200
    @test nsame == n
end

@testset "P29: a backward message is flat along null(∂f/∂x)" begin
    rng = Xoshiro(2902)
    nflat = 0
    for trial in 1:50
        nin, nout = rand(rng, 2:4), rand(rng, 1:4)
        r = rand(rng, 1:min(nin - 1, nout))                  # rank below nin: flat directions
        W, c, x0 = P29Rank.random_W(rng, nout, nin, r), randn(rng, nout), randn(rng, nin)
        fx, jx = first(P29Rank.maps(W, c))
        A = jx(x0)
        f = NonlinearFactor(fx, nin => nout; noise = 0.3)
        prior = GaussianBelief(x0, Matrix(1.0I, nin, nin))     # expansion point x0 (η = Λ x0)
        y = GaussianBelief(randn(rng, nout), Matrix(2.0I, nout, nout))
        msg = first(Mycelium.factor_message(f, :x, nothing, (y = y,), prior, NamedTuple(), NamedTuple()))
        @test msg isa ExtendedGaussian
        msg isa ExtendedGaussian || continue
        N = nullspace(A)
        @test size(msg.D, 2) == size(N, 2)
        @test norm(A * msg.D) <= 1e-8 * (1 + opnorm(A))   # the flat directions are those of null(A)
        # against the exact Jacobian: the same message
        g = NonlinearFactor(fx, nin => nout; noise = 0.3, jacobian = jx)
        ex = first(Mycelium.factor_message(g, :x, nothing, (y = y,), prior, NamedTuple(), NamedTuple()))
        @test ex isa ExtendedGaussian
        @test GaussianBelief(msg).Λ ≈ GaussianBelief(ex).Λ atol = 1e-5 rtol = 1e-5
        @test GaussianBelief(msg).η ≈ GaussianBelief(ex).η atol = 1e-5 rtol = 1e-5
        nflat += 1
    end
    @test nflat == 50
end

@testset "P29: a well-conditioned Jacobian costs 2nin evaluations of f" begin
    calls = Ref(0)
    W = [1.0 0.2 0.0; 0.1 1.0 0.3]
    f = NonlinearFactor(x -> (calls[] += 1; tanh.(W * x)), 3 => 2; noise = 0.1)
    Lenticulum._jacobian(f, [0.1, -0.2, 0.3])
    @test calls[] == 6
    calls[] = 0
    g = NonlinearFactor(x -> (calls[] += 1; tanh.(W[[1, 1], :] * x)), 3 => 2; noise = 0.1)   # rank 1
    J = Lenticulum._jacobian(g, [0.1, -0.2, 0.3])
    @test calls[] == 12                                      # the second step, for the error estimate
    @test P29Rank.rk(J) == 1
end

@testset "P29: known limit, a Jacobian below the difference error" begin
    # A saturated tanh: the exact Jacobian is about 1e-12 (rank 1), below what central differences
    # resolve. The cleaned Jacobian is cut to zero: it never claims a higher rank, but it says f
    # is constant here, which it is not quite.
    W = [30.0 0.0; 0.0 30.0]
    f = NonlinearFactor(x -> tanh.(W * x), 2 => 2; noise = 0.1)
    x0 = [1.0, 0.0]
    J = Lenticulum._jacobian(f, x0)
    @test P29Rank.rk(J) <= 2
    @test J[:, 2] ≈ [0.0, 30.0] rtol = 1e-6                 # the unsaturated direction is kept
    @test P29Rank.rk(J) == 1
end

@testset "P29: ForwardDiff through a cleaned Jacobian" begin
    # the forward message's mean as a function of the input mean, for a rank-deficient W;
    # the dual derivative against central differences of the value
    W = [1.0 2.0; 0.5 1.0]                                   # rank 1
    f = NonlinearFactor(x -> sin.(W * x), 2 => 2; noise = 0.2)
    S = [0.3 0.1; 0.1 0.2]
    meanof(m) = (msg = first(Mycelium.factor_message(f, :y, nothing, (x = GaussianBelief(S \ m, inv(S)),),
                                                     nothing, NamedTuple(), NamedTuple()));
                 msg.Λ \ msg.η)
    m = [0.2, -0.1]
    Jd = ForwardDiff.jacobian(meanof, m)
    h = 1e-5
    Jc = reduce(hcat, ((meanof(m .+ h .* e) .- meanof(m .- h .* e)) ./ (2h) for e in eachcol(Matrix(1.0I, 2, 2))))
    @test Jd ≈ Jc atol = 1e-4 rtol = 1e-4
end

@testset "P29 review: a rank that changes at the expansion point keeps its derivative" begin
    # f(x) = x.^2: J = 2 Diagonal(x), rank 1 at [0, 1], rank 0 at [0, 0]. Along x0 + t v the
    # derivative of J is 2 Diagonal(v), also in the directions the cut removes at t = 0.
    f = NonlinearFactor(x -> x .^ 2, 2 => 2; noise = 0.1)
    v = [1.0, 2.0]
    for x0 in ([0.0, 1.0], [0.0, 0.0])
        dJ = ForwardDiff.derivative(t -> vec(Lenticulum._jacobian(f, x0 .+ t .* v)), 0.0)
        @test reshape(dJ, 2, 2) ≈ [2.0 0.0; 0.0 4.0] atol = 1e-6
    end
end

@testset "P29 review: the error bound is per entry, not global" begin
    # rows of very different size, or steps of very different size: the small singular value is
    # resolvable from the accuracy of its own entries
    for (g, x, s2) in ((x -> [1e-3 * x[1], 1e7 + x[2]], [0.5, 0.5], 1e-3),
                       (x -> [1e-3 * x[1], x[2]], [0.0, 1e6], 1e-3))
        f = NonlinearFactor(g, 2 => 2; noise = 0.1)
        J = Lenticulum._jacobian(f, x)
        @test P29Rank.rk(J) == 2
        @test svdvals(J)[2] ≈ s2 rtol = 1e-3
    end
end

@testset "P29 review: a non-finite f gives its non-finite Jacobian, no error" begin
    f = NonlinearFactor(y -> 1 ./ y, 2 => 2; noise = 0.1)
    @test !all(isfinite, Lenticulum._jacobian(f, [0.0, 1.0]))
    g = NonlinearFactor(y -> [NaN, y[1]], 1 => 2; noise = 0.1)
    @test !all(isfinite, Lenticulum._jacobian(g, [0.3]))
end

@testset "I17: the rank tolerance follows the element type (Float32 maps)" begin
    # a Float32 map that is singular up to Float32 rounding: eps(Float64) would call it full rank
    A32 = Float32[1 2; 2 4]
    s32 = svdvals(A32)
    @test rank(A32) == 1 && s32[2] < 2 * eps(Float32) * s32[1]
    @test Lenticulum._rank_deficient_svd(A32) && Lenticulum._rank_deficient(A32)
    # the same rounding-level singular value in Float64 arithmetic is real information
    A64 = Float64[1 2; 2 4 + 2.6e-8]
    s64 = svdvals(A64)
    @test s64[2] > 2 * eps(Float64) * s64[1]
    @test !Lenticulum._rank_deficient(A64)
    # a full-rank Float32 map stays full rank
    @test !Lenticulum._rank_deficient(Float32[1 0; 0 1]) && !Lenticulum._rank_deficient(Float32[1 2; 3 4])
    # rectangular: more columns than rows is always deficient
    @test Lenticulum._rank_deficient(Float32[1 2 3; 4 5 6])
    # the public no-input forward message: with flat x and y = A x + noise, the output stays
    # informative orthogonal to range(A) — for v = [-2, 1]/sqrt(5), v'y carries the noise variance
    f = GaussianFactor(2 => 2)
    ps = (A = A32, b = zeros(Float32, 2))
    msg, _ = Mycelium.factor_message(f, :y, nothing, (x = TrivialBelief(),), nothing, ps, (;))
    @test !(msg isa TrivialBelief)
    g = msg isa LenticulumCore.ExtendedGaussian ? GaussianBelief(msg) : msg
    v = [-2.0, 1.0] / sqrt(5)
    @test v' * Matrix(g.Λ) * v ≈ 1 / 0.1 rtol = 1e-5            # noise = 0.1, so Q = 0.1 I
    # the Float64-exact singular map takes the same path (it was never misclassified)
    msg64, _ = Mycelium.factor_message(f, :y, nothing, (x = TrivialBelief(),), nothing,
                                      (A = Float64[1 2; 2 4], b = zeros(2)), (;))
    @test !(msg64 isa TrivialBelief)
end
