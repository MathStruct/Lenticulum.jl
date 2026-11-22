# I1: compilation per structure in the AD paths (`Mycelium/src/passing.md`, "Compilation per
# structure"). Needs DifferentiationInterface, ForwardDiff and Zygote loaded (`parameter_ad.jl`).
using LenticulumCore: _flatten, _unflatten, offset_tree, _eltype, _graph_ad, _graph_parameter_gradient,
    _parameter_gradient
const ChainRulesCore = Zygote.ChainRulesCore
struct _I1Box; x::Any; end
mutable struct _I1Cycle; next::Any; v::Any; end

# The recursion `offset_tree` was before I1, as the oracle.
_old_offsets(x::AbstractArray{<:Number}, i) = (i, i + length(x))
_old_offsets(x::Number, i) = (i, i + 1)
_old_seq(::Tuple{}, i) = ((), i)
function _old_seq(xs::Tuple, i)
    o, j = _old_offsets(first(xs), i)
    rest, k = _old_seq(Base.tail(xs), j)
    return (o, rest...), k
end
function _old_offsets(x::Union{NamedTuple,Tuple}, i)
    r, j = _old_seq(values(x), i)
    return (x isa NamedTuple ? NamedTuple{keys(x)}(r) : r), j
end

# A random tree: nested NamedTuples and Tuples, arrays, scalars and empty subtrees.
function _random_tree(rng, depth, T = Float64)
    n = rand(rng, 0:4)
    leaf() = rand(rng, 1:4) == 1 ? T(randn(rng)) : rand(rng, Bool) ? randn(rng, T, rand(rng, 0:3)) :
        randn(rng, T, rand(rng, 1:2), rand(rng, 1:3))
    vals = Tuple(depth > 0 && rand(rng, 1:3) == 1 ? _random_tree(rng, depth - 1, T) : leaf() for _ in 1:n)
    rand(rng, Bool) && return vals
    return NamedTuple{Tuple(Symbol(:k, i) for i in 1:n)}(vals)
end

@testset "I1: offset_tree, _eltype and _flatten against the old recursion" begin
    rng = Xoshiro(11)
    for trial in 1:200
        T = trial % 3 == 0 ? Float32 : Float64
        x = _random_tree(rng, 3, T)
        o = offset_tree(x)
        @test o === first(_old_offsets(x, 1))
        @test typeof(o) === typeof(first(_old_offsets(x, 1)))
        θ = _flatten(x)
        @test _unflatten(θ, o, x) == x
    end
    @test offset_tree((;)) === (;)
    @test offset_tree(()) === ()
    @test _eltype((;)) === Union{}
    @test _eltype((a = Float32[1], b = (c = 2.0,))) === Float64
    @test _eltype((a = (;), b = (Float32[1],))) === Float32
    # A graph-wide tree, one field per factor.
    big = NamedTuple{Tuple(Symbol(:f, i) for i in 1:120)}(Tuple(isodd(i) ? (A = ones(2, 2), b = zeros(2)) : (;) for i in 1:120))
    @test offset_tree(big) === first(_old_offsets(big, 1))
    @test length(_flatten(big)) == 60 * 6
end

_i1_loss(t::Number) = t^2 + sin(t)
_i1_loss(t::AbstractArray) = sum(abs2, t) + sum(t[1:2:end])
_i1_loss(t::Union{Tuple,NamedTuple}) = isodd(length(t)) && length(t) > 1 ? _i1_loss(t[1]) :
    isempty(t) ? 0.0 : sum(map(_i1_loss, values(t)))

@testset "I1: the rule of _unflatten" begin
    rng = Xoshiro(12)
    for trial in 1:60
        x = _random_tree(rng, 3)
        θ = _flatten(x)
        isempty(θ) && continue
        o = offset_tree(x)
        # A loss that reads every other entry twice, so that cotangents are accumulated, and
        # skips the subtrees of odd length, so that some are missing.
        lossθ(θ) = _i1_loss(_unflatten(θ, o, x))
        gz = only(Zygote.gradient(lossθ, θ))
        gf = ForwardDiff.gradient(lossθ, θ)
        @test gz === nothing ? iszero(gf) : gz ≈ gf   # `nothing`: a loss that reads no leaf
    end
    # Cotangents with fields left out, a zero, a thunk, and a scalar leaf.
    x = (a = [1.0, 2.0], b = (c = 3.0, d = [4.0 5.0]), e = (;))
    θ, o = _flatten(x), offset_tree(x)
    y, back = ChainRulesCore.rrule(_unflatten, θ, o, x)
    @test y == x
    Δ = ChainRulesCore.Tangent{typeof(x)}(; b = ChainRulesCore.Tangent{typeof(x.b)}(; c = 7.0))
    @test back(Δ)[2] == [0, 0, 7, 0, 0]
    @test back(ChainRulesCore.ZeroTangent())[2] isa ChainRulesCore.AbstractZero
    @test back(ChainRulesCore.@thunk((a = [1.0, 1.0], b = nothing, e = nothing)))[2] == [1, 1, 0, 0, 0]
    @test back((a = nothing, b = (c = nothing, d = [2.0 3.0]), e = nothing))[2] == [0, 0, 0, 2, 3]
    # A top-level tuple, and Float32 coordinates.
    x32 = (Float32[1, 2], (Float32(3),))
    θ32 = _flatten(x32)
    g32 = only(Zygote.gradient(θ -> sum(abs2, _unflatten(θ, offset_tree(x32), x32)[1]) + _unflatten(θ, offset_tree(x32), x32)[2][1], θ32))
    @test g32 == Float32[2, 4, 1]
    @test eltype(g32) === Float32
end

@testset "I1: review of the rule (Codex findings 3 and 5)" begin
    # 3: the cotangent's element type is θ's promoted with the cotangents', as the plain pullbacks give.
    x = (a = Float32[1, 2], b = (c = 3.0f0,))
    θ, o = _flatten(x), offset_tree(x)
    _, back = ChainRulesCore.rrule(_unflatten, θ, o, x)
    d = back((a = [1.0, 2.0], b = (c = 0.5,)))[2]
    @test eltype(d) === Float64 && d == [1, 2, 0.5]
    @test eltype(back((a = Float32[1, 1], b = nothing))[2]) === Float32
    xr = (a = [1 // 2],)
    _, backr = ChainRulesCore.rrule(_unflatten, _flatten(xr), offset_tree(xr), xr)
    @test backr((a = [big(1) // big(10)^400],))[2] == [big(1) // big(10)^400]
    # 5: a cotangent with the tree's fields in another order, or some of them, is read by name.
    y = (p = [1.0], q = [2.0], r = [3.0])
    _, backy = ChainRulesCore.rrule(_unflatten, _flatten(y), offset_tree(y), y)
    @test backy((r = [30.0], p = [10.0], q = [20.0]))[2] == [10, 20, 30]
    @test backy(ChainRulesCore.Tangent{typeof(y)}(; r = [3.0]))[2] == [0, 0, 3]
    @test backy((p = [1.0], q = nothing, r = [3.0]))[2] == [1, 0, 3]
    # Forward over reverse: ForwardDiff through a Zygote gradient gives dual cotangents for plain
    # coordinates (VariationalDiffusion's mixed derivatives failed on a `Float64` buffer).
    z = (w = [0.5, -1.0], v = (s = 2.0,))
    θz, oz = _flatten(z), offset_tree(z)
    gt(t) = only(Zygote.gradient(θ -> t * _i1_loss(_unflatten(θ, oz, z)) + t^2 * sum(θ), θz))
    @test ForwardDiff.derivative(gt, 1.5) ≈ (gt(1.5 + 1e-6) - gt(1.5 - 1e-6)) / 2e-6 rtol = 1e-6
end

# Codex finding 2 (review of I1), not fixed and before I1 as well (LP-076): under a caller's
# ForwardDiff, a loss that does not depend on the parameters leaves the gradient's coordinates
# after the first unwritten (ForwardDiff writes as many as the caller's dual has partials), e.g.
# `ForwardDiff.derivative(y -> _flatten(_parameter_gradient(AutoForwardDiff(), (p, y) -> y,
# (a = [0.7, -0.2, 0.1],), y)), 0.3)`. Not a test: the memory is often zero by chance.

@testset "I1: one dual type for the graph drivers" begin
    X = Base.get_extension(LenticulumCore, :LenticulumCoreForwardDiffDIExt)
    @test X !== nothing
    tree(n) = NamedTuple{Tuple(Symbol(:f, i) for i in 1:n)}(Tuple((A = ones(2, 2), b = zeros(2)) for _ in 1:n))
    a1, a2 = _graph_ad(AutoForwardDiff(), zeros(30), (tree(5), (;))), _graph_ad(AutoForwardDiff(), zeros(60), (tree(10), (s = 1,)))
    @test typeof(a1) === typeof(a2)
    @test a1.tag === ForwardDiff.Tag(X._GraphTag(), Float64)
    @test a1 isa AutoForwardDiff{12}
    @test _graph_ad(AutoForwardDiff(), zeros(5), ()) isa AutoForwardDiff{5}
    @test _graph_ad(AutoForwardDiff(), zeros(0), ()) isa AutoForwardDiff{1}
    @test _graph_ad(AutoForwardDiff(chunksize = 3), zeros(30), ()) isa AutoForwardDiff{3}
    @test _graph_ad(AutoForwardDiff(), zeros(Float32, 30), ()).tag === ForwardDiff.Tag(X._GraphTag(), Float32)
    # Left as it was: a tag of the caller's, coordinates that are not plain floats, other backends.
    tagged = AutoForwardDiff(tag = ForwardDiff.Tag(sin, Float64))
    @test _graph_ad(tagged, zeros(30), ()) === tagged
    @test _graph_ad(AutoForwardDiff(), zeros(BigFloat, 3), ()) === AutoForwardDiff()
    @test _graph_ad(AutoZygote(), zeros(30), ()) === AutoZygote()
    @test _graph_ad(nothing, zeros(30), ()) === nothing
    # A dual anywhere in the contexts: a dual coordinate, an array of duals, a field of a struct,
    # a `Vector{Any}`, a `Dict`, a closure, behind a cycle.
    d = ForwardDiff.Dual{ForwardDiff.Tag{typeof(cos),Float64}}(1.0, 1.0)
    c = _I1Cycle(nothing, d); c.next = c
    for ctx in ((d,), ([d, d],), (_I1Box((p = 1, q = [d])),), (Any[1.0, Any[d]],), (Dict(:a => d),),
                ((let y = d; t -> t * y; end),), (c,), (Ref(d),), ((a = 1.0, b = (c = [d],)),))
        @test _graph_ad(AutoForwardDiff(), zeros(30), ctx) === AutoForwardDiff()
    end
    # and none: the same contexts with plain numbers, and objects the walk skips.
    c2 = _I1Cycle(nothing, 1.0); c2.next = c2
    for ctx in ((1.0,), ([1.0],), (_I1Box((p = 1, q = [2.0])),), (Any[1.0, Any[:a]],), (Dict(:a => 1.0),),
                (c2,), (ReentrantLock(), sin, Float64, "s", stdout))
        @test _graph_ad(AutoForwardDiff(), zeros(30), ctx) !== AutoForwardDiff()
    end
end

@testset "I1: graph gradients equal the plain ones, and nest under a caller's ForwardDiff" begin
    rng = Xoshiro(13)
    ps = (f1 = (A = randn(rng, 2, 2), b = randn(rng, 2)), f2 = (;), f3 = (w = randn(rng, 5),), f4 = (s = 0.3,))
    loss(p, c, M) = c * sum(abs2, M * p.f1.A * p.f1.b) + sum(sin, p.f3.w) * p.f4.s + c^2 * p.f4.s
    M = randn(rng, 2, 2)
    for ad in (AutoForwardDiff(), AutoZygote(), nothing)
        g0 = _parameter_gradient(ad, loss, ps, 1.7, M)
        g1 = _graph_parameter_gradient(ad, loss, ps, 1.7, M)
        @test _flatten(g1) ≈ _flatten(g0) rtol = (ad === nothing ? 1e-6 : 1e-12)
        @test typeof(g1) === typeof(g0)
    end
    # The graph tag is created above, before the caller's tag below; the caller's dual is in a
    # context, so the graph call keeps the default tag and the nesting is right.
    dgdc(c) = _flatten(_graph_parameter_gradient(AutoForwardDiff(), loss, ps, c, M))
    D = ForwardDiff.derivative(dgdc, 1.7)
    h = 1e-6
    @test D ≈ (dgdc(1.7 + h) - dgdc(1.7 - h)) / 2h rtol = 1e-6
    # and with the dual inside the parameters themselves.
    dgdA(t) = _flatten(_graph_parameter_gradient(AutoForwardDiff(), loss, merge(ps, (f4 = (s = t,),)), 1.7, M))
    @test ForwardDiff.derivative(dgdA, 0.3) ≈ (dgdA(0.3 + h) - dgdA(0.3 - h)) / 2h rtol = 1e-6
end
