# I30: the graph-wide trees out of Zygote's signatures (`MyceliumLearning/src/gradient_couplings.md`,
# "The graph-wide trees out of Zygote's signatures (I30)"). Needs `_i1_chain` (`structure_compile.jl`).
# The values against the old code: `coupled_keys.jl` and `coupled_spec.jl` keep the old functions as
# oracles; the 39 records of the I30 baseline (`scratch/i30/` of the work package) matched except
# where stated in the note.
using ADTypes, DifferentiationInterface, ForwardDiff, Zygote

const _I30_LC = LenticulumCore
const _I30_ML = MyceliumLearning

# Zygote's pullback of the replay's glue, on a chain of `T` steps: its type, and the gradient.
function _i30_pullback(T)
    g, ps, st = _i1_chain(T)
    ts = _I30_ML._realised_tasks(g, tree_schedule(g), 1, ps, st)
    tape, s, sr = _I30_ML._reference_tape(g, ts, ps, st, nothing)
    frozen = _I30_ML._frozen_beliefs(s, g, ps, sr)
    θ, offs = _I30_LC._flatten(ps), _I30_LC.offset_tree(ps)
    ob, lb, sb = _I30_LC._Erased(offs), _I30_LC._Erased(ps), _I30_LC._Erased(st)
    loss = θ -> _I30_ML._coupled_loss(_I30_LC._unflatten_erased(θ, ob, lb), g, ts, tape, sb, AutoZygote(), 0.0, frozen)
    _, back = Zygote.pullback(loss, θ)
    reference = _I30_LC._flatten(free_energy_gradient(MessageStore(g), g, ps, st; ad = AutoZygote(), schedule = tree_schedule(g)))
    return typeof(back), only(back(1.0)), reference
end

@testset "I30: the replay's Pullback type does not depend on the structure" begin
    (B3, d3, r3), (B4, d4, r4) = _i30_pullback(3), _i30_pullback(4)
    @test B3 === B4
    @test d3 == r3 && d4 == r4
end

# Codex's probe of the design's step 3 (`gradient_couplings.md`): a key with an active `Float32` and a
# frozen integer above 2^24, and a key frozen whole. The old per-key pinning rounded the frozen value to
# `Float32` (16_777_216), so the loss was constant in `active`; the graph-wide pinning keeps 16_777_217.
struct _I30Source <: AbstractLenticulumFactor end
LenticulumCore.channels(::_I30Source) = (LenticulumCore.Channel(:x, nothing),)
LenticulumCore.supported_polarities(::_I30Source) = (Polarity(x = Unobserved()),)
LenticulumCore.supports_polarity(f::_I30Source, p::Polarity) = p in LenticulumCore.supported_polarities(f)
LenticulumCore.islearnable(::_I30Source) = true
Mycelium.factor_message(::_I30Source, t::Symbol, pol, inputs, prior, ps, st) = (GaussianBelief([0.0], ones(1, 1)), st)
Mycelium.local_free_energy(::_I30Source, beliefs, ps, st) =
    (hasproperty(ps, :frozen) ? (ps.frozen - 16_777_216) * ps.active : zero(ps.active), st)

@testset "I30: pinning in flat coordinates" begin
    b = GraphBuilder()
    for (key, freeze) in ((:mixed, (:frozen,)), (:all_frozen, :all))
        variable!(b, key)
        factor!(b, key, _I30Source(); freeze)
        connect!(b, key, :x, key; direction = Emitting(), coupling = ExactCoupling())
    end
    g = build(b)
    ps = (mixed = (active = 2f0, frozen = 16_777_217), all_frozen = (active = 1.0,))
    sched = forward_backward_schedule(g)
    for ad in (AutoZygote(), AutoForwardDiff(), nothing)
        G = _I30_LC._flatten(free_energy_gradient(MessageStore(g), g, ps, (;); ad, schedule = sched))
        @test G[1] ≈ 1 atol = 1e-8            # the deliberate change: 0 under the per-key pinning
        @test iszero(G[2]) && iszero(G[3])    # frozen coordinates
    end
    # one element type: a frozen coordinate under ForwardDiff is a dual with zero partials
    pin = _I30_ML._graph_pin(g, ps)
    θ = _I30_LC._flatten(ps)
    D = ForwardDiff.Dual{:i30}.(θ, 1.0)
    p = pin(D)
    @test eltype(p) === eltype(D)
    @test ForwardDiff.partials.(p[2:3]) == [ForwardDiff.Partials((0.0,)), ForwardDiff.Partials((0.0,))]
    @test ForwardDiff.value.(p) == θ
    # nothing frozen: no pinning at all
    g2, ps2, _ = _i1_chain(2)
    @test _I30_ML._graph_pin(g2, ps2) === identity
end

@testset "I30: the box" begin
    X = Base.get_extension(LenticulumCore, :LenticulumCoreForwardDiffDIExt)
    # the guard of the fixed ForwardDiff tag sees a caller's dual inside a box
    d = ForwardDiff.Dual(0.3, 1.0)
    @test X._has_dual((_I30_LC._Erased((a = [d],)),), IdDict{Any,Nothing}())
    @test !X._has_dual((_I30_LC._Erased((a = [0.3],)),), IdDict{Any,Nothing}())
    @test _I30_LC._graph_ad(AutoForwardDiff(), zeros(2), (_I30_LC._Erased((a = d,)),)) === AutoForwardDiff()
    # no generic tree function accepts a box: a path that forgets to unbox fails
    @test_throws MethodError _I30_LC._flatten(_I30_LC._Erased((a = [1.0],)))
    # the rule of `_unflatten_erased` takes every form of the box's cotangent
    ps = (a = [1.0, 2.0], b = (c = 3.0,))
    θ, offs = _I30_LC._flatten(ps), _I30_LC.offset_tree(ps)
    gz = only(Zygote.gradient(t -> (q = _I30_LC._unflatten_erased(t, _I30_LC._Erased(offs), _I30_LC._Erased(ps)).tree;
        sum(abs2, q.a) + q.b.c), θ))
    @test gz == [2.0, 4.0, 1.0]
    # the rules of the reads give the primal's values on a box of a tree that is not a `NamedTuple`
    # (GLM's review: the state's fallback read `_subtrees`, wrong for an `absent` other than `NamedTuple()`)
    rc = Zygote.ChainRulesCore
    g, _, _ = _i1_chain(1)
    nodes = g.factors
    @test first(rc.rrule(Mycelium._states, _I30_LC._Erased(nothing), nodes, :absent)) == Mycelium._states(nothing, nodes, :absent)
    @test first(rc.rrule(Mycelium._states, _I30_LC._Erased(nothing), nodes)) == Mycelium._states(nothing, nodes)
end
