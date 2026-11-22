using LinearAlgebra

# A known static sweep on one message per direction. This makes the Anderson update
# and a rejected extrapolation predictable before running any inference.
struct AndersonAffineSweep{F} <: AbstractSchedule
    update::F
end
function Mycelium.sweep!(s::MessageStore, g::FactorGraph, sched::AndersonAffineSweep, ps, st;
                         damping::Real = 0.0)
    s.iteration += 1
    residual = 0.0
    for msgs in (s.to_factor, s.to_variable)
        old = msgs[1].belief
        new = sched.update(old)
        residual = max(residual, belief_distance(old, new))
        msgs[1] = Message(new, s.iteration)
    end
    return residual, st
end

@testset "Anderson mixing and safeguards" begin
    g = chain_graph()
    nt = (;)
    canonical(η, Λ, improper = false) = LenticulumCore.GaussianBelief(η, Λ, improper)
    function seed(b)
        store = MessageStore(g)
        store.to_factor[1] = store.to_variable[1] = Message(b, 0)
        return store
    end

    @test_throws ArgumentError AndersonSchedule(ResidualSchedule())
    @test_throws ArgumentError AndersonSchedule(tree_schedule(g); m = 0)
    @test_throws ArgumentError AndersonSchedule(tree_schedule(g); ridge = -1)

    Λ₀ = [4.0 0.6; 0.6 3.0]
    Λ★ = [2.0 -0.4; -0.4 1.0]
    η★ = [2.0, -3.0]
    # G(x) = (x + x★)/2 in canonical coordinates, so the second type-II
    # step cancels its one residual direction, independently of the scaling.
    inner = AndersonAffineSweep(b -> canonical((b.η + η★) / 2, (b.Λ + Λ★) / 2, b.improper))
    start = canonical(zeros(2), Λ₀, true)
    an = AndersonSchedule(inner; m = 1)
    s, other = seed(start), seed(start)
    r, state = sweep!(s, g, an, nt, nt)
    @test state === nt
    @test r ≈ belief_distance(start, inner.update(start))
    @test s.to_variable[1].belief.η ≈ η★ / 2
    r, _ = sweep!(s, g, an, nt, nt)
    @test r ≈ belief_distance(inner.update(start), inner.update(inner.update(start)))
    for msgs in (s.to_factor, s.to_variable)
        @test msgs[1].belief.η ≈ η★
        @test msgs[1].belief.Λ ≈ Λ★
        @test msgs[1].belief.improper
        @test msgs[1].iteration == 2
    end
    @test anderson_restarts(an, s) == 0
    sweep!(other, g, an, nt, nt)
    @test other.to_variable[1].belief.η ≈ η★ / 2
    # An unsupported message anywhere in the store invalidates the history.
    s.to_factor[2] = Message(TrivialBelief(), s.iteration)
    sweep!(s, g, an, nt, nt)
    s.to_factor[2] = nothing
    s.to_factor[1] = s.to_variable[1] = Message(start, s.iteration)
    sweep!(s, g, an, nt, nt)
    @test s.to_variable[1].belief.η ≈ η★ / 2
    sweep!(s, g, an, nt, nt)
    @test s.to_variable[1].belief.η ≈ η★

    # A new message invalidates the old history, even for the same store.
    s.to_factor[2] = Message(start, s.iteration)
    s.to_factor[1] = s.to_variable[1] = Message(start, s.iteration)
    sweep!(s, g, an, nt, nt)
    @test s.to_variable[1].belief.η ≈ η★ / 2
    sweep!(s, g, an, nt, nt)
    @test s.to_variable[1].belief.η ≈ η★
    sweep!(s, g, an, nt, nt)
    sweep!(s, g, an, nt, nt)
    @test s.to_variable[1].belief.Λ ≈ Λ★

    fp32 = AndersonAffineSweep(b -> canonical(b.η ./ 2f0 .+ 1f0, b.Λ))
    s, ref = (seed(canonical(Float32[0], fill(1f0, 1, 1))) for _ in 1:2)
    an32 = AndersonSchedule(fp32)
    for η in (1f0, 1.5f0, 1.75f0)
        @test sweep!(s, g, an32, nt, nt) == sweep!(ref, g, fp32, nt, nt)
        @test s.to_variable[1].belief.η == ref.to_variable[1].belief.η == [η]
        @test eltype(s.to_variable[1].belief.η) === Float32
    end

    # The scalar precision map p ↦ (p−1)/2 extrapolates to −1 on sweep 2.
    # Its first three unmixed images from 15 are 7, 3, 1, all admissible.
    plain = AndersonAffineSweep(b -> canonical(b.η, (b.Λ .- 1) / 2, b.improper))
    guarded = AndersonSchedule(plain)
    s = seed(canonical([0.0], fill(15.0, 1, 1)))
    for (p, restarts) in ((7, 0), (3, 1), (1, 1))
        sweep!(s, g, guarded, nt, nt)
        @test only(s.to_variable[1].belief.Λ) ≈ p
        @test anderson_restarts(guarded, s) == restarts
    end
    @test anderson_restarts(guarded, other) == 0
    @test !Mycelium._anderson_admissible([Inf, 1.0], [0.0, 1.0], [(:to_variable, 1, 1)])
end

@testset "Anderson and plain schedules on a Gaussian tree" begin
    nt = (;)
    for T in (Float64, Float32)
        b = GraphBuilder()
        variable!(b, :x); variable!(b, :y)
        factor!(b, :a, PriorFactor(:x, LenticulumCore.Gaussian(T[1], fill(T(2), 1, 1)), _ -> 0))
        factor!(b, :b, PriorFactor(:y, LenticulumCore.Gaussian(T[3], fill(T(1), 1, 1)), _ -> 0))
        factor!(b, :relay, RelayFactor(:x, :y))
        connect!(b, :a, :x, :x; direction = Emitting())
        connect!(b, :b, :y, :y; direction = Emitting())
        connect!(b, :relay, :x, :x); connect!(b, :relay, :y, :y)
        g = build(b)
        plain = tree_schedule(g)
        an = AndersonSchedule(plain)
        for schedule in (plain, an)
            marginals, report, _, _ = infer!(g, schedule, nt, nt; damping = 0.25, tol = 1e-8)
            @test report.converged
            for belief in (marginals.x, marginals.y)
                # x = y, so the density on the common coordinate is the product of the priors.
                @test only(LenticulumCore.belief_mean(belief)) ≈ T(7/3)
                @test only(LenticulumCore.belief_cov(belief)) ≈ T(2/3)
            end
        end
    end
end

@testset "Anderson on an empty store and at an exact fixed point" begin
    nt = (;)
    g = build(GraphBuilder())
    an = AndersonSchedule(SequentialSchedule(MessageTask[]))
    s = MessageStore(g)
    for _ in 1:3
        @test sweep!(s, g, an, nt, nt) == (0.0, nt)
    end
    @test anderson_restarts(an, s) == 0

    # With ridge zero, an exact fixed point still has γ = 0 as a least-squares solution.
    g = chain_graph()
    s = MessageStore(g)
    b = LenticulumCore.Gaussian(1.0, 1.0)
    s.to_factor[1] = s.to_variable[1] = Message(b, 0)
    an = AndersonSchedule(AndersonAffineSweep(identity); ridge = 0)
    @test sweep!(s, g, an, nt, nt) == (0.0, nt)
    # An unregularised zero residual history currently throws SingularException.
    @test_broken sweep!(s, g, an, nt, nt) == (0.0, nt)
    @test belief_distance(s.to_variable[1].belief, b) == 0
end
