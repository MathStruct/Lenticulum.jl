using LinearAlgebra

@testset "Belief blocks agree with their columns" begin
    product = Mycelium.combine_blocks
    distance = Mycelium.block_distance
    relative = Mycelium.block_relative_residual
    gb(η, Λ, improper = false) = LenticulumCore.GaussianBelief(η, Λ, improper)
    function check_distance(a, b)
        pairs = zip(replicas(a), replicas(b))
        @test distance(a, b) ≈ maximum(p -> belief_distance(p...), pairs; init = 0.0)
        @test relative(a, b, 1e-3, 1e-2) ≈
              maximum(p -> relative_residual(p..., 1e-3, 1e-2), pairs; init = 0.0)
    end
    function check_product(a, b; atol = 1e-12)
        for (left, right) in ((a, b), (b, a))
            result = product(left, right)
            expected = [combine(x, y) for (x, y) in zip(replicas(left), replicas(right))]
            @test nreplicas(result) == length(expected)
            for (x, y) in zip(replicas(result), expected)
                @test belief_distance(x, y) ≈ 0 atol = atol
                if y isa LenticulumCore.GaussianBelief
                    @test x.improper == y.improper
                elseif y isa MixtureBelief
                    @test [c.improper for c in x.components] == [c.improper for c in y.components]
                end
            end
        end
    end

    empty = belief_block(Any[])
    @test empty isa VectorBlock
    @test isempty(replicas(empty))
    @test isempty(replicas(product(empty, empty)))
    check_distance(empty, empty)
    one = belief_block([DiracBelief(Float32[1, 2])])
    @test one isa PointBlock
    @test only(replicas(one)).value == Float32[1, 2]
    @test eltype(only(replicas(one)).value) === Float32

    trivial = belief_block(fill(TrivialBelief(), 2))
    point = belief_block([DiracBelief(Float32[1, 2]), DiracBelief(Float32[4, 6])])
    point2 = PointBlock([1.0 1.0; 2.0 2.0])
    @test product(trivial, trivial) === trivial
    check_product(trivial, point)
    check_product(point, PointBlock(copy(point.X)))
    @test_throws ArgumentError product(point, point2)
    check_distance(trivial, trivial)
    check_distance(point, point2)
    check_distance(point, point)
    @test isinf(distance(nothing, trivial))
    @test isinf(relative(nothing, trivial, 0.1, 0.1))
    @test isinf(distance(point, one))
    @test isinf(relative(point, one, 0.1, 0.1))

    Λ = Float32[2 0.25; 0.25 1]
    gauss = belief_block([gb(Float32[2, 1], Λ, true), gb(Float32[1, 3], Λ)])
    flat = belief_block([gb(zeros(2), zeros(2, 2), true), gb(zeros(2), zeros(2, 2))])
    mixed = belief_block([gb(zeros(2), zeros(2, 2)), gb([3.0, -1], [4.0 0; 0 3])])
    @test gauss isa GaussianBlock && gauss.Λ === Λ
    @test flat.Λ isa AbstractVector   # equal matrices with distinct identities stay per replica
    shared_zero = GaussianBlock(zeros(2, 2), zeros(2, 2), [true, false])
    @test Mycelium._uninformative(shared_zero) === true
    @test Mycelium._uninformative(mixed) === missing
    @test Mycelium._uninformative(gauss) === false
    check_product(point, gauss)
    check_product(gauss, shared_zero)
    check_product(gauss, mixed)
    @test product(gauss, shared_zero).Λ isa AbstractMatrix{Float64}
    check_distance(gauss, shared_zero)
    check_distance(gauss, mixed)
    wrong_dimension = GaussianBlock(zeros(1, 2), ones(1, 1), [false, false])
    @test_throws DimensionMismatch product(gauss, wrong_dimension)
    @test isinf(relative(gauss, wrong_dimension, 0.1, 0.1))
    @test_throws DimensionMismatch GaussianBlock(zeros(2, 2), Λ, [false])
    @test_throws DimensionMismatch GaussianBlock(zeros(2, 2), [Λ], [false, false])

    cats = belief_block([CategoricalBelief(Float32[1, 3]), CategoricalBelief(Float32[2, 1])])
    cats2 = CategoricalBlock(log.([0.6 0.1; 0.4 0.9]))
    @test cats isa CategoricalBlock
    check_product(cats, cats2; atol = 2e-7)
    check_distance(cats, cats2)
    labelled = CategoricalBlock(cats2.logp, [:low, :high])
    @test_throws ArgumentError product(cats, labelled)
    @test isinf(distance(cats, labelled))
    @test_throws ArgumentError product(CategoricalBlock(reshape([0.0, -Inf], 2, 1)),
                                       CategoricalBlock(reshape([-Inf, 0.0], 2, 1)))

    hetero = belief_block([TrivialBelief(), DiracBelief(2)])
    hetero2 = belief_block([TrivialBelief(), DiracBelief(4)])
    @test hetero isa VectorBlock
    @test Mycelium._uninformative(hetero) === missing
    @test isinf(distance(hetero, empty))
    @test isinf(relative(hetero, empty, 0.1, 0.1))
    check_distance(hetero, hetero2)
    check_product(hetero, belief_block([DiracBelief(1), TrivialBelief()]))
    for b in (trivial, point, cats)
        @test Mycelium._uninformative(b) == all(LenticulumCore.isuninformative, replicas(b))
    end

    η = reshape(Float32[1, 2, -2, 0, 0, 1, 3, -1], 2, 2, 2)
    weights = Float32[0.25 0.75; 0.75 0.25]
    mix(Λ, improper = false) = belief_block(
        [MixtureBelief([gb(η[:, k, i], Λ, improper) for k in 1:2], weights[:, i]) for i in 1:2])
    m = mix(Λ)
    @test m isa MixtureBlock && m.Λ === Λ
    @test LenticulumCore.mixture_weights(replica(m, 2)) ≈ weights[:, 2]
    @test replica(m, 2).components[1].η == η[:, 1, 2]

    changed_η = copy(η)
    changed_η[2, 1, 2] += 0.125f0
    changed_logw = log.(reverse(weights; dims = 1))
    different = [Λ, Float32[1 0; 0 0]]
    cm = ComponentMixtureBlock(η, different, log.(weights), [false, true])
    @test nreplicas(cm) == 2
    @test replica(cm, 2).components[2].Λ === different[2]
    @test replica(cm, 2).components[2].improper
    @test replica(cm, 2).components[2].η == η[:, 2, 2]
    @test belief_block(replicas(cm)) isa VectorBlock
    @test isinf(distance(cm, ComponentMixtureBlock(η[:, :, 1:1], different, log.(weights[:, 1:1]), [false, true])))
    for variant in (MixtureBlock(changed_η, Λ, m.logw, false),
                    MixtureBlock(η, 2Λ, m.logw, false),
                    MixtureBlock(η, Λ, changed_logw, false))
        check_distance(m, variant)
    end
    for variant in (ComponentMixtureBlock(changed_η, different, cm.logw, cm.improper),
                    ComponentMixtureBlock(η, [Λ, 2 * different[2]], cm.logw, cm.improper),
                    ComponentMixtureBlock(η, different, changed_logw, cm.improper))
        check_distance(cm, variant)
    end
    for b in (m, cm)
        @test Mycelium._uninformative(b) === false
        check_product(b, point)
        @test_throws DimensionMismatch product(b, one)
        @test_throws DimensionMismatch product(one, b)
    end
    @test_throws DimensionMismatch ComponentMixtureBlock(η[:, :, 1], different, log.(weights), [false, true])
    @test_throws DimensionMismatch ComponentMixtureBlock(η, different, log.(weights[:, 1:1]), [false, true])
    @test_throws DimensionMismatch ComponentMixtureBlock(η, different, log.(weights), [false])
    @test_throws DimensionMismatch ComponentMixtureBlock(η, [Λ, ones(Float32, 1, 1)], log.(weights), [false, true])

    cavity = GaussianBlock([0.2 1.0; 0.5 -0.4], [3.0 0.5; 0.5 2], [false, false])
    check_product(m, cavity; atol = 2e-7)
    check_product(mix(zeros(Float32, 2, 2), true), cavity)
    check_product(m, mixed)
    check_product(cm, cavity)
    one_component = MixtureBlock(η[:, 1:1, :], Λ, zeros(Float32, 1, 2), false)
    check_product(one_component, cavity; atol = 2e-7)
end

@testset "EdgeMessages materialises a block before an edge write" begin
    # The row's columns need not be consecutive edges.
    msgs = Mycelium.EdgeMessages(3, [[3, 1]])
    @test_throws ArgumentError Mycelium.EdgeMessages(3, [[3, 1], [1, 2]])
    @test Mycelium._row_block(msgs, 1) === nothing
    @test !Mycelium._isset(msgs, 3)
    msgs[3] = Message(DiracBelief([10]), 2)
    @test Mycelium._row_block(msgs, 1) === missing
    msgs[1] = Message(DiracBelief([20]), 2)
    @test Mycelium._row_block(msgs, 1).X == reshape([10, 20], 1, 2)
    Mycelium._set_block!(msgs, 1, PointBlock(reshape([30, 40], 1, 2)), 7)
    copied = copy(msgs)
    @test copied isa Vector{Any}
    @test copied[3].belief.value == [30]
    @test copied[1].belief.value == [40]
    @test Mycelium._isset(copied, 3)
    @test Mycelium._isset(msgs, 1)
    @test !Mycelium._isset(msgs, 2)
    msgs[1] = Message(DiracBelief([50]), 8)
    @test msgs.blocks[1] === nothing
    @test msgs[1].belief.value == [50] && msgs[1].iteration == 8
    @test msgs[3].belief.value == [30] && msgs[3].iteration == 7
    @test copied[1].belief.value == [40] && copied[1].iteration == 7
    s = MessageStore(copied, Any[nothing, nothing, nothing], 7)
    @test isempty(s.to_variable.rows)
    @test s.to_variable[3].belief.value == [30]
    Mycelium._set_block!(msgs, 1, PointBlock(reshape([60, 70], 1, 2)), 9)
    fill!(msgs, nothing)
    @test all(isnothing, msgs)
    @test Mycelium._row_block(msgs, 1) === nothing
end
