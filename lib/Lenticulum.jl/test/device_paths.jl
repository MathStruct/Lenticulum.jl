using Adapt, JLArrays, LinearAlgebra, Random, Test
using Lenticulum, LenticulumCore, Mycelium, MyceliumLearning
using ADTypes, DifferentiationInterface, Zygote

isdefined(@__MODULE__, :EltypeMatrix) || include("../../LenticulumCore.jl/test/support/eltype_matrix.jl")
isdefined(@__MODULE__, :DeviceMatrix) || include("../../LenticulumCore.jl/test/support/device_matrix.jl")

struct Q20Residual <: AbstractLenticulumFactor end
LenticulumCore.energy(::Q20Residual, x, a, y, ps, st) = (ps.A * y .^ 2, st)

struct Q20ScoreSource{DiagonalPrecision} <: AbstractLenticulumFactor end
LenticulumCore.channels(::Q20ScoreSource) = (LenticulumCore.Channel(:x, 2),)
LenticulumCore.supported_polarities(::Q20ScoreSource) = (LenticulumCore.Polarity(x = Unobserved()),)
LenticulumCore.supports_polarity(f::Q20ScoreSource, p::LenticulumCore.Polarity) =
    p in supported_polarities(f)
LenticulumCore.islearnable(::Q20ScoreSource) = true
Mycelium.local_free_energy(::Q20ScoreSource, beliefs, ps, st) = (0.0, st)
Mycelium.factor_message(::Q20ScoreSource{true}, target::Symbol, pol, inputs, prior, ps, st) =
    (GaussianBelief(ps.h, Diagonal(ps.q)), st)
Mycelium.factor_message(::Q20ScoreSource{false}, target::Symbol, pol, inputs, prior, ps, st) =
    (GaussianBelief(ps.h, ps.q), st)

function q20_score_graph(source)
    b = GraphBuilder()
    variable!(b, :x, 2)
    factor!(b, :source, source)
    factor!(b, :loss, LossFactor((:x,), x -> sum(abs2, x .+ 0.2) / 2))
    connect!(b, :source, :x, :x; direction = Emitting(), coupling = ExactCoupling())
    connect!(b, :loss, :x, :x; direction = Absorbing(), coupling = ScoreFunctionCoupling(baseline = 0.3))
    return build(b)
end

@testset "Q20 device paths outside the factor matrix" begin
    JLArrays.allowscalar(false)
    D = DeviceMatrix.device_array()
    dev(x) = adapt(D, x)
    function check(out, ref; rtol = 1e-10)
        @test DeviceMatrix.device_agrees(out, ref, rtol)
        @test DeviceMatrix.on_device(DeviceMatrix.device_payload(out))
    end
    μ, Σ = [0.4, -0.3], [0.8 0.1; 0.1 0.6]
    next = Gaussian(-μ, 1.5Σ)
    prior = Gaussian(μ, Σ)
    mixture = MixtureBelief([prior, next, Gaussian(2μ, 2Σ)], [0.65, 0.349999999, 1e-9])
    @testset "mixture reduction" begin
        reduce(m) = reduce_mixture(m; max_components = 1)
        check(reduce(dev(mixture)), reduce(mixture))
        diagonal = MixtureBelief([Gaussian(μ, Diagonal([0.8, 0.6])),
                                  Gaussian(-μ, Diagonal([1.2, 0.9]))], [0.65, 0.35])
        # Merging diagonal components generally produces a dense covariance.
        dense = MixtureBelief([Gaussian(μ, [0.8 0; 0 0.6]),
                               Gaussian(-μ, [1.2 0; 0 0.9])], [0.65, 0.35])
        check(reduce(dev(diagonal)), reduce(dense))
    end
    @testset "mixture log density" begin
        check(Mycelium.belief_logdensity(dev(mixture), D(μ)), Mycelium.belief_logdensity(mixture, μ))
    end
    @testset "CPU inference on touched paths" begin
        @test (@inferred LenticulumCore._host_values(μ)) === μ
        @test (@inferred Mycelium.belief_logdensity(mixture, μ)) isa Float64
        @test (@inferred MyceliumLearning._draw(Xoshiro(4), prior)) isa Vector{Float64}
        @test (@inferred ep_divide(prior, next)) isa GaussianBelief
        @test (@inferred Union{Nothing,typeof(prior)} ep_project(mixture)) isa GaussianBelief
    end
    @testset "EP projection" begin
        for b in (mixture, MixtureBelief(ExtendedGaussian.([μ, -μ], [Σ, 1.5Σ]), [0.65, 0.35]))
            check(ep_project(dev(b)), ep_project(b))
        end
    end
    @testset "EP site division" begin
        for (p, c) in ((prior, next), (next, prior),
                       (ExtendedGaussian(μ, Σ), ExtendedGaussian(-μ, 1.5Σ)))
            check(ep_divide(dev(p), dev(c)), ep_divide(p, c))
        end
    end
    @testset "Sampson localisation" begin
        metric(y, A, cov) = localise(SampsonNorm(; Σ = cov, λ = 0.2), Q20Residual(),
                                    nothing, nothing, y, (; A), (;)).M
        check(metric(D(μ), D(Σ), D(Σ)), metric(μ, Σ, Σ); rtol = 1e-8)
    end
    @testset "ScoreFunctionCoupling sampling" begin
        # Both draws consume the same host RNG stream, as the replay does.
        @testset "$(nameof(typeof(b)))" for b in (prior, CategoricalBelief([0.2, 0.5, 0.3]))
            check(MyceliumLearning._draw(Xoshiro(4), dev(b)), MyceliumLearning._draw(Xoshiro(4), b))
        end
    end
    @testset "ScoreFunctionCoupling replay" begin
        estimate(g, ps) = free_energy_gradient(MessageStore(g), g, ps, (;);
            ad = AutoZygote(), schedule = forward_backward_schedule(g), rng = Xoshiro(19), nsamples = 3)
        broken = Dict(:dense => "Reverse AD through the dense device Cholesky is unavailable")
        for (name, source, q) in ((:diagonal, Q20ScoreSource{true}(), [1.0, 1.0]),
                                  (:dense, Q20ScoreSource{false}(), [1.0 0.0; 0.0 1.0]))
            g = q20_score_graph(source)
            ps = (source = (h = [0.8, 0.4], q = q), loss = (;))
            ref = estimate(g, ps)
            agrees() = begin
                out = estimate(dev((g, ps))...)
                out.nsamples == ref.nsamples &&
                    DeviceMatrix.device_agrees((out.value, out.standard_error), (ref.value, ref.standard_error), 1e-10) &&
                    DeviceMatrix.on_device((out.value, out.standard_error))
            end
            if haskey(broken, name)
                @testset "$(broken[name])" begin
                    @test_broken agrees()
                end
            else
                @test agrees()
            end
        end
    end
    @testset "scale-shift parameters" begin
        f = ScaleShiftFactor(:x, :z, 3; groups = [1, 2, 1])
        X = [0.2 0.5 -1.0; 2.0 1.0 1.4; 0.7 1.3 0.8]
        check(scale_shift_parameters(f, D(X)), scale_shift_parameters(f, X))
    end
    @testset "pin coordinates" begin
        pin(v) = Mycelium._pin_coordinates(v, [2])
        check(pin(D(μ)), pin(μ))
        ref = only(Zygote.gradient(v -> sum(abs2, pin(v)), μ))
        out = only(Zygote.gradient(v -> sum(abs2, pin(v)), D(μ)))
        @test ref == [0.0, 2μ[2]]
        check(out, ref)
    end
    @testset "direct constructors" begin
        for (name, construct) in (:Gaussian => ((m, Q) -> Gaussian(m, Q)),
                          :GaussianFactor => ((m, Q) -> GaussianFactor(2 => 2; noise = Q).Q),
                          :GaussianPrior => ((m, Q) -> GaussianPrior(:x, m, Q; learn_noise = :full).Σ),
                          :OrdinalProbitLikelihood => ((m, Q) -> OrdinalProbitLikelihood(m).cutpoints),
                          :SquaredNorm => ((m, Q) -> SquaredNorm(Q).M),
                          :LinearConstraint => ((m, Q) -> LinearConstraintFactor((x = 2,), 2; noise = Q).Q),
                          :HardConstraint => ((m, Q) -> HardConstraintFactor((x = 2,), 2; noise = Q).Q),
                          :NonlinearFactor => ((m, Q) -> NonlinearFactor(identity, 2 => 2; noise = Q).Q),
                          :SwitchingConstraint => ((m, Q) -> SwitchingConstraintFactor((x = 2,), 2, 2; noise = Q).Q),
                          :RegimeFactor => ((m, Q) -> RegimeFactor((x = 2,), 2, 2; noise = Q).Q),
                          :LikelihoodFactor => ((m, Q) -> begin
                              f = LikelihoodFactor(OrdinalProbitLikelihood([-1.0, 1.0]), 2; A = Q[1:1, :])
                              (; f.A, f.b)
                          end))
            @testset "$name" begin
                m = reverse(μ)
                check(construct(D(m), D(Σ)), construct(m, Σ))
            end
        end
    end
end
