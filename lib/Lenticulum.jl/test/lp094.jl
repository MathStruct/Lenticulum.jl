# LP-094: findings of the reviews of Q19 (element types on every message path). Each testset
# fails on the code Codex delivered and passes with the fix.
module LP094
using Test, LinearAlgebra, Lenticulum, LenticulumCore, Mycelium, ForwardDiff

tot(e::GradedEnergy) = sum(e.parts)
tot(v::AbstractVector) = sum(tot, v)
tot(x::Number) = x

# The numeric leaf types of a result, ignoring labels, flags and shapes. Integers and Bools are
# neutral (they do not widen a float), as in the rule.
nums(x) = Set{Type}()
nums(x::Union{Bool,Integer}) = Set{Type}()
nums(x::Number) = Set([typeof(x)])
nums(x::AbstractArray{<:Number}) = eltype(x) <: Union{Bool,Integer} ? Set{Type}() : Set([eltype(x)])
nums(x::Union{Tuple,NamedTuple,AbstractArray}) = mapreduce(nums, union, x; init = Set{Type}())
nums(e::GradedEnergy) = nums(e.parts)
nums(x) = isstructtype(typeof(x)) ?
    mapreduce(nums, union, (getfield(x, n) for n in fieldnames(typeof(x))); init = Set{Type}()) :
    Set{Type}()

@testset "LP-094: reviews of Q19" begin
    @testset "finding 1: a dual prior mean keeps its derivative in the frozen belief" begin
        # Q19's first version built x − μ in the covariance's type, which the mean does not
        # enter: Float64(::Dual) threw. The oracle is a central difference.
        msgs = (; x = GaussianBelief([0.4], fill(2.0, 1, 1)))
        function F(m, plate)
            f = GaussianPrior(:x, [m], fill(0.8, 1, 1))
            plate || return tot(first(Mycelium.belief_free_energy(f, Mycelium.factor_belief(f, msgs, (;), (;)), (;), (;))))
            fs = [f, f]
            B = Mycelium.factor_beliefs(fs, map(b -> Mycelium.belief_block(fill(b, 2)), msgs), (;), (;))
            return tot(first(Mycelium.belief_free_energies(fs, B, (;), (;))))
        end
        for plate in (false, true)
            h = 1e-6
            fd = (F(0.3 + h, plate) - F(0.3 - h, plate)) / 2h
            @test ForwardDiff.derivative(m -> F(m, plate), 0.3) ≈ fd rtol = 1e-6
        end
    end

    @testset "finding 2: BigFloat without GenericLinearAlgebra raises the documented ArgumentError" begin
        # Run standalone (`julia --project=scratch/q19r/env lib/Lenticulum.jl/test/lp094.jl`),
        # this process has not loaded GenericLinearAlgebra, so the guard must fire. When the
        # full suite has loaded it (eltypes.jl does `using GenericLinearAlgebra`), the same
        # calls must succeed instead.
        f = LinearConstraintFactor((x = 1, y = 1), 1; noise = BigFloat(0.4))
        ps = (A = (x = BigFloat[-1.25;;], y = BigFloat[1.0;;]), c = BigFloat[0.1])
        p = first(p for p in supported_polarities(f) if isempty(observed_channels(p)))
        t = first(unobserved_channels(p))
        msg = () -> first(Mycelium.factor_message(f, t, p, (;), Gaussian(BigFloat(0.5), BigFloat(0.8)), ps, (;)))
        flat = () -> Mycelium._flat_factor(Matrix{BigFloat}([1.0 0.0; 0.0 0.0]), nothing)
        # the guard's own condition: true once GenericLinearAlgebra is loaded anywhere in the process
        if !hasmethod(LinearAlgebra.svd!, Tuple{Matrix{BigFloat}})
            e1 = try msg(); nothing catch e; e end
            @test e1 isa ArgumentError
            @test occursin("GenericLinearAlgebra", sprint(showerror, e1))
            e2 = try flat(); nothing catch e; e end
            @test e2 isa ArgumentError
            @test occursin("GenericLinearAlgebra", sprint(showerror, e2))
        else
            @test msg() isa LenticulumCore.GaussianBelief
            @test flat() isa Mycelium._Flat
        end
    end

    @testset "finding 3: a plate whose replicas differ in element type keeps each replica's offset type" begin
        fs = [LinearConstraintFactor((x = 1, y = 1), 1; noise = 0.4f0),
              LinearConstraintFactor((x = 1, y = 1), 1; noise = 0.4)]
        ps = (A = (x = [-2;;], y = [1;;]), c = [16777217])       # 2^24 + 1: rounds in Float32
        p = first(p for p in supported_polarities(fs[1]) if observed_channels(p) == (:x,))
        ins = [(x = Gaussian(0.5f0, 0.8f0),), (x = Gaussian(0.5, 0.8),)]
        out, _ = Mycelium.factor_messages(fs, :y, p, ins, [nothing, nothing], ps, (;))
        ref1, _ = Mycelium.factor_message(fs[1], :y, p, ins[1], nothing, ps, (;))
        ref2, _ = Mycelium.factor_message(fs[2], :y, p, ins[2], nothing, ps, (;))
        @test out[1].η == ref1.η && out[1].Λ == ref1.Λ
        @test out[2].η == ref2.η && out[2].Λ == ref2.Λ
        @test eltype(out[1].η) === Float32
        @test eltype(out[2].η) === Float64
    end

    @testset "finding 4: GaussianPrior takes its element type from the call, not the stored mean's storage" begin
        # (a) an integer stored mean does not widen a Float32 call
        f = GaussianPrior(:x, [1], fill(0.8f0, 1, 1))
        msgs = (; x = Gaussian(0.5f0, 0.8f0))
        b = Mycelium.factor_belief(f, msgs, (;), (;))
        e = first(Mycelium.belief_free_energy(f, b, (;), (;)))
        l = first(Mycelium.local_free_energy(f, msgs, (;), (;)))
        @test nums(b) == Set([Float32])
        @test nums(e) == Set([Float32])
        @test nums(l) == Set([Float32])
        @test tot(e) ≈ tot(l) rtol = 1f-6
        # a plain Float64 stored mean is configuration: converted, it does not widen a Float32 call
        # (the first fix of 4b added it to the working type and widened energy and lfe)
        f64 = GaussianPrior(:x, [0.3], fill(0.8f0, 1, 1))
        b64 = Mycelium.factor_belief(f64, msgs, (;), (;))
        @test nums(b64) == Set([Float32])
        @test nums(first(Mycelium.belief_free_energy(f64, b64, (;), (;)))) == Set([Float32])
        @test nums(first(Mycelium.local_free_energy(f64, msgs, (;), (;)))) == Set([Float32])
        @test nums(first(LenticulumCore.energy(f64, nothing, nothing, Float32[0.4], (;), (;)))) == Set([Float32])
        # (b) a dual stored mean keeps its derivative in local_free_energy and energy
        msgs64 = (; x = GaussianBelief([0.4], fill(2.0, 1, 1)))
        F(m) = tot(first(Mycelium.local_free_energy(GaussianPrior(:x, [m], fill(0.8, 1, 1)), msgs64, (;), (;))))
        G(m) = tot(first(LenticulumCore.energy(GaussianPrior(:x, [m], fill(0.8, 1, 1)), nothing, nothing, [0.4], (;), (;))))
        h = 1e-6
        @test ForwardDiff.derivative(F, 0.3) ≈ (F(0.3 + h) - F(0.3 - h)) / 2h rtol = 1e-6
        @test ForwardDiff.derivative(G, 0.3) ≈ (G(0.3 + h) - G(0.3 - h)) / 2h rtol = 1e-6
    end

    @testset "finding 5: integer logw does not widen the switching or regime factors" begin
        msgs = (x = Gaussian(0.5f0, 0.8f0), y = Gaussian(0.9f0, 0.7f0), k = CategoricalBelief(Float32[0.65, 0.35]))
        f = SwitchingConstraintFactor((x = 1, y = 1), 2, 1; noise = 0.4f0)
        ps = (A = (x = [-2;;], y = [1;;]), b = [1 2], logw = [0, 0])
        e = first(Mycelium.local_free_energy(f, msgs, ps, (;)))
        @test nums(e) == Set([Float32])
        @test eltype(Mycelium.cluster_form(f, ps).logw) === Float32
        p = first(p for p in supported_polarities(f) if :x in unobserved_channels(p) && :k in observed_channels(p))
        m = first(Mycelium.factor_message(f, :x, p, (y = msgs.y, k = msgs.k), msgs.x, ps, (;)))
        @test nums(m) == Set([Float32])
        g = RegimeFactor((x = 1, y = 1), 2, 1; noise = 0.4f0)
        psg = (A = (x = cat([-2;;], [-1;;]; dims = 3), y = cat([1;;], [2;;]; dims = 3)),
               b = [1 2], logw = [0, 0])
        eg = first(Mycelium.local_free_energy(g, msgs, psg, (;)))
        @test nums(eg) == Set([Float32])
        @test eltype(Mycelium.cluster_form(g, psg).logw) === Float32
    end

    @testset "finding 6: integer parameters with a declared output do not widen" begin
        f = LinearConstraintFactor((x = 1, y = 1), 1; noise = 0.4f0, output = :y)
        ps = (A = (x = [-2;;],), c = [1])
        msgs = (x = Gaussian(0.5f0, 0.8f0), y = Gaussian(0.9f0, 0.7f0))
        e = first(Mycelium.local_free_energy(f, msgs, ps, (;)))
        @test nums(e) == Set([Float32])
        p = first(p for p in supported_polarities(f) if observed_channels(p) == (:y,))
        m = first(Mycelium.factor_message(f, :x, p, (y = msgs.y,), nothing, ps, (;)))
        @test nums(m) == Set([Float32])
        # the switching factor shares _coeffs/_fixed_at
        g = SwitchingConstraintFactor((x = 1, y = 1), 2, 1; noise = 0.4f0, output = :y)
        psg = (A = (x = [-2;;],), b = [1 2], logw = [0, 0])
        gm = (x = Gaussian(0.5f0, 0.8f0), y = Gaussian(0.9f0, 0.7f0), k = CategoricalBelief(Float32[0.65, 0.35]))
        eg = first(Mycelium.local_free_energy(g, gm, psg, (;)))
        @test nums(eg) == Set([Float32])
    end

    @testset "finding 7: ScaleShiftFactor with integer parameters stays in the messages' type" begin
        f = ScaleShiftFactor(:x, :y, 1)
        ps = (b = [0], logs = [0])
        msgs = (x = Gaussian(0.5f0, 0.8f0), y = Gaussian(0.9f0, 0.7f0))
        e = first(Mycelium.local_free_energy(f, msgs, ps, (;)))
        @test nums(e) == Set([Float32])
        p = first(p for p in supported_polarities(f) if :x in unobserved_channels(p))
        m = first(Mycelium.factor_message(f, :x, p, (y = msgs.y,), nothing, ps, (;)))
        @test nums(m) == Set([Float32])
        b = Mycelium.factor_belief(f, msgs, ps, (;))
        be = first(Mycelium.belief_free_energy(f, b, ps, (;)))
        @test nums(be) == Set([Float32])
    end

    @testset "finding 9: a dual stored A in LikelihoodFactor keeps its derivative" begin
        msgs = (x = Gaussian(0.5, 0.8), y = DiracBelief([2]))
        F(a) = tot(first(Mycelium.local_free_energy(
            LikelihoodFactor(OrdinalProbitLikelihood([-1.0, 1.0]), 1; A = a, b = [0.2]), msgs, (;), (;))))
        h = 1e-6
        d = ForwardDiff.derivative(F, 1.1)
        @test d ≈ (F(1.1 + h) - F(1.1 - h)) / 2h rtol = 1e-5
        # a Float64 stored A with a Float32 input still converts to Float32 (intended, Q4)
        f = LikelihoodFactor(OrdinalProbitLikelihood([-1.0, 1.0]), 1; A = 1.1, b = [0.2])
        e = first(Mycelium.local_free_energy(f, (x = Gaussian(0.5f0, 0.8f0), y = DiracBelief([2])), (;), (;)))
        @test nums(e) == Set([Float32])
    end
end
end
