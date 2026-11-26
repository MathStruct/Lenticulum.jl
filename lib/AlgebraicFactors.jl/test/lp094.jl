# LP-094 (part 2), AlgebraicFactors: the element-type findings of the Q19 reviews that live in
# this package. Run standalone with `julia --project=scratch/q19r/env
# lib/AlgebraicFactors.jl/test/lp094.jl`.
module LP094
using Test, AlgebraicFactors, LenticulumCore, Mycelium

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

@testset "LP-094: reviews of Q19 (AlgebraicFactors)" begin
    @testset "finding 8: ToricFactor's factor_belief and belief_free_energy take the messages' type" begin
        f = ToricFactor((:a, :b), (2, 2), [1 1 1 1; 0 0 1 1; 0 1 0 1]; normalised = true)
        ps = (η = [1, -1, 2],)      # integer η must not widen a Float32 call
        msgs = (a = CategoricalBelief(Float32[0.4, 0.6]), b = CategoricalBelief(Float32[0.7, 0.3]))
        b = Mycelium.factor_belief(f, msgs, ps, (;))
        @test eltype(b.logq) === Float32
        e = first(Mycelium.belief_free_energy(f, b, ps, (;)))
        @test nums(e) == Set([Float32])
        l = first(Mycelium.local_free_energy(f, msgs, ps, (;)))
        @test nums(l) == Set([Float32])
    end
end
end
