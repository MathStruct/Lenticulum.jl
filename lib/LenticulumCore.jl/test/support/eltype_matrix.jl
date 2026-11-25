module EltypeMatrix

using Test
using InteractiveUtils: subtypes
using ForwardDiff
using LenticulumCore

export typed, wired_cases, EltypeCase, check_eltype_matrix, check_factor_coverage

struct FixtureTag end
const Dual = ForwardDiff.Dual{FixtureTag,Float64,1}
const TYPES = (Float32, Float64, BigFloat, Dual)

# Seed a nonzero direction without perturbing structural zeros or symmetry.
typed(::Type{Dual}, x::AbstractFloat) = ForwardDiff.Dual{FixtureTag}(Float64(x), Float64(x) / 8)
typed(::Type{T}, x::AbstractFloat) where {T} = T(x)
typed(T, x) = x
typed(T, x::Union{NamedTuple,Tuple,AbstractArray}) = map(v -> typed(T, v), x)
typed(T, x::AbstractArray{<:AbstractFloat}) = typed.(Ref(T), x)
typed(T, b::DiracBelief) = DiracBelief(typed(T, b.value))
typed(T, b::GaussianBelief) = GaussianBelief(typed(T, b.η), typed(T, b.Λ), b.improper)
typed(T, b::ExtendedGaussian) =
    ExtendedGaussian(typed(T, b.μ), typed(T, b.Σ), typed(T, b.D))
typed(T, b::SampleBelief) = SampleBelief(typed(T, b.samples), typed(T, b.weights))
typed(T, b::CategoricalBelief) = CategoricalBelief(Val(:log), typed(T, b.logp), b.labels)
typed(T, b::MixtureBelief) = MixtureBelief(Val(:log), typed(T, b.components), typed(T, b.logw))

struct EltypeCase{F,C}
    name::Symbol
    make::F
    cases::C
    κ::Float64
end
EltypeCase(name, make, cases; κ = 1.0) = EltypeCase(name, make, cases, Float64(κ))

"""
Build the operations from the same incoming beliefs used by the conformance tests.
Each polarity gets a separate operation, so a broken solve cannot hide the other outputs.
"""
function wired_cases(message, local_energy, f; pointwise = nothing)
    cases = Pair{Symbol,Function}[]
    for p in supported_polarities(f), target in unobserved_channels(p)
        observed = observed_channels(p)
        suffix = isempty(observed) ? "latent" : join(observed, "_")
        push!(cases, Symbol(:message_, target, :_given_, suffix) => function (data)
            inputs = NamedTuple{observed_channels(p)}(map(n -> data.msgs[n], observed_channels(p)))
            return first(message(data.factor, target, p, inputs, data.msgs[target], data.ps, data.st))
        end)
    end
    if pointwise !== nothing
        push!(cases, :energy => data -> first(energy(data.factor, data.pointwise..., data.ps, data.st)))
    end
    push!(cases, :local_free_energy => data -> first(local_energy(data.factor, data.msgs, data.ps, data.st)))
    return cases
end

factor_family(f) = Base.typename(typeof(f)).wrapper
function check_factor_coverage(fixtures, modules; exclusions = Dict())
    function visit(T)
        isabstracttype(T) || return (T,)
        return Tuple(Iterators.flatten(visit.(subtypes(T))))
    end
    actual = Set(T for T in visit(AbstractLenticulumFactor) if parentmodule(T) in modules)
    covered = Set(factor_family(f.make(Float64, Float64).factor) for f in fixtures)
    @test isempty(intersect(covered, keys(exclusions)))
    @test all(!isempty, values(exclusions))
    @test union(covered, keys(exclusions)) == actual
end

# Discrete labels, flags and shapes are compared exactly. Only numerical payloads
# carry a working element type. Compare flat spaces via their projector.
payload(b) = b
payload(b::TrivialBelief) = (;)
payload(b::DiracBelief) = (; b.value)
payload(b::GaussianBelief) = (; b.η, b.Λ, b.improper)
payload(b::ExtendedGaussian) = (; b.μ, b.Σ, flat = b.D * b.D')
payload(b::CategoricalBelief) = (; b.logp, b.labels)
payload(b::MixtureBelief) = (; b.components, b.logw)
payload(b::SampleBelief) = (; b.samples, b.weights)
payload(e::GradedEnergy) = e.parts

primal(x) = x
primal(x::Dual) = ForwardDiff.value(x)
finite_partials(x) = true
finite_partials(x::Dual) = all(isfinite, ForwardDiff.partials(x))

function agrees(a::CategoricalBelief, b::CategoricalBelief, T, rtol)
    return isequal(a.labels, b.labels) && agrees(a.logp, b.logp, T, rtol)
end

# Some Schur complements return a mixture of constant likelihoods where another
# precision coalesces it to one constant likelihood. Test the numeric storage
# in both cases, including the weights if the result kept the mixture.
flat_message(b, atol) = false
flat_message(b::GaussianBelief, atol) =
    b.improper && all(x -> abs(primal(x)) <= atol, b.η) && all(x -> abs(primal(x)) <= atol, b.Λ)
flat_message(b::MixtureBelief, atol) = all(c -> flat_message(c, atol), b.components)

typed_flat(b::GaussianBelief, T, d) =
    length(b.η) == d && size(b.Λ) == (d, d) && eltype(b.η) === T && eltype(b.Λ) === T &&
    all(finite_partials, b.η) && all(finite_partials, b.Λ)
typed_flat(b::MixtureBelief, T, d) =
    eltype(b.logw) === T && all(finite_partials, b.logw) && all(c -> typed_flat(c, T, d), b.components)
flat_dimension(b::GaussianBelief) = length(b.η)
flat_dimension(b::MixtureBelief) = flat_dimension(first(b.components))

function agrees_flat(a, b, T, rtol)
    return flat_message(a, rtol) && flat_message(b, rtol) && typed_flat(a, T, flat_dimension(b))
end

function agrees(a, b, T, rtol)
    a, b = payload(a), payload(b)
    if b isa AbstractArray
        a isa AbstractArray && size(a) == size(b) || return false
        if eltype(b) <: AbstractFloat
            return eltype(a) === T && all(finite_partials, a) && isapprox(primal.(a), b; rtol)
        end
        return all(agrees(x, y, T, rtol) for (x, y) in zip(a, b))
    elseif b isa Union{NamedTuple,Tuple}
        ((a isa NamedTuple && b isa NamedTuple) || (a isa Tuple && b isa Tuple)) && keys(a) == keys(b) || return false
        return all(agrees(x, y, T, rtol) for (x, y) in zip(a, b))
    elseif b isa AbstractFloat
        return a isa T && finite_partials(a) && isapprox(primal(a), b; rtol)
    end
    return isequal(a, b)
end

function check_eltype_matrix(fixtures; broken = Dict(), untyped = (), flat = ())
    @testset "Factor element types" begin
        for f in fixtures
            base = f.make(Float64, Float64)
            @testset "$(f.name)" begin
                for (name, run) in f.cases
                    ref = run(base)
                    for (P, T, label) in ((Float32, Float32, :Float32), (Float64, Float64, :Float64),
                                          (BigFloat, BigFloat, :BigFloat), (Dual, Dual, :Dual),
                                          (Float32, Float64, :mixed))
                        label === :mixed && !(name in (:local_free_energy, :free_energy)) && continue
                        rtol = label in (:Float32, :mixed) ? 1f-4 * f.κ :
                               label === :BigFloat ? 1e-10 : 1e-12
                        reason = get(broken, (f.name, name, label), nothing)
                        expected = (f.name, name) in untyped ? Float64 : T
                        compare = (f.name, name) in flat ? agrees_flat : agrees
                        @testset "$name $label" begin
                            if isnothing(reason)
                                @test compare(run(f.make(P, T)), ref, expected, rtol)
                            else
                                @testset "$reason" begin
                                    @test_broken compare(run(f.make(P, T)), ref, expected, rtol)
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end

end
