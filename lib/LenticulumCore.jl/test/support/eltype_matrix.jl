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

# The Dual fixtures seed every stored float x with the partial x/8: the direction of scaling every
# float by (1 + t/8). `Scaled{s}` builds the same fixture in Float64 at t = s·STEP, so a central
# difference of the two scaled runs is the reference for each partial (LP-094 finding 10).
struct Scaled{s} end
const STEP = 1e-6
typed(::Type{Scaled{s}}, x::AbstractFloat) where {s} = Float64(x) * (1 + s * STEP / 8)
Base.zero(::Type{<:Scaled}) = 0.0
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

# The inference packages pass their APIs in so these helpers also load in Core's test environment.
function inference_cases(inference, gradient_message, f; pointwise, mean_field = true, raw_blocks = false)
    M = inference
    cases = wired_cases(M.factor_message, M.local_free_energy, f.factor; pointwise)
    for p in supported_polarities(f.factor), target in unobserved_channels(p)
        observed = observed_channels(p)
        suffix = isempty(observed) ? "latent" : join(observed, "_")
        inputs(d) = NamedTuple{observed}(map(n -> d.msgs[n], observed))
        prior(d) = d.msgs[target]
        add(op, run) = push!(cases, Symbol(op, :_, target, :_given_, suffix) => run)

        add(:gradient_message, d -> first(gradient_message(nothing, d.factor, target, p,
                                                           inputs(d), prior(d), d.ps, d.st)))
        add(:plate_message_vector, d -> first(M.factor_messages(fill(d.factor, 2), target, p,
                                                               fill(inputs(d), 2), fill(prior(d), 2), d.ps, d.st)))
        add(:plate_message_block, function (d)
            block, _ = M.factor_messages(fill(d.factor, 2), target, p, message_blocks(M, inputs(d)),
                                         fill(prior(d), 2), d.ps, d.st)
            return raw_blocks ? block : [M.replica(block, i) for i in 1:2]
        end)

        args = (f.factor, target, p, inputs(f))
        if mean_field && factor_method(M.mean_field_message, args..., f.ps, f.st)
            add(:mean_field_message, d -> first(M.mean_field_message(d.factor, target, p, inputs(d), d.ps, d.st)))
        end
        if any(b -> b isa GaussianBelief, inputs(f)) &&
                factor_method(M.mixture_message, args..., prior(f), f.ps, f.st)
            add(:mixture_message, d -> first(M.mixture_message(d.factor, target, p,
                map(mixture_input, inputs(d)), prior(d), d.ps, d.st)))
        end
    end
    if factor_method(M.factor_belief, f.factor, f.msgs, f.ps, f.st)
        belief(d) = M.factor_belief(d.factor, d.msgs, d.ps, d.st)
        plate_beliefs(d) = M.factor_beliefs(fill(d.factor, 2), message_blocks(M, d.msgs), d.ps, d.st)
        push!(cases, :factor_belief => belief,
              :belief_free_energy => d -> first(M.belief_free_energy(d.factor, belief(d), d.ps, d.st)),
              :factor_beliefs => plate_beliefs,
              :belief_free_energies => d -> first(M.belief_free_energies(fill(d.factor, 2),
                                                                        plate_beliefs(d), d.ps, d.st)))
    end
    return cases
end

# The default methods for these operations do not implement a factor rule.
factor_method(op, args...) = Base.unwrap_unionall(which(op, Base.typesof(args...)).sig).parameters[2] !== Any

message_blocks(M, msgs) = map(b -> M.belief_block(fill(b, 2)), msgs)

mixture_input(b) = b
function mixture_input(b::GaussianBelief)
    w = eltype(b.η)
    shifted = GaussianBelief(b.η .* w(0.75), b.Λ)
    weights = LenticulumCore._on_device(LenticulumCore._device_leaf(b), w[0.625, 0.375])
    return MixtureBelief([b, shifted], weights)
end

# Frozen factor beliefs are not AbstractBeliefs. Their fixtures opt in to field-wise
# comparison so their stored numerical values participate in the same matrix.
belief_fields(b) = NamedTuple{fieldnames(typeof(b))}(ntuple(i -> getfield(b, i), fieldcount(typeof(b))))

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

function agrees_flat(a::AbstractVector, b::AbstractVector, T, rtol)
    return length(a) == length(b) && all(agrees_flat(x, y, T, rtol) for (x, y) in zip(a, b))
end

function agrees(a, b, T, rtol)
    a, b = payload(a), payload(b)
    if b isa Pair
        return a isa Pair && agrees(a.first, b.first, T, rtol) && agrees(a.second, b.second, T, rtol)
    elseif b isa AbstractDict
        return a isa AbstractDict && keys(a) == keys(b) && all(k -> agrees(a[k], b[k], T, rtol), keys(b))
    elseif b isa AbstractArray
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

# A fixture can opt all-latent calls out of promotion when the rule does not
# read the target cavity. Some factors use the cavity even without observations.
latent_cases(fixtures) = [(f.name, op) for f in fixtures for (op, _) in f.cases
                         if endswith(string(op), "_given_latent")]

# The numerical leaves of a result in a fixed order, through the same payloads as `agrees`.
leaves(x) = Any[]
leaves(x::Number) = Any[x]
leaves(x::Union{Bool,Integer}) = Any[]
leaves(x::AbstractArray{<:Number}) = Any[vec(x)...]
leaves(x::AbstractArray{<:Union{Bool,Integer}}) = Any[]
function leaves(x)
    y = payload(x)
    y === x && !(x isa Union{Tuple,NamedTuple,AbstractArray,Pair,AbstractDict}) && return Any[]
    return leaves_of(y)
end
leaves_of(y) = leaves(y)
leaves_of(y::Pair) = vcat(leaves(y.first), leaves(y.second))
leaves_of(y::AbstractDict) = reduce(vcat, (leaves(y[k]) for k in sort!(collect(keys(y)); by = string)); init = Any[])
leaves_of(y::Union{Tuple,NamedTuple,AbstractArray}) = reduce(vcat, (leaves(v) for v in y); init = Any[])
partial(x::Dual) = ForwardDiff.partials(x)[1]
partial(x) = zero(Float64)

"""
Each partial of a Dual result against the central difference of the fixture built at
(1 ± STEP/8) times its stored floats. Returns the mismatching leaves (empty when they agree).
"""
function partial_mismatches(f, run, result)
    up, down = run(f.make(Scaled{1}, Scaled{1})), run(f.make(Scaled{-1}, Scaled{-1}))
    a, u, d = leaves(result), leaves(up), leaves(down)
    length(a) == length(u) == length(d) || return [(:length, length(a), length(u))]
    bad = Tuple{Int,Float64,Float64}[]
    for i in eachindex(a)
        fd = (Float64(u[i]) - Float64(d[i])) / (2 * STEP)
        g = partial(a[i])
        isapprox(g, fd; rtol = 1e-5, atol = 1e-6 * max(1, abs(Float64(primal(a[i]))))) || push!(bad, (i, g, fd))
    end
    return bad
end

function check_eltype_matrix(fixtures; broken = Dict(), untyped = (), flat = (), parameter_only = (),
                             unchecked_partials = Dict())
    @testset "Factor element types" begin
        for f in fixtures
            base = f.make(Float64, Float64)
            @testset "$(f.name)" begin
                for (name, run) in f.cases
                    ref = run(base)
                    for (P, T, label) in ((Float32, Float32, :Float32), (Float64, Float64, :Float64),
                                          (BigFloat, BigFloat, :BigFloat), (Dual, Dual, :Dual),
                                          (Float32, Float64, :mixed))
                        rtol = label in (:Float32, :mixed) ? 1f-4 * f.κ :
                               label === :BigFloat ? 1e-10 : 1e-12
                        reason = get(broken, (f.name, name, label), nothing)
                        expected = (f.name, name) in untyped ? Float64 :
                                   (f.name, name) in parameter_only ? P : promote_type(P, T)
                        compare = (f.name, name) in flat ? agrees_flat : agrees
                        @testset "$name $label" begin
                            if isnothing(reason)
                                result = run(f.make(P, T))
                                @test compare(result, ref, expected, rtol)
                                if label === :Dual && !haskey(unchecked_partials, (f.name, name))
                                    bad = partial_mismatches(f, run, result)
                                    isempty(bad) || @info "partials differ from the central difference" f.name name bad
                                    @test isempty(bad)
                                end
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
