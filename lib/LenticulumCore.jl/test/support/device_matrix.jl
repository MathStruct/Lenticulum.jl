module DeviceMatrix

using Adapt, JLArrays, StaticArrays, Test, LinearAlgebra
using LenticulumCore
using ..EltypeMatrix: payload

export check_device_matrix, device_payload, device_agrees, device_array

# The device array type under test: JLArray (scalar indexing off, on the CPU) unless the runner
# defines `Main.LENTICULUM_DEVICE_ARRAY` first, as `gpu/runtests.jl` does with CuArray (Q8).
device_array() = isdefined(Main, :LENTICULUM_DEVICE_ARRAY) ? Main.LENTICULUM_DEVICE_ARRAY : JLArray

struct Views end
struct SmallStatic end

Adapt.adapt_storage(::Views, x::Array{<:Number}) = view(x, axes(x)...)
Adapt.adapt_storage(::SmallStatic, x::Array{<:Number}) =
    all(<=(4), size(x)) ? SArray{Tuple{size(x)...}}(x) : x

# Labels and the collection of mixture components are structural. Traverse
# the components and the numeric arrays inside each belief.
device_payload(x) = x
device_payload(x::Union{Tuple,NamedTuple}) = map(device_payload, x)
device_payload(x::AbstractArray{<:Number}) = x
device_payload(x::AbstractArray) = map(device_payload, x)
device_payload(b::AbstractBelief) = (p = payload(b); p === b ? b : device_payload(p))
device_payload(b::CategoricalBelief) = (; b.logp)
device_payload(b::ExtendedGaussian) = (; b.μ, b.Σ, b.D)
device_payload(b::GradedEnergy) = device_payload(payload(b))

on_device(x) = true
on_device(x::AbstractArray{<:Number}) = x isa device_array()
on_device(x::Diagonal) = on_device(x.diag)
on_device(x::Union{Tuple,NamedTuple,AbstractArray}) = all(on_device, x)

device_agrees(a, b, rtol) = _agrees(payload(a), payload(b), rtol)
device_agrees(a::CategoricalBelief, b::CategoricalBelief, rtol) =
    isequal(a.labels, b.labels) && _agrees(a.logp, b.logp, rtol)

_agrees(a, b, rtol) = isequal(a, b)
host_array(a) = Array(a)
host_array(a::Diagonal) = Array(Diagonal(Array(a.diag)))
_agrees(a::Number, b::Number, rtol) = isapprox(a, b; rtol, atol = rtol / 100)
function _agrees(a::AbstractArray{<:Number}, b::AbstractArray{<:Number}, rtol)
    size(a) == size(b) || return false
    A, B = host_array(a), host_array(b)
    return isequal(A, B) || isapprox(A, B; rtol, atol = rtol / 100)
end
function _agrees(a::AbstractArray, b::AbstractArray, rtol)
    return size(a) == size(b) && all(device_agrees(x, y, rtol) for (x, y) in zip(a, b))
end
function _agrees(a::Union{Tuple,NamedTuple}, b::Union{Tuple,NamedTuple}, rtol)
    return keys(a) == keys(b) && all(device_agrees(x, y, rtol) for (x, y) in zip(a, b))
end

# A Schur complement with zero precision can round differently between
# backends, changing whether identical constant likelihoods are coalesced.
constant_likelihood(x, atol) = false
constant_likelihood(::TrivialBelief, atol) = true
constant_likelihood(b::GaussianBelief, atol) =
    b.improper && maximum(abs, b.η; init=zero(eltype(b.η))) <= atol && maximum(abs, b.Λ; init=zero(eltype(b.Λ))) <= atol
constant_likelihood(b::MixtureBelief, atol) = all(c -> constant_likelihood(c, atol), b.components)

function check_device_matrix(fixtures; subset = false, broken = Dict(), flat = ())
    JLArrays.allowscalar(false)
    @testset "Factor array storage" begin
        for fixture in fixtures
            base = fixture.make(Float64, Float64)
            cases = subset ? Iterators.take(fixture.cases, 1) : fixture.cases
            @testset "$(fixture.name)" begin
                for (name, run) in cases
                    ref = run(base)
                    D = device_array()
                    for (backend, to, T) in ((:Array, Array, Float64), (:view, Views(), Float64),
                                             (:SArray, SmallStatic(), Float64), (:device, D, Float64),
                                             (:device32, D, Float32))
                        rtol = T === Float32 ? 1e-4 : 1e-10
                        @testset "$name $backend" begin
                            check() = begin
                                result = run(adapt(to, fixture.make(T, T)))
                                agrees = if (fixture.name, name) in flat
                                    constant_likelihood(result, rtol) && constant_likelihood(ref, rtol)
                                else
                                    device_agrees(result, ref, rtol)
                                end
                                agrees &&
                                    (to !== D || on_device(device_payload(result)))
                            end
                            reason = get(broken, (fixture.name, name, backend), nothing)
                            if reason === nothing
                                @test check()
                            else
                                @testset "$reason" begin
                                    @test_broken check()
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
