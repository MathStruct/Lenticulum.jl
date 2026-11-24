module Q5ADMatrix

using Test
using ADTypes: AutoZygote, AutoForwardDiff, AutoEnzyme, AutoMooncake
import DifferentiationInterface as DI
import FiniteDifferences as FD
import Enzyme, ForwardDiff, Mooncake, Zygote

export matrix_row, gradient_row, skip_row, fast_tier, slow_tier

const BACKENDS = (
    "Zygote" => AutoZygote(),
    "ForwardDiff" => AutoForwardDiff(),
    "Enzyme" => AutoEnzyme(; mode=Enzyme.Reverse),
    "Mooncake" => AutoMooncake(; config=nothing),
)
const ENZYME_FORWARD = "Enzyme.Forward" => AutoEnzyme(; mode=Enzyme.Forward)

fast_tier() = get(ENV, "LENTICULUM_SLOW_TESTS", "false") != "only"
slow_tier() = get(ENV, "LENTICULUM_SLOW_TESTS", "false") in ("true", "only")

# Structural equality, for checking that a backend left its fixture unchanged:
# arrays and numbers by value, everything else field by field (closures too).
same(a::Number, b::Number) = isequal(a, b)
same(a::AbstractArray, b::AbstractArray) =
    typeof(a) == typeof(b) && size(a) == size(b) && all(i -> same(a[i], b[i]), eachindex(a, b))
same(a::AbstractDict, b::AbstractDict) =
    length(a) == length(b) && all(k -> haskey(b, k) && same(a[k], b[k]), keys(a))
same(a::Union{Symbol,AbstractString,Nothing,Missing,Module,Type}, b) = isequal(a, b)
function same(a, b)
    typeof(a) == typeof(b) || return false
    isbits(a) && return isequal(a, b)
    for i in 1:nfields(a)
        isdefined(a, i) == isdefined(b, i) || return false
        isdefined(a, i) && !same(getfield(a, i), getfield(b, i)) && return false
    end
    return true
end

function matrix_row(name, differentiate, expected; broken=(), mutates=(), skipped=(), slow=(), forward=false,
                    forwarddiff_scenario=nothing, check_fixture=true)
    backends = forward ? (BACKENDS..., ENZYME_FORWARD) : BACKENDS
    selected = filter(backends) do (name, _)
        return name in slow ? slow_tier() : fast_tier()
    end
    isempty(selected) && return
    @testset "AD matrix: $name" begin
        reference = deepcopy(expected)()
        for (name, ad) in selected
            @testset "$name" begin
                if name in skipped
                    @test_skip differentiate(ad)
                    continue
                end
                # A backend can change captured data, before throwing or while
                # returning a correct value (Enzyme writes adjoints into Const
                # memory reached through a dynamic call, LP-090). Each backend gets
                # its own copy of the fixture, and the copy is checked afterwards.
                run_backend = deepcopy(differentiate)
                snapshot = deepcopy(run_backend)
                value = try
                    run_backend(ad)
                catch err
                    err isa InterruptException && rethrow()
                    name in broken || rethrow()
                    nothing
                end
                if name in broken
                    # A broken backend is expected to throw. If it returns, a
                    # correct value is an unexpected pass and a wrong value a
                    # failure: both are loud, so a broken row never hides a
                    # wrong derivative (LP-089).
                    if value === nothing
                        @test_broken false
                    elseif isapprox(value, reference; rtol=1e-6)
                        @test_broken true
                    else
                        @test isapprox(value, reference; rtol=1e-6)
                    end
                else
                    @test isapprox(value, reference; rtol=1e-6)
                    if name == "ForwardDiff" && forwarddiff_scenario !== nothing
                        forwarddiff_scenario(ad, copy(reference))
                    end
                end
                # The fixture must be unchanged after the call (also after a throw).
                # Graph rows pass check_fixture=false: inference fills the graph's
                # caches and the message store, so their fixtures change legitimately.
                if !check_fixture
                elseif name in mutates
                    @test_broken same(run_backend, snapshot)
                else
                    @test same(run_backend, snapshot)
                end
            end
        end
    end
end

# Treat the fixture captured by the loss as constant data in Enzyme. This is
# the same DI context convention that LenticulumCore uses for its public AD APIs.
apply_loss(x, f) = f(x)
gradient(f, ad, x) = DI.gradient(f, ad, x)
gradient(f, ad::AutoEnzyme, x) = DI.gradient(apply_loss, ad, x, DI.Constant(f))

function gradient_row(name, f, x; reference=nothing, scenario=nothing, kwargs...)
    # The stencil can temporarily modify its input. Keep the fixture independent
    # of that buffer in case the loss also reads x through a held-fixed argument.
    expected() = reference === nothing ? only(FD.grad(FD.central_fdm(5, 1), f, copy(x))) : reference
    run_scenario = scenario === nothing ? nothing : (ad, reference) -> begin
        loss, argument = deepcopy((f, x))
        example = scenario.Scenario{:gradient,:out}(loss, argument; res1=reference, name)
        scenario.test_differentiation(ad, [example]; rtol=1e-6)
    end
    differentiate(ad) = gradient(f, ad, x)
    matrix_row(name, differentiate, expected; forwarddiff_scenario=run_scenario, kwargs...)
end

function skip_row(name; slow=false)
    (slow ? slow_tier() : fast_tier()) || return
    @testset "AD matrix: $name" begin
        for (backend, _) in BACKENDS
            @testset "$backend" begin
                @test_skip false
            end
        end
    end
end

end
