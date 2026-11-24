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

function matrix_row(name, differentiate, expected; broken=(), unstable=(), skipped=(), slow=(), forward=false,
                    forwarddiff_scenario=nothing)
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
                # A backend can change captured data before throwing.
                # Copy the whole fixture before invoking each backend.
                run_backend = deepcopy(differentiate)
                if name in unstable
                    # The outcome depends on what was compiled before (an
                    # Enzyme-internal error in one order, a correct value in
                    # another). A throw is recorded as broken, a returned value
                    # must be right.
                    value = try
                        run_backend(ad)
                    catch err
                        err isa InterruptException && rethrow()
                        nothing
                    end
                    if value === nothing
                        @test_broken false
                    else
                        @test isapprox(value, reference; rtol=1e-6)
                    end
                elseif name in broken
                    # A broken backend is expected to throw. If it returns, a
                    # correct value is an unexpected pass and a wrong value a
                    # failure: both are loud, so a broken row never hides a
                    # wrong derivative (LP-089).
                    value = try
                        run_backend(ad)
                    catch err
                        err isa InterruptException && rethrow()
                        nothing
                    end
                    if value === nothing
                        @test_broken false
                    elseif isapprox(value, reference; rtol=1e-6)
                        @test_broken true
                    else
                        @test isapprox(value, reference; rtol=1e-6)
                    end
                else
                    @test isapprox(run_backend(ad), reference; rtol=1e-6)
                    if name == "ForwardDiff" && forwarddiff_scenario !== nothing
                        forwarddiff_scenario(ad, copy(reference))
                    end
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
