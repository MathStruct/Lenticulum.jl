"""
Reproduce the explicit-integrator table in `src/neuralode.md`:
    julia --project=lib/ImplicitLayers.jl lib/ImplicitLayers.jl/examples/flow_roundtrip.jl

BigFloat (256 bits) separates truncation error from cancellation/roundoff for the gentle
RK4 case. Endpoint = 3, T = 1. Residual = |3 - forward(reverse(3))| / 3.
"""
module FlowRoundTripMeasurements
using ImplicitLayers
using Printf

function measure(Integrator, a, n)
    vf(u, t) = a .* u
    endpoint = BigFloat[3]
    itg = Integrator(n)
    start = integrate(vf, endpoint, big"1", big"0", itg)
    back = integrate(vf, start, big"0", big"1", itg)
    exactstart = endpoint .* exp(-a)
    forward = integrate(vf, endpoint, big"0", big"1", itg)
    exactend = endpoint .* exp(a)
    return (; residual = abs(only(back - endpoint)) / 3,
            start_error = abs(only(start - exactstart)) / abs(only(exactstart)),
            forward_error = abs(only(forward - exactend)) / abs(only(exactend)))
end

function print_table()
    setprecision(256) do
        println("| scheme | a | N | relative residual | relative start error |")
        println("|---|---:|---:|---:|---:|")
        for (name, Integrator) in (("Euler", EulerIntegrator), ("RK4", RK4Integrator))
            for a in (big"0.5", big"-5"), n in (1, 25, 50, 100, 200)
                row = measure(Integrator, a, n)
                @printf("| %s | %.1f | %d | %.6e | %.6e |\n", name, a, n,
                        row.residual, row.start_error)
            end
        end
    end
end
end

if abspath(PROGRAM_FILE) == @__FILE__
    FlowRoundTripMeasurements.print_table()
end
