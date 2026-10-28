# Run with --project=lib/LenticulumCore.jl/test.
# Intermittent NaN in forward-mode Enzyme Jacobians (LP-002). Lenticulum code is not
# involved: `Enzyme.jacobian(Forward, (u,A,x) -> u .- A*x, y, Const(A), Const(x))` returns
# a NaN entry in about 5% of calls (Enzyme 0.13.216, Julia 1.13.1); reverse mode does not.
# DI's `AutoEnzyme()` Jacobian takes the forward path, so `energy_jacobian(...; ad =
# AutoEnzyme())` inherits it. Use `AutoEnzyme(; mode = Enzyme.Reverse)` for Jacobians.
# Reported upstream: https://github.com/EnzymeAD/Enzyme.jl/issues/3884
using Enzyme, LinearAlgebra

function enzyme_forward_nan_probe(n = 2000)
    A, x, y = [0.3 0.2; 0.1 -0.4], [1.2, -0.7], [0.4, 0.9]
    f(u, A, x) = u .- A * x
    counts = map((Forward, Reverse)) do mode
        count(1:n) do _
            J = first(Enzyme.jacobian(mode, f, y, Const(A), Const(x)))
            !(all(isfinite, J) && J ≈ I(2))
        end
    end
    @show VERSION pkgversion(Enzyme) n
    println("forward failures: ", counts[1], ", reverse failures: ", counts[2])
    return counts
end

if abspath(PROGRAM_FILE) == @__FILE__
    enzyme_forward_nan_probe()
end
