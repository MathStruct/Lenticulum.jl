module CoreADMatrix

using LenticulumCore, LinearAlgebra, Test
include("support/ad_matrix.jl")
using .Q5ADMatrix
include("support/parameter_residual.jl")

@testset verbose=true "LenticulumCore AD backend matrix" begin
    f = ParameterResidual()
    x, y, st = [1.2, -0.7], [0.4, 0.9], (calls=0,)
    p = [0.2, -0.1]
    parameters(v) = (block=(A=[0.3 0.2; 0.1 -0.4], empty=(;)), b=v, c=(0.4,))
    gradient_row("energy/ParameterResidual", v -> sum(abs2, first(energy(f, x, nothing, y, parameters(v), st))), p;
                 forward=true)

    # The container factors define free-energy components and scalarisations,
    # and have no raw pointwise energy method in LenticulumCore.
    skip_row("energy/ComposedFactor")
    skip_row("energy/TensorFactor")

    ps = parameters(p)
    flatten = LenticulumCore._flatten
    gradient(ad) = flatten(LenticulumCore.energy_gradient(f, x, nothing, y, ps, st; ad))
    matrix_row("energy_gradient", gradient, () -> gradient(nothing))

    jacobian(ad) = energy_jacobian(f, x, nothing, y, ps, st; wrt=:ps, ad)
    matrix_row("energy_jacobian", jacobian, () -> jacobian(nothing))

    # Use a loss that reads a scalar leaf and an array leaf through the public API.
    tree = (block=(w=p,), scale=0.7)
    objective(q) = q.scale * sum(abs2, q.block.w)
    parameter(ad) = flatten(parameter_gradient(objective, tree; ad))
    matrix_row("parameter_gradient", parameter, () -> parameter(nothing))
end

end
