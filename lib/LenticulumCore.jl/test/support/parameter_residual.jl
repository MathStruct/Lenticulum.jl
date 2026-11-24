struct ParameterResidual <: AbstractLenticulumFactor end
LenticulumCore.energy(::ParameterResidual, x, a, y, ps, st) =
    (y .- ps.block.A * x .- ps.b .- ps.c[1], (calls = st.calls + 1,))
LenticulumCore.pointwise_scalarisation(::ParameterResidual) = SquaredNorm()
LenticulumCore.free_energy(f::ParameterResidual, x, y, ps, st) = scalar_energy(f, x, nothing, y, ps, st)
