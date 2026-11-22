module ImplicitSolversForwardDiffExt

using ImplicitSolvers: ImplicitSolvers
using ForwardDiff: Dual, value

ImplicitSolvers.primal(x::Dual) = ImplicitSolvers.primal(value(x))

end
