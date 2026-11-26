using GenericLinearAlgebra
using Test, Lenticulum, LenticulumCore

isdefined(@__MODULE__, :EltypeMatrix) || include("../../LenticulumCore.jl/test/support/eltype_matrix.jl")
isdefined(@__MODULE__, :LenticulumEltypeFixtures) || include("support/eltype_fixtures.jl")

@testset "Lenticulum element-type fixtures" begin
    fixtures = LenticulumEltypeFixtures.fixtures(; extended = true)
    EltypeMatrix.check_factor_coverage(fixtures, (Lenticulum, LenticulumCore))
    # All-latent Split has no numerical data, parameters or stored numeric configuration.
    untyped = [(:SplitFactor, op) for f in fixtures if f.name === :SplitFactor
               for (op, _) in f.cases if endswith(string(op), "_given_latent")]
    # With only the discrete channel observed, or no observations, the continuous
    # regime messages are constant likelihoods. Their unit-scale Schur complements
    # round to zero within the matrix tolerance. Both canonical forms are valid.
    flat = [(:RegimeFactor, op) for f in fixtures if f.name === :RegimeFactor
            for (op, _) in f.cases if any(s -> endswith(string(op), s),
                                         ("_x_given_k", "_x_given_latent", "_y_given_k", "_y_given_latent"))]
    # Hard-constraint BP uses the target cavity for its working type even on
    # an all-latent call. Its mean-field delegation has no target cavity.
    parameter_only = filter(EltypeMatrix.latent_cases(fixtures)) do (factor, op)
        return factor ∉ (:HardConstraintFactor, :HardConstraintNoisy) ||
               startswith(string(op), "mean_field_message_")
    end
    # Partials checked against a central difference, except where a value is stopped or laid out
    # differently on purpose (LP-094 finding 10).
    unchecked_partials = Dict{Tuple{Symbol,Symbol},String}()
    for op in (:factor_belief, :belief_free_energy, :factor_beliefs, :belief_free_energies)
        unchecked_partials[(:ScaleShiftFactor, op)] =
            "the frozen belief stops its input energy's gradient by design; the free energy's derivative " *
            "with the belief held fixed matches a central difference (LP-094, Mistral's probe B2)"
    end
    unchecked_partials[(:RegimeFactor, :factor_beliefs)] =
        "the Dual path returns one frozen belief per replica, the float path one stacked belief: the " *
        "leaves differ in layout (`agrees` compares them replica by replica; `factor_belief` is checked)"
    EltypeMatrix.check_eltype_matrix(fixtures; untyped, flat, parameter_only, unchecked_partials)
end
