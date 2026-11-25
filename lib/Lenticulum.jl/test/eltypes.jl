using Test, Lenticulum, LenticulumCore

isdefined(@__MODULE__, :EltypeMatrix) || include("../../LenticulumCore.jl/test/support/eltype_matrix.jl")
isdefined(@__MODULE__, :LenticulumEltypeFixtures) || include("support/eltype_fixtures.jl")

@testset "Lenticulum element-type fixtures" begin
    fixtures = LenticulumEltypeFixtures.fixtures()
    EltypeMatrix.check_factor_coverage(fixtures, (Lenticulum, LenticulumCore))
    broken = Dict{Tuple{Symbol,Symbol,Symbol},String}()
    for f in fixtures, (op, _) in f.cases
        if f.name in (:HardConstraintFactor, :ScaleShiftFactor)
            broken[(f.name, op, :BigFloat)] = "Hard-direction eigen and SVD kernels require LAPACK"
        elseif f.name === :SplitFactor && !endswith(string(op), "_given_latent")
            broken[(f.name, op, :BigFloat)] = "Extended-Gaussian spectral kernels require LAPACK"
        end
    end
    for op in (:message_x_given_latent, :message_y_given_latent)
        broken[(:LinearConstraintFactor, op, :BigFloat)] = "The flat residual metric uses a full LAPACK SVD"
    end
    broken[(:LinearConstraintFactor, :local_free_energy, :BigFloat)] =
        "The joint precision rank check uses LAPACK eigenvalues"
    for op in (:message_x_given_k, :message_x_given_latent, :message_y_given_k,
               :message_y_given_latent, :message_k_given_x, :message_k_given_y, :message_k_given_latent)
        broken[(:SwitchingConstraintFactor, op, :BigFloat)] = "The flat residual metric uses a full LAPACK SVD"
    end
    broken[(:RegimeFactor, :message_k_given_latent, :BigFloat)] =
        "Mycelium's diffuse cluster integral uses a LAPACK eigendecomposition"
    for (name, ops) in ((:LikelihoodFactor, (:local_free_energy,)),
                        (:PoissonEP, (:message_x_given_y, :message_y_given_x, :local_free_energy)))
        for op in ops
            broken[(name, op, :BigFloat)] = "The Gauss–Hermite rule uses a LAPACK tridiagonal eigensolver"
        end
    end

    # All-latent Split has no numerical data, parameters or stored numeric configuration.
    untyped = [(:SplitFactor, Symbol(:message_, target, :_given_latent)) for target in (:x, :y, :z)]
    # With only the discrete channel observed, or no observations, the continuous
    # regime messages are constant likelihoods. Their unit-scale Schur complements
    # round to zero within the matrix tolerance. Both canonical forms are valid.
    flat = [(:RegimeFactor, Symbol(:message_, target, :_given_, given))
            for target in (:x, :y) for given in (:k, :latent)]
    EltypeMatrix.check_eltype_matrix(fixtures; broken, untyped, flat)
end
