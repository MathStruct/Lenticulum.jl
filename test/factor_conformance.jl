# L-SPEC-001 §1–§5, with explicit fixtures so no integration or energy is guessed.
_in_energyspace(x::Real, ::ScalarEnergySpace) = true
_in_energyspace(x::AbstractVector{<:Real}, E::EuclideanEnergySpace) = length(x) == dimension(E)
function _in_energyspace(x::GradedEnergy, E::GradedEnergySpace)
    return keys(x) == keys(E.parts) &&
           all(_in_energyspace(x[k], E.parts[k]) for k in keys(x))
end
_in_energyspace(x, E::LenticulumCore.AbstractEnergySpace) = false

# Exercise override forwarding without assuming a particular grading or a polarity-free space.
_conformance_scalarisation(::ScalarEnergySpace) = WeightedSum(2.0)
_conformance_scalarisation(E::EuclideanEnergySpace) = WeightedSum(fill(2.0, dimension(E)))
_conformance_scalarisation(E::GradedEnergySpace) =
    GradedScalarisation(map(_conformance_scalarisation, E.parts))

"""
    check_factor_conformance(f, p; π, inputs, ps, st, pointwise, game_prior, game_y, msgs)

Check the testable L-SPEC-001 §1–§5 interface for a factor and polarity. Fixtures explicitly
supply the cavity, observed inputs, pointwise arguments, and game/graph beliefs: the game
API's input/output order need not match the inversion polarity. Numerical agreement checks
the factor's declared algorithm, including its documented approximations; it does not certify
that an approximate inversion or quadrature is exact.
"""
function check_factor_conformance(f, p; π, inputs, ps, st, pointwise,
                                  game_prior, game_y, msgs)
    @testset "$(nameof(typeof(f))): $(observed_channels(p)) → $(unobserved_channels(p))" begin
        rng = Xoshiro(42)
        @test LuxCore.initialparameters(rng, f) isa NamedTuple
        @test LuxCore.initialstates(rng, f) isa NamedTuple
        if applicable(LuxCore.parameterlength, f)
            @test LuxCore.parameterlength(f) ≥ 0
        end
        if applicable(LuxCore.statelength, f)
            @test LuxCore.statelength(f) ≥ 0
        end
        chans = LenticulumCore.channels(f)
        @test !isempty(chans)
        @test Tuple(channelname(c) for c in chans) == keys(p)
        @test ispartition(p)
        @test supports_polarity(f, p)
        @test p in supported_polarities(f)
        @test islearnable(f) isa Bool
        @test isfrozen(f) isa Bool
        for trait in (energyspace, pointwise_energyspace)
            @test trait(f, p) isa LenticulumCore.AbstractEnergySpace
            @test trait(f) isa LenticulumCore.AbstractEnergySpace
        end
        for trait in (scalarisation, pointwise_scalarisation)
            @test trait(f, p) isa LenticulumCore.AbstractScalarisation
        end

        l, _ = energy(f, pointwise..., ps, st)
        @test _in_energyspace(l, pointwise_energyspace(f, p))
        sl, _ = scalar_energy(f, pointwise..., ps, st; polarity = p)
        @test sl isa Real && isfinite(sl)
        @test sl ≈ scalarise(pointwise_scalarisation(f, p), l)

        lens, _ = assemble(f, p, ps, st)
        @test lens isa LenticulumCore.AbstractBayesianLens
        site, _ = likelihood_message(lens, π, inputs, ps, st)
        post, _ = invert(lens, π, inputs, ps, st)
        @test site isa LenticulumCore.AbstractBelief
        @test post isa LenticulumCore.AbstractBelief
        @test belief_distance(combine(π, site), post) ≤ 1e-8
        target = only(unobserved_channels(p))
        graph_site, _ = factor_message(f, target, p, inputs, π, ps, st)
        @test belief_distance(graph_site, site) ≤ 1e-8

        F, _ = free_energy(f, game_prior, game_y, ps, st)
        H, _ = LenticulumCore.entropy(f, game_prior, game_y, ps, st)
        @test _in_energyspace(F, energyspace(f, p))
        @test _in_energyspace(H, energyspace(f, p))
        sf, _ = scalar_free_energy(f, game_prior, game_y, ps, st; polarity = p)
        @test sf isa Real && isfinite(sf)
        @test sf ≈ scalarise(scalarisation(f, p), F)
        @test isfinite(scalarise(scalarisation(f, p), H))
        σ = _conformance_scalarisation(energyspace(f, p))
        custom, _ = scalar_free_energy(f, game_prior, game_y, ps, st; σ)
        @test custom ≈ scalarise(σ, F)
        local_F, _ = local_free_energy(f, msgs, ps, st)
        @test _in_energyspace(local_F, energyspace(f, p))
        @test scalarise(scalarisation(f, p), local_F) ≈ sf
    end
end
