# L-SPEC-001 §1–§5, with explicit fixtures so no integration or energy is guessed.
_in_energyspace(x::Real, ::ScalarEnergySpace) = true
_in_energyspace(x::AbstractVector{<:Real}, E::EuclideanEnergySpace) = length(x) == dimension(E)
function _in_energyspace(x::GradedEnergy, E::GradedEnergySpace)
    return keys(x) == keys(E.parts) &&
           all(_in_energyspace(x[k], E.parts[k]) for k in keys(x))
end
_in_energyspace(x, E::LenticulumCore.AbstractEnergySpace) = false

# Scalar agreement can conceal cancellation between graded components.
_energy_isapprox(a::GradedEnergy, b::GradedEnergy) =
    keys(a) == keys(b) && all(_energy_isapprox(a[k], b[k]) for k in keys(a))
_energy_isapprox(a, b) = isapprox(a, b; atol = 1e-10, rtol = 1e-8)

# Exercise override forwarding without assuming a particular grading or a polarity-free space.
_conformance_scalarisation(::ScalarEnergySpace) = WeightedSum(2.0)
_conformance_scalarisation(E::EuclideanEnergySpace) = WeightedSum(fill(2.0, dimension(E)))
_conformance_scalarisation(E::GradedEnergySpace) =
    GradedScalarisation(map(_conformance_scalarisation, E.parts))

"""
    check_factor_conformance(f, p; π, inputs, ps, st, pointwise, game_prior, game_y, msgs,
                             direct_message = false, expected_site, expected_posterior)

Check the testable L-SPEC-001 §1–§5 interface for a factor and polarity. Fixtures explicitly
supply the cavity, observed inputs, pointwise arguments, and game/graph beliefs: the game
API's input/output order need not match the inversion polarity. Numerical agreement checks
the factor's declared algorithm, including its documented approximations; it does not certify
that an approximate inversion or quadrature is exact.

For a direct-message factor, set `direct_message = true` and supply independently computed
`expected_site` and `expected_posterior` beliefs. This path checks the extrinsic message,
the posterior obtained by pooling it with `π`, and exclusion of the target's prior and
incoming message. It makes no claim that the factor implements a Bayesian lens (whose
inversion would also need a joint latent posterior).

The optional `pointwise_gap`, `game_gap`, and `message_gap` strings explicitly omit only
that section for a documented unsupported contract. `polarity_gap` omits only enumeration
membership for a polarity accepted by `supports_polarity` but missing from the list.
Callers must test/record the gap separately; these runs are partial conformance.
`p = nothing` is reserved for sinks (no polarities), with an explicit `message_gap`.
Independent site/posterior fixtures may also be given on the lens path. All other checks
are shared; the default path retains the assemble/invert/likelihood_message checks.
"""
function check_factor_conformance(f, p; π, inputs, ps, st, pointwise = nothing,
                                  game_prior = nothing, game_y = nothing, msgs,
                                  direct_message = false,
                                  expected_site = nothing, expected_posterior = nothing,
                                  pointwise_gap = nothing, game_gap = nothing,
                                  message_gap = nothing, polarity_gap = nothing)
    for gap in (pointwise_gap, game_gap, message_gap, polarity_gap)
        gap === nothing || (gap isa AbstractString && !isempty(gap)) ||
            throw(ArgumentError("conformance gaps must be nonempty documented reasons"))
    end
    @testset "$(nameof(typeof(f))): $(p === nothing ? "sink" : string(observed_channels(p), " → ", unobserved_channels(p)))" begin
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
        if p === nothing
            @test isempty(supported_polarities(f))
            @test message_gap !== nothing
        else
            @test Tuple(channelname(c) for c in chans) == keys(p)
            @test ispartition(p)
            @test supports_polarity(f, p)
            if polarity_gap === nothing
                @test p in supported_polarities(f)
            end
        end
        @test islearnable(f) isa Bool
        @test isfrozen(f) isa Bool
        for trait in (energyspace, pointwise_energyspace)
            @test (p === nothing ? trait(f) : trait(f, p)) isa LenticulumCore.AbstractEnergySpace
            @test trait(f) isa LenticulumCore.AbstractEnergySpace
        end
        for trait in (scalarisation, pointwise_scalarisation)
            @test (p === nothing ? trait(f) : trait(f, p)) isa LenticulumCore.AbstractScalarisation
        end

        if pointwise_gap === nothing
            l, _ = LenticulumCore.energy(f, pointwise..., ps, st)
            @test _in_energyspace(l, pointwise_energyspace(f, p))
            sl, _ = scalar_energy(f, pointwise..., ps, st; polarity = p)
            @test sl isa Real && isfinite(sl)
            @test sl ≈ scalarise(localise(pointwise_scalarisation(f, p), f, pointwise..., ps, st), l)

        end

        if message_gap === nothing
            target = only(unobserved_channels(p))
            if direct_message
                expected_site isa LenticulumCore.AbstractBelief ||
                    throw(ArgumentError("direct-message conformance requires an independent expected_site"))
                expected_posterior isa LenticulumCore.AbstractBelief ||
                    throw(ArgumentError("direct-message conformance requires an independent expected_posterior"))
                site, next = factor_message(f, target, p, inputs, π, ps, st)
                @test site isa LenticulumCore.AbstractBelief
                @test next isa NamedTuple
                @test belief_distance(site, expected_site) ≤ 1e-8
                post = combine(π, site)
                @test post isa LenticulumCore.AbstractBelief
                @test belief_distance(post, expected_posterior) ≤ 1e-8
                without_prior, _ = factor_message(f, target, p, inputs, TrivialBelief(), ps, st)
                @test belief_distance(site, without_prior) ≤ 1e-8
                with_target = merge(inputs, NamedTuple{(target,)}((π,)))
                excluded, _ = factor_message(f, target, p, with_target, π, ps, st)
                @test belief_distance(site, excluded) ≤ 1e-8
            else
                lens, _ = assemble(f, p, ps, st)
                @test lens isa LenticulumCore.AbstractBayesianLens
                site, _ = likelihood_message(lens, π, inputs, ps, st)
                post, _ = invert(lens, π, inputs, ps, st)
                @test site isa LenticulumCore.AbstractBelief
                @test post isa LenticulumCore.AbstractBelief
                @test belief_distance(combine(π, site), post) ≤ 1e-8
                if expected_site !== nothing
                    @test belief_distance(site, expected_site) ≤ 1e-8
                end
                if expected_posterior !== nothing
                    @test belief_distance(post, expected_posterior) ≤ 1e-8
                end
                graph_site, _ = factor_message(f, target, p, inputs, π, ps, st)
                @test belief_distance(graph_site, site) ≤ 1e-8
            end

        end

        local_F, _ = local_free_energy(f, msgs, ps, st)
        @test _in_energyspace(local_F, energyspace(f))
        @test isfinite(scalarise(scalarisation(f), local_F))
        if game_gap === nothing
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
            @test _energy_isapprox(local_F, F)
            @test scalarise(scalarisation(f, p), local_F) ≈ sf
        end
    end
end
