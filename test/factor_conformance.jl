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
In wired scope, `p = nothing` is reserved for sinks (no polarities), with an explicit `message_gap`.
Independent site/posterior fixtures may also be given on the lens path. A joint lens
requires `expected_target_sites`, a NamedTuple of independent per-target graph fixtures. All other checks
are shared; the default path retains the assemble/invert/likelihood_message checks.

`scope = :game` checks only parameter/state traits and the game algebra, requiring independent
`expected_free_energy` and `expected_entropy` fixtures. Use it for unwired product games or
stochastic sources whose repeated messages are different draws. It does not claim channel,
pointwise, message or graph conformance. Distributional observations can be passed directly
as `game_y` in either game or wired scope. Optional expected game fixtures on the default
`:wired` path also check every energy coordinate and the graph value against the closed form.

`scope = :bound` checks parameters/states and `free_energy_upper_bound` only. Supply
`bound = (evidence = x, kwargs = (; times), exact = exact_nll)`, where `exact` is an independent
finite scalar negative log evidence for the bound API's normalised model. The result must be
an evaluated `FreeEnergyUpperBound`, finite and at least `exact` (roundoff tolerance 1e-10).
Monte Carlo estimates are never accepted as bounds. These two scopes require `p = nothing`;
unlike a wired sink, they do not assert that supported polarities are empty.
"""
function check_factor_conformance(f, p; π = nothing, inputs = nothing, ps, st, pointwise = nothing,
                                  game_prior = nothing, game_y = nothing, msgs = nothing,
                                  direct_message = false,
                                  expected_site = nothing, expected_posterior = nothing,
                                  expected_target_sites = nothing,
                                  expected_free_energy = nothing, expected_entropy = nothing,
                                  scope = :wired, bound = nothing,
                                  pointwise_gap = nothing, game_gap = nothing,
                                  message_gap = nothing, polarity_gap = nothing)
    scope in (:wired, :game, :bound) || throw(ArgumentError("unknown conformance scope"))
    if scope !== :wired
        p === nothing || throw(ArgumentError("game/bound scope requires p=nothing"))
        all(isnothing, (pointwise_gap, game_gap, message_gap, polarity_gap)) ||
            throw(ArgumentError("game/bound scope does not accept wired-section gaps"))
    end
    if scope === :game
        expected_free_energy !== nothing && expected_entropy !== nothing ||
            throw(ArgumentError("game scope requires independent free-energy and entropy fixtures"))
    elseif scope === :bound
        bound isa NamedTuple && keys(bound) == (:evidence, :kwargs, :exact) ||
            throw(ArgumentError("bound scope requires (evidence, kwargs, exact) fixture"))
        bound.kwargs isa NamedTuple && bound.exact isa Real && isfinite(bound.exact) ||
            throw(ArgumentError("bound fixture requires keyword NamedTuple and finite exact value"))
    end
    for gap in (pointwise_gap, game_gap, message_gap, polarity_gap)
        gap === nothing || (gap isa AbstractString && !isempty(gap)) ||
            throw(ArgumentError("conformance gaps must be nonempty documented reasons"))
    end
    @testset "$(nameof(typeof(f))): $(scope !== :wired ? string(scope) : p === nothing ? "sink" : string(observed_channels(p), " → ", unobserved_channels(p)))" begin
        rng = Xoshiro(42)
        @test LuxCore.initialparameters(rng, f) isa NamedTuple
        @test LuxCore.initialstates(rng, f) isa NamedTuple
        if applicable(LuxCore.parameterlength, f)
            @test LuxCore.parameterlength(f) ≥ 0
        end
        if applicable(LuxCore.statelength, f)
            @test LuxCore.statelength(f) ≥ 0
        end
        @test islearnable(f) isa Bool
        @test isfrozen(f) isa Bool
        if scope === :bound
            B, next = LenticulumCore.free_energy_upper_bound(f, bound.evidence, ps, st; bound.kwargs...)
            @test B isa LenticulumCore.FreeEnergyUpperBound
            @test next isa NamedTuple
            @test !(B isa Real)
            @test isfinite(B.value)
            @test B.value ≥ bound.exact - 1e-10
            @test_throws MethodError scalarise(scalarisation(f), B)
        end
        if scope === :wired
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
                if direct_message
                    target = only(unobserved_channels(p))
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
                    targets = unobserved_channels(p)
                    if length(targets) == 1
                        graph_site, _ = factor_message(f, only(targets), p, inputs, π, ps, st)
                        @test belief_distance(graph_site, site) ≤ 1e-8
                    else
                        expected_target_sites isa NamedTuple || throw(ArgumentError(
                            "joint lens conformance requires independent expected_target_sites"))
                        for target in targets
                            graph_site, _ = factor_message(f, target, p, inputs, π, ps, st)
                            @test belief_distance(graph_site, getfield(expected_target_sites, target)) ≤ 1e-8
                        end
                    end
                end

            end

            local_F, _ = local_free_energy(f, msgs, ps, st)
            @test _in_energyspace(local_F, energyspace(f))
            @test isfinite(scalarise(scalarisation(f), local_F))
            if expected_free_energy !== nothing
                @test _energy_isapprox(local_F, expected_free_energy)
            end
        end # wired scope
        if scope === :game || (scope === :wired && game_gap === nothing)
            @test energyspace(f) isa LenticulumCore.AbstractEnergySpace
            @test scalarisation(f) isa LenticulumCore.AbstractScalarisation
            F, next_F = free_energy(f, game_prior, game_y, ps, st)
            H, next_H = LenticulumCore.entropy(f, game_prior, game_y, ps, st)
            @test next_F isa NamedTuple
            @test next_H isa NamedTuple
            @test _in_energyspace(F, energyspace(f, p))
            @test _in_energyspace(H, energyspace(f, p))
            sf, _ = scalar_free_energy(f, game_prior, game_y, ps, st; polarity = p)
            @test sf isa Real && isfinite(sf)
            @test sf ≈ scalarise(scalarisation(f, p), F)
            @test isfinite(scalarise(scalarisation(f, p), H))
            if expected_free_energy !== nothing
                @test _in_energyspace(expected_free_energy, energyspace(f, p))
                @test _energy_isapprox(F, expected_free_energy)
            end
            if expected_entropy !== nothing
                @test _in_energyspace(expected_entropy, energyspace(f, p))
                @test _energy_isapprox(H, expected_entropy)
            end
            σ = _conformance_scalarisation(energyspace(f, p))
            custom, _ = scalar_free_energy(f, game_prior, game_y, ps, st; σ)
            @test custom ≈ scalarise(σ, F)
            if scope === :wired
                @test _energy_isapprox(local_F, F)
                @test scalarise(scalarisation(f, p), local_F) ≈ sf
            end
        end
    end
end
