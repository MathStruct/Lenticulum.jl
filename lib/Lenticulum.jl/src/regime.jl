# ---------------------------------------------------------------------------
# The regime factor:   0 = Σᵢ A_{i,k} xᵢ − b_k + ε,   ε ~ N(0, Q_k),   k ∈ 1:K, P(k) = w_k
#
# A `SwitchingConstraintFactor` whose coefficients and noise depend on the outcome as well as its
# right-hand side (`switching.md` §5, the first item of "What it does not do"). Each outcome
# gives a Gaussian with its own precision, so the factor is stated only through
# `Mycelium.cluster_form` with per-outcome A and Q (`cluster.md`, "Per-outcome coefficients and
# noise"). Inside a cluster node it is exact; outside one its messages and free energy are those
# of a one-factor cluster (`Mycelium.as_cluster`).
#
# See `regime.md`.
# ---------------------------------------------------------------------------

"""
    RegimeFactor(dims::NamedTuple, K::Int, m::Int; switch = :k, noise = 0.1, learn_noise = false,
                 output = nothing, output_coeff = I, init_coeff, init_offsets)

The switching soft linear relation whose coefficients and noise depend on the outcome
``k \\in \\{1, \\dots, K\\}`` (a regime):

```math
f(x_1, \\ldots, x_n, k) \\;=\\; w_k\\, \\mathcal N\\Bigl(\\textstyle\\sum_i A_{i,k} x_i - b_k;\\ 0,\\ Q_k\\Bigr).
```

`dims` names the continuous channels, `m` is the residual dimension, the outcome is the channel
`switch`. The parameters are `(A, b, logw)`: `A` a `NamedTuple` of `m × dims[i] × K` arrays,
`b` the `m × K` right-hand sides, `logw` the unnormalised log-weights. `noise` is shared by the
regimes as a scalar, a real vector (a diagonal) or an `m × m` matrix, or one per regime as a
vector of `K` matrices or an `m × m × K` array.
With `learn_noise = true` the noise is a parameter `L` too (`m × m × K`: ``Q_k = L_k L_k^\\top``,
``L_k`` lower triangular with the exponential of `L[:, :, k]`'s diagonal on its diagonal),
initialised at `noise`.

`output = :y` (`dims[:y] == m`) fixes the coefficient of `y` at `output_coeff` in every regime and
drops it from the parameters: the factor is then a normalised conditional density of `y` given
the other channels, a mixture of regressions. Without it, learning every coefficient (above all
with `learn_noise`) has no lower bound: scaling `A` and `L` together lowers ``-\\log f`` (LP-022).

Mixtures of regressions (E1's crossing lines: `A_{z,k} = [-a_k\\ 1]`), Gaussian mixtures with one
covariance per component, mixtures of factor analysers (``z = W_k x + \\mu_k + \\varepsilon_k`` with a
Gaussian `x`) and switching linear dynamics are instances. The factor is exact inside a
[`Mycelium.cluster`](@ref) node. Outside one it sends exact messages and has its exact Bethe
free energy as a one-factor cluster ([`Mycelium.as_cluster`](@ref)), which needs every
continuous input but the target proper or observed, and a message to a continuous channel
whose regimes give improper components only if they share their precision.

See `regime.md`.
"""
struct RegimeFactor{names,D<:Tuple,Q<:AbstractArray,W,B} <: LenticulumCore.AbstractLenticulumFactor
    dims::NamedTuple{names,D}
    K::Int
    m::Int
    switch::Symbol
    Q::Q                      # m × m × K, the fixed noise or the initial one
    learn_noise::Bool
    init_coeff::W
    init_offsets::B
    output::Union{Symbol,Nothing}
    output_coeff::Union{Matrix{Float64},Nothing}
end

function RegimeFactor(dims::NamedTuple, K::Int, m::Int; switch::Symbol = :k, noise = 0.1, learn_noise::Bool = false,
                      init_coeff = (rng, m, n) -> randn(rng, m, n) ./ sqrt(n),
                      init_offsets = (rng, m, K) -> randn(rng, m, K),
                      output::Union{Symbol,Nothing} = nothing, output_coeff = I)
    isempty(dims) && throw(ArgumentError("a regime factor needs at least one continuous channel"))
    all(d -> d isa Int && d > 0, values(dims)) || throw(ArgumentError("channel dimensions must be positive Ints; got $(dims)"))
    switch in keys(dims) && throw(ArgumentError("the switch channel :$switch is also a continuous channel"))
    K > 0 || throw(ArgumentError("the number of regimes K must be positive; got $K"))
    m > 0 || throw(ArgumentError("the residual dimension m must be positive; got $m"))
    per = noise isa AbstractArray{<:Real,3} ? [noise[:, :, k] for k in axes(noise, 3)] :
          noise isa AbstractVector && !(eltype(noise) <: Real) ? noise : fill(noise, K)
    length(per) == K || throw(ArgumentError("noise: $(length(per)) covariances for $K regimes"))
    Q = zeros(m, m, K)
    for k in 1:K
        Qk = _as_cov(per[k], m)
        size(Qk) == (m, m) || throw(ArgumentError("noise covariance $k must have size ($m, $m)"))
        LenticulumCore._nearly_symmetric(Qk) || throw(ArgumentError("noise covariance $k must be symmetric"))
        Qk = Matrix(LenticulumCore._symmetric_part(Qk))
        isposdef(Symmetric(Qk)) || throw(ArgumentError("noise covariance $k must be positive definite"))
        Q[:, :, k] = Qk
    end
    return RegimeFactor(dims, K, m, switch, Q, learn_noise, init_coeff, init_offsets, output, _output_coeff(dims, m, output, output_coeff))
end

function LuxCore.initialparameters(rng::AbstractRNG, f::RegimeFactor)
    A = NamedTuple{_learned(f)}(map(n -> cat((f.init_coeff(rng, f.m, f.dims[n]) for _ in 1:f.K)...; dims = 3), _learned(f)))
    base = (A = A, b = f.init_offsets(rng, f.m, f.K), logw = zeros(f.K))
    f.learn_noise || return base
    L = zeros(f.m, f.m, f.K)
    for k in 1:f.K
        C = cholesky(Symmetric(f.Q[:, :, k])).L
        L[:, :, k] = LinearAlgebra.tril(Matrix(C), -1) + Diagonal(log.(diag(C)))
    end
    return merge(base, (L = L,))
end
LuxCore.initialstates(::AbstractRNG, ::RegimeFactor) = NamedTuple()
LuxCore.parameterlength(f::RegimeFactor) =
    f.K * (f.m * sum(f.dims[n] for n in _learned(f); init = 0) + f.m + 1) + (f.learn_noise ? f.K * f.m * (f.m + 1) ÷ 2 : 0)
# the output's fixed coefficient, the same in every regime
_fixed_output(f::RegimeFactor) = f.output_coeff === nothing ? nothing : repeat(f.output_coeff, 1, 1, f.K)

_allnames(f::RegimeFactor{names}) where {names} = (names..., f.switch)
LenticulumCore.channels(f::RegimeFactor{names}) where {names} =
    (map((n, d) -> LenticulumCore.Channel(n, d), names, values(f.dims))..., LenticulumCore.Channel(f.switch, f.K))
LenticulumCore.supported_polarities(f::RegimeFactor) = _one_unobserved_polarities(_allnames(f))
LenticulumCore.supports_polarity(f::RegimeFactor, p::LenticulumCore.Polarity) =
    Set(keys(p)) == Set(_allnames(f)) && length(LenticulumCore.unobserved_channels(p)) == 1
LenticulumCore.hasnopolarity(::RegimeFactor) = false
LenticulumCore.isunidirectional(::RegimeFactor) = false
LenticulumCore.islearnable(::RegimeFactor) = true
LenticulumCore.energyspace(::RegimeFactor) = LenticulumCore.GradedEnergySpace((
    fit = LenticulumCore.ScalarEnergySpace(), complexity = LenticulumCore.ScalarEnergySpace(),
    negentropy = LenticulumCore.ScalarEnergySpace()))
LenticulumCore.scalarisation(::RegimeFactor) = LenticulumCore.GradedScalarisation((
    fit = LenticulumCore.IdentityScalarisation(), complexity = LenticulumCore.IdentityScalarisation(),
    negentropy = LenticulumCore.IdentityScalarisation()))

# L_k with the exponential of its diagonal; the strictly lower part masked (no mutation, for AD)
function _regime_cholesky(R::AbstractMatrix)
    m = size(R, 1)
    mask = [i > j ? 1.0 : 0.0 for i in 1:m, j in 1:m]
    return R .* mask .+ Diagonal(exp.(diag(R)))
end

"""
    regime_covariances(f::RegimeFactor, ps) -> Array

The noise covariances ``Q_k`` as an `m × m × K` array: the fixed ones, or ``L_k L_k^\\top`` from the
parameter `L` with `learn_noise = true`.
"""
function regime_covariances(f::RegimeFactor, ps)
    f.learn_noise || return f.Q
    return cat(map(k -> (L = _regime_cholesky(ps.L[:, :, k]); L * L'), 1:f.K)...; dims = 3)
end

Mycelium.cluster_switch(f::RegimeFactor) = f.switch
Mycelium.cluster_form(f::RegimeFactor, ps) =
    Mycelium.ClusterForm(_coeffs(f, ps), ps.b, regime_covariances(f, ps), _logweights(ps), f.switch)

# Outside a cluster: a one-factor cluster over the factor's channels.
Mycelium.uses_cavity(::RegimeFactor) = true
Mycelium.factor_message(f::RegimeFactor, target::Symbol, polarity, inputs, prior, ps, st) =
    Mycelium.factor_message(Mycelium.as_cluster(f), target, polarity, inputs, prior, (factor = ps,), st)
Mycelium.factor_belief(f::RegimeFactor, msgs, ps, st) = Mycelium.factor_belief(Mycelium.as_cluster(f), msgs, (factor = ps,), st)
Mycelium.belief_free_energy(f::RegimeFactor, b::Mycelium.FrozenClusterBelief, ps, st) =
    Mycelium.belief_free_energy(Mycelium.as_cluster(f), b, (factor = ps,), st)
Mycelium.local_free_energy(f::RegimeFactor, msgs, ps, st) = Mycelium.belief_free_energy(f, Mycelium.factor_belief(f, msgs, ps, st), ps, st)

# Over a plate: the replicas as tied one-factor clusters, batched when their inputs share their
# shape (`cluster_batch.jl`; replica by replica otherwise). Equal factors share one cluster.
const _as_clusters = Mycelium._as_clusters
Mycelium.factor_messages(fs::AbstractVector{<:RegimeFactor}, target::Symbol, polarity, inputs::AbstractVector,
                         priors::AbstractVector, ps, st) =
    Mycelium.factor_messages(_as_clusters(fs), target, polarity, inputs, priors, (factor = ps,), st)
Mycelium.factor_beliefs(fs::AbstractVector{<:RegimeFactor}, msgs::NamedTuple, ps, st) =
    Mycelium.factor_beliefs(_as_clusters(fs), msgs, (factor = ps,), st)
Mycelium.belief_free_energies(fs::AbstractVector{<:RegimeFactor}, B::Mycelium.FrozenClusterBeliefs, ps, st) =
    Mycelium.belief_free_energies(_as_clusters(fs), B, (factor = ps,), st)
Mycelium.local_free_energies(fs::AbstractVector{<:RegimeFactor}, msgs::NamedTuple, ps, st) =
    Mycelium.local_free_energies(_as_clusters(fs), msgs, (factor = ps,), st)

# Without a declared output a constraint factor is normalised in its residual only (LP-022):
# `free_energy_gradient` warns when one learns its coefficients on data.
Mycelium.learns_unnormalised(f::Union{LinearConstraintFactor,SwitchingConstraintFactor,RegimeFactor}) = f.output === nothing

