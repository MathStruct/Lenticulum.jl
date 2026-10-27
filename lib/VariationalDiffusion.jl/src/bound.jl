# A normalised finite VP chain, separate from the RED-Diff graph surrogate.
# q(x_i|x_{i-1}) = N(a_i x_{i-1}, β_i I), p(x_T)=N(0,I),
# pθ(x_{i-1}|x_i) = N((x_i-β_i εθ(x_i,t_i)/σ_i)/a_i, β_i I).

function _bound_chain(f, x, times)
    f.predictor.schedule isa VPSDE || throw(ArgumentError("diffusion bound requires VPSDE"))
    f.predictor.blocks === nothing || throw(ArgumentError("diffusion bound requires a shared noise time"))
    length(x) == statedim(f) || throw(DimensionMismatch("evidence must contain all joint channels"))
    all(isfinite,x) || throw(ArgumentError("evidence must be finite"))
    ts = collect(float.(times))
    !isempty(ts) && all(t -> isfinite(t) && 0 < t <= 1, ts) && all(>(0),diff(ts)) ||
        throw(ArgumentError("times must be strictly increasing in (0,1]"))
    α = alpha.(Ref(f.predictor.schedule), ts)
    σ² = sigma.(Ref(f.predictor.schedule), ts).^2
    a = α ./ vcat(1,α[1:end-1])
    β = 1 .- a.^2
    all(b -> 0 < b < 1, β) || throw(ArgumentError("chain transitions need representable variances in (0,1)"))
    return ts, α, σ², a, β
end

_bound_predictor_tree(::DiffusionFactor, ps, st) = (ps,st)
_bound_predictor_tree(::_ProxDMFactor, ps, st) = (ps.predictor,st.predictor)

# Returns a scalar contribution at x_i. The q posterior covariance is integrated exactly.
function _bound_term(x, z, ε, i, α, σ², a, β)
    n = length(x)
    ε isa AbstractVector && length(ε) == n || throw(DimensionMismatch("bound predictor must return a vector matching the joint dimension"))
    all(isfinite,ε) || throw(ArgumentError("bound predictor must return finite noise predictions"))
    mθ = (z .- β[i]/sqrt(σ²[i]) .* ε) ./ a[i]
    if i == 1
        return (sum(abs2,x-mθ)/β[i] + n*log(2π*β[i]))/2
    end
    v = β[i]*σ²[i-1]/σ²[i]
    mq = α[i-1]*β[i]/σ²[i] .* x + a[i]*σ²[i-1]/σ²[i] .* z
    return (sum(abs2,mq-mθ)/β[i] + n*(v/β[i]-1+log(β[i]/v)))/2
end
_bound_terminal(x, α, σ²) = (sum(abs2,α[end].*x) + length(x)*(σ²[end]-1-log(σ²[end])))/2

"""
    free_energy_upper_bound(f::DiffusionFactor, x, ps, st; times) -> (FreeEnergyUpperBound, st)

Exact negative ELBO of a finite, normalised VP Gaussian reverse chain for **full joint
point evidence** `x` (Lebesgue coordinates). `times` explicitly defines the model: terminal
N(0,I), reverse variance βᵢ I including the decoder, mean `(z-βᵢ εθ(z,tᵢ)/σᵢ)/aᵢ`.
The variational law is the forward chain conditioned on `x`. All normalisers are included.

Analytic expectation is supported for a one-component `GaussianMixtureEps` with the default
input adapter and no device transfer. Other predictors must use the explicitly sampled
`estimate_free_energy_upper_bound`. This bound concerns this finite chain's negative log
evidence; it is not a bound on the graph's `(clamp,score)` surrogate or a continuous-time
likelihood. No partial-evidence marginalisation or prox posterior entropy is claimed.
"""
function LenticulumCore.free_energy_upper_bound(f::DiffusionFactor, x::AbstractVector, ps, st; times)
    ts, α, σ², a, β = _bound_chain(f,x,times)
    pred = f.predictor
    pred.model isa GaussianMixtureEps && size(pred.model.μ₀,2) == 1 &&
        pred.input === default_input && pred.device === nothing || throw(ArgumentError(
            "exact diffusion ELBO expectation requires the analytic single-Gaussian score; use estimate_free_energy_upper_bound for other predictors"))
    isequal(pred.model.schedule,pred.schedule) || throw(ArgumentError(
        "exact diffusion ELBO requires the analytic score and predictor to share a schedule"))
    pp, sp = _bound_predictor_tree(f,ps,st)
    length(pp.μ[:,1]) == length(x) || throw(DimensionMismatch("Gaussian score dimension"))
    all(isfinite,pp.μ) && isfinite(pred.model.s) || throw(ArgumentError("Gaussian score parameters must be finite"))
    value = _bound_terminal(x,α,σ²)
    for i in eachindex(ts)
        z = α[i].*x
        e, next = epsilon(pred,z,ts[i],pp,sp)
        isequal(next,sp) || throw(ArgumentError("bound requires inference-mode predictor state"))
        # ε(z,t)=Kz+d. Only the analytic family certifies this affine identity.
        K = sqrt(σ²[i])/(α[i]^2*pred.model.s^2+σ²[i])
        M = (1-β[i]/sqrt(σ²[i])*K)/a[i]
        D = i == 1 ? -M : a[i]*σ²[i-1]/σ²[i]-M
        value += _bound_term(x,z,e,i,α,σ²,a,β) + length(x)*σ²[i]*D^2/(2β[i])
    end
    return LenticulumCore.FreeEnergyUpperBound(value), st
end

"""
    estimate_free_energy_upper_bound(f, x, ps, st; times, nsamples=1024, rng)

Monte Carlo negative-ELBO estimate for the same finite chain. Returns
`FreeEnergyBoundEstimate(value, standard_error, nsamples)`; an individual estimate is **not**
a numerical upper bound. Predictor calls must preserve inference-mode state. Each replicate
uses independent forward marginals at every time; the q posterior covariance is analytic.
"""
function LenticulumCore.estimate_free_energy_upper_bound(f::DiffusionFactor, x::AbstractVector, ps, st;
                                                        times, nsamples::Int=1024, rng::AbstractRNG)
    nsamples >= 2 || throw(ArgumentError("nsamples must be at least two"))
    ts, α, σ², a, β = _bound_chain(f,x,times)
    pp, sp = _bound_predictor_tree(f,ps,st)
    vals = zeros(nsamples)
    for k in eachindex(vals)
        vals[k] = _bound_terminal(x,α,σ²)
        for i in eachindex(ts)
            z = α[i].*x + sqrt(σ²[i]).*randn(rng,length(x))
            e, next = epsilon(f.predictor,z,ts[i],pp,sp)
            isequal(next,sp) || throw(ArgumentError("bound requires inference-mode predictor state"))
            vals[k] += _bound_term(x,z,e,i,α,σ²,a,β)
        end
    end
    m = sum(vals)/nsamples
    se = sqrt(sum(abs2,vals .- m)/(nsamples*(nsamples-1)))
    return LenticulumCore.FreeEnergyBoundEstimate(m,se,nsamples), st
end

for op in (:free_energy, :entropy)
    @eval LenticulumCore.$op(::DiffusionFactor, π, y, ps, st) = throw(ArgumentError(
        "DiffusionFactor has no exact game entropy/free energy; use free_energy_upper_bound for the normalised finite-chain ELBO (full point evidence), or its explicitly labelled Monte Carlo estimate"))
end
