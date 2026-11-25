module MStepFixtures

using LinearAlgebra, Lenticulum, MyceliumLearning
using ..EltypeMatrix: EltypeCase

function moments(T, channels, K)
    d = length(channels)
    n = T[k + 1 for k in 1:K]
    means = [T(j + k) / 4 for j in 1:d, k in 1:K]
    m = means .* transpose(n)
    S = cat((n[k] .* (Matrix{T}(I, d, d) + means[:, k] * means[:, k]') for k in 1:K)...; dims = 3)
    return MyceliumLearning.MomentStats(collect(channels), fill(1, d), n, m, S)
end

function fixtures()
    make(run, name) = EltypeCase(name, run, [:m_step => d -> m_step(d.factor, d.moments, d.ps, ())])
    matrix(T, x) = fill(T(x), 1, 1)
    blocks(T) = (; x = matrix(T, -0.5))
    lin(P, T) = (factor = LinearConstraintFactor((x=1, y=1), 1; noise=P(0.4), output=:y),
                 moments = moments(T, (:x, :y), 1), ps = (A=blocks(P), c=P[0.05]))
    sw(P, T) = (factor = SwitchingConstraintFactor((x=1, y=1), 2, 1; noise=P(0.4), output=:y),
                moments = moments(T, (:x, :y), 2), ps = (A=blocks(P), b=P[0.05 0.1], logw=P[0, 0]))
    regime(P, T) = (factor = RegimeFactor((x=1, y=1), 2, 1; noise=P(0.4), output=:y, learn_noise=true),
                    moments = moments(T, (:x, :y), 2),
                    ps = (A=(x=reshape(P[-0.5, -0.6], 1, 1, 2),), b=P[0.05 0.1], logw=P[0, 0], L=zeros(P, 1, 1, 2)))
    gaussian(P, T) = (factor = GaussianFactor(1 => 1; noise=P(0.4), learn_noise=:full),
                      moments = moments(T, (:x, :y), 1),
                      ps = (A=matrix(P, 0.5), b=P[0.05], noise=(log_diagonal=P[0], lower=P[])))
    prior(P, T) = (factor = GaussianPrior(:x, P(0.2), P(0.4); learn_noise=:diagonal),
                   moments = moments(T, (:x,), 1), ps = (noise=(log_precision=P[0],),))
    return [make(lin, :LinearConstraintMStep), make(sw, :SwitchingConstraintMStep),
            make(regime, :RegimeMStep), make(gaussian, :GaussianFactorMStep), make(prior, :GaussianPriorMStep)]
end

end
