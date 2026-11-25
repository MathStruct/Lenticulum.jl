# Fixtures of a dimension above one, several channels, diagonal and dense noise together
# (Mistral's Q7 review, LP-092). The repository's element-type fixtures are one-dimensional
# almost throughout; two defects sat on these wider shapes.
module LenticulumWideFixtures
using LinearAlgebra, Lenticulum, LenticulumCore, Mycelium, MyceliumLearning
using ..EltypeMatrix

function wired(name, make; energy = true)
    f = make(Float64, Float64)
    ops = wired_cases(Mycelium.factor_message, Mycelium.local_free_energy, f.factor;
                      pointwise = energy ? f.pointwise : nothing)
    return EltypeCase(name, make, ops)
end
function data(factor, P, T, ps, msgs, pointwise = nothing; st = (;))
    return (; factor, ps = typed(P, ps), st, msgs = typed(T, msgs), pointwise = typed(T, pointwise))
end

const G3 = Gaussian([0.1, 0.2, -0.3], [1.0 0.1 0.0; 0.1 0.8 0.2; 0.0 0.2 0.9])
const G2 = Gaussian([0.5, -0.2], [0.9 0.15; 0.15 0.7])
const G1 = Gaussian(0.4, 0.6)
const D3 = DiracBelief([0.3, -0.1, 0.2])
const D2 = DiracBelief([0.6, 0.1])
const Q2d = [0.5 0.1; 0.1 0.4]

function fixtures()
    cases = Any[]
    push!(cases, wired(:GaussianFactor32, (P, T) -> data(
        GaussianFactor(3 => 2; noise = Q2d), P, T,
        (A = [0.9 0.1 0.0; 0.0 0.8 0.2], b = [0.0, 0.1]),
        (; x = G3, y = G2), (D3.value, nothing, D2.value))))
    push!(cases, wired(:GaussianFactorDiag, (P, T) -> data(
        GaussianFactor(2 => 2; noise = [0.4, 0.6]), P, T,
        (A = [1.1 0.2; 0.0 0.9], b = [0.1, -0.2]),
        (; x = G2, y = G2), (D2.value, nothing, D2.value))))
    push!(cases, wired(:GaussianPrior3, (P, T) -> data(
        GaussianPrior(:x, [0.3, 0.1, 0.2], [1.2 0.1 0.0; 0.1 0.9 0.1; 0.0 0.1 1.0]), P, T,
        (;), (; x = G3), (nothing, nothing, D3.value))))
    push!(cases, wired(:LinearConstraint32, (P, T) -> data(
        LinearConstraintFactor((x = 3, y = 2), 2; noise = Q2d), P, T,
        (A = (x = [1.0 0.2 0.0; 0.1 0.9 0.3], y = [1.0 0.1; 0.2 1.0]), c = [0.1, -0.2]),
        (; x = G3, y = G2), ((x = D3.value, y = D2.value), nothing, nothing))))
    push!(cases, wired(:LinearConstraintDiag, (P, T) -> data(
        LinearConstraintFactor((x = 2, y = 2), 2; noise = [0.3, 0.5]), P, T,
        (A = (x = [1.2 0.1; 0.0 0.8], y = [1.0 0.0; 0.1 1.0]), c = [0.05, 0.1]),
        (; x = G2, y = G2), ((x = D2.value, y = D2.value), nothing, nothing))))
    push!(cases, wired(:HardConstraint22, (P, T) -> data(
        HardConstraintFactor((x = 2, y = 2), 2; noise = 0.0), P, T,
        (A = (x = [1.0 0.1; 0.0 1.0], y = [1.0 0.0; 0.2 1.0]), c = [0.1, 0.0]),
        (; x = G2, y = G2)); energy = false))
    push!(cases, wired(:ScaleShift3, (P, T) -> data(
        ScaleShiftFactor(:x, :y, 3), P, T,
        (b = [0.2, -0.1, 0.3], logs = [0.3, 0.1, -0.2]),
        (; x = G3, y = G3)); energy = false))
    push!(cases, wired(:SplitFactor3, (P, T) -> data(
        SplitFactor(:z, (x = [1, 2], y = [3])), P, T, (;),
        (; x = G2, y = G1, z = G3)); energy = false))
    nl = (x -> x .^ 2 .+ x, x -> Diagonal(2 .* x .+ 1))
    push!(cases, wired(:Nonlinear22, (P, T) -> data(
        NonlinearFactor(nl[1], 2 => 2; jacobian = nl[2], noise = Q2d), P, T, (;),
        (; x = G2, y = G2), (D2.value, nothing, D2.value))))
    push!(cases, wired(:Unscented22, (P, T) -> data(
        NonlinearFactor(nl[1], 2 => 2; method = Unscented(), noise = Q2d), P, T, (;),
        (; x = G2, y = G2), (D2.value, nothing, D2.value))))
    swmsgs = (; x = G2, y = G2, k = CategoricalBelief([0.5, 0.3, 0.2]))
    swpts = (merge(map(belief_mean, (; x = G2, y = G2)), (k = 2,)), nothing, nothing)
    push!(cases, wired(:Switching32, (P, T) -> data(
        SwitchingConstraintFactor((x = 2, y = 2), 3, 2; noise = Q2d), P, T,
        (A = (x = [1.0 0.1; 0.0 1.0], y = [1.0 0.0; 0.2 0.9]),
         b = [0.1 0.2 -0.1; -0.2 0.0 0.3], logw = [0.2, -0.1, 0.0]),
        swmsgs, swpts)))
    A3 = cat([1.0 0.1; 0.0 1.0], [0.8 0.0; 0.1 0.9], [0.9 0.2; 0.0 0.7]; dims = 3)
    push!(cases, wired(:Regime32, (P, T) -> data(
        RegimeFactor((x = 2, y = 2), 3, 2; noise = Q2d), P, T,
        (A = (x = A3,), b = [0.1 0.2 -0.1; -0.2 0.0 0.3], logw = [0.2, -0.1, 0.0]),
        swmsgs); energy = false))
    push!(cases, wired(:OrdinalProbit2, (P, T) -> data(
        LikelihoodFactor(OrdinalProbitLikelihood([-1.0, 1.0]), 2; A = [0.7 0.3], b = [0.2]), P, T, (;),
        (; x = G2, y = DiracBelief([2])), (D2.value, nothing, [2]))))
    push!(cases, wired(:PoissonEP2, (P, T) -> data(
        LikelihoodFactor(PoissonLikelihood(; maxcount = 5), 2; A = [0.6 0.4], b = [0.1],
                         method = ExpectationPropagation(order = 7)), P, T, (;),
        (; x = G2, y = DiracBelief([2])), (D2.value, nothing, [2]))))
    push!(cases, wired(:Composed22, (P, T) -> data(
        ComposedFactor(GaussianFactor(2 => 2; noise = Q2d, channels = (:x, :u)),
                       GaussianFactor(2 => 2; noise = [0.3 0.05; 0.05 0.5], channels = (:u, :y))),
        P, T, (first = (A = [1.1 0.0; 0.1 0.9], b = [0.0, 0.1]),
               second = (A = [0.9 0.2; 0.0 1.0], b = [0.1, -0.1])),
        (; x = G2, y = G2), (D2.value, nothing, D2.value);
        st = (first = (;), second = (;)))))
    gauss(P) = GaussianFactor(2 => 2; noise = Q2d)
    prior(P) = GaussianPrior(:x, [0.3, 0.1], [1.0 0.2; 0.2 0.8])
    tensor(P, T) = (factor = TensorFactor((edge = gauss(P), source = prior(P))),
                    ps = (edge = (A = [1.0 0.1; 0.2 0.9], b = [0.0, 0.2]), source = (;)),
                    st = (edge = (;), source = (;)),
                    prior = typed(T, (edge = G2, source = TrivialBelief())),
                    observation = typed(T, (edge = D2, source = G2)))
    push!(cases, EltypeCase(:TensorFactor2, tensor, [
        :free_energy => d -> first(free_energy(d.factor, d.prior, d.observation, d.ps, d.st)),
        :entropy => d -> first(LenticulumCore.entropy(d.factor, d.prior, d.observation, d.ps, d.st)),
    ]))
    return cases
end

function moments2(T, chs, dims, K)
    d = sum(dims)
    n = T[k + 1 for k in 1:K]
    m = [T(j + k) / 4 for j in 1:d, k in 1:K]
    S = cat((n[k] .* (Matrix{T}(I, d, d) + m[:, k] * m[:, k]') for k in 1:K)...; dims = 3)
    return MyceliumLearning.MomentStats(collect(chs), dims, n, m, S)
end

function mstep_fixtures()
    make(run, name) = EltypeCase(name, run, [:m_step => d -> m_step(d.factor, d.moments, d.ps, ())])
    lin(P, T) = (factor = LinearConstraintFactor((x = 3, y = 2), 2; noise = P.(Q2d), output = :y),
                 moments = moments2(T, (:x, :y), [3, 2], 1),
                 ps = (A = (x = P.([1.0 0.2 0.0; 0.1 0.9 0.3]),), c = P[0.05, 0.1]))
    sw(P, T) = (factor = SwitchingConstraintFactor((x = 3, y = 2), 2, 2; noise = P.(Q2d), output = :y),
                moments = moments2(T, (:x, :y), [3, 2], 2),
                ps = (A = (x = P.([1.0 0.2 0.0; 0.1 0.9 0.3]),), b = P[0.05 0.1; -0.1 0.0], logw = P[0.0, 0.0]))
    regime(P, T) = (factor = RegimeFactor((x = 3, y = 2), 2, 2; noise = P.(Q2d), output = :y, learn_noise = true),
                    moments = moments2(T, (:x, :y), [3, 2], 2),
                    ps = (A = (x = reshape(P.(cat([1.0 0.2 0.0; 0.1 0.9 0.3], [0.8 0.1 0.0; 0.0 0.7 0.2]; dims = 3)), 2, 3, 2),),
                          b = P[0.05 0.1; -0.1 0.0], logw = P[0.0, 0.0], L = zeros(P, 2, 2, 2)))
    gauss(P, T) = (factor = GaussianFactor(3 => 2; noise = P.(Q2d), learn_noise = :full),
                   moments = moments2(T, (:x, :y), [3, 2], 1),
                   ps = (A = P.([1.0 0.2 0.0; 0.1 0.9 0.3]), b = P[0.05, 0.1],
                         noise = (log_diagonal = P[0.0, 0.0], lower = P[0.1])))
    prior(P, T) = (factor = GaussianPrior(:x, P.([0.2, -0.1, 0.3]), Diagonal(P.([0.4, 0.6, 0.5])); learn_noise = :diagonal),
                   moments = moments2(T, (:x,), [3], 1),
                   ps = (noise = (log_precision = P[0.0, 0.0, 0.0],),))
    return [make(lin, :LinearConstraintMStep32), make(sw, :SwitchingConstraintMStep32),
            make(regime, :RegimeMStep32), make(gauss, :GaussianFactorMStep32), make(prior, :GaussianPriorMStep3)]
end

end # module
