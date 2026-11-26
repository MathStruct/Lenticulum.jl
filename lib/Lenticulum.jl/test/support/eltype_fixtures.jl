module LenticulumEltypeFixtures

using LinearAlgebra, Lenticulum, LenticulumCore, Mycelium, MyceliumLearning
using ..EltypeMatrix

for B in (Lenticulum.FrozenGaussianBelief, Lenticulum.FrozenGaussianBeliefs,
          Lenticulum.FrozenSwitchingBeliefs, Lenticulum.ScaleShiftBelief,
          Mycelium.FrozenClusterBelief, Mycelium.FrozenClusterBeliefs)
    @eval EltypeMatrix.payload(b::$B) = EltypeMatrix.belief_fields(b)
end

# The cluster's Dual path returns separate frozen beliefs. Compare each replica
# to its view of the stacked float result, including the shared covariance and entropy.
function EltypeMatrix.agrees(a::AbstractVector, b::Mycelium.FrozenClusterBeliefs, T, rtol)
    nA, n = size(b.q)
    length(a) == n || return false
    return all(1:n) do i
        cols = ((i - 1) * nA + 1):(i * nA)
        points = Dict(k => v[:, i] for (k, v) in b.points)
        ref = Mycelium.FrozenClusterBelief(b.q[:, i], b.M[:, cols], b.Σ, b.H,
                                         b.class, b.free, b.offset, points, b.assignments)
        EltypeMatrix.agrees(a[i], ref, T, rtol)
    end
end

function fixture(name, make, extended; energy = true, mean_field = true)
    f = make(Float64, Float64)
    pointwise = energy ? f.pointwise : nothing
    ops = extended ?
        EltypeMatrix.inference_cases(Mycelium, MyceliumLearning._gradient_message, f; pointwise, mean_field) :
        wired_cases(Mycelium.factor_message, Mycelium.local_free_energy, f.factor; pointwise)
    return EltypeCase(name, make, ops)
end

function data(factor, P, T, ps, msgs, pointwise = nothing; st = (;))
    return (; factor, ps = typed(P, ps), st, msgs = typed(T, msgs), pointwise = typed(T, pointwise))
end

function fixtures(; extended = false)
    wired(name, make; kwargs...) = fixture(name, make, extended; kwargs...)
    px = Gaussian(0.5, 0.8)
    py = Gaussian(0.9, 0.7)
    x = DiracBelief([0.5])
    y = DiracBelief([0.9])
    both = (; x = px, y = py)
    xy = (x.value, nothing, y.value)
    relation = (A = (; x = fill(-1.25, 1, 1), y = fill(1.0, 1, 1)), c = [0.1])
    af = (A = fill(1.25, 1, 1), b = [0.1])
    maps = (x -> x .^ 2 .+ x, x -> reshape(2 .* x .+ 1, 1, 1))
    switching = (A = relation.A, b = [0.1 0.3], logw = [0.2, -0.1])
    regime = (A = map(A -> cat(A, A .* 0.8; dims = 3), relation.A),
              b = switching.b, logw = switching.logw)
    switch_msgs = merge(both, (k = CategoricalBelief([0.65, 0.35]),))
    switch_points = (merge(map(belief_mean, both), (k = 1,)), nothing, nothing)
    gaussian(P) = GaussianFactor(1 => 1; noise = typed(P, 0.4))
    prior(P) = GaussianPrior(:x, typed(P, 0.3), typed(P, 0.8))
    cases = [
        wired(:GaussianFactor, (P,T) -> data(gaussian(P), P, T, af, both, xy)),
        wired(:GaussianPrior, (P,T) -> data(prior(P), P, T, (;), (; x = px), (nothing, nothing, x.value))),
        wired(:LinearConstraintFactor, (P,T) -> data(
            LinearConstraintFactor((x = 1, y = 1), 1; noise = typed(P, 0.4)), P, T,
            relation, both, ((x = x.value, y = y.value), nothing, nothing))),
        wired(:HardConstraintFactor, (P,T) -> data(
            HardConstraintFactor((x = 1, y = 1), 1; noise = zero(P)), P, T, relation, both); energy = false, mean_field = false),
        wired(:ScaleShiftFactor, (P,T) -> data(ScaleShiftFactor(:x, :y, 1), P, T,
            (b = [0.2], logs = [0.3]), both); energy = false, mean_field = false),
        wired(:SplitFactor, (P,T) -> data(SplitFactor(:z, (x = [1], y = [2])), P, T, (;),
            merge(both, (z = Gaussian([0.4, 0.8], [0.7 0.1; 0.1 0.8]),))); energy = false),
        wired(:NonlinearFactor, (P,T) -> data(
            NonlinearFactor(maps[1], 1 => 1; jacobian = maps[2], noise = typed(P, 0.4)), P, T,
            (;), (; x = px, y), xy)),
        wired(:UnscentedNonlinear, (P,T) -> data(
            NonlinearFactor(maps[1], 1 => 1; method = Unscented(), noise = typed(P, 0.4)), P, T,
            (;), (; x = px, y), xy)),
        wired(:SwitchingConstraintFactor, (P,T) -> data(
            SwitchingConstraintFactor((x = 1, y = 1), 2, 1; noise = typed(P, 0.4)), P, T,
            switching, switch_msgs, switch_points)),
        wired(:RegimeFactor, (P,T) -> data(
            RegimeFactor((x = 1, y = 1), 2, 1; noise = typed(P, 0.4)), P, T,
            regime, switch_msgs); energy = false),
        wired(:LikelihoodFactor, (P,T) -> data(LikelihoodFactor(
            OrdinalProbitLikelihood(typed(P, [-1.0, 1.0])), 1; A = typed(P, 1.1), b = typed(P, [0.2])),
            P, T, (;), (x = px, y = DiracBelief([2])), (x.value, nothing, [2]))),
        wired(:PoissonEP, (P,T) -> data(LikelihoodFactor(
            PoissonLikelihood(; maxcount = 5), 1; method = ExpectationPropagation(order = 9)),
            P, T, (;), (x = px, y = DiracBelief([2])), (x.value, nothing, [2]))),
        wired(:ComposedFactor, (P,T) -> data(ComposedFactor(
            GaussianFactor(1 => 1; noise = typed(P, 0.4), channels = (:x, :u)),
            GaussianFactor(1 => 1; noise = typed(P, 0.3), channels = (:u, :y))),
            P, T, (first = af, second = af), both, xy; st = (first = (;), second = (;)))),
    ]
    extended && append!(cases, [
        wired(:GaussianPriorLearned, (P,T) -> data(
            GaussianPrior(:x, 0.3, 0.8; learn_noise = :diagonal), P, T,
            (noise = (log_precision = [0.2],),), (; x = px), (nothing, nothing, x.value))),
        wired(:GaussianFactorLearned, (P,T) -> data(
            GaussianFactor(2 => 2; noise = [0.4 0.1; 0.1 0.5], learn_noise = :full), P, T,
            (A = [1.25 0.2; 0.1 0.8], b = [0.1, 0.2],
             noise = (log_diagonal = [0.2, 0.3], lower = [0.15])),
            (x = Gaussian([0.5, 0.4], [0.8 0.1; 0.1 0.7]),
             y = Gaussian([0.9, 0.1], [0.7 0.1; 0.1 0.8])),
            ([0.5, 0.4], nothing, [0.9, 0.1]))),
        wired(:LinearConstraintInteger, (P,T) -> data(
            LinearConstraintFactor((x = 1, y = 1), 1; noise = typed(P, 0.4)), P, T,
            (A = (x = [-2;;], y = [1;;]), c = [1]), both, ((x = x.value, y = y.value), nothing, nothing))),
        wired(:GaussianFactorInteger, (P,T) -> data(
            gaussian(P), P, T, (A = [2;;], b = [1]), both, xy)),
        wired(:HardConstraintNoisy, (P,T) -> data(
            HardConstraintFactor((x = 1, y = 1), 1; noise = typed(P, 0.4)), P, T,
            relation, both); energy = false),
    ])
    tensor(P,T) = (factor = TensorFactor((edge = gaussian(P), source = prior(P))),
                   ps = (edge = typed(P, af), source = (;)), st = (edge = (;), source = (;)),
                   prior = typed(T, (edge = px, source = TrivialBelief())),
                   observation = typed(T, (edge = y, source = px)))
    push!(cases, EltypeCase(:TensorFactor, tensor, [
        :free_energy => d -> first(free_energy(d.factor, d.prior, d.observation, d.ps, d.st)),
        :entropy => d -> first(LenticulumCore.entropy(d.factor, d.prior, d.observation, d.ps, d.st)),
    ]))
    return cases
end

end
