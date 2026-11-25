module LenticulumEltypeFixtures

using LinearAlgebra, Lenticulum, LenticulumCore, Mycelium
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

function fixtures()
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
            HardConstraintFactor((x = 1, y = 1), 1; noise = zero(P)), P, T, relation, both); energy = false),
        wired(:ScaleShiftFactor, (P,T) -> data(ScaleShiftFactor(:x, :y, 1), P, T,
            (b = [0.2], logs = [0.3]), both); energy = false),
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
