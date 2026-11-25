module AlgebraicEltypeFixtures

using AlgebraicFactors, LenticulumCore, Mycelium
using ..EltypeMatrix

function wired(name, make)
    f = make(Float64, Float64)
    ops = wired_cases(Mycelium.factor_message, Mycelium.local_free_energy, f.factor; f.pointwise)
    return EltypeCase(name, make, ops)
end

function data(factor, P, T, ps, msgs, pointwise)
    return (; factor, ps = typed(P, ps), st = (;), msgs = typed(T, msgs), pointwise = typed(T, pointwise))
end

function fixtures()
    return [
        wired(:ToricFactor, (P,T) -> data(
            ToricFactor((:a, :b), (2, 2), [1 1 1 1; 0 0 1 1; 0 1 0 1]; normalised = true),
            P, T, (η = [0.3, -0.2, 0.15],),
            (a = CategoricalBelief([0.4, 0.6]), b = CategoricalBelief([0.7, 0.3])),
            ((1, 2), nothing, nothing))),
        wired(:PolynomialFactor, (P,T) -> data(
            PolynomialFactor((x = 1, y = 1), ExponentTable(2, 1), 1;
                             Σ = typed(P, [0.03 0.01; 0.01 0.05]), λ = typed(P, 0.2)),
            P, T, (Θ = [-0.4 -1.3 2.0],),
            (x = Gaussian(0.2, 0.7), y = DiracBelief([0.8])),
            ((x = [0.2], y = [0.8]), nothing, nothing))),
        wired(:PluckerFactor, (P,T) -> data(
            PluckerFactor(2), P, T, (;),
            (z = DiracBelief([2.0, -1.0]), w = Gaussian([0.3, 0.5], [0.7 0.1; 0.1 0.4]), p = TrivialBelief()),
            ((z = [2.0, -1.0], w = [0.3, 0.5]), nothing, nothing))),
    ]
end

end
