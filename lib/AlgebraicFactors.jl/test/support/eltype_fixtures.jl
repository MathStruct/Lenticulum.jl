module AlgebraicEltypeFixtures

using AlgebraicFactors, LenticulumCore, Mycelium, MyceliumLearning
using ..EltypeMatrix

for B in (AlgebraicFactors.ToricBelief, AlgebraicFactors.PolynomialBelief)
    @eval EltypeMatrix.payload(b::$B) = EltypeMatrix.belief_fields(b)
end

function fixture(name, make, extended)
    f = make(Float64, Float64)
    ops = extended ?
        EltypeMatrix.inference_cases(Mycelium, MyceliumLearning._gradient_message, f; f.pointwise) :
        wired_cases(Mycelium.factor_message, Mycelium.local_free_energy, f.factor; f.pointwise)
    return EltypeCase(name, make, ops)
end

function data(factor, P, T, ps, msgs, pointwise)
    return (; factor, ps = typed(P, ps), st = (;), msgs = typed(T, msgs), pointwise = typed(T, pointwise))
end

function fixtures(; extended = false)
    wired(name, make) = fixture(name, make, extended)
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
