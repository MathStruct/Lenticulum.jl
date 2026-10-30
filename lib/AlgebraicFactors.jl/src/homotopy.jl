"""
    HomotopyBranches(; real_tol=1e-8, singular_tol=1e-8, certify=true)

Use all isolated roots for a square PolynomialFactor target, after loading
`HomotopyContinuation`. Pass as `PolynomialFactor(...; branches=HomotopyBranches())`;
`:single` remains the default. Reality means max(abs(imag(root))) ≤ real_tol.
Singularity uses σmin(Ju) ≤ singular_tol*max(1,σmax(Ju)), or HC's singular flag.
Singular roots are reported but never weighted. Inference throws if there are no
regular real roots; inspect `polynomial_branches` for diagnostics even then.
Only proper Gaussian or trivial (Lebesgue-flat) target priors are supported.
Certification uses HC's interval/Krawczyk certificates, not α-theory.
"""
struct HomotopyBranches
    real_tol::Float64
    singular_tol::Float64
    certify::Bool
    function HomotopyBranches(;real_tol=1e-8,singular_tol=1e-8,certify=true)
        all(t->isfinite(t)&&t>0,(real_tol,singular_tol)) || throw(ArgumentError("tolerances must be finite and positive"))
        new(real_tol,singular_tol,certify)
    end
end

"""
    polynomial_branches(f, target, inputs, prior, ps; method=HomotopyBranches())

Inspect a square polynomial solve. Requires `using HomotopyContinuation`.
Returns `roots` (distinct finite candidates, with real/singular/certified flags,
`detJu`, `A`, site, posterior component and weight), `paths` (including failed
and at-infinity flags), `weights`, `components`, `likelihood`, and `posterior`.
Unweighted roots have zero weight; absent mixtures are `nothing`.

For regular real roots, S=Ju⁻¹ A Ju⁻ᵀ and posterior weights are proportional to
N(root; μ,P+S)/abs(detJu), or 1/abs(detJu) for a Lebesgue-flat prior, computed
in log space. The extrinsic likelihood is a mixture of *normalised* branch
sites N(root,S), with masses proportional to 1/abs(detJu). Mycelium's overlap
reweighting then reproduces the posterior. Its masses differ from posterior
masses, as required to avoid applying the prior twice.

This is a root-local Gaussian approximation, not the exact nonlinear posterior
at finite noise. Positive-dimensional square systems are outside its contract;
path diagnostics and individual certificates do not certify completeness.
"""
function polynomial_branches(f,target,inputs,prior,ps;method=HomotopyBranches())
    return _polynomial_branches(method,f,target,inputs,prior,ps)
end
_polynomial_branches(args...) = throw(ArgumentError("load HomotopyContinuation with `using HomotopyContinuation` to enumerate polynomial branches"))
_homotopy_message(args...) = throw(ArgumentError("load HomotopyContinuation with `using HomotopyContinuation` to enumerate polynomial branches"))

"""
    branch_entropy_bounds(m::MixtureBelief)

Return `(label_entropy, lower, upper)` in nats for a proper Gaussian mixture:
Σw H(component) ≤ H(mixture) ≤ Σw H(component)+H(w). No exact mixture entropy
or game-level entropy is implied. Zero masses contribute zero to H(w).
"""
function branch_entropy_bounds(m::LenticulumCore.MixtureBelief)
    w = LenticulumCore.mixture_weights(m)
    h = -sum((iszero(v) ? zero(v) : v*l) for (v,l) in zip(w,m.logw))
    hs = map(m.components) do c
        c isa LenticulumCore.GaussianBelief && LenticulumCore.isproper(c) ||
            throw(ArgumentError("entropy bounds require proper Gaussian components"))
        q = length(c.η)
        # Core's convention: the precision is its symmetric part (LP-019 fifth review).
        (q*(1+log(2π))-LinearAlgebra.logdet(LenticulumCore._symmetric_part(c.Λ)))/2
    end
    lower = sum(w.*hs)
    return (label_entropy=h,lower=lower,upper=lower+h)
end
