module AlgebraicFactorsHomotopyContinuationExt
using AlgebraicFactors, HomotopyContinuation, LenticulumCore, Mycelium, LinearAlgebra
const AF = AlgebraicFactors
const HC = HomotopyContinuation

function AF._polynomial_branches(method::HomotopyBranches, f::PolynomialFactor,
                                 target::Symbol, inputs, prior, ps)
    target in keys(f.dims) || throw(ArgumentError("unknown target :$target"))
    q = f.dims[target]
    f.k == q || throw(ArgumentError("homotopy requires square k=q; use §3's single-branch site (branches=:single) otherwise"))
    πprior = prior === nothing ? TrivialBelief() : prior
    πprior isa TrivialBelief || (πprior isa GaussianBelief && isproper(πprior) && length(πprior.η)==q) ||
        throw(ArgumentError("homotopy target prior must be trivial or a proper Gaussian of dimension $q"))
    size(ps.Θ)==(f.k,size(f.table.exponents,2)) && all(isfinite,ps.Θ) || throw(ArgumentError("invalid Θ"))
    # Construct each basis polynomial by recurrence, including homogeneous Chebyshev
    # tables whose lower-degree terms cannot be converted within the original span.
    u = HC.ModelKit.variables(:u,1:q)
    z = Any[]
    for n in keys(f.dims)
        append!(z,n===target ? u : first(AF._polynomial_moments(get(inputs,n,nothing),f.dims[n])))
    end
    basisvalue(x,d) = if f.table.basis === :monomial
        x^d
    elseif d==0
        1
    else
        a,b = 1,x
        for _ in 2:d
            a,b = b,2*x*b-a
        end
        b
    end
    E = f.table.exponents
    v = [prod(basisvalue(z[i],E[i,j]) for i in axes(E,1)) for j in axes(E,2)]
    F = HC.System([sum(ps.Θ[i,j]*v[j] for j in eachindex(v)) for i in 1:f.k];variables=u)
    # HC interprets homogeneous input projectively, excluding the affine origin.
    # Translate homogeneous systems before solving so y²=0 retains its double root.
    shift = zeros(q)
    if HC.is_homogeneous(F)
        shift = collect(1.0:q)
        F = HC.System(F(u .+ shift);variables=u)
        HC.is_homogeneous(F) && throw(ArgumentError("degenerate homogeneous system: isolated affine roots required"))
    end
    result = HC.solve(F;show_progress=false, seed=UInt32(4004), start_system=:total_degree, threading=false)
    paths = [(root=HC.solution(p).+shift,real=maximum(abs,imag.(HC.solution(p)))≤method.real_tol,
              singular=HC.is_singular(p),at_infinity=HC.is_at_infinity(p),failed=HC.is_failed(p),
              return_code=p.return_code) for p in HC.path_results(result)]
    candidates = HC.results(result;only_nonsingular=false)
    roots = NamedTuple[]
    sites,components = GaussianBelief[],GaussianBelief[]
    logmass,logpost = Float64[],Float64[]
    # HC.results removes repeated singular endpoints using HC's multiplicity data.
    for p in candidates
        solverroot = HC.solution(p)
        root = solverroot .+ shift
        realroot = maximum(abs,imag.(root))≤method.real_tol
        certified = method.certify ? HC.is_certified(only(HC.certificates(HC.certify(F,[solverroot];show_progress=false)))) : nothing
        x = real.(root)
        if realroot
            _,Ju,A = AF._polynomial_linearisation(f,target,inputs,x,ps)
        else
            complexz = ComplexF64[]
            targetrange = 1:0
            for n in keys(f.dims)
                if n === target
                    targetrange = length(complexz)+1:length(complexz)+q
                    append!(complexz,root)
                else
                    append!(complexz,first(AF._polynomial_moments(get(inputs,n,nothing),f.dims[n])))
                end
            end
            Ju = (ps.Θ*veronese_jacobian(f.table,complexz))[:,targetrange]
            A = nothing # residual covariance is defined only for real branch sites
        end
        sv = svdvals(Ju)
        singular = HC.is_singular(p) || minimum(sv)≤method.singular_tol*max(1,maximum(sv))
        detJu = det(Ju)
        site,component = nothing,nothing
        index = 0
        if realroot && !singular
            site,_ = AF._polynomial_site(f,target,inputs,x,ps)
            component = Mycelium.combine(πprior,site)
            push!(sites,site);push!(components,component)
            push!(logmass,-first(logabsdet(Ju)))
            overlap = πprior isa TrivialBelief ? 0.0 : last(Mycelium._product(site,πprior))
            push!(logpost,last(logmass)+overlap)
            index=length(sites)
        end
        push!(roots,(root=root,real=realroot,singular=singular,at_infinity=false,failed=false,
                     certified=certified,detJu=detJu,A=A,site=site,component=component,index=index))
    end
    normalise(l) = l .- (maximum(l)+log(sum(exp.(l .- maximum(l)))))
    if isempty(sites)
        weights=Float64[];likelihood=nothing;posterior=nothing
    else
        likelihood=MixtureBelief(Val(:log),sites,normalise(logmass))
        posterior=MixtureBelief(Val(:log),components,normalise(logpost))
        weights=mixture_weights(posterior)
    end
    roots = [merge(r,(weight=r.index==0 ? 0.0 : weights[r.index],)) for r in roots]
    return (;roots,paths,weights,components,likelihood,posterior)
end

function AF._homotopy_message(method::HomotopyBranches,f::PolynomialFactor,target,inputs,prior,ps,st)
    data=AF._polynomial_branches(method,f,target,inputs,prior,ps)
    data.likelihood===nothing && throw(ArgumentError("no regular real root: inspect polynomial_branches for singular roots and path diagnostics (§4.5)"))
    failed = count(p->p.failed,data.paths)
    at_infinity = count(p->p.at_infinity,data.paths)
    singular = count(r->r.real&&r.singular,data.roots)
    if failed+at_infinity+singular>0
        @warn "homotopy inference excluded paths or singular roots; inspect polynomial_branches for diagnostics" failed at_infinity singular
    end
    return data.likelihood,st
end
end
