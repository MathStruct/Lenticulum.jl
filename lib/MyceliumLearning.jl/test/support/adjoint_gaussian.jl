# Unnormalised Gaussian potentials, independent of top-level GaussianFactor.
# Unary exp(-q*x²/2+h*x), pair exp(-k*(x-y)²/2).
struct AdjointUnary <: AbstractLenticulumFactor end
struct AdjointPair <: AbstractLenticulumFactor end
LenticulumCore.channels(::AdjointUnary) = (LenticulumCore.Channel(:x,nothing),)
LenticulumCore.channels(::AdjointPair) = (LenticulumCore.Channel(:x,nothing),LenticulumCore.Channel(:y,nothing))
LenticulumCore.supported_polarities(::AdjointUnary) = (Polarity(x=Unobserved()),)
LenticulumCore.supported_polarities(::AdjointPair) = (Polarity(x=Unobserved(),y=Observed()),Polarity(x=Observed(),y=Unobserved()))
LenticulumCore.supports_polarity(f::Union{AdjointUnary,AdjointPair},p::Polarity) = p in supported_polarities(f)
LenticulumCore.islearnable(::Union{AdjointUnary,AdjointPair}) = true
Mycelium.uses_cavity(::Union{AdjointUnary,AdjointPair}) = true
Mycelium.factor_message(::AdjointUnary,target::Symbol,pol,inputs,prior,ps,st) = (GaussianBelief([ps.h],fill(ps.q,1,1)), st)
function Mycelium.factor_message(::AdjointPair,target::Symbol,pol,inputs,prior,ps,st)
    b = inputs[target === :x ? :y : :x]
    q,h = only(b.Λ),only(b.η)
    return GaussianBelief([ps.k*h/(ps.k+q)],fill(ps.k*q/(ps.k+q),1,1)),st
end

adjoint_qh(::Union{Nothing,TrivialBelief}) = (0.0,0.0)
adjoint_qh(b::GaussianBelief) = (only(b.Λ),only(b.η))
function Mycelium.local_free_energy(::AdjointUnary,beliefs,ps,st)
    q,h = adjoint_qh(beliefs.x)
    C = inv(ps.q+q); m = C*(ps.h+h)
    return ps.q*(m*m+C)/2-ps.h*m-log(2π*exp(1)*C)/2,st
end
function Mycelium.local_free_energy(::AdjointPair,beliefs,ps,st)
    qx,hx = adjoint_qh(beliefs.x); qy,hy = adjoint_qh(beliefs.y)
    qx+qy == 0 && return 0.0,st # no normalisable joint yet during startup
    Λ = [ps.k+qx -ps.k; -ps.k ps.k+qy]
    C = inv(Λ); m = C*[hx,hy]
    H = (2*(1+log(2π))-logdet(Λ))/2
    return ps.k*((m[1]-m[2])^2+C[1,1]+C[2,2]-2C[1,2])/2-H,st
end

function adjoint_gaussian_graph(;loopy=false,coupling=ExactCoupling(),cut=false,diagonal=false,freeze=())
    b=GraphBuilder()
    for v in (:x,:y,:z); variable!(b,v); end
    for (f,v) in ((:u,:x),(:v,:y),(:w,:z))
        factor!(b,f,AdjointUnary();freeze=f===:u ? freeze : ())
        connect!(b,f,:x,v;coupling)
    end
    pairs=loopy ? ((:a,:x,:y),(:b,:y,:z),(:c,:z,:x)) : ((:a,:x,:y),(:b,:y,:z))
    for (f,x,y) in pairs
        factor!(b,f,AdjointPair())
        connect!(b,f,:x,x;coupling=f===:a && diagonal ? DiagonalCoupling() : coupling,detach=f===:a && cut)
        connect!(b,f,:y,y;coupling)
    end
    g=build(b)
    p=(u=(q=0.7,h=0.4),v=(q=0.9,h=-0.3),w=(q=0.6,h=0.1),a=(k=2.0,),b=(k=1.3,))
    loopy && (p=merge(p,(c=(k=1.7,),)))
    st=NamedTuple{keys(p)}(map(_ -> (;),values(p)))
    return g,p,st
end
adjoint_flat(p) = LenticulumCore._flatten(p)
function adjoint_reference_loss(p,g,st,sched,n,damping)
    s=MessageStore(g)
    for _ in 1:n; _,st=sweep!(s,g,sched,p,st;damping); end
    return first(scalar_free_energy(s,g,p,st))
end
