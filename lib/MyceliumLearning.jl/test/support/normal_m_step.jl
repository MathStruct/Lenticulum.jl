# A one-channel learnable normal, used only inside a cluster. The M-step oracle is the
# observed sample itself, independently of the initial mean.
struct P32Normal <: LenticulumCore.AbstractLenticulumFactor end
LenticulumCore.channels(::P32Normal) = (LenticulumCore.Channel(:x,1),)
LenticulumCore.islearnable(::P32Normal) = true
function Mycelium.cluster_form(::P32Normal, ps)
    T = eltype(ps.μ)
    Mycelium.ClusterForm((x=ones(T,1,1),),reshape(ps.μ,1,1),ones(T,1,1),T[0],nothing)
end
MyceliumLearning.m_step(::P32Normal, stats::MomentStats, ps, frozen) =
    (μ=stats.m[:,1] ./ stats.n[1],)
