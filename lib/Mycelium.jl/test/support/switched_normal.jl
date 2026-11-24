# Scalar switched normal, written by hand, with a separate variance for each component.
struct P36Switched <: AbstractLenticulumFactor end
LenticulumCore.channels(::P36Switched) = (LenticulumCore.Channel(:y,1), LenticulumCore.Channel(:k,2))
Mycelium.cluster_switch(::P36Switched) = :k
LuxCore.initialparameters(::AbstractRNG, ::P36Switched) = (mu=[-0.7,1.1], q=[0.4,0.8], logw=log.([0.3,0.7]))
Mycelium.cluster_form(::P36Switched, ps) = ClusterForm((y=ones(1,1),), reshape(ps.mu,1,2), reshape(ps.q,1,1,2), ps.logw, :k)

# Explicit channel dimensions are required by OverlayFactor/as_cluster.
struct P36Fixed <: AbstractLenticulumFactor
    mu::Vector{Float64}
    cov::Matrix{Float64}
end
LenticulumCore.channels(f::P36Fixed) = (LenticulumCore.Channel(:y,length(f.mu)),)
LuxCore.parameterlength(::P36Fixed) = 0
Mycelium.cluster_form(f::P36Fixed, ps) = ClusterForm((y=Matrix(1.0I,length(f.mu),length(f.mu)),),reshape(f.mu,:,1),f.cov,[0.],nothing)
LuxCore.parameterlength(::P36Switched) = 6
