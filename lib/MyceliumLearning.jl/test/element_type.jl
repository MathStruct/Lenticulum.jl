using LinearAlgebra
using Random

@testset "P32 EM statistics preserve and promote element types" begin
    nt = NamedTuple()
    stats = map((Float32, Float64)) do T
        b = GraphBuilder(); variable!(b,:x,1)
        f = PriorFactor(:x, LenticulumCore.Gaussian(T[0.5],fill(T(2),1,1)), _ -> false)
        factor!(b,:p,f); connect!(b,:p,:x,:x)
        gc = cluster(build(b),:c,[:p]); c = gc.factors[1].factor
        belief = Mycelium.factor_belief(c,nt,nt,nt)
        st = MyceliumLearning._moment_stats(c,1,Mycelium.cluster_form(f,nt),belief)
        @test st isa MomentStats{T}
        @test eltype(st.n) === T && eltype(st.m) === T && eltype(st.S) === T
        st
    end
    for name in (:n,:m,:S)
        @test getfield(stats[1],name) ≈ getfield(stats[2],name) rtol=1e-4
    end
    @test MomentStats([:x],[1],1) isa MomentStats{Float64}
    # Use exactly representable input sums: Cholesky-based collected covariances can differ
    # by one Float32 ulp even for an exactly representable mathematical covariance.
    stats = (MomentStats([:x],[1],Float32[1],fill(0.5f0,1,1),fill(2.25f0,1,1,1)),
             MomentStats([:x],[1],[1.0],fill(0.5,1,1),fill(2.25,1,1,1)))
    wide = MomentStats{Float64}([:x],[1],1)
    wide.n .= 1 + 2.0^-40; wide.m .= 2.0^-40; wide.S .= 3 + 2.0^-40
    ref = MomentStats(stats[2].channels,stats[2].dims,copy(stats[2].n),copy(stats[2].m),copy(stats[2].S))
    ref = MyceliumLearning._add!(ref,wide)
    pool = MyceliumLearning._add!(stats[1],wide)
    @test pool isa MomentStats{Float64}
    for name in (:n,:m,:S)
        @test getfield(pool,name) == getfield(ref,name)
    end
    # The wide destination is retained and the narrow argument is not modified.
    @test stats[1].n == Float32[1]
    reverse = MomentStats{Float64}([:x],[1],1)
    reverse.n .= wide.n; reverse.m .= wide.m; reverse.S .= wide.S
    @test MyceliumLearning._add!(reverse,stats[1]) === reverse
    @test reverse.S == ref.S
end

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

@testset "P32 Float32 fit_em parameters and history" begin
    b = GraphBuilder(); variable!(b,:x,1)
    factor!(b,:p,P32Normal()); connect!(b,:p,:x,:x)
    factor!(b,:d,DataFactor(:x,Float32[0.75]));connect!(b,:d,:x,:x;direction=Emitting())
    gc = cluster(build(b),:c,[:p])
    ps = cluster_parameters(gc,(p=(μ=Float32[-0.5],),))
    out,Fs,_ = fit_em(gc,ps,NamedTuple();schedule=SequentialSchedule(all_tasks(gc)),iters=1,tol=-Inf)
    @test out.c.p.μ == Float32[0.75]
    @test eltype(out.c.p.μ) === Float32
    @test eltype(Fs) === Float32
    @test length(Fs) == 2
    @test Fs[2] < Fs[1]
    # A later wider energy widens the history without rounding it.
    h = MyceliumLearning._push_energy!(Union{}[],1f0)
    h = MyceliumLearning._push_energy!(h,1 + 2.0^-40)
    @test h == [1.0,1 + 2.0^-40]
end

@testset "P33: accelerated EM with an empty fixed subtree" begin
    b = GraphBuilder(); variable!(b,:x,1)
    factor!(b,:p,P32Normal());connect!(b,:p,:x,:x)
    fixed = PriorFactor(:x,LenticulumCore.Gaussian(Float32[0],fill(2f0,1,1)),_ -> false)
    factor!(b,:fixed,fixed);connect!(b,:fixed,:x,:x)
    factor!(b,:d,DataFactor(:x,Float32[0.75]));connect!(b,:d,:x,:x;direction=Emitting())
    gc = cluster(build(b),:c,[:p,:fixed])
    ps = cluster_parameters(gc,(p=(μ=Float32[-0.5],),fixed=NamedTuple()))
    out,Fs,_ = fit_em(gc,ps,NamedTuple();schedule=SequentialSchedule(all_tasks(gc)),
                      iters=1,tol=-Inf,accelerate=true)
    @test eltype(out.c.p.μ) === Float32
    @test eltype(Fs) === Float32
end

@testset "P33 accelerated EM: well-conditioned data and fixed empty subtree" begin
    function run(T)
        b=GraphBuilder()
        # Sixty observations with spread, sharing one learned mean; exact M-step is their mean.
        X=[0.7 + sin(i)*0.4 for i=1:60]
        plate!(b,:rep,60) do p,i
            variable!(p,:x,1)
            factor!(p,:p,P32Normal());connect!(p,:p,:x,:x)
            factor!(p,:fixed,PriorFactor(:x,LenticulumCore.Gaussian(T[0],fill(T(2),1,1)),_ -> false));connect!(p,:fixed,:x,:x)
            factor!(p,:d,DataFactor(:x,T[X[i]]));connect!(p,:d,:x,:x;direction=Emitting())
        end
        g=cluster(build(b),:rep,:c,[:p,:fixed];params=:shared)
        ps=(shared=(p=(μ=T[-0.5],),fixed=(;)),)
        out,Fs,_=fit_em(g,ps,(;);schedule=plate_schedule(g),iters=3,tol=-Inf,accelerate=true)
        @test eltype(out.shared.p.μ) === T
        @test eltype(Fs) === T
        @test out.shared.p.μ ≈ [sum(X)/length(X)] rtol=1e-4
        out.shared.p.μ
    end
    @test run(Float32) ≈ run(Float64) rtol=1e-4
end
