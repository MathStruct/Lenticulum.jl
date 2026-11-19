# P8 (LP-024 item 7): the graph free energy is in the element type of its messages. A graph whose
# messages are all Float32 has a Float32 free energy, on the node-by-node path and on a plate's
# rows, and it agrees with the same graph in Float64. Before P8 every total started at `0.0`, and
# the zero energies of data factors and points, the categorical entropy's `0 log 0` and the
# Gaussian entropy's `log(2π)` were Float64 constants, so a Float32 graph summed to Float64.

using LinearAlgebra
using ForwardDiff, Zygote

# x (Gaussian, degree 2: prior and relay), y (Gaussian, degree 2: relay and prior), c (categorical,
# degree 2: two priors), z (a point, degree 2: data and prior).
function p8_graph(T; plate::Union{Nothing,Int} = nothing)
    gauss(μ, λ) = LenticulumCore.GaussianBelief(T.(λ .* μ), Matrix{T}(λ * I, 2, 2))
    quad(μ, λ) = x -> λ * sum(abs2, x .- T.(μ)) / 2 - log(λ) + log(2 * T(π))
    function add!(b, i)
        variable!(b, :x, 2); variable!(b, :y, 2); variable!(b, :c); variable!(b, :z, 2)
        factor!(b, :px, PriorFactor(:x, gauss([1, -1], T(2)), quad([1, -1], T(2))))
        factor!(b, :r, RelayFactor(:x, :y))
        factor!(b, :py, PriorFactor(:y, gauss([0, 1], T(1)), quad([0, 1], T(1))))
        factor!(b, :pc1, PriorFactor(:c, CategoricalBelief(T[0.25, 0.75]), k -> T(k)))
        factor!(b, :pc2, PriorFactor(:c, CategoricalBelief(T[0.6, 0.4]), k -> T(2) * k))
        factor!(b, :dz, DataFactor(:z, T[0.5, i]))
        factor!(b, :pz, PriorFactor(:z, gauss([0, 0], T(1)), quad([0, 0], T(1))))
        connect!(b, :px, :x, :x); connect!(b, :r, :x, :x); connect!(b, :r, :y, :y)
        connect!(b, :py, :y, :y); connect!(b, :pc1, :c, :c); connect!(b, :pc2, :c, :c)
        connect!(b, :dz, :z, :z; direction = Emitting()); connect!(b, :pz, :z, :z)
    end
    b = GraphBuilder()
    plate === nothing ? add!(b, 0) : plate!((p, i) -> add!(p, i), b, :rep, plate)
    return validate(build(b))
end

function p8_energy(T; plate = nothing)
    g = p8_graph(T; plate)
    nt = NamedTuple()
    sched = plate === nothing ? spanning_tree_schedule(g) : plate_schedule(g)
    _, report, _, s = infer!(g, sched, nt, nt)
    report.converged || error("P8 test graph did not converge")
    F, _ = LenticulumCore.scalar_free_energy(s, g, nt, nt)
    Ff, _ = factor_free_energies(s, g, nt, nt)
    V = variable_corrections(s, g)
    return F, Ff, V
end

@testset "P8: the graph free energy keeps the element type of its messages" begin
    @test Mycelium.variable_entropy(DiracBelief(Float32[1, 2])) + 1f0 isa Float32
    @test Mycelium.variable_entropy(TrivialBelief()) + 1f0 isa Float32
    @test Mycelium.variable_entropy(CategoricalBelief(Float32[0, 1])) isa Float32
    @test Mycelium.variable_entropy(LenticulumCore.GaussianBelief(Float32[0, 0], Matrix{Float32}(I, 2, 2))) isa Float32
    @test LenticulumCore.support_entropy(LenticulumCore.ExtendedGaussian(Float32[0, 0], Matrix{Float32}(I, 2, 2))) isa Float32
    @test first(local_free_energy(DataFactor(:z, Float32[1]), NamedTuple(), nothing, nothing)) + 1f0 isa Float32

    for plate in (nothing, 3)
        F32, Ff32, V32 = p8_energy(Float32; plate)
        F64, _, _ = p8_energy(Float64; plate)
        @test F32 isa Float32
        @test F32 ≈ F64 rtol = 1e-5
        @test all(e -> !(e isa Float64), values(Ff32.parts))
        @test all(e -> !(e isa Float64), values(V32.parts))
    end

    # review (Codex): a graph whose every term is an exact zero still returns a float
    b = GraphBuilder(); variable!(b, :x, 1)
    factor!(b, :d, DataFactor(:x, Float32[1])); connect!(b, :d, :x, :x; direction = Emitting())
    g = validate(build(b)); nt = NamedTuple()
    _, _, _, s = infer!(g, spanning_tree_schedule(g), nt, nt)
    @test first(LenticulumCore.scalar_free_energy(s, g, nt, nt)) isa AbstractFloat
    # review (Codex): a mixture of categoricals of different element types keeps the widest
    m = LenticulumCore.MixtureBelief(CategoricalBelief[CategoricalBelief(Float32[1, 0]), CategoricalBelief([1e-100, 1.0])], Float32[0, 1])
    @test Mycelium.variable_entropy(m) ≈ 1e-100 * log(1e100) rtol = 1e-6   # 1 − 1e-100 rounds to 1 in Float64; was -0f0
    # review (Codex): derivatives in the precision, ForwardDiff and Zygote, against the closed forms
    # H = (1 + log 2π)/2 − log(λ)/2 and log Z = η²/(2λ) − log(λ)/2 + log(2π)/2
    H(λ) = Mycelium.variable_entropy(LenticulumCore.GaussianBelief([0.3], fill(λ, 1, 1)))
    logZ(λ) = LenticulumCore.logpartition(LenticulumCore.GaussianBelief([0.3], fill(λ, 1, 1)))
    for λ in (0.5, 2.0)
        @test ForwardDiff.derivative(H, λ) ≈ -1 / (2λ)
        @test only(Zygote.gradient(H, λ)) ≈ -1 / (2λ)
        @test ForwardDiff.derivative(logZ, λ) ≈ -0.09 / (2λ^2) - 1 / (2λ)
        @test only(Zygote.gradient(logZ, λ)) ≈ -0.09 / (2λ^2) - 1 / (2λ)
    end
end

# P32: one draw, two precisions, with an independently enumerated discrete oracle.
using Random
function p32_cluster(T, pa, pb, table, μ)
    b = GraphBuilder()
    for (v, d) in ((:a, 2), (:b, 2), (:x, 1)); variable!(b, v, d); end
    for (name, v, p) in ((:pa, :a, pa), (:pb, :b, pb))
        factor!(b, name, PriorFactor(v, CategoricalBelief(T.(p)), _ -> false))
        connect!(b, name, v, v)
    end
    factor!(b, :tab, TableFactor((a=2, b=2), T.(table)))
    connect!(b, :tab, :a, :a); connect!(b, :tab, :b, :b)
    factor!(b, :px, PriorFactor(:x, LenticulumCore.Gaussian(T[μ], fill(T(2),1,1)), _ -> false))
    connect!(b, :px, :x, :x)
    factor!(b, :out, PriorFactor(:x, LenticulumCore.Gaussian(T[0], fill(T(3),1,1)), _ -> false))
    connect!(b, :out, :x, :x)
    gc = cluster(build(b), :c, [:pa, :pb, :tab, :px])
    return gc, gc.factors[gc.factor_index[:c]].factor
end
@testset "P32 cluster and Gaussian overlap element types" begin
    rng = Xoshiro(32); pa = rand(rng,2); pa ./= sum(pa)
    pb = rand(rng,2); pb ./= sum(pb); tab = randn(rng,2,2); μ = randn(rng)
    P = [pa[i]*pb[j]*exp(tab[i,j]) for i=1:2,j=1:2]; Z = sum(P); P ./= Z
    results = map((Float32, Float64)) do T
        gc,c = p32_cluster(T,pa,pb,tab,μ); nt = NamedTuple()
        jt = Mycelium._cluster_joint(c, nt, nt, nothing)
        @test eltype(jt.H) === T
        @test Mycelium._cluster_batchable([c,c],[nt,nt],nothing,nt) === true
        msg = first(Mycelium.factor_message(c,:x,nothing,nt,nothing,nt,nt))
        @test eltype(msg.η) === T
        @test eltype(msg.Λ) === T
        s = MessageStore(gc)
        m = cluster_marginal(s,gc,:c,:a,nt)
        @test eltype(m.logp) === T
        @test exp.(m.logp) ≈ vec(sum(P;dims=2)) rtol=(T === Float64 ? 1e-12 : 1e-3)
        E = first(Mycelium.local_free_energy(c,nt,nt,nt))
        @test all(x -> x isa T || x === false, values(E.parts))
        @test sum(values(E.parts)) ≈ -log(Z) rtol=(T === Float64 ? 1e-12 : 1e-3)
        mx = cluster_marginal(s,gc,:c,:x,nt)
        for comp in (mx isa LenticulumCore.MixtureBelief ? mx.components : [mx])
            @test eltype(comp.η) === T
            @test LenticulumCore.belief_mean(comp) ≈ [μ] rtol=1e-3
            @test LenticulumCore.belief_cov(comp) ≈ fill(2.0,1,1) rtol=1e-3
        end
        (msg,m,E)
    end
    @test results[1][1].η ≈ results[2][1].η rtol=1e-3
    @test results[1][2].logp ≈ results[2][2].logp rtol=1e-3
    @test collect(values(results[1][3].parts)) ≈ collect(values(results[2][3].parts)) rtol=1e-3
    # A purely discrete cluster has an exact zero Gaussian fit term.
    bld = GraphBuilder(); variable!(bld,:a,2); variable!(bld,:b,2)
    factor!(bld,:table,TableFactor((a=2,b=2),Float32.(tab)))
    connect!(bld,:table,:a,:a);connect!(bld,:table,:b,:b)
    discrete = cluster(build(bld),:c,[:table]).factors[1].factor
    ed = first(Mycelium.local_free_energy(discrete,NamedTuple(),NamedTuple(),NamedTuple()))
    @test ed[:fit] === false
    @test all(x -> x isa Float32 || x === false, values(ed.parts))
    gc,c = p32_cluster(Float32,pa,pb,tab,μ); nt=NamedTuple()
    @test Mycelium._joint_numtype(c, [], nt, nothing) === Float64
    @test Mycelium._numtype(nt) === Bool
    impossible = TableFactor((a=2,b=2),Float32[-Inf -Inf; 0 1])
    imsg = first(Mycelium.factor_message(impossible,:a,nothing,(b=DiracBelief(1),),nothing,nt,nt))
    @test eltype(imsg.logp) === Float32
    @test imsg.logp == Float32[-Inf,0]
    inttable = TableFactor((a=2,b=2),[0 1; 2 3])
    intmsg = first(Mycelium.factor_message(inttable,:a,nothing,(b=DiracBelief(1),),nothing,nt,nt))
    @test eltype(intmsg.logp) === Float64
    cat32 = CategoricalBelief(Float32[0.25,0.75])
    mixedtable = first(Mycelium.factor_message(inttable,:a,nothing,(b=cat32,),nothing,nt,nt))
    @test eltype(mixedtable.logp) === Float32
    p2 = exp.(Float64.(inttable.logtable)) * Float64[0.25,0.75]
    @test exp.(mixedtable.logp) ≈ p2 ./ sum(p2) rtol=1e-5
    tab3 = reshape(collect(0:7),2,2,2)
    three = TableFactor((a=2,b=2,c=2),tab3)
    # Missing b is neutral even though it is visited before the Float32 c input.
    mixed3 = first(Mycelium.factor_message(three,:a,nothing,(c=cat32,),nothing,nt,nt))
    @test eltype(mixed3.logp) === Float32
    p3 = vec(sum(exp.(Float64.(tab3)) .* reshape(Float64[0.25,0.75],1,1,2);dims=(2,3)))
    @test exp.(mixed3.logp) ≈ p3 ./ sum(p3) rtol=1e-5
    incoming = LenticulumCore.Gaussian([0.0],fill(3.0,1,1))
    @test eltype(Mycelium._cluster_joint(c,(x=incoming,),nt,nothing).H) === Float64
    d = ForwardDiff.Dual(0.3,1.0)
    incomingdual = GaussianBelief([d],fill(one(d),1,1))
    @test eltype(Mycelium._cluster_joint(c,(x=incomingdual,),nt,nothing).H) === typeof(d)
    for T in (Float32,Float64)
        a = LenticulumCore.Gaussian(T[0.2],fill(T(2),1,1))
        b = LenticulumCore.Gaussian(T[-0.4],fill(T(3),1,1))
        _, z = Mycelium.combine_with_overlap(a,b)
        @test z isa T
        @test z ≈ -0.6^2/10 - log(10π)/2 rtol=1e-5
        @test Mycelium.belief_logdensity(a,T[0.7]) isa T
        u = GaussianBelief(T[0],fill(T(0.5),1,1),true)
        _, zi = Mycelium.combine_with_overlap(u,u)
        @test zi isa T
        @test zi ≈ log(2π)/2 rtol=1e-5
        flat = GaussianBelief(T[0],zeros(T,1,1),true)
        @test last(Mycelium.combine_with_overlap(flat,flat)) isa T
        point = LenticulumCore.extended(DiracBelief(T[0.7]))
        @test last(Mycelium.combine_with_overlap(a,point)) isa T
    end
end

@testset "P33 sample reweighting and damping" begin
    X = reshape(Float32[-1,-0.5,0,0.25,0.5,1,1.5,2],1,:)
    g32 = LenticulumCore.Gaussian(Float32[0.2],fill(1.5f0,1,1))
    g64 = LenticulumCore.Gaussian(Float64[0.2],fill(1.5,1,1))
    for weights in (nothing, Float32[1,2,3,4,4,3,2,1])
        a = LenticulumCore.SampleBelief(X,weights)
        a64 = LenticulumCore.SampleBelief(Float64.(X),weights === nothing ? nothing : Float64.(weights))
        w = Mycelium.combine(a,g32).weights
        @test eltype(w) === Float32
        @test w ≈ Mycelium.combine(a64,g64).weights rtol=1e-5
        @test eltype(Mycelium.combine(a,g64).weights) === Float64
        @test eltype(Mycelium._sample_weights(a)) === Float32
        b = LenticulumCore.SampleBelief(X,reverse(Float32[1,2,3,4,4,3,2,2]))
        @test eltype(Mycelium.damp(a,b,0.3f0).weights) === Float32
        @test Mycelium.belief_distance(a,b) isa Float32
        av = LenticulumCore.SampleBelief(collect(eachcol(X)),weights)
        @test eltype(Mycelium._sample_weights(av)) === Float32
        @test Mycelium._sample_weights(av) == Mycelium._sample_weights(a)
    end
    # GH-21: moments remain differentiable; reweighting adds no mutation to the tape.
    f(x) = sum(Mycelium.combine(LenticulumCore.SampleBelief(reshape(x,1,:)),g32).weights .* x)
    x = vec(X)
    @test ForwardDiff.gradient(f,x) ≈ only(Zygote.gradient(f,x)) rtol=1e-5
end

# A Gaussian residual x-y-μ[k], used to test the stacked path independently of Lenticulum.
struct P33Residual <: LenticulumCore.AbstractLenticulumFactor end
LenticulumCore.islearnable(::P33Residual) = true
LenticulumCore.channels(::P33Residual) = (LenticulumCore.Channel(:x,1), LenticulumCore.Channel(:y,1), LenticulumCore.Channel(:k,2))
Mycelium.cluster_switch(::P33Residual) = :k
function Mycelium.cluster_form(::P33Residual, ps)
    T = eltype(ps.μ)
    Mycelium.ClusterForm((x=ones(T,1,1), y=-ones(T,1,1)), reshape(ps.μ,1,2), fill(T(0.75),1,1), log.(T[0.4,0.6]), :k)
end
function p33_plate(T)
    b = GraphBuilder()
    py = PriorFactor(:y,LenticulumCore.Gaussian(T[0.1],fill(T(1.25),1,1)),_ -> false)
    plate!(b,:rep,4) do p,i
        variable!(p,:x,1); variable!(p,:y,1); variable!(p,:k,2)
        factor!(p,:r,P33Residual()); connect!(p,:r,:x,:x);connect!(p,:r,:y,:y);connect!(p,:r,:k,:k)
        factor!(p,:py,py);connect!(p,:py,:y,:y)
        factor!(p,:d,DataFactor(:x,T[i/4]));connect!(p,:d,:x,:x;direction=Emitting())
    end
    g = cluster(build(b),:rep,:c,[:r,:py];params=:mix)
    return g, (mix=(r=(μ=T[-0.5,1],),py=(;)),)
end
function p33_plate_results(T)
    g,ps = p33_plate(T); nt=(;)
    s=MessageStore(g); for _=1:2; sweep!(s,g,plate_schedule(g),ps,nt); end
    pl=only(plates(g)); cs=[g.factors[g.factor_index[replica_name(:c,i)]].factor for i=1:4]
    inputs=[(x=DiracBelief(T[i/4]),) for i=1:4]
    @test Mycelium._cluster_batchable(cs,inputs,nothing,ps.mix)
    mixed=[i==2 ? (x=DiracBelief(Float64[i/4]),) : inputs[i] for i=1:4]
    T===Float32 && @test !Mycelium._cluster_batchable(cs,mixed,nothing,ps.mix)
    dualps=(r=(μ=ForwardDiff.Dual.(ps.mix.r.μ,one(T)),),py=(;))
    @test !Mycelium._cluster_batchable(cs,inputs,nothing,dualps)
    dualins=[(x=DiracBelief(ForwardDiff.Dual.(T[i/4],one(T))),) for i=1:4]
    @test !Mycelium._cluster_batchable(cs,dualins,nothing,ps.mix)
    B=Mycelium.factor_beliefs(cs,(x=Mycelium.belief_block([m.x for m in inputs]),),ps.mix,nt)
    @test B isa Mycelium.FrozenClusterBeliefs{T}
    jt,_,pts,nA=Mycelium._batch_joint(cs,inputs,ps.mix,nothing)
    l,sel=Mycelium._discrete_parts(cs[1],jt,:k,Dict{Any,Any}())
    kb=Mycelium._discrete_block(:k,l,sel,nA,4)
    @test eltype(kb.logp) === T
    @test exp.(kb.logp) ≈ B.q rtol=1e-5
    blockmsg=first(Mycelium.factor_messages(cs,:x,nothing,nt,fill(nothing,4),ps.mix,nt))
    @test eltype(blockmsg.logw) === T
    wide=[(x=DiracBelief([Float64(i/4)+2.0^-40]),) for i=1:4]
    @test Mycelium._cluster_batchable(cs,wide,nothing,ps.mix)
    widejt,_,widepts,_=Mycelium._batch_joint(cs,wide,ps.mix,nothing)
    @test eltype(widejt.H) === Float64
    @test widepts[:x] == reduce(hcat,[m.x.value for m in wide])
    Eb=first(Mycelium.belief_free_energies(cs,B,ps.mix,nt))
    Ep=[first(Mycelium.local_free_energy(cs[i],inputs[i],ps.mix,nt)) for i=1:4]
    for i=1:4
        @test all(v -> v isa T, values(Eb[i].parts))
        @test collect(values(Eb[i].parts)) ≈ collect(values(Ep[i].parts)) rtol=1e-5
        per=Mycelium.factor_belief(cs[i],inputs[i],ps.mix,nt)
        cols=(2i-1):(2i)
        @test B.M[:,cols] ≈ per.M rtol=1e-5
        @test B.q[:,i] ≈ per.q rtol=1e-5
        # Enumeration: x|k ~ N(μ[k]+0.1, 2); posterior on k and marginal of y.
        prob=[p*exp(-(T(i/4)-μ-T(0.1))^2/4)/sqrt(T(4)*T(π)) for (p,μ) in zip(T[0.4,0.6],ps.mix.r.μ)]
        @test sum(values(Eb[i].parts)) ≈ -log(sum(prob)) rtol=1e-5
        mk=cluster_marginal(s,g,replica_name(:c,i),:k,ps)
        @test eltype(mk.logp) === T
        @test exp.(mk.logp) ≈ prob/sum(prob) rtol=1e-5
    end
    mb=first(Mycelium.factor_messages(cs,:x,nothing,fill(nt,4),fill(nothing,4),ps.mix,nt))
    for i=1:4
        mr=first(Mycelium.factor_message(cs[i],:x,nothing,nt,nothing,ps.mix,nt))
        @test eltype(mb[i].logw) === T
        @test mb[i].logw ≈ mr.logw rtol=1e-5
        for (a,b) in zip(mb[i].components,mr.components)
            @test eltype(a.η) === T && eltype(a.Λ) === T
            @test a.η ≈ b.η rtol=1e-5
            @test a.Λ ≈ b.Λ rtol=1e-5
        end
    end
    f(μ)=sum(sum(values(e.parts)) for e in first(Mycelium.belief_free_energies(cs,B,(r=(μ=μ,),py=(;)),nt)))
    grad=only(Zygote.gradient(f,ps.mix.r.μ))
    @test eltype(grad) === T
    @test grad ≈ ForwardDiff.gradient(f,ps.mix.r.μ) rtol=1e-5
    F=first(LenticulumCore.scalar_free_energy(s,g,ps,nt))
    @test F isa T
    return F,grad,mb[1].logw,B.M
end
@testset "P33 cluster plate element types" begin
    a,b=p33_plate_results(Float32),p33_plate_results(Float64)
    for (x,y) in zip(a,b); @test x ≈ y rtol=1e-3; end
end
