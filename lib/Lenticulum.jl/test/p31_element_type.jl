# P31: paired graphs from the same seeded draw; no conversion of incoming data.
module P31ElementType
using Test, Lenticulum, LenticulumCore, Mycelium, MyceliumLearning, LinearAlgebra, Random, LuxCore
using ForwardDiff, Zygote
const nt = NamedTuple()
meanof(b::DiracBelief) = b.value
meanof(b::MixtureBelief) = belief_mean(moment_match(b))
meanof(b) = belief_mean(b)
covof(b::DiracBelief) = zeros(eltype(b.value), length(b.value), length(b.value))
covof(b::MixtureBelief) = belief_cov(moment_match(b))
covof(b) = belief_cov(b)

# P32 fixed Regime inference through cluster.jl:574,582.
# P32 fixed MyceliumLearning/em.jl MomentStats storage.
function typecheck(ok, upstream = false)
    if upstream
        @test ok
    else
        @test ok
    end
end
function checkbelief(b, T; upstream = false)
    if b isa GaussianBelief
        typecheck(eltype(b.η) === T, upstream); typecheck(eltype(b.Λ) === T, upstream)
    elseif b isa ExtendedGaussian
        for x in (b.μ, b.Σ, b.D); typecheck(eltype(x) === T, upstream); end
    elseif b isa CategoricalBelief
        typecheck(eltype(b.logp) === T, upstream)
    elseif b isa MixtureBelief
        typecheck(eltype(b.logw) === T, upstream)
        for c in b.components; checkbelief(c, T; upstream); end
    elseif b isa DiracBelief
        # A discrete outcome is an index, not floating data.
        b.value isa Integer || @test eltype(b.value) === T
    else
        @test b === nothing || b isa TrivialBelief
    end
end
function energies(s, g, ps, st; upstream = false)
    scalar(x) = x isa Float32 || x === false
    ff = first(factor_free_energies(s, g, ps, st))
    for (name, e) in pairs(ff.parts)
        isreg = g.factors[g.factor_index[name]].factor isa RegimeFactor
        for v in values(e isa GradedEnergy ? e.parts : (x = e,))
            typecheck(scalar(v), upstream && isreg)
        end
    end
    for v in values(variable_corrections(s, g).parts)
        # P32 fixed the categorical variable's Float64 correction from cluster outputs.
        if upstream && !scalar(v)
            @test scalar(v) # P32 fixed cluster.jl:574,582
        else
            @test scalar(v)
        end
    end
end
function runpair(make; upstream = false, plate = false, oracle = nothing)
    g32, p32, st32 = make(Float32); g64, p64, st64 = make(Float64)
    schedule(g) = plate ? plate_schedule(g) : SequentialSchedule(all_tasks(g))
    m32, r32, _, s32 = infer!(g32, schedule(g32), p32, st32; tol = 1e-6, maxsweeps = 100)
    m64, r64, _, s64 = infer!(g64, schedule(g64), p64, st64; tol = 1e-10, maxsweeps = 100)
    @test r32.converged; @test r64.converged
    for v in g32.variables
        b32, b64 = m32[v.name], m64[v.name]
        # P32 fixed Regime outputs and their downstream cavities.
        checkbelief(b32, Float32; upstream = upstream && !(b32 isa DiracBelief))
        if b32 isa CategoricalBelief
            @test probabilities(b32) ≈ probabilities(b64) rtol=1e-3 atol=1e-5
        elseif !(b32 isa DiracBelief && b32.value isa Integer)
            @test meanof(b32) ≈ meanof(b64) rtol=1e-3 atol=1e-5
            @test covof(b32) ≈ covof(b64) rtol=1e-3 atol=1e-5
        end
    end
    for (outgoing, msgs) in ((true, s32.to_variable), (false, s32.to_factor)), (i, msg) in enumerate(msgs)
        b = msg === nothing ? nothing : msg.belief
        # Use the edge direction to distinguish unaffected prior/data emissions.
        f = g32.factors[g32.edges[i].factor].factor
        affected = upstream && (outgoing ? f isa RegimeFactor : f isa GaussianPrior)
        checkbelief(b, Float32; upstream = affected)
    end
    F32 = first(LenticulumCore.scalar_free_energy(s32, g32, p32, st32))
    F64 = first(LenticulumCore.scalar_free_energy(s64, g64, p64, st64))
    typecheck(F32 isa Float32, upstream)
    @test F32 ≈ F64 rtol=1e-3 atol=1e-5
    energies(s32, g32, p32, st32; upstream)
    oracle === nothing || oracle(m64, F64, p64)
    return g32, p32, st32, s32
end

function switching(T; output = false, regime = false, plate = false, mixed = false)
    rng = Xoshiro(31)
    offsets = randn(rng, 1, 2) ./ 3
    data = randn(rng, 1) ./ 3
    gb = GraphBuilder()
    function add!(b, i)
        variable!(b, :x, 1); variable!(b, :y, 1); variable!(b, :k, 2)
        factor!(b, :dx, DataFactor(:x, T[0.2])); connect!(b, :dx, :x, :x; direction = Emitting())
        f = regime ? RegimeFactor((x = 1, y = 1), 2, 1; noise = T(0.4), output = output ? :y : nothing) :
            SwitchingConstraintFactor((x = 1, y = 1), 2, 1; noise = T(0.4), output = output ? :y : nothing)
        factor!(b, :f, f)
        for c in (:x, :y, :k); connect!(b, :f, c, c); end
        factor!(b, :d, DataFactor(:y, mixed ? data : T.(data))); connect!(b, :d, :y, :y; direction = Emitting())
    end
    plate ? plate!((b, i) -> add!(b, i), gb, :rep, 3) : add!(gb, 1)
    g = validate(build(gb)); ps, st = LuxCore.setup(Xoshiro(0), g)
    A = output ? (x = fill(T(-0.7), 1, 1),) : (x = fill(T(-0.7), 1, 1), y = ones(T, 1, 1))
    regime && (A = map(a -> repeat(a, 1, 1, 2), A))
    ps = merge(ps, (f = (A = A, b = T.(offsets), logw = T[0.1, -0.2]),))
    return g, ps, st
end
# Independent finite enumeration of the two outcomes with continuous data clamped.
function switch_oracle(m, F, ps; plate = false)
    y = meanof(m[plate ? replica_name(:y, 1) : :y])[1]
    x = meanof(m[plate ? replica_name(:x, 1) : :x])[1]
    lp = ps.f.logw .- Lenticulum._logsumexp(ps.f.logw)
    l = lp .- ((y .- 0.7*x .- vec(ps.f.b)).^2 ./ 0.4 .+ log(2π*0.4)) ./ 2
    q = exp.(l .- Lenticulum._logsumexp(l))
    @test probabilities(m[plate ? replica_name(:k, 1) : :k]) ≈ q atol=1e-9
    @test F ≈ -(plate ? 3 : 1)*Lenticulum._logsumexp(l) atol=1e-9
end

function nonlinear(T, method; jacobian = false)
    gb = GraphBuilder(); variable!(gb, :x, 1); variable!(gb, :y, 1)
    factor!(gb, :p, GaussianPrior(:x, T[0.2], fill(T(0.8), 1, 1))); connect!(gb, :p, :x, :x)
    # Affine is a polynomial with an exact dense oracle for both approximation rules.
    f(x) = 2 .* x .+ one(eltype(x))/5
    j = jacobian ? x -> fill(2one(eltype(x)), 1, 1) : nothing
    factor!(gb, :f, NonlinearFactor(f, 1 => 1; noise = T(0.4), method, jacobian = j))
    connect!(gb, :f, :x, :x); connect!(gb, :f, :y, :y)
    factor!(gb, :d, DataFactor(:y, T[0.3])); connect!(gb, :d, :y, :y; direction = Emitting())
    g = validate(build(gb)); ps, st = LuxCore.setup(Xoshiro(0), g); return g, ps, st
end

function likelihood(T, kind, method; observed = true)
    lik = kind === :poisson ? PoissonLikelihood(; maxcount = 5) : kind === :bivariate ? BivariatePoissonLikelihood(; maxcount = 4) : OrdinalProbitLikelihood(T[-0.5, 0.7])
    A = kind === :bivariate ? reshape(T[0.4, -0.2, 0.1], 3, 1) : fill(T(0.7), 1, 1)
    gb = GraphBuilder(); variable!(gb, :x, 1); variable!(gb, :y, obsdim(lik))
    factor!(gb, :p, GaussianPrior(:x, T[0.2], fill(T(0.6), 1, 1))); connect!(gb, :p, :x, :x)
    factor!(gb, :f, LikelihoodFactor(lik, 1; A, method)); connect!(gb, :f, :x, :x); connect!(gb, :f, :y, :y)
    if observed
        factor!(gb, :d, DataFactor(:y, fill(T(1), obsdim(lik)))); connect!(gb, :d, :y, :y; direction = Emitting())
    end
    g = validate(build(gb)); ps, st = LuxCore.setup(Xoshiro(0), g); return g, ps, st
end

@testset "P31 element types" begin
@testset "switching and regime" begin
    for output in (false, true), plate in (false, true)
        runpair(T -> switching(T; output, plate); plate, oracle = (m,F,p) -> switch_oracle(m,F,p; plate))
        runpair(T -> switching(T; output, plate, regime = true); plate, upstream = true,
                oracle = (m,F,p) -> switch_oracle(m,F,p; plate))
    end
    g, ps, st = switching(Float32; mixed = true)
    m, _, _, s = infer!(g, tree_schedule(g), ps, st)
    checkbelief(first(Mycelium.factor_message(g.factors[g.factor_index[:f]].factor, :x, nothing, (y = DiracBelief(Float64[0.3]),), nothing, ps.f, nt)), Float64)
    @test first(LenticulumCore.scalar_free_energy(s, g, ps, st)) isa Float64
    for T in (Float32, Float64), F in (SwitchingConstraintFactor, RegimeFactor)
        f = F((x = 1, y = 1), 2, 1; noise = T(0.4), output = :y)
        @test eltype(f.output_coeff) === T
        @test eltype(f.Q) === T
    end
end
@testset "switching continuous paths" begin
    function continuous(T)
        local f, p
        f=SwitchingConstraintFactor((x=1,y=1),2,1;noise=T(0.4),output=:y)
        p=(A=(x=fill(T(-0.7),1,1),),b=reshape(T[-0.3,0.4],1,2),logw=T[0.1,-0.2])
        x=Gaussian(T[0.2],fill(T(0.8),1,1));y=DiracBelief(T[0.3]);k=CategoricalBelief(T[0.4,0.6])
        return f,p,(x=x,y=y,k=k)
    end
    f64,p64,b64=continuous(Float64)
    f,p,b=continuous(Float32)
    for target in (:x,:y,:k)
        inputs=Base.structdiff(b,NamedTuple{(target,)})
        inputs64=Base.structdiff(b64,NamedTuple{(target,)})
        msg=first(Mycelium.factor_message(f,target,nothing,inputs,nothing,p,nt))
        msg64=first(Mycelium.factor_message(f64,target,nothing,inputs64,nothing,p64,nt))
        checkbelief(msg,Float32)
        if msg isa CategoricalBelief
            @test probabilities(msg) ≈ probabilities(msg64) rtol=1e-3
        else
            @test meanof(msg) ≈ meanof(msg64) rtol=1e-3
            @test covof(msg) ≈ covof(msg64) rtol=1e-3
        end
        mf=first(Mycelium.mean_field_message(f,target,nothing,inputs,p,nt))
        checkbelief(mf,Float32)
        # Plate Gaussian and point blocks, including a free continuous channel.
        blocks=map(inputs) do bi
            Mycelium.belief_block(fill(bi,3))
        end
        out=first(Mycelium.factor_messages(fill(f,3),target,nothing,blocks,fill(nothing,3),p,nt))
        for i in 1:3; checkbelief(Mycelium.replica(out,i),Float32); end
        expanded=first(Mycelium.mixture_message(f,target,nothing,inputs,nothing,p,nt))
        checkbelief(expanded,Float32)
    end
    # Mixture inputs are expanded with their own type, also for linear constraints.
    mix=MixtureBelief([b.x,Gaussian(Float32[-0.2],fill(0.5f0,1,1))],Float32[0.4,0.6])
    for target in (:y,:k)
        inputs=target===:y ? (x=mix,k=b.k) : (x=mix,y=b.y)
        checkbelief(first(Mycelium.mixture_message(f,target,nothing,inputs,nothing,p,nt)),Float32)
    end
    for msgs in (b,merge(b,(k=DiracBelief(1),)),(x=b.x,y=b.y))
        E=first(Mycelium.local_free_energy(f,msgs,p,nt))
        @test all(x->x isa Float32 || x===false,values(E.parts))
        fb=Mycelium.factor_belief(f,msgs,p,nt)
        @test all(x->x isa Float32 || x===false,values(first(Mycelium.belief_free_energy(f,fb,p,nt)).parts))
    end
    blocks=map(b) do bi; Mycelium.belief_block(fill(bi,3)); end
    frozen=Mycelium.factor_beliefs(fill(f,3),blocks,p,nt)
    @test eltype(frozen.M)===Float32
    @test eltype(frozen.Σ)===Float32
    @test eltype(frozen.q)===Float32
    for E in first(Mycelium.belief_free_energies(fill(f,3),frozen,p,nt))
        @test all(x->x isa Float32 || x===false,values(E.parts))
    end
    msg=first(Mycelium.factor_message(f,:x,nothing,(y=b.y,k=b.k),nothing,p,nt))
    post=Mycelium.combine(b.x,msg)
    @test eltype(post.logw)===Float32 # P32 fixed Mycelium/src/gaussian.jl:298 log(2π) in _overlap
    for c in post.components; checkbelief(c,Float32); end
    # Independent Gaussian conditioning per outcome, with a categorical input.
    S=0.4+0.7^2*0.8; residuals = 0.3 .- 0.7*0.2 .- vec(p64.b)
    lp = log.([0.4,0.6]) .+ p64.logw .- Lenticulum._logsumexp(p64.logw) .- residuals.^2 ./ (2S)
    q = exp.(lp .- Lenticulum._logsumexp(lp))
    @test mixture_weights(post) ≈ q rtol=1e-3
    for k in 1:2
        @test belief_mean(post.components[k]) ≈ [0.2+0.8*0.7/S*residuals[k]] rtol=1e-3
        @test belief_cov(post.components[k]) ≈ fill(0.8-(0.8*0.7)^2/S,1,1) rtol=1e-3
    end
end
@testset "nonlinear" begin
    for method in (Linearization(), Unscented()), jacobian in (false, true)
        runpair(T -> nonlinear(T, method; jacobian); oracle = (m,F,p) -> begin
            S = 4*0.8 + 0.4; d = 0.3 - (2*0.2+0.2)
            @test belief_mean(m[:x]) ≈ [0.2 + 1.6/S*d] atol=1e-9
            @test belief_cov(m[:x]) ≈ fill(0.8-1.6^2/S, 1, 1) atol=1e-9
            @test F ≈ (d^2/S+log(2π*S))/2 atol=1e-9
        end)
        # A genuinely nonlinear function also preserves the type in both directions.
        f = NonlinearFactor(x -> sin.(x), 1 => 1; noise = 0.4f0, method)
        prior = Gaussian(Float32[0.2], fill(0.3f0, 1, 1))
        checkbelief(first(Mycelium.factor_message(f, :y, nothing, (x = prior,), nothing, nt, nt)), Float32)
        checkbelief(first(Mycelium.factor_message(f, :x, nothing, (y = DiracBelief(Float32[0.4]),), prior, nt, nt)), Float32)
    end
end
@testset "likelihoods" begin
    for kind in (:poisson, :bivariate, :ordinal), method in (Laplace(), ExpectationPropagation(order = 12)), observed in (false, true)
        runpair(T -> likelihood(T, kind, method; observed))
    end
end
@testset "hard shared plates" begin
    function hard(T)
        local g, ps, st
        gb = GraphBuilder()
        plate!(gb, :rep, 3) do b, i
            variable!(b, :x, 2); variable!(b, :y, 2)
            for c in (:x, :y)
                factor!(b, Symbol(:p,c), GaussianPrior(c, T[0.1,-0.2], Matrix{T}(I,2,2)))
                connect!(b, Symbol(:p,c), c, c)
            end
            factor!(b, :f, HardConstraintFactor((x=2,y=2),1; noise=zero(T)))
            connect!(b,:f,:x,:x); connect!(b,:f,:y,:y)
        end
        g = validate(build(gb)); ps, st = LuxCore.setup(Xoshiro(0),g)
        ps = merge(ps, (f=(A=(x=T[1 0.2],y=T[-0.3 1]),c=T[0.3]),))
        return g,ps,st
    end
    runpair(hard; plate=true, oracle=(m,F,p)->begin
        G = hcat(p.f.A.x,p.f.A.y); μ = repeat([0.1,-0.2],2)
        V=G*G'; d=p.f.c-G*μ
        post=μ+G'*(V\d); S=I-G'*(V\G)
        @test belief_mean(m[replica_name(:x,1)]) ≈ post[1:2] atol=1e-9
        @test belief_cov(m[replica_name(:x,1)]) ≈ S[1:2,1:2] atol=1e-9
        @test F ≈ 3*(dot(d,V\d)+logdet(2π*V))/2 atol=1e-9
    end)
    for T in (Float32,Float64)
        f=HardConstraintFactor((x=2,y=2),1;noise=zero(T))
        p=(A=(x=T[1 0.2],y=T[-0.3 1]),c=T[0.3])
        bs=[Gaussian(T[i/10,-0.2],Matrix{T}(I,2,2)) for i in 1:3]
        parts=Lenticulum._hard_channel_parts(bs,2,3)
        @test parts !== nothing
        # Prove the shared kernel is reached, not only the fallback.
        shared=Lenticulum._hard_shared_messages(fill(f,3),:y,p,3,n->bs,Lenticulum._hard_channel_parts)
        @test shared !== nothing
        for i in 1:3
            checkbelief(shared[i],T)
            oracle=first(Mycelium.factor_message(f,:y,nothing,(x=bs[i],),nothing,p,nt))
            @test shared[i].μ ≈ extended(oracle).μ rtol=1e-4 atol=1e-5
            @test shared[i].Σ ≈ extended(oracle).Σ rtol=1e-4 atol=1e-5
            @test shared[i].D * shared[i].D' ≈ extended(oracle).D * extended(oracle).D' rtol=1e-4 atol=1e-5
        end
    end
end
@testset "exposed parameters and coupling" begin
    function exposed(T)
        local g, ps, st
        gb=GraphBuilder()
        for c in (:x,:y,:w); variable!(gb,c,1); end
        factor!(gb,:p,GaussianPrior(:w,T[0.2],fill(T(0.8),1,1)));connect!(gb,:p,:w,:w)
        factor!(gb,:f,expose(GaussianFactor(1=>1;noise=T(0.4)),(:A=>:w,)))
        connect!(gb,:f,:x,:x;direction=Absorbing());connect!(gb,:f,:y,:y;direction=Absorbing());connect!(gb,:f,:w,:w)
        for (c,v) in ((:x,T[0.7]),(:y,T[0.3]))
            factor!(gb,Symbol(:d,c),DataFactor(c,v));connect!(gb,Symbol(:d,c),c,c;direction=Emitting())
        end
        g=validate(build(gb));ps,st=LuxCore.setup(Xoshiro(0),g)
        return g,merge(ps,(f=(b=T[0.1],),)),st
    end
    runpair(exposed;oracle=(m,F,p)->begin
        S=0.4+0.7^2*0.8; d=0.3-0.1-0.7*0.2
        @test belief_mean(m[:w]) ≈ [0.2+0.8*0.7/S*d] atol=1e-9
        @test belief_cov(m[:w]) ≈ fill(0.8-(0.8*0.7)^2/S,1,1) atol=1e-9
        @test F ≈ (d^2/S+log(2π*S))/2 atol=1e-9
    end)
    for T in (Float32,Float64)
        f=GaussianFactor(1=>1;noise=T(0.4))
        msgs=(x=DiracBelief(T[0.7]),y=DiracBelief(T[0.3]),A=Gaussian(T[0.2],fill(T(0.8),1,1)),b=DiracBelief(T[0.1]))
        for target in (:A,:b,:y)
            checkbelief(first(Mycelium.parameter_message(f,target,nothing,msgs,nt,nt)),T)
        end
        e=first(Mycelium.parameter_free_energy(f,msgs,nt,nt))
        @test all(x->x isa T || x === false,values(e.parts))
        points=merge(msgs,(A=DiracBelief(T[0.2]),))
        @test all(x->x isa T || x === false,values(first(Mycelium.parameter_free_energy(f,points,nt,nt)).parts))
    end
    g,p,st=switching(Float32);g64,p64,_=switching(Float64)
    for normalize in (false,true)
        w=coupling_weights(g,p;normalize);w64=coupling_weights(g64,p64;normalize)
        @test valtype(typeof(w)) === Float32
        @test collect(values(w)) ≈ collect(values(w64)) rtol=1e-3 atol=1e-5
    end
end
@testset "M steps and EM upstream limit" begin
    # The numerical solve itself accepts Float32; its answer has an independent regression oracle.
    rng=Xoshiro(31);X=randn(rng,20,2);Y=X*[0.7,-0.3].+0.1
    for T in (Float32,Float64)
        Φ=T.(hcat(X,ones(20),Y));S=Φ'*Φ
        W=reshape(T[0,0,0,1],1,4)
        Lenticulum._solve_blocks!(W,S,Bool[1,1,1,0],:oracle)
        @test eltype(W) === T
        @test -vec(W[1,1:3]) ≈ hcat(X,ones(20))\Y rtol=1e-3 atol=1e-5
        @test eltype(Lenticulum._normalised_logw(T[3,7])) === T
    end
    # One actual EM iteration. MomentStats and the cluster E-step inject Float64 upstream.
    function emgraph(T)
        local g, ps, st
        gb=GraphBuilder()
        for i in 1:20
            x,y,k=(replica_name(c,i) for c in (:x,:y,:k))
            variable!(gb,x,1);variable!(gb,y,1);variable!(gb,k,2)
            for (c,v,s) in ((:x,T[X[i,1]],x),(:y,T[Y[i]],y))
                d=replica_name(Symbol(:d,c),i)
                factor!(gb,d,DataFactor(c,v));connect!(gb,d,c,s;direction=Emitting())
            end
            f=replica_name(:f,i)
            factor!(gb,f,RegimeFactor((x=1,y=1),2,1;noise=T(0.4),output=:y);params=:f)
            connect!(gb,f,:x,x);connect!(gb,f,:y,y);connect!(gb,f,:k,k)
        end
        g=validate(build(gb));ps,st=LuxCore.setup(Xoshiro(0),g)
        emps=(A=(x=fill(T(-0.5),1,1,2),),b=reshape(T[-0.2,0.2],1,2),logw=T[0,0])
        return g,merge(ps,(f=emps,)),st
    end
    g,p,st=emgraph(Float32);g64,p64,st64=emgraph(Float64)
    out,Fs,s=fit_em(g,p,st;schedule=SequentialSchedule(all_tasks(g)),iters=1,tol=-Inf)
    out64,Fs64,_=fit_em(g64,p64,st64;schedule=SequentialSchedule(all_tasks(g64)),iters=1,tol=-Inf)
    @test length(Fs)==2
    @test eltype(Fs) === Float32
    for (_, stats, _) in values(moment_statistics(s,g,out,st))
        @test stats isa MomentStats{Float32}
    end
    @test !isempty(moment_statistics(s,g,out,st))
    @test out.f.b != p.f.b
    # Independent one-step EM for the two regressions with shared fixed noise.
    Xm = hcat(X[:,1],ones(20))
    lr = hcat((-(Y .- 0.5.*X[:,1] .- c).^2 ./ 0.8 for c in (-0.2,0.2))...)
    R = exp.(lr .- maximum(lr;dims=2)); R ./= sum(R;dims=2)
    for k in 1:2
        θ = (Xm' * Diagonal(R[:,k]) * Xm) \ (Xm' * (R[:,k].*Y))
        @test -out64.f.A.x[1,1,k] ≈ θ[1] atol=1e-10
        @test out64.f.b[1,k] ≈ θ[2] atol=1e-10
        @test exp(out64.f.logw[k]) ≈ sum(R[:,k])/20 atol=1e-10
    end
    for a in (out.f.A.x,out.f.b,out.f.logw)
        @test eltype(a) === Float32 # P32 fixed MyceliumLearning/src/em.jl MomentStats
    end
    @test out.f.A.x ≈ out64.f.A.x rtol=1e-3 atol=1e-5
    @test out.f.b ≈ out64.f.b rtol=1e-3 atol=1e-5
    @test out.f.logw ≈ out64.f.logw rtol=1e-3 atol=1e-5
    @test Fs ≈ Fs64 rtol=1e-3
end
@testset "Float32 derivatives" begin
    for T in (Float32,Float64)
        f=SwitchingConstraintFactor((x=1,),2,1;noise=T(0.4))
        msgs=(x=DiracBelief(T[0.3]),k=CategoricalBelief(T[0.4,0.6]))
        fun(b)=first(Mycelium.local_free_energy(f,msgs,(A=(x=ones(T,1,1),),b=reshape([b,-b],1,2),logw=T[0.1,-0.2]),nt))[:fit]
        d=ForwardDiff.derivative(fun,T(0.2));dz=only(Zygote.gradient(fun,T(0.2)))
        @test d isa T;@test dz isa T
        @test d ≈ dz rtol=1e-4
        nl=NonlinearFactor(x->sin.(x),1=>1;noise=T(0.4))
        jac(x)=only(first(Lenticulum.linearize(nl,[x],zeros(T,1,1))))
        @test ForwardDiff.derivative(jac,T(0.2)) isa T
    end
end
end
end # module

# P32 review (LP-061, finding M3): Float32 EM on a clustered regime plate tracks Float64. The
# reviewer's 12-point problem is degenerate (each regime fits its points exactly, a noise variance
# collapses and F falls without bound, non-monotonically even in Float64); with 60 points the
# Float32 and Float64 histories agree to about 1e-6 (7e-7 measured over 15 iterations).
function p32_em_plate(T, N)
    rng = Xoshiro(4)
    Wt = [randn(rng, 3, 2) for _ in 1:3]
    μt = [[0.0, 3.0, 0.0], [3.0, -2.0, 1.0], [-3.0, 0.0, 2.0]]
    Z = reduce(hcat, [(k = mod1(i, 3); Wt[k] * randn(rng, 2) .+ μt[k] .+ 0.2 .* randn(rng, 3)) for i in 1:N])
    b = GraphBuilder()
    for i in 1:N
        r(s) = replica_name(s, i)
        variable!(b, r(:z), 3); variable!(b, r(:x), 2); variable!(b, r(:k), 3)
        factor!(b, r(:d), DataFactor(:z, T.(Z[:, i]))); connect!(b, r(:d), :z, r(:z); direction = Emitting())
        factor!(b, r(:pr), GaussianPrior(:x, zeros(T, 2), Matrix{T}(I, 2, 2)); params = :pr); connect!(b, r(:pr), :x, r(:x))
        factor!(b, r(:reg), RegimeFactor((z = 3, x = 2), 3, 3; learn_noise = true, output = :z, noise = T(0.5)); params = :reg)
        connect!(b, r(:reg), :z, r(:z)); connect!(b, r(:reg), :x, r(:x)); connect!(b, r(:reg), :k, r(:k))
    end
    gc = cluster(as_plate(build(b), :pts, N), :pts, :c, [:pr, :reg]; params = :mix)
    f = RegimeFactor((z = 3, x = 2), 3, 3; learn_noise = true, output = :z, noise = T(0.5))
    p0 = map(x -> x isa AbstractArray ? T.(x) : x, merge(LuxCore.initialparameters(Xoshiro(2), f), (b = Z[:, 1:3],)))
    p0 = merge(p0, (A = map(a -> T.(a), p0.A),))
    nt = NamedTuple()
    ps = merge(NamedTuple(replica_name(:d, i) => nt for i in 1:N), (mix = (pr = nt, reg = p0),))
    return gc, ps
end
@testset "P32: Float32 EM on a regime plate tracks Float64" begin
    nt = NamedTuple()
    (g32, ps32), (g64, ps64) = p32_em_plate(Float32, 60), p32_em_plate(Float64, 60)
    p32, F32, _ = fit_em(g32, ps32, nt; schedule = plate_schedule(g32), iters = 6, tol = 0)
    p64, F64, _ = fit_em(g64, ps64, nt; schedule = plate_schedule(g64), iters = 6, tol = 0)
    @test eltype(F32) === Float32 && eltype(F64) === Float64
    @test all(x -> eltype(x) === Float32, (p32.mix.reg.b, p32.mix.reg.logw, p32.mix.reg.L, p32.mix.reg.A.x))
    @test F32 ≈ F64 rtol = 1e-5
    @test all(diff(F64) .<= 0)
    @test p32.mix.reg.b ≈ p64.mix.reg.b rtol = 1e-4
end
