using JLArrays

# Two independent coordinates per variable: every covariance/precision stays
# diagonal, so this tests replay without relying on device dense factorisation.
struct ScheduleDeviceUnary <: AbstractLenticulumFactor end
struct ScheduleDevicePair <: AbstractLenticulumFactor end
LenticulumCore.channels(::ScheduleDeviceUnary) = channels(AdjointUnary())
LenticulumCore.channels(::ScheduleDevicePair) = channels(AdjointPair())
LenticulumCore.supported_polarities(::ScheduleDeviceUnary) = supported_polarities(AdjointUnary())
LenticulumCore.supported_polarities(::ScheduleDevicePair) = supported_polarities(AdjointPair())
LenticulumCore.supports_polarity(f::Union{ScheduleDeviceUnary,ScheduleDevicePair}, p::Polarity) = p in supported_polarities(f)
LenticulumCore.islearnable(::Union{ScheduleDeviceUnary,ScheduleDevicePair}) = true
Mycelium.uses_cavity(::Union{ScheduleDeviceUnary,ScheduleDevicePair}) = true
Mycelium.factor_message(::ScheduleDeviceUnary,t::Symbol,pol,inputs,prior,p,st) =
    (GaussianBelief(p.h,Diagonal(p.q)),st)
function Mycelium.factor_message(::ScheduleDevicePair,t::Symbol,pol,inputs,prior,p,st)
    b = inputs[t === :x ? :y : :x]
    return GaussianBelief(p.k .* b.η ./ (p.k .+ b.Λ.diag),
        Diagonal(p.k .* b.Λ.diag ./ (p.k .+ b.Λ.diag))),st
end

function Mycelium.local_free_energy(::ScheduleDeviceUnary,bs,p,st)
    q,h = bs.x.Λ.diag,bs.x.η
    C = inv.(p.q .+ q); m = C .* (p.h .+ h)
    T = eltype(m)
    return sum(p.q .* (m.^2 .+ C) ./ 2 .- p.h .* m .-
        log.(T(2)*T(π)*exp(one(T)) .* C) ./ 2),st
end
function Mycelium.local_free_energy(::ScheduleDevicePair,bs,p,st)
    qx,hx = bs.x.Λ.diag,bs.x.η; qy,hy = bs.y.Λ.diag,bs.y.η
    det = qx .* qy .+ p.k .* (qx .+ qy)
    Cx = (p.k .+ qy) ./ det; Cy = (p.k .+ qx) ./ det; Cxy = p.k ./ det
    mx = Cx .* hx .+ Cxy .* hy; my = Cxy .* hx .+ Cy .* hy
    T = eltype(mx)
    return sum(p.k .* ((mx.-my).^2 .+ Cx .+ Cy .- 2 .* Cxy) ./ 2 .-
        (T(2)*(one(T)+log(T(2)*T(π))) .- log.(det)) ./ 2),st
end

function schedule_device_graph(T, coupling; device=true, loopy=true)
    cv(x) = device ? JLArray(T.(x)) : T.(x)
    b = GraphBuilder()
    for v in (:x,:y,:z); variable!(b,v); end
    for (f,v) in ((:u,:x),(:v,:y),(:w,:z))
        factor!(b,f,ScheduleDeviceUnary()); connect!(b,f,:x,v;coupling)
    end
    pairs = loopy ? ((:a,:x,:y),(:b,:y,:z),(:c,:z,:x)) : ((:a,:x,:y),(:b,:y,:z))
    for (f,x,y) in pairs
        factor!(b,f,ScheduleDevicePair())
        connect!(b,f,:x,x;coupling); connect!(b,f,:y,y;coupling)
    end
    p = (u=(q=cv([0.7,0.8]),h=cv([0.4,-0.2])),
         v=(q=cv([0.9,0.6]),h=cv([-0.3,0.1])),
         w=(q=cv([0.6,1.1]),h=cv([0.1,0.2])),a=(k=cv([2.0,1.6]),),b=(k=cv([1.3,1.2]),))
    loopy && (p=merge(p,(c=(k=cv([1.7,1.0]),),)))
    st=NamedTuple{keys(p)}(map(_ -> (;),values(p)))
    return build(b),p,st
end

@testset "LP-002 C device schedule replay and implicit boundary" begin
    JLArrays.allowscalar(false)
    errors=Dict(Float32=>0.0,Float64=>0.0)
    for T in (Float32,Float64)
        @test LenticulumCore.isuninformative(GaussianBelief(JLArray(zeros(T,2)),Diagonal(JLArray(zeros(T,2)))))
        @test !LenticulumCore.isuninformative(GaussianBelief(JLArray(ones(T,2)),Diagonal(JLArray(zeros(T,2)))))
        @test !LenticulumCore.isuninformative(GaussianBelief(JLArray(zeros(T,2)),Diagonal(JLArray(ones(T,2)))))
    end
    # Cover both precisions, topologies, couplings, schedules and damping with
    # representative combinations rather than a redundant Cartesian product.
    cases=((Float32,false,ExactCoupling(),:flooding,0.0),
           (Float32,false,PathwiseCoupling(),:residual,0.35),
           (Float32,true,ExactCoupling(),:residual,0.0),
           (Float32,true,PathwiseCoupling(),:flooding,0.35),
           (Float64,false,ExactCoupling(),:residual,0.0),
           (Float64,false,PathwiseCoupling(),:flooding,0.35),
           (Float64,true,ExactCoupling(),:flooding,0.0),
           (Float64,true,PathwiseCoupling(),:residual,0.35))
    for (T,loopy,coupling,kind,damping) in cases
        g,p,st=schedule_device_graph(T,coupling;loopy)
        gc,pc,sc=schedule_device_graph(T,coupling;loopy,device=false)
        if coupling isa ExactCoupling
            ref=free_energy_gradient(MessageStore(gc),gc,pc,sc;ad=AutoForwardDiff(),
                schedule=flooding_schedule(gc),nsweeps=100)
            for (ad,solver) in ((AutoForwardDiff(),:dense),(AutoZygote(),:krylov))
                d=free_energy_gradient(MessageStore(gc),gc,pc,sc;ad,adjoint=:implicit,
                    schedule=flooding_schedule(gc),linear_solver=solver,
                    tol=T===Float32 ? 1e-6 : 1e-12,linear_tol=T===Float32 ? 2e-6 : 1e-10)
                @test LenticulumCore._flatten(d) ≈ LenticulumCore._flatten(ref) atol=T===Float32 ? 3e-5 : 1e-9 rtol=T===Float32 ? 3e-5 : 1e-9
            end
        end
        sched=kind === :flooding ? flooding_schedule(g) : ResidualSchedule(maxtasks=7)
        let
            n=12
            ops=Mycelium._realised_tasks(g,sched,n,p,st;damping)
            # Compare the SAME realised sequence, including residual tie choices.
            csched=sched isa ResidualSchedule ? SequentialSchedule(ops) : flooding_schedule(gc)
            ref=free_energy_gradient(MessageStore(gc),gc,pc,sc;ad=AutoForwardDiff(),
                schedule=csched,nsweeps=sched isa ResidualSchedule ? 1 : n,damping)
            for ad in (AutoForwardDiff(),AutoZygote())
                grad=free_energy_gradient(MessageStore(g),g,p,st;ad,schedule=sched,nsweeps=n,damping)
                flat=LenticulumCore._flatten(grad)
                @test flat isa JLArray
                @test eltype(flat) === T
                @test Array(flat) ≈ LenticulumCore._flatten(ref) atol=T===Float32 ? 3e-5 : 1e-9 rtol=T===Float32 ? 3e-5 : 1e-9
                errors[T]=max(errors[T],norm(Array(flat)-LenticulumCore._flatten(ref))/max(1,norm(LenticulumCore._flatten(ref))))
            end
        end
        # Dense graph coordinates intentionally remain a CPU interface. The
        # message-coordinate representation has no device adapter; do not copy
        # device data to the host silently, even for a Krylov solve.
        for solver in (:dense,:krylov)
            @test_throws (msg -> occursin("device coordinate packing is unsupported",msg)) free_energy_gradient(
                MessageStore(g),g,p,st;ad=AutoZygote(),schedule=flooding_schedule(g),
                adjoint=:implicit,linear_solver=solver,tol=T===Float32 ? 1e-5 : 1e-11)
        end
    end
    @info "LP-002 C device replay maximum scaled error to CPU realised-sequence reference" errors
end

@testset "LP-002 C device Dirac replay and CPU vector fixed point" begin
    JLArrays.allowscalar(false)
    for T in (Float32,Float64), coupling in (ExactCoupling(),PathwiseCoupling())
        b=GraphBuilder(); variable!(b,:x)
        factor!(b,:source,AdjointDirac()); factor!(b,:loss,LossFactor((:x,),x->sum(abs2,x)/2))
        connect!(b,:source,:x,:x;direction=Emitting(),coupling)
        connect!(b,:loss,:x,:x;direction=Absorbing(),coupling)
        g=build(b); a=T[0.7,-0.2]
        p=(source=(a=JLArray(a),),loss=(;)); pc=(source=(a=a,),loss=(;))
        st=(source=(;),loss=(;))
        for sched in (flooding_schedule(g),ResidualSchedule(maxtasks=2)), damping in (0.0,0.35),
                ad in (AutoForwardDiff(),AutoZygote())
            d=free_energy_gradient(MessageStore(g),g,p,st;ad,schedule=sched,nsweeps=3,damping)
            @test d.source.a isa JLArray
            @test eltype(d.source.a) === T
            @test Array(d.source.a) ≈ a rtol=1e-6
        end
        for ad in (AutoForwardDiff(),AutoZygote()), solver in (:dense,:krylov)
            d=free_energy_gradient(MessageStore(g),g,pc,st;ad,schedule=flooding_schedule(g),
                adjoint=:implicit,linear_solver=solver,linear_tol=1e-6)
            @test d.source.a ≈ a rtol=1e-6
        end
        @test_throws (msg -> occursin("device coordinate packing is unsupported",msg)) free_energy_gradient(
            MessageStore(g),g,p,st;schedule=flooding_schedule(g),adjoint=:implicit)
    end
end
