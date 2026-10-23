# Does compiling on another thread stall a running loop? A control loop that does not allocate
# runs on one thread and records the longest gap between its iterations, while another thread
# idles, compiles fresh functions, or compiles them with the GC switched off.
# Recorded in `Parallelism and Compilation.md` §5.
#
# Run:  julia -t 2,1 --gcthreads=2,1 bench/jit_latency.jl
#
# Two default threads are needed (the loop and the compiler); since Julia 1.12 the main thread
# belongs to the interactive pool, so both tasks are spawned on :default. A busy loop spawned on
# :interactive can land on the main thread and starve it.
using Printf

const NFUNCS = parse(Int, get(ENV, "NFUNCS", "300"))
Threads.nthreads(:default) ≥ 2 || error("run with at least two default threads: julia -t 2,1 --gcthreads=2,1 bench/jit_latency.jl")

# the control loop: arithmetic only; every 2000 iterations it records the worst gap seen
function control_loop(stop, worst_gaps)
    last, worst, acc, n = time_ns(), UInt64(0), 0.0, 0
    while !stop[]
        for i in 1:1000
            acc += sin(i * 1e-3)
        end
        t = time_ns()
        worst = max(worst, t - last)
        last = t
        n += 1
        if n % 2000 == 0
            push!(worst_gaps, worst)
            worst = UInt64(0)
        end
    end
    return acc
end

# the background: compile NFUNCS new methods, each with a different body, and call each once
function compile_fresh(k, offset)
    for i in 1:k
        name = Symbol(:fresh_, offset + i)
        f = Core.eval(Main, :(function $name(x::Vector{Float64})
            s = 0.0
            for (j, v) in enumerate(x)
                s += v^$(i % 7 + 2) * j + $(i) * sin(v)
            end
            return s
        end))
        Base.invokelatest(f, rand(10))
    end
end

function run_bench(name, background)
    stop, gaps = Threads.Atomic{Bool}(false), UInt64[]
    loop = Threads.@spawn :default control_loop(stop, gaps)
    sleep(0.5)
    gc0 = Base.gc_time_ns()
    t = @elapsed background()
    stop[] = true
    fetch(loop)
    sort!(gaps)
    worst = isempty(gaps) ? 0.0 : last(gaps) / 1e6
    typical = isempty(gaps) ? 0.0 : gaps[max(1, end ÷ 2)] / 1e6
    @printf("%-40s worst gap %7.2f ms   typical window maximum %.3f ms   background %.1f s   GC %.3f s\n",
            name, worst, typical, t, (Base.gc_time_ns() - gc0) / 1e9)
    flush(stdout)
end

@printf("Julia %s, %d default + %d interactive threads, %d GC threads, %d functions\n", VERSION,
        Threads.nthreads(:default), Threads.nthreads(:interactive), Threads.ngcthreads(), NFUNCS)
run_bench("idle", () -> sleep(3))
run_bench("compiling on another thread", () -> fetch(Threads.@spawn :default compile_fresh(NFUNCS, 0)))
run_bench("compiling, GC disabled meanwhile", () -> fetch(Threads.@spawn :default begin
    GC.enable(false)
    try
        compile_fresh(NFUNCS, NFUNCS)
    finally
        GC.enable(true)
    end
end))
