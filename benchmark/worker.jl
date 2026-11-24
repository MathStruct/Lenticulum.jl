using BenchmarkTools, JSON
using Dates: now, UTC
using Statistics: median
include(joinpath(@__DIR__, "common.jl"))

function measure_suite!(results, failures, pkg, pattern, seconds)
    suite = try
        load_suite(pkg)
    catch err
        err isa InterruptException && rethrow()
        failures[pkg] = Dict(pkg => sprint(showerror, err))
        return
    end
    leaves = sort!(BenchmarkTools.leaves(Base.invokelatest(getproperty, suite, :SUITE)); by=first)
    trials = results[pkg] = BenchmarkGroup()
    for ((package, group, name), benchmark) in leaves
        key = join((package, group, name), '/')
        occursin(pattern, key) || continue
        try
            tune!(benchmark; seconds=min(seconds, 0.15))
            # `tune!` does nothing for a case with fixed `evals`, and `run` does not warm up, so the
            # first sample would fill the graph's caches (polarities, emit plans): warm up first.
            BenchmarkTools.warmup(benchmark; verbose=false)
            trial = run(benchmark; seconds, samples=smoke() ? 4 : 1000)
            trials[group][name] = trial
        catch err
            err isa InterruptException && rethrow()
            get!(() -> Dict{String,String}(), failures, pkg)[key] = sprint(showerror, err)
        end
    end
end

function measure_latency!(latency, failures, pkg, pattern, directory)
    names = if pkg == "MyceliumLearning"
        ("using", "kalman_infer_and_gradient")
    elseif pkg == "VariationalDiffusion"
        ("using", "relation_query", "relation_gradient")
    elseif pkg == "Lenticulum"
        ("using", "regime_plate_fit_em")
    else
        ("using",)
    end
    script = joinpath(BENCHMARK_DIR, "latency.jl")
    repeats = smoke() ? 1 : 3
    for name in names
        key = "latency/$pkg/$name"
        occursin(pattern, key) || continue
        samples = Float64[]
        for _ in 1:repeats
            seconds_file = joinpath(directory, "latency-seconds")
            log = joinpath(directory, "latency-log")
            cmd = julia_worker(`--project=$(dirname(Base.active_project())) $script $pkg $name $seconds_file`)
            process = open(io -> run(pipeline(ignorestatus(cmd); stdout=io, stderr=io)), log, "w")
            if !success(process)
                message = strip(read(log, String))
                get!(() -> Dict{String,String}(), failures, pkg)[key] =
                    isempty(message) ? "Latency process failed: $process" : message
                break
            end
            push!(samples, parse(Float64, read(seconds_file, String)))
        end
        if length(samples) == repeats
            latency[key] = (; minimum=minimum(samples), median=median(samples), samples)
        end
    end
end

function main(args)
    libs, out, names, pattern, seconds, started = args
    packages = split(names, ',')
    regex = Regex(pattern)
    trials = BenchmarkGroup()
    latency = Dict{String,Any}()
    failures = Dict{String,Dict{String,String}}()
    sources = get(TOML.parsefile(Base.active_project()), "sources", Dict())
    for pkg in packages
        if !haskey(sources, pkg)
            project = joinpath(libs, pkg * ".jl", "Project.toml")
            reason = isfile(project) ? "A required local package is absent at this revision" :
                     "Package is absent at this revision"
            failures[pkg] = Dict(pkg => reason)
            continue
        end
        start = time_ns()
        println("Warm suite ", pkg)
        flush(stdout)
        measure_suite!(trials, failures, pkg, regex, parse(Float64, seconds))
        println("  ", round((time_ns() - start) / 1e9; digits=2), " s")
    end
    mktempdir() do directory
        for pkg in packages
            haskey(sources, pkg) || continue
            println("First-call latency ", pkg)
            flush(stdout)
            measure_latency!(latency, failures, pkg, regex, directory)
        end
    end
    git = `git -C $libs`
    metadata = Dict(
        "commit" => readchomp(`$git rev-parse HEAD`),
        "dirty" => !isempty(readchomp(`$git status --porcelain -- .`)),
        "julia" => string(VERSION),
        "threads" => Threads.nthreads(),
        "interactive_threads" => Threads.nthreads(:interactive),
        "cpu" => first(Sys.cpu_info()).model,
        "date" => string(now(UTC)) * "Z",
        "wall_seconds" => (time_ns() - parse(UInt64, started)) / 1e9,
    )
    saved = IOBuffer()
    BenchmarkTools.save(saved, trials)
    # Embed the standard BenchmarkTools serialization alongside latency and metadata.
    result = (; metadata, trials=JSON.parse(String(take!(saved))), latency, failures)
    open(out, "w") do io
        JSON.json(io, result)
        println(io)
    end
    for (pkg, cases) in sort!(collect(failures); by=first), (key, message) in sort!(collect(cases); by=first)
        println(stderr, pkg, " failed at ", key, ": ", first(split(message, '\n')))
    end
    println("Results in ", out)
end

main(ARGS)
