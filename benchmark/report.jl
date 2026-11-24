using BenchmarkTools, JSON, Printf
using Statistics: median

struct BenchmarkRun
    metadata::Vector{Any}
    trials::Dict{String,BenchmarkTools.Trial}
    latency::Dict{String,Any}
    failures::Vector{Any}
end

function read_run(paths)
    metadata, failures = Any[], Any[]
    trials = Dict{String,BenchmarkTools.Trial}()
    latency = Dict{String,Any}()
    for path in paths
        data = JSON.parsefile(path)
        push!(metadata, data["metadata"])
        push!(failures, data["failures"])
        group = only(BenchmarkTools.load(IOBuffer(JSON.json(data["trials"]))))
        for (parts, trial) in BenchmarkTools.leaves(group)
            key = join(parts, '/')
            if !haskey(trials, key) || time(median(trial)) < time(median(trials[key]))
                trials[key] = trial
            end
        end
        for (key, sample) in data["latency"]
            if !haskey(latency, key) || sample["median"] < latency[key]["median"]
                latency[key] = sample
            end
        end
    end
    return BenchmarkRun(metadata, trials, latency, failures)
end

function metadata_table(io, label, result)
    for (i, meta) in enumerate(result.metadata)
        n = length(result.metadata) > 1 ? " ($i)" : ""
        commit = meta["commit"] * (meta["dirty"] ? " (dirty libraries)" : "")
        @printf(io, "| %s%s | %s | %s | %d + %d | %s | %s | %.1f s |\n", label, n,
                commit, meta["julia"], meta["threads"], meta["interactive_threads"], meta["cpu"],
                meta["date"], meta["wall_seconds"])
    end
end

common_keys(a, b) = sort!(collect(intersect(keys(a), keys(b))))

function error_summary(message)
    lines = split(message, '\n'; keepempty=false)
    actual_error = findfirst(line -> startswith(line, "ERROR:"), lines)
    return lines[something(actual_error, 1)]
end

function coverage(io, base, head)
    for (label, a, b) in (("New keys", head, base), ("Removed keys", base, head))
        keys_a = union(keys(a.trials), keys(a.latency))
        keys_b = union(keys(b.trials), keys(b.latency))
        println(io, "\n### ", label, "\n")
        missing = sort!(collect(setdiff(keys_a, keys_b)))
        isempty(missing) && println(io, "None.")
        for key in missing
            println(io, "- `", key, "`")
        end
    end
    println(io, "\n### Failed suites\n")
    if all(isempty, base.failures) && all(isempty, head.failures)
        println(io, "None.")
    end
    for (side, result) in (("Base", base), ("Head", head)), (i, failed) in enumerate(result.failures)
        for pkg in sort!(collect(keys(failed))), key in sort!(collect(keys(failed[pkg])))
            message = error_summary(failed[pkg][key])
            println(io, "- ", side, " run ", i, " — `", pkg, "`, `", key, "` — ", message)
        end
    end
end

function warm_table(io, pkg, base, head, thresholds, noise, budgeted)
    println(io, "\n### ", pkg, "\n")
    println(io, "| Benchmark | Base median (µs) | Head median (µs) | H/B | Allocations B → H | Bytes B → H | Flag |")
    println(io, "|---|---:|---:|---:|---:|---:|---|")
    flagged = false
    for key in common_keys(base.trials, head.trials)
        startswith(key, pkg * "/") || continue
        b, h = median(base.trials[key]), median(head.trials[key])
        flags = String[]
        config = thresholds["time"]
        guard = max(config["relative"], config["noise_multiplier"] * get(noise, key, config["default_noise"]))
        if isregression(time, judge(h, b; time_tolerance=guard))
            push!(flags, "Time regression")
        elseif isregression(time, judge(b, h; time_tolerance=guard))
            push!(flags, "Time improvement")
        end
        m = any(re -> occursin(re, key), budgeted) ? 0.0 : thresholds["allocations"]["relative"]
        allocs(h) > allocs(b) * (1 + m) && push!(flags, "Allocation increase")
        flagged |= !isempty(flags)
        case = split(key, '/'; limit=2)[2]
        @printf(io, "| `%s` | %.3f | %.3f | %.3f | %d → %d | %d → %d | %s |\n",
                case, time(b) / 1e3, time(h) / 1e3, time(h) / time(b),
                allocs(b), allocs(h), memory(b), memory(h), join(flags, ", "))
    end
    return flagged
end

function latency_table(io, base, head, threshold)
    keys = common_keys(base.latency, head.latency)
    isempty(keys) && return false
    println(io, "\n### First-call latency\n")
    println(io, "| Workload | Base median (s) | Head median (s) | H/B | Minimum B → H (s) | Flag |")
    println(io, "|---|---:|---:|---:|---:|---|")
    flagged = false
    for key in keys
        b, h = base.latency[key], head.latency[key]
        ratio = h["median"] / b["median"]
        regression = ratio > 1 + threshold
        flagged |= regression
        @printf(io, "| `%s` | %.3f | %.3f | %.3f | %.3f → %.3f | %s |\n",
                key, b["median"], h["median"], ratio, b["minimum"], h["minimum"],
                regression ? "Latency regression" : "")
    end
    return flagged
end

function write_noise(io, base, head, path, default_noise)
    noise = Dict(key => abs(log(time(median(head.trials[key])) / time(median(base.trials[key]))))
                 for key in common_keys(base.trials, head.trials))
    mkpath(dirname(path))
    open(io -> TOML.print(io, Dict("noise" => noise); sorted=true), path, "w")
    println(io, "\n## A/A warm timing noise\n")
    if isempty(noise)
        println(io, "No shared warm benchmarks measured.")
    else
        values_by_key = collect(values(noise))
        @printf(io, "%d keys. Median |log ratio| %.4f, maximum %.4f.\n\n",
                length(noise), median(values_by_key), maximum(values_by_key))
        @printf(io, "Keys above %.3f\n\n", default_noise)
        above = sort!(filter(p -> last(p) > default_noise, collect(noise)); by=last, rev=true)
        isempty(above) && println(io, "None.")
        for (key, value) in above
            @printf(io, "- `%s` %.4f\n", key, value)
        end
    end
    println("Noise in ", path)
end

function write_report(options, report, runs, revisions, head_label)
    thresholds = TOML.parsefile(joinpath(BENCHMARK_DIR, "thresholds.toml"))
    noise_path = joinpath(BENCHMARK_DIR, "noise.toml")
    noise = isfile(noise_path) ? TOML.parsefile(noise_path)["noise"] : Dict{String,Float64}()
    budgeted = Regex.(thresholds["allocations"]["budgeted"])
    results = read_run.(runs)
    head = last(results)
    io = IOBuffer()
    println(io, "# Benchmark comparison\n")
    println(io, "| Revision | Commit | Julia | Threads (default + interactive) | CPU | UTC | Run wall time |")
    println(io, "|---|---|---|---:|---|---|---:|")
    for (rev, base) in zip(revisions, results)
        metadata_table(io, "Base " * rev, base)
    end
    metadata_table(io, "Head " * head_label, head)
    smoke() && println(io, "\nSmoke mode uses short trials and one latency sample per key.")
    haskey(options, "--interleave") && println(io, "\nEach row uses the faster median of its two runs per revision.")
    flagged = false
    for (rev, base) in zip(revisions, results)
        println(io, "\n## ", head_label, " vs ", rev)
        for pkg in selected_packages(options)
            flagged |= warm_table(io, pkg, base, head, thresholds, noise, budgeted)
        end
        flagged |= latency_table(io, base, head, thresholds["latency"]["relative"])
        coverage(io, base, head)
    end
    if haskey(options, "--aa")
        path = abspath(get(options, "--noise", joinpath(dirname(report), "noise.toml")))
        write_noise(io, first(results), head, path, thresholds["time"]["default_noise"])
    end
    contents = String(take!(io))
    write(report, contents)
    if haskey(ENV, "GITHUB_STEP_SUMMARY")
        open(io -> println(io, contents), ENV["GITHUB_STEP_SUMMARY"], "a")
    end
    println("Report in ", report)
    any(!isempty, head.failures) && exit(1)
    haskey(options, "--strict") && flagged && exit(1)
end
