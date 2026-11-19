# Runs every benchmark in `bench/` and the examples of Lenticulum and VariationalDiffusion at small size,
# each in its own Julia process with `LENTICULUM_SMOKE=true` (Feature Backlog P15). It checks that
# they still run, not what they measure: a script's numbers at small size mean nothing, and its own
# checks (against a dense solve, a closed form) are whatever the script does.
#
# Run:  julia --project=bench bench/smoke.jl                 # all of them
#       julia --project=bench bench/smoke.jl bp_grid plates  # some, by file name without `.jl`
# (once: julia --project=bench -e 'using Pkg; Pkg.instantiate()')
#
# Each script's output goes to `<out>/<name>.log`, a table of the results to `<out>/summary.md` and to
# stdout; `<out>` is `$OUT`, or `~/.cache/lenticulum/smoke/<date-time>/`. A script that writes files
# (profiles) is given `OUT=<out>/<name>/`. A script fails if it exits with a nonzero code or runs
# longer than `$SMOKE_TIMEOUT` seconds (default 1200, compilation included); the runner exits with 1
# if any failed. Set `GITHUB_STEP_SUMMARY` (CI does) and the table is appended to it as well.
#
# The switch: a script reads `get(ENV, "LENTICULUM_SMOKE", "false") == "true"` and, if set, runs
# its sizes, repetitions and iterations at the smallest values that still go through every code path
# it has. Every new benchmark or example must have one (the runner finds new files by itself).
using Dates, Printf

const ROOT = dirname(@__DIR__)
const BENCH = joinpath(ROOT, "bench")

# The folders run, each with the environment its scripts run in and the files left out. Not run: the
# `lp002_*` scripts (diagnostics of AD backends' limits, opt-in, some expected to fail), the examples of
# ImplicitLayers (not yet reviewed, `vault/Project/Refactor 2026-11.md`), `gpu_energy.jl` (needs CUDA) and
# `VariationalDiffusion.jl/examples/mnist/` (its own environment and data).
const FOLDERS = [
    (dir = BENCH, project = BENCH, skip = ["divider.jl", "kalman_chain.jl", "smoke.jl"]),   # helpers, included
    (dir = joinpath(ROOT, "lib", "Lenticulum.jl", "examples"), project = BENCH, skip = String[]),
    (dir = joinpath(ROOT, "lib", "VariationalDiffusion.jl", "examples"), project = BENCH, skip = ["gpu_energy.jl", "calibration_setup.jl"]),
]

# Julia flags per script; the default is `-t auto`.
const FLAGS = Dict("jit_latency" => `-t 2,1 --gcthreads=2,1`)

function scripts()
    found = [(name = splitext(f)[1], path = joinpath(F.dir, f), project = F.project) for F in FOLDERS
             for f in sort(readdir(F.dir)) if endswith(f, ".jl") && !startswith(f, "lp002_") && !(f in F.skip)]
    names = [s.name for s in found]
    allunique(names) || error("two scripts share a name: ", names)
    found
end

function run_one(name, path, project, out, timeout)
    log = joinpath(out, name * ".log")
    dir = joinpath(out, name)
    flags = get(FLAGS, name, `-t auto`)
    cmd = `$(Base.julia_cmd()) --project=$project $flags $path`
    cmd = addenv(cmd, "LENTICULUM_SMOKE" => "true", "OUT" => dir)
    t0 = time()
    status = open(log, "w") do io
        p = run(pipeline(Cmd(cmd; dir = ROOT); stdout = io, stderr = io); wait = false)
        while process_running(p)
            if time() - t0 > timeout
                kill(p); sleep(5); process_running(p) && kill(p, Base.SIGKILL)
                wait(p)
                return "timeout"
            end
            sleep(0.5)
        end
        wait(p)
        p.exitcode == 0 ? "ok" : "exit $(p.exitcode)"
    end
    (; name, path = relpath(path, ROOT), status, seconds = time() - t0, log)
end

function main(args)
    all = scripts()
    if !isempty(args)
        known = [s.name for s in all]
        unknown = setdiff(args, known)
        isempty(unknown) || error("no such script: ", join(unknown, ", "), "; known: ", join(known, ", "))
        all = filter(s -> s.name in args, all)
    end
    out = get(ENV, "OUT", joinpath(homedir(), ".cache", "lenticulum", "smoke", Dates.format(now(), "yyyy-mm-dd-HHMM")))
    mkpath(out)
    timeout = parse(Float64, get(ENV, "SMOKE_TIMEOUT", "1200"))
    results = map(all) do s
        r = run_one(s.name, s.path, s.project, out, timeout)
        @printf("%-28s %-9s %7.1f s\n", s.name, r.status, r.seconds); flush(stdout)
        r
    end
    failed = filter(r -> r.status != "ok", results)
    table = IOBuffer()
    println(table, "## Benchmarks and examples at small size\n")
    println(table, "$(length(results) - length(failed)) of $(length(results)) ran; `LENTICULUM_SMOKE=true`, Julia $(VERSION).\n")
    println(table, "| script | result | seconds |\n|---|---|---|")
    for r in results
        @printf(table, "| `%s` | %s | %.0f |\n", r.path, r.status == "ok" ? "ok" : "**$(r.status)**", r.seconds)
    end
    text = String(take!(table))
    write(joinpath(out, "summary.md"), text)
    haskey(ENV, "GITHUB_STEP_SUMMARY") && open(io -> write(io, text), ENV["GITHUB_STEP_SUMMARY"], "a")
    println("\nLogs and summary in ", out)
    for r in failed
        println("\n── ", r.name, " (", r.status, "), last lines of ", r.log)
        foreach(println, last(readlines(r.log), 30))
    end
    isempty(failed) || exit(1)
end

main(ARGS)
