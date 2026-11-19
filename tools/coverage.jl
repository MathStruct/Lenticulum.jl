# A line-coverage report of the packages' test suites (Feature Backlog P15): which source lines no test
# of the package's own suite reaches.
#
# Run:  julia --project=tools tools/coverage.jl --run                     # the five core packages
#       julia --project=tools tools/coverage.jl --run lib/Mycelium.jl     # some packages
#       julia --project=tools tools/coverage.jl lib/Mycelium.jl           # only process existing .cov files
# (once: julia --project=tools -e 'using Pkg; Pkg.instantiate()')
#
# With `--run`, each package's suite runs as `Pkg.test(coverage = true)` in its own process (log in
# `<out>/<package>-test.log`); Pkg then counts only the lines of that package's own directory, so a
# line of Mycelium reached only by Lenticulum's tests counts as not covered. A failing suite stops the
# report for that package (its counts would be partial). Device tests count only with
# `LENTICULUM_DEVICE_TESTS=true` (Mycelium, MyceliumLearning; the CI job sets it). Without `--run`,
# the `.cov` files already beside the sources are read. Either way they are deleted afterwards (`--keep` keeps them).
#
# Writes to `<out>` (`$OUT`, or `~/.cache/lenticulum/coverage/<date-time>/`):
#   lcov.info     every package's `src/` and `ext/`, for an editor or `genhtml`
#   summary.md    per package and per file: covered / coverable lines, sorted from the least covered
#   uncovered.md  per file, the line ranges no test reached
# and appends `summary.md` to `$GITHUB_STEP_SUMMARY` when that is set (CI).
using CoverageTools, Dates, Logging, Printf

const ROOT = dirname(@__DIR__)
const CORE = ["lib/LenticulumCore.jl", "lib/Mycelium.jl", "lib/MyceliumLearning.jl",
              "lib/Lenticulum.jl", "lib/VariationalDiffusion.jl"]

function run_suite(pkg, out)
    log = joinpath(out, basename(pkg) * "-test.log")
    cmd = `$(Base.julia_cmd()) -t auto --project=$pkg -e 'using Pkg; Pkg.test(coverage = true)'`
    cmd = addenv(Cmd(cmd; dir = ROOT, ignorestatus = true), "JULIA_NUM_THREADS" => "auto")
    p = open(io -> run(pipeline(cmd; stdout = io, stderr = io)), log, "w")
    p.exitcode == 0 || println("  ", pkg, ": suite failed (exit ", p.exitcode, "), see ", log)
    p.exitcode == 0
end

source_dirs(pkg) = filter(isdir, [joinpath(ROOT, pkg, "src"), joinpath(ROOT, pkg, "ext")])

function line_ranges(lines)
    isempty(lines) && return ""
    parts, a, b = String[], lines[1], lines[1]
    for l in lines[2:end]
        if l == b + 1; b = l; else push!(parts, a == b ? "$a" : "$(a)–$(b)"); a = b = l; end
    end
    push!(parts, a == b ? "$a" : "$(a)–$(b)")
    join(parts, ", ")
end

quietly(f) = with_logger(f, ConsoleLogger(stderr, Logging.Warn))   # CoverageTools logs every file

pct(c, t) = t == 0 ? "–" : @sprintf("%.1f %%", 100c / t)

function main(args)
    run_tests = "--run" in args
    keep = "--keep" in args
    pkgs = filter(a -> !startswith(a, "--"), args)
    isempty(pkgs) && (pkgs = CORE)
    pkgs = [rstrip(relpath(abspath(p), ROOT), '/') for p in pkgs]
    for p in pkgs
        isfile(joinpath(ROOT, p, "Project.toml")) || error("not a package: ", p)
    end
    out = get(ENV, "OUT", joinpath(homedir(), ".cache", "lenticulum", "coverage", Dates.format(now(), "yyyy-mm-dd-HHMM")))
    mkpath(out)

    allcov = FileCoverage[]
    summary = IOBuffer(); uncovered = IOBuffer()
    println(summary, "## Line coverage by each package's own suite\n")
    println(summary, "Julia $(VERSION), $(Dates.format(now(), "yyyy-mm-dd")). A line counts as covered if the package's own tests reach it.\n")
    println(summary, "| package | covered | coverable | |\n|---|---|---|---|")
    perfile = IOBuffer()
    failed = String[]
    for pkg in pkgs
        if run_tests
            println("Testing ", pkg, " with coverage …"); flush(stdout)
            ok = run_suite(pkg, out)
            if !ok
                push!(failed, pkg)
                println(summary, "| `$pkg` | suite failed | | |")
                keep || quietly(() -> clean_folder(joinpath(ROOT, pkg)))
                continue
            end
        end
        hascov = any(d -> any(f -> endswith(f, ".cov"), reduce(vcat, [fs for (_, _, fs) in walkdir(d)]; init = String[])),
                     source_dirs(pkg))
        hascov || error("no .cov files under ", pkg, "/src or ext: run with --run, or run its suite with coverage first")
        cov = quietly(() -> reduce(vcat, [process_folder(d) for d in source_dirs(pkg)]; init = FileCoverage[]))
        foreach(fc -> fc.filename = relpath(abspath(fc.filename), ROOT), cov)
        append!(allcov, cov)
        c, t = get_summary(cov)
        println(summary, "| `$pkg` | $c | $t | $(pct(c, t)) |")
        println(perfile, "\n### `$pkg`\n\n| file | covered | coverable | |\n|---|---|---|---|")
        rows = map(cov) do fc
            fc_c, fc_t = get_summary(fc)
            (fc.filename, fc_c, fc_t)
        end
        sort!(rows; by = r -> (r[3] == 0 ? 2.0 : r[2] / r[3], r[1]))
        for (f, fc_c, fc_t) in rows
            fc_t == 0 && continue
            println(perfile, "| `$f` | $fc_c | $fc_t | $(pct(fc_c, fc_t)) |")
        end
        println(uncovered, "\n## `$pkg`\n")
        for fc in sort(cov; by = fc -> fc.filename)
            missed = [i for (i, n) in enumerate(fc.coverage) if n !== nothing && n == 0]
            isempty(missed) || println(uncovered, "- `", fc.filename, "`: ", line_ranges(missed))
        end
        keep || quietly(() -> clean_folder(joinpath(ROOT, pkg)))
    end
    text = String(take!(summary)) * String(take!(perfile))
    write(joinpath(out, "summary.md"), text)
    write(joinpath(out, "uncovered.md"), "# Lines no test reached\n" * String(take!(uncovered)))
    LCOV.writefile(joinpath(out, "lcov.info"), allcov)
    haskey(ENV, "GITHUB_STEP_SUMMARY") && open(io -> write(io, text), ENV["GITHUB_STEP_SUMMARY"], "a")
    print(text)
    println("\nReport in ", out)
    isempty(failed) || (println("Suites failed: ", join(failed, ", ")); exit(1))
end

main(ARGS)
