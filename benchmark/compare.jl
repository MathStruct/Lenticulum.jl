using Pkg
include(joinpath(@__DIR__, "common.jl"))

const USAGE = """
Usage: julia --project=benchmark benchmark/compare.jl --base REV [--head REV] [--also-base REV]
       julia --project=benchmark benchmark/compare.jl --aa REV [--noise FILE.toml]
Common options: [--report FILE.md] [--packages A,B] [--filter REGEX]
                [--samples-time SECONDS] [--interleave] [--strict]
Without --head, the comparison uses the current checkout's library files.
"""

function checkout(git, parent, label, rev)
    worktree = joinpath(parent, label)
    run(`$git worktree add --detach --quiet $worktree $rev`)
    return worktree
end

function measure(checkout_dir, directory, label, options)
    out = joinpath(directory, label * ".json")
    libs = joinpath(checkout_dir, "lib")
    runner = joinpath(BENCHMARK_DIR, "run.jl")
    forwarded = [arg for key in ("--packages", "--filter", "--samples-time") if haskey(options, key)
                 for arg in (key, options[key])]
    println("Measuring ", label)
    flush(stdout)
    run(julia_worker(`$runner --libs $libs --out $out $forwarded`))
    return out
end

function compare(options, temporary, report_path)
    git = `git -C $REPO_ROOT`
    aa = haskey(options, "--aa")
    revisions = aa ? [options["--aa"]] : [options["--base"]]
    haskey(options, "--also-base") && push!(revisions, options["--also-base"])
    head = aa ? options["--aa"] : get(options, "--head", nothing)

    # Keep detached worktree bookkeeping in a temporary local clone. Git shares
    # the object store with the source and needs only read access to it.
    mirror = joinpath(temporary, "repo.git")
    run(`$git clone --quiet --shared --bare -- $REPO_ROOT $mirror`)
    mirror_git = `git --git-dir=$mirror`
    worktrees = Dict{String,String}()
    function worktree(rev)
        hash = readchomp(`$git rev-parse --verify --end-of-options $(rev * "^{commit}")`)
        return get!(worktrees, hash) do
            checkout(mirror_git, temporary, hash, hash)
        end
    end

    runs = [String[] for _ in 1:(length(revisions) + 1)]
    try
        trees = worktree.(revisions)
        push!(trees, isnothing(head) ? REPO_ROOT : worktree(head))
        labels = length(revisions) == 1 ? ("base", "head") : ("base", "also-base", "head")
        for repeat in 1:(haskey(options, "--interleave") ? 2 : 1)
            for (i, (tree, label)) in enumerate(zip(trees, labels))
                name = repeat == 1 ? label : label * "-2"
                push!(runs[i], measure(tree, dirname(report_path), name, options))
            end
        end
    finally
        for worktree_dir in values(worktrees)
            run(`$mirror_git worktree remove --force $worktree_dir`)
        end
    end
    return runs, revisions, isnothing(head) ? "working tree" : head
end

function main(args)
    args == ["--help"] && return print(USAGE)
    options = parse_options(args,
        ("--base", "--also-base", "--head", "--aa", "--noise", "--report", "--packages", "--filter", "--samples-time");
        switches=("--interleave", "--strict"))
    aa = haskey(options, "--aa")
    aa != haskey(options, "--base") || error(USAGE)
    aa && any(k -> haskey(options, k), ("--head", "--also-base", "--interleave")) && error(USAGE)
    !aa && haskey(options, "--noise") && error("--noise is the output of --aa")
    selected_packages(options)
    sample_budget(options)
    Regex(get(options, "--filter", ""))
    report = abspath(get(options, "--report", joinpath(REPO_ROOT, "outputs", "benchmarks", "report.md")))
    mkpath(dirname(report))
    return mktempdir() do temporary
        runs, revisions, head = compare(options, temporary, report)

        # Report in a tools-only environment so even an incompatible base can
        # be read with the head's BenchmarkTools and JSON serialization.
        tools = joinpath(temporary, "report")
        mkdir(tools)
        project = TOML.parsefile(PROJECT_FILE)
        for table in ("deps", "compat")
            filter!(pair -> first(pair) in ("BenchmarkTools", "JSON"), project[table])
        end
        delete!(project, "sources")
        write_project(tools, project)
        instantiate(tools)
        Pkg.activate(tools; io=devnull)
        report_writer = include(joinpath(BENCHMARK_DIR, "report.jl"))
        Base.invokelatest(report_writer, options, report, runs, revisions, head)
    end
end

main(ARGS)
