include(joinpath(@__DIR__, "common.jl"))

const USAGE = """
Usage: julia benchmark/run.jl --libs DIR --out FILE.json
                              [--packages A,B] [--filter REGEX] [--samples-time SECONDS]
"""

function revision_project(libs)
    project = TOML.parsefile(PROJECT_FILE)
    files = Dict(pkg => joinpath(libs, pkg * ".jl", "Project.toml") for pkg in PACKAGES)
    package_projects = Dict(pkg => TOML.parsefile(file) for (pkg, file) in files if isfile(file))

    # Old revisions may predate a package. Omit it and any packages requiring it
    # so the remaining suites can still use Pkg's normal resolver.
    while true
        unavailable = setdiff(PACKAGES, keys(package_projects))
        blocked = [pkg for (pkg, p) in package_projects
                   if any(name -> haskey(get(p, "deps", Dict()), name), unavailable)]
        isempty(blocked) && break
        foreach(pkg -> delete!(package_projects, pkg), blocked)
    end
    for pkg in PACKAGES
        if haskey(package_projects, pkg)
            project["sources"][pkg]["path"] = dirname(files[pkg])
        else
            for table in ("deps", "sources", "compat")
                delete!(project[table], pkg)
            end
        end
    end
    return project
end

function main(args)
    args == ["--help"] && return print(USAGE)
    options = parse_options(args, ("--libs", "--out", "--packages", "--filter", "--samples-time"))
    all(k -> haskey(options, k), ("--libs", "--out")) || error(USAGE)
    libs = abspath(options["--libs"])
    isdir(libs) || error("No library directory: ", libs)
    out = abspath(options["--out"])
    packages = selected_packages(options)
    pattern = get(options, "--filter", "")
    Regex(pattern)
    seconds = sample_budget(options)
    mkpath(dirname(out))

    started = time_ns()
    mktempdir() do project_dir
        write_project(project_dir, revision_project(libs))
        instantiate(project_dir)
        worker = joinpath(BENCHMARK_DIR, "worker.jl")
        run(julia_worker(`--project=$project_dir $worker $libs $out $(join(packages, ",")) $pattern $seconds $started`))
    end
end

main(ARGS)
