using TOML

const BENCHMARK_DIR = @__DIR__
const REPO_ROOT = dirname(BENCHMARK_DIR)
const PROJECT_FILE = joinpath(BENCHMARK_DIR, "Project.toml")
const PACKAGES = ("ImplicitSolvers", "LenticulumCore", "Mycelium", "MyceliumLearning",
                  "Lenticulum", "VariationalDiffusion")

smoke() = get(ENV, "LENTICULUM_SMOKE", "false") == "true"

function parse_options(args, names; switches=())
    options = Dict{String,String}()
    i = 1
    while i <= length(args)
        name = args[i]
        haskey(options, name) && error("Option given twice: ", name)
        if name in switches
            options[name] = ""
        elseif name in names
            i < length(args) && !startswith(args[i + 1], "--") || error("Missing value for ", name)
            i += 1
            options[name] = args[i]
        else
            error("Unknown option: ", name)
        end
        i += 1
    end
    return options
end

function selected_packages(options)
    packages = split(get(options, "--packages", join(PACKAGES, ",")), ',')
    all(in(PACKAGES), packages) || error("Unknown package in --packages. Available: ", join(PACKAGES, ", "))
    allunique(packages) || error("--packages contains a duplicate")
    return packages
end

function sample_budget(options)
    seconds = parse(Float64, get(options, "--samples-time", "0.6"))
    isfinite(seconds) && seconds > 0 || error("--samples-time must be finite and positive")
    return smoke() ? min(seconds, 0.04) : seconds
end

function julia_worker(args)
    threads = string(Threads.nthreads(:default), ',', Threads.nthreads(:interactive))
    cmd = `$(Base.julia_cmd()) --startup-file=no --threads=$threads $args`
    # A fixed BLAS pool keeps the CPU workloads and package-load environment comparable.
    return addenv(cmd, "OPENBLAS_NUM_THREADS" => "1", "JULIA_LOAD_PATH" => "@:@stdlib")
end

function write_project(dir, project)
    open(io -> TOML.print(io, project; sorted=true), joinpath(dir, "Project.toml"), "w")
end

function instantiate(dir)
    run(julia_worker(`--project=$dir -e 'using Pkg; Pkg.instantiate(; update_registry=false)'`))
end

function load_suite(pkg)
    suite = Module(gensym(Symbol(pkg, "Benchmarks")))
    Core.eval(suite, :(include(path) = Base.include($suite, path)))
    path = joinpath(REPO_ROOT, "lib", pkg * ".jl", "benchmark", "benchmarks.jl")
    Base.include(suite, path)
    return suite
end
