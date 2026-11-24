using TOML

function check_tooling(out)
    path = @__DIR__
    report = joinpath(out, "report.md")
    noise = joinpath(out, "noise.toml")
    compare = joinpath(path, "compare.jl")
    cmd = `$(Base.julia_cmd()) --project=$path --threads=1 $compare --aa HEAD --packages ImplicitSolvers --report $report --noise $noise`
    run(addenv(cmd, "LENTICULUM_SMOKE" => "true"))

    @assert occursin("## A/A warm timing noise", read(report, String))
    @assert haskey(TOML.parsefile(noise)["noise"], "ImplicitSolvers/linear/dense")
    @assert filesize(joinpath(out, "base.json")) > 0
    @assert filesize(joinpath(out, "head.json")) > 0
    println("Benchmark tooling smoke passed")
end

function check_warmup()
    path = @__DIR__
    run(`$(Base.julia_cmd()) --project=$path -e 'using Pkg; Pkg.instantiate()'`)
    run(`$(Base.julia_cmd()) --project=$path --threads=1 $(joinpath(path, "warmup_check.jl"))`)
end

check_warmup()
if isempty(ARGS)
    mktempdir(check_tooling)
else
    check_tooling(abspath(only(ARGS)))
end
