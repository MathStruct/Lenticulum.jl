using Documenter
using Lenticulum

makedocs(;
    sitename="Lenticulum.jl",
    modules=[Lenticulum],
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/DanielBoigk/Lenticulum.jl",
)
