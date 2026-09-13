using Documenter

using Lenticulum
using LenticulumCore
using Mycelium
using VariationalDiffusion
using ImplicitLayers
using Adversarial

makedocs(;
    sitename = "Lenticulum.jl",
    authors = "Daniel Boigk",
    modules = [
        Lenticulum,
        LenticulumCore,
        Mycelium,
        VariationalDiffusion,
        ImplicitLayers,
        Adversarial,
    ],
    format = Documenter.HTML(;
        canonical = "https://MathStruct.github.io/Lenticulum.jl",
        edit_link = "master",
        assets = String[],
        # LenticulumCore and Mycelium have large interfaces, so their reference pages are
        # genuinely long. They are split into sections by source file rather than paginated.
        size_threshold_warn = 150 * 1024,
    ),
    pages = [
        "Home" => "index.md",
        "Getting started" => "getting-started.md",
        "Vocabulary" => "vocabulary.md",
        "Theory vault" => "theory.md",
        "Packages" => [
            "LenticulumCore" => "packages/lenticulumcore.md",
            "Mycelium" => "packages/mycelium.md",
            "Lenticulum" => "packages/lenticulum.md",
            "VariationalDiffusion" => "packages/variationaldiffusion.md",
            "ImplicitLayers" => "packages/implicitlayers.md",
            "Adversarial" => "packages/adversarial.md",
        ],
    ],
    # Docstrings cite the Obsidian vault with [[wiki links]] and reference notes that are not
    # part of this site. Those are deliberate: the theory lives in `markdown/`, not here.
    checkdocs = :exports,
    warnonly = [:missing_docs, :cross_references],
)

# --- The Obsidian vault, rendered with Quartz, deployed inside this site ---------------
# `docs/vault/build.sh` stages the vault and builds it into docs/build/vault/, which
# deploydocs then ships along with everything else, so it lands at <site>/dev/vault/.
# It needs Node ≥ 22; if `npx` is not on the path the API docs still build on their own.
let script = joinpath(@__DIR__, "vault", "build.sh")
    if Sys.which("npx") === nothing
        @warn "npx not found — skipping the theory vault (docs/vault/build.sh). The API docs are unaffected."
    else
        @info "Building the theory vault with Quartz"
        run(`$script`)
    end
end

deploydocs(;
    repo = "github.com/MathStruct/Lenticulum.jl",
    devbranch = "master",
)
