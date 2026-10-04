using Documenter
using Literate

# --- Tutorials: docs/literate/*.jl → a Documenter page and a Jupyter notebook each ---------
# The page is executed by Documenter (@example blocks, so outputs and plots appear in the
# docs); the notebook is written unexecuted, for download, and linked from the top of the page.
const LITERATE = joinpath(@__DIR__, "literate")
const TUTORIALS = joinpath(@__DIR__, "src", "tutorials")
tutorial_pages = String[]
for file in sort(filter(endswith(".jl"), readdir(LITERATE)))
    name = splitext(file)[1]
    badge(content) = replace(content, r"^(# # .*\n)"m =>
        SubstitutionString("\\1#\n#md # [Download as a Jupyter notebook]($(name).ipynb)\n" *
            "#nb # *Setup:* run this in a Julia environment where Lenticulum's packages are\n" *
            "#nb # developed (see the README) and the other packages of the first code cell are added.\n"); count = 1)
    Literate.markdown(joinpath(LITERATE, file), TUTORIALS; documenter = true, preprocess = badge)
    Literate.notebook(joinpath(LITERATE, file), TUTORIALS; execute = false)
    push!(tutorial_pages, "tutorials/$(name).md")
end

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
        "Tutorials" => tutorial_pages,
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
    # part of this site. Those are deliberate: the theory lives in `vault/`, not here.
    checkdocs = :exports,
    warnonly = [:missing_docs, :cross_references],
)

# --- The Obsidian vault, rendered with Quartz, deployed inside this site ---------------
# `docs/site/build.sh` stages the vault and builds it into docs/build/vault/, which
# deploydocs then ships along with everything else, so it lands at <site>/dev/vault/.
# It needs Node ≥ 22; if `npx` is not on the path the API docs still build on their own.
let script = joinpath(@__DIR__, "site", "build.sh")
    if Sys.which("npx") === nothing
        @warn "npx not found — skipping the theory vault (docs/site/build.sh). The API docs are unaffected."
    else
        @info "Building the theory vault with Quartz"
        run(`$script`)
    end
end

deploydocs(;
    repo = "github.com/MathStruct/Lenticulum.jl",
    devbranch = "master",
)
