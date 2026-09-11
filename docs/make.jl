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

deploydocs(;
    repo = "github.com/MathStruct/Lenticulum.jl",
    devbranch = "master",
)
