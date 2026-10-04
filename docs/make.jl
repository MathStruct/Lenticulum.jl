using Documenter
using Literate
using SHA: sha1

# --- Tutorials: docs/literate/*.jl → a page and a Jupyter notebook each --------------------
# Each tutorial is executed once by Literate (outputs and plots written into plain Markdown),
# and the result is cached under docs/cache/tutorials, keyed by a hash of the script, the docs
# environment and the source of the Lenticulum packages it loads. An unchanged tutorial is
# copied from the cache instead of re-run, so the slow ones (they train networks) only run
# when they or the code they exercise change. CI keeps docs/cache between runs
# (.github/workflows/Docs.yml). Set LENTICULUM_RERUN_TUTORIALS=1 to ignore the cache.
# The notebook is written unexecuted, for download, and linked from the top of the page.
const ROOT = dirname(@__DIR__)
const LITERATE = joinpath(@__DIR__, "literate")
const TUTORIALS = joinpath(@__DIR__, "src", "tutorials")
const CACHE = joinpath(@__DIR__, "cache", "tutorials")
const PKGDIRS = Dict("Lenticulum" => ROOT, "LenticulumCore" => joinpath(ROOT, "lib", "LenticulumCore.jl"),
                     "Mycelium" => joinpath(ROOT, "lib", "Mycelium.jl"),
                     "VariationalDiffusion" => joinpath(ROOT, "lib", "VariationalDiffusion.jl"),
                     "ImplicitLayers" => joinpath(ROOT, "lib", "ImplicitLayers.jl"),
                     "Adversarial" => joinpath(ROOT, "lib", "Adversarial.jl"))

function sources(dir)
    files = String[]
    for sub in ("src", "ext")
        isdir(joinpath(dir, sub)) || continue
        for (root, _, fs) in walkdir(joinpath(dir, sub)), f in fs
            endswith(f, ".jl") && push!(files, joinpath(root, f))
        end
    end
    return sort(files)
end

# script + docs environment + every Lenticulum package the script loads (and the core two)
function tutorial_key(script)
    src = read(script, String)
    used = Set(["LenticulumCore", "Mycelium"])
    for m in eachmatch(r"^(?:using|import)\s+([^\n]+)"m, src), w in eachmatch(r"\w+", m[1])
        haskey(PKGDIRS, w.match) && push!(used, w.match)
    end
    io = IOBuffer()
    write(io, src, read(joinpath(@__DIR__, "Project.toml")))
    for pkg in sort(collect(used)), f in sources(PKGDIRS[pkg])
        write(io, relpath(f, ROOT), read(f))
    end
    return bytes2hex(sha1(take!(io)))[1:16]
end

# Literate's plain-Markdown flavour strips Documenter's (@ref) and (@id …) links; protect them
protect(s) = replace(s, "(@" => "(LITERATE-AT-")
restore(s) = replace(s, "(LITERATE-AT-" => "(@")

rm(TUTORIALS; force = true, recursive = true)
mkpath(TUTORIALS)
tutorial_pages = String[]
for file in sort(filter(endswith(".jl"), readdir(LITERATE)))
    name = splitext(file)[1]
    script = joinpath(LITERATE, file)
    badge(content) = replace(content, r"^(# # .*\n)"m =>
        SubstitutionString("\\1#\n#md # [Download as a Jupyter notebook]($(name).ipynb)\n" *
            "#nb # *Setup:* run this in a Julia environment where Lenticulum's packages are\n" *
            "#nb # developed (see the README) and the other packages of the first code cell are added.\n"); count = 1)
    cached = joinpath(CACHE, "$(name)-$(tutorial_key(script))")
    if isdir(cached) && get(ENV, "LENTICULUM_RERUN_TUTORIALS", "0") != "1"
        @info "Tutorial $(name): reusing the cached rendering"
    else
        @info "Tutorial $(name): executing"
        tmp = mktempdir()
        Literate.markdown(script, tmp; flavor = Literate.CommonMarkFlavor(), execute = true,
                          preprocess = protect ∘ badge, postprocess = restore)
        for old in filter(d -> startswith(d, "$(name)-"), isdir(CACHE) ? readdir(CACHE) : String[])
            rm(joinpath(CACHE, old); recursive = true)               # drop stale renderings
        end
        mkpath(dirname(cached))
        mv(tmp, cached; force = true)
    end
    foreach(f -> cp(joinpath(cached, f), joinpath(TUTORIALS, f); force = true), readdir(cached))
    Literate.notebook(script, TUTORIALS; execute = false)
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
