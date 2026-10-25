using Documenter
using DocumenterCitations
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
                     "Adversarial" => joinpath(ROOT, "lib", "Adversarial.jl"),
                     "AlgebraicFactors" => joinpath(ROOT, "lib", "AlgebraicFactors.jl"))

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

# notebooks have no bibliography: [key](@cite) becomes "(key)", and the @bibliography block goes
uncite(s) = replace(replace(s, r"\[([^\]]+)\]\(@cite[tp]?\)" => s"(\1)"),
                    r"^# ```@bibliography\n(?:# .*\n)*?# ```\n"m => "# See the references page of the documentation.\n")

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
    Literate.notebook(script, TUTORIALS; execute = false, preprocess = uncite)
    push!(tutorial_pages, "tutorials/$(name).md")
end

using Lenticulum
using LenticulumCore
using Mycelium
using VariationalDiffusion
using ImplicitLayers
using Adversarial
using AlgebraicFactors

# Where this build lives (see the vault section at the end): master's site, or the preview of
# the development branch. The pages link to the vault through LENTICULUM_VAULT_URL.
const SITE = "https://mathstruct.org/Lenticulum.jl"
const PREVIEW_BRANCH = "dev-danielb"
const BRANCH = get(ENV, "GITHUB_REF_NAME", "")
const DEVURL = BRANCH == PREVIEW_BRANCH ? "preview" : "dev"
const VAULT_URL = DEVURL == "dev" ? "$(SITE)/" : "$(SITE)/preview/"
const API_URL = "$(SITE)/api/$(DEVURL)/"
ENV["LENTICULUM_VAULT_URL"] = VAULT_URL

# One bibliography for the docs and the vault (docs/site/bib2vault.py renders it there)
const BIB = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style = :authoryear)

makedocs(;
    plugins = [BIB],
    sitename = "Lenticulum.jl",
    authors = "Daniel Boigk",
    modules = [
        Lenticulum,
        LenticulumCore,
        Mycelium,
        VariationalDiffusion,
        ImplicitLayers,
        Adversarial,
        AlgebraicFactors,
    ],
    format = Documenter.HTML(;
        canonical = "https://mathstruct.org/Lenticulum.jl/api/dev",
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
        "References" => "references.md",
        "Packages" => [
            "LenticulumCore" => "packages/lenticulumcore.md",
            "Mycelium" => "packages/mycelium.md",
            "Lenticulum" => "packages/lenticulum.md",
            "VariationalDiffusion" => [
                "Overview" => "packages/variationaldiffusion/index.md",
                "Predictors and backends" => "packages/variationaldiffusion/predictors.md",
                "Implicit inference" => "packages/variationaldiffusion/implicit.md",
                "Factors" => "packages/variationaldiffusion/factors.md",
                "ProxDM" => "packages/variationaldiffusion/proxdm.md",
            ],
            "ImplicitLayers" => "packages/implicitlayers.md",
            "Adversarial" => "packages/adversarial.md",
            "AlgebraicFactors" => "packages/algebraicfactors.md",
        ],
    ],
    # Docstrings cite the Obsidian vault with [[wiki links]] and reference notes that are not
    # part of this site. Those are deliberate: the theory lives in `vault/`, not here.
    checkdocs = :exports,
    warnonly = [:missing_docs, :cross_references],
)

# --- The Obsidian vault, rendered with Quartz, at the front of the site -------------------
# The site is mathstruct.org/Lenticulum.jl: the vault is its front page, and this API
# documentation lives under api/ (deploydocs' `dirname`), Documenter's versions inside it.
# `master` deploys the API docs to api/dev/ and the vault to the root; the development branch
# deploys a preview, api/preview/ and preview/, so a change can be checked on the real site
# (Quartz renders some things differently from Obsidian) before it is merged. The preview has no
# comment threads, so it does not open discussions that duplicate the real ones.
#
# `docs/site/build.sh` builds the vault into docs/vault-build/, outside docs/build/, so that
# deploydocs does not ship a second copy of it; the Docs workflow deploys it
# (docs/site/deploy_vault.sh). It needs Node ≥ 22; without `npx` the API docs build alone.
let script = joinpath(@__DIR__, "site", "build.sh")
    if Sys.which("npx") === nothing
        @warn "npx not found — skipping the theory vault (docs/site/build.sh). The API docs are unaffected."
    else
        @info "Building the theory vault with Quartz"
        withenv("VAULT_BASEURL" => replace(VAULT_URL, r"^https://" => "", r"/$" => ""),
                "VAULT_API_URL" => API_URL,
                "VAULT_COMMENTS" => DEVURL == "dev" ? "1" : "0") do
            run(`$script`)
        end
    end
end

deploydocs(;
    repo = "github.com/MathStruct/Lenticulum.jl",
    dirname = "api",
    devbranch = BRANCH == PREVIEW_BRANCH ? PREVIEW_BRANCH : "master",
    devurl = DEVURL,
    # both, whichever branch deploys: api/versions.js lists them
    versions = ["stable" => "v^", "v#.#", "dev" => "dev", "preview" => "preview"],
)
