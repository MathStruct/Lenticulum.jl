# AGENTS.md

Instructions for AI coding agents working in this repository. Humans: see
[CONTRIBUTING.md](CONTRIBUTING.md); everything here applies to you too.

## Read this first

The author moved fast and did not check everything. Much of the code and many of the notes were
written quickly, a lot of it with AI agents, and not all of it has been reviewed by a person.

- Treat a claim in a note as unverified unless a test or a measurement in the repository backs
  it. Check before you build on it.
- When you find something wrong (a wrong formula, a stale number, a note that overstates a
  result, a test that does not test what it says), fix it or say so plainly. Do not work around
  it silently.
- Do not add claims you have not checked. A result goes into a note with the test or the run
  that produced it; anything else is labelled as a conjecture or an expectation.

Development happens on the branch `dev-danielb`, by one person; `master` is what the
documentation site publishes, and `dev-danielb` is merged into it once the change has been
checked. The site is mathstruct.org/Lenticulum.jl: the vault at the root, the API docs under
`api/dev/`. A push to `dev-danielb` deploys a preview, the vault to `<site>/preview/` and the
API docs to `<site>/api/preview/` (`docs/make.jl` chooses from the branch; `docs/site/README.md`). Work on `dev-danielb` unless told otherwise,
and do not commit, push or merge unless you are asked to.

## The repository

| where | what |
|---|---|
| `src/` | the top-level package `Lenticulum` (linear-Gaussian and constraint factors) |
| `lib/*.jl/` | the other packages: `LenticulumCore`, `Mycelium` (graphs, messages, free energy), `VariationalDiffusion` (diffusion models as implicit learners), `ImplicitLayers`, `Adversarial`; each with its own `Project.toml`, `test/` and, for some, `examples/` and `ext/` |
| `vault/` | the Obsidian vault: concept, design and application notes |
| `lib/*/src/*.md`, `src/*.md` | implementation notes, one beside each source file |
| `docs/` | Documenter pages, Literate tutorials (`docs/literate/`), the shared bibliography `docs/src/refs.bib`, and the Quartz build of the vault (`docs/site/`) |
| `bench/` | benchmarks, each recorded in a note |
| `meta/` | the author's local working material; untracked, never commit it |

`vault/Start Here.md` gives the conventions; read it before writing a note.

## Working conventions

**Tests.** Each package is its own environment, and the packages find each other through
`[sources]` paths:

```sh
julia --project=lib/VariationalDiffusion.jl -e 'using Pkg; Pkg.test()'   # one package
julia --project=. -e 'using Pkg; Pkg.test()'                              # the top-level package
```

The VariationalDiffusion suite takes about ten minutes. To iterate on one testset, run the
file's header (everything before the outer `@testset`) plus that testset as a script with
`--project=lib/VariationalDiffusion.jl/test`. Tests compare against closed forms (Gaussian
mixtures, linear-Gaussian models) wherever one exists; keep it that way.

**Implementation notes.** Something new in `lib/*/src/x.jl` gets or extends `x.md` beside it:
the theory, the implementation, what it does not do, and the difficulties met on the way.
Numbers in a note come from a script in the repository (an example, a benchmark, a test).

**Notes in the vault.** A tag line, a summary in a blockquote, then `> Sources:` (papers with
exact theorem numbers, then `code:`) and `> Theory (CT-ML wiki):`. Display math has its `$$`
fences on their own lines (the build refuses truncating blocks); write `t_{\min}`, not `t_\min`,
for KaTeX. Wikilinks resolve by note name.

**Citations.** Add the work to `docs/src/refs.bib` first, with metadata checked against arXiv or
Crossref. Then:

```sh
python3 docs/site/link_sources.py   # adds the "Bibliography:" line under each note's sources
python3 docs/site/bib2vault.py      # vault/Bibliography.md
python3 docs/site/bib2cff.py        # the references of CITATION.cff
python3 docs/site/zenodo.py         # .zenodo.json, from CITATION.cff and contributors.toml
```

Known bug: `link_sources.py` also rewrites the Bibliography lines of a few unrelated notes
(`lib/VariationalDiffusion.jl/src/implicit.md`, `sampler.md`, *Backpropagation through Implicit
Inference*), dropping links that were there. Check `git diff` after running it and restore
those files.

**Docs.** `julia --project=docs docs/make.jl` builds the API docs and the vault (Node ≥ 22);
`docs/site/build.sh` builds the vault alone, and `docs/site/build.sh --serve` serves it at
`http://localhost:8080` with live reload, rendered exactly as the website renders it (Quartz,
KaTeX, the TikZ diagrams); Obsidian's rendering differs in places. Both builds must pass.

**Long runs and outputs.** Logs, trained models and other large outputs go to
`~/.cache/lenticulum/`, not into the repository. Do not delete other sessions' files there.

**Julia pitfalls met here.**
- On a GPU, a `Float64` constant in a Float32 computation widens it to Float64 (and cuDNN
  rejects it); cast constants to the element type of the input.
- Indexing a device array with an index vector inside AD scalar-indexes; use broadcasting.
- Device code is tested with JLArrays and `JLArrays.allowscalar(false)`.
- Since Julia 1.12 the main thread is in the `:interactive` pool; a busy task spawned there can
  starve it (`bench/jit_latency.jl`).

**Commits.** When asked to commit, write what changed and why; an agent's commit names it as a
co-author. The bibliography, `CITATION.cff` and `.zenodo.json` are generated: commit them
together with the `refs.bib` change that produced them.

## Credit

Contributors are credited as described in CONTRIBUTING.md. Do not add anyone to
`CITATION.cff` or `contributors.toml` without their consent.
