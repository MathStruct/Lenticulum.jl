# AGENTS.md

Instructions for AI coding agents working in this repository. Humans: see
[CONTRIBUTING.md](CONTRIBUTING.md); everything here applies to you too.

## Read this first

`vault/Project/Status.md` says where the work stands: the branch, what is next and with which
facts, what is open. Read it after this file; update it before a session ends.

The author moved fast and did not check everything. Much of the code and many of the notes were
written quickly, a lot of it with AI agents, and not all of it has been reviewed by a person.

- Treat a claim in a note as unverified unless a test or a measurement in the repository backs
  it. Check before you build on it.
- When you find something wrong (a wrong formula, a stale number, a note that overstates a
  result, a test that does not test what it says), fix it or say so plainly. Do not work around
  it silently.
- Do not add claims you have not checked. A result goes into a note with the test or the run
  that produced it; anything else is labelled as a conjecture or an expectation.

Branches: `master` is what the site publishes (mathstruct.org/Lenticulum.jl: the vault at the
root, the API docs under `api/dev/`); `dev` is the shared branch where others' work comes
together, by pull request; `dev-danielb` is the maintainer's own branch, and a
push to it deploys a preview of the site, the vault to `<site>/preview/` and the API docs to
`<site>/api/preview/` (`docs/make.jl` chooses from the branch; `docs/site/README.md`).
Working for the maintainer, use `dev-danielb`; working for anyone else, use a branch of your
own and target `dev`. The restructured repository (`refactor`, November 2026) is merged into
`dev-danielb`; `master` follows when the maintainer merges it. An agent working in the
maintainer's checkout commits its work to `dev-danielb` at the end of each task, without being
asked, once the task's gates pass (see Commits); it does not push or merge unless asked.

**Helper agents and worktrees.** An agent that implements a work package for the maintainer
(another model, a subagent) works in its own worktree beside this one,
`../Lenticulum.jl-<agent>`, on a branch `<agent>/<topic>` made from `dev-danielb`
(`git worktree add ../Lenticulum.jl-mistral -b mistral/<topic> dev-danielb`). Codex (GPT)
implements in `../Lenticulum.jl-codex`, on branches `codex/<topic>`
(`git worktree add ../Lenticulum.jl-codex -b codex/<topic> dev-danielb`). The Manifests are
not tracked, so instantiate each environment there before testing. It does not commit unless the
brief says so, and never pushes; the maintainer's session reads the branch and merges it. A
reviewer that runs tests uses the same worktree (`../Lenticulum.jl-mistral`, branch
`mistral/review`): the change under review is applied there, and scratch tests that show a
finding are written there, never in the maintainer's checkout. Reviews never go into the
repository: a reviewer writes its answer to a folder next to it, `../Lenticulum.jl-Review-<date>/`
(one file for a small review), or the maintainer's scratch directory, and the findings are then
triaged in the vault (see Reviews below).

## The repository

The library is built around three paths, and work on them comes first: **A** Gaussian factor
graphs (exact inference, state estimation), **B** relation learners (a diffusion model queried in
any direction), **C** structure and regimes (switching factors, EM, structure search).
`vault/Project/Refactor 2026-11.md` says what is in the core, what is verified, and what is
not yet reviewed.

| where | what |
|---|---|
| `lib/ImplicitSolvers.jl/` | shared numerical solvers: tolerance, linear systems, roots, implicit adjoints and reference differences |
| `lib/LenticulumCore.jl/` | what a factor is: channels, polarities, energies, the belief types |
| `lib/Mycelium.jl/` | inference: graphs, messages, schedules, plates, cluster nodes, the free energy |
| `lib/MyceliumLearning.jl/` | learning: free-energy gradients, couplings, the graph adjoint, EM, the optimiser step |
| `lib/Lenticulum.jl/` | the factors: linear-Gaussian, nonlinear, constraint, switching, regime, likelihood; M-steps; `examples/` |
| `lib/VariationalDiffusion.jl/` | diffusion models as implicit learners |
| `lib/ImplicitLayers.jl/`, `lib/AdversarialFactors.jl/`, `lib/AlgebraicFactors.jl/` | off the three paths and **not yet reviewed**: DEQs and neural ODEs, implicit generative models, toric models. Tested weekly, on demand and when their own files change (`.github/workflows/Weekly.yml`), not on every push |
| `vault/` | the Obsidian vault: `Foundations/`, one folder per path, `Design/`, `Project/`, `Research Questions/`, `Parked/` |
| `lib/*/src/*.md` | implementation notes, one beside each source file |
| `docs/` | Documenter pages, Literate tutorials (`docs/literate/`), the shared bibliography `docs/src/refs.bib`, and the Quartz build of the vault (`docs/site/`) |
| `bench/` | benchmarks, each recorded in a note; `julia --project=bench bench/<name>.jl` (its Manifest is not tracked: `julia --project=bench -e 'using Pkg; Pkg.instantiate()'` once). The examples of Lenticulum and VariationalDiffusion run in the same environment. `julia --project=bench bench/smoke.jl` runs every benchmark and example at small size (`LENTICULUM_SMOKE=true`; weekly in CI, `.github/workflows/Examples.yml`): a new benchmark or example must honour that switch |
| `tools/` | `julia --project=tools tools/coverage.jl --run [lib/<Pkg>.jl …]`: line coverage of each package's own suite (summary per file, uncovered ranges, `lcov.info`, to `~/.cache/lenticulum/coverage/`); weekly in CI (`Weekly.yml`, job `coverage`) |

Each package has its own `Project.toml` and `test/`. Dependencies go one way:
`ImplicitSolvers` has no project dependencies and is intended for `ImplicitLayers`, `VariationalDiffusion` and `MyceliumLearning`.
`LenticulumCore` ← `Mycelium` ← `MyceliumLearning` ← `Lenticulum`; `VariationalDiffusion`,
`AdversarialFactors` and `AlgebraicFactors` depend on the first two, `ImplicitLayers` on the first three. `vault/Start Here.md` gives the conventions; read it before writing a note.

## Working conventions

**Tests.** Each package is its own environment, and the packages find each other through
`[sources]` paths:

```sh
export JULIA_NUM_THREADS=auto   # Pkg.test starts a new process; it takes threads from here, not from -t
julia -t auto --project=lib/Lenticulum.jl -e 'using Pkg; Pkg.test()'             # one package
julia -t auto --project=lib/VariationalDiffusion.jl -e 'using Pkg; Pkg.test()'
```

Run times (2026-11-10, one machine, thread count not recorded): LenticulumCore 1 min, Mycelium
1 min, MyceliumLearning 6 min, VariationalDiffusion 6 min. Lenticulum's suite has two tiers
(2026-11-21, 8 threads): the fast tier, the default, about 26–29 min; the slow tier, the tests that differentiate
through replays, plates and element types, about 22 min more (28 min alone, with its own compilation). `LENTICULUM_SLOW_TESTS=true` runs both,
`=only` the slow tier alone (`lib/Lenticulum.jl/test/runtests.jl` says which files and testsets are
slow; a new test that compiles for minutes goes there). CI runs the fast tier on every push and the slow
tier beside it in `Slow.yml`: weekly, on pull requests, and on pushes that touch the AD code (the
workflow's `paths`). Locally, run the slow tier too before a commit that touches those paths. The suites are
independent; run up to three at once, each with its output sent to a file, and read the tail of
a file only when its suite failed. While working, run the narrowest gate that covers the change:
to iterate on one testset, run the file's header (everything before the outer `@testset`) plus
that testset as a script. Run the suites of every package the change touches, and of the
packages that depend on it, once before a commit. Tests compare
against closed forms (Gaussian mixtures, linear-Gaussian models, enumeration) wherever one
exists; keep it that way.

**Static checks.** Every suite ends with `test/aqua.jl` (Aqua.jl: ambiguities, unbound type
parameters, undefined exports, stale dependencies, compat bounds, piracy, persistent tasks). An
exception goes into that file with its reason; a new dependency needs a `[compat]` entry.

**A new factor type, or a new message path, comes with an oracle test on random instances.**
The defects of the reviews so far sat where a feature was added without one
(`lib/Lenticulum.jl/test/gaussian_oracle.jl` is the model: random trees against the dense joint
solve). A known limit is recorded as `@test_broken`, not only in a note.

**Reviews.** The findings of a review, by a person or a model, are triaged in a problem note
under `vault/Project/Problems/` (LP-023 and LP-024 are the model): each finding, whether it was
right, what was done, and the test that shows it. A finding is a claim until a test reproduces
it; one that cannot be reproduced is recorded as such, not fixed. The test of a fix stays as its
regression test.

**Feature backlog.** `vault/Project/Feature Backlog/` is kept current without being asked:
`Feature Backlog.md` is the index (conventions, the dependency graph), each section has its own
note (`Backlog P The Three Paths.md`, … one per id letter), and done rows go to the month's
`Backlog Done <yyyy-mm>.md`. A feature asked for by a specification, a problem, a review or a
measurement gets a row in its section's note, and the row moves to the done note with the commit
that implements it. Read the index and the section you need, not the whole folder.

**Say what you did not check.** A commit message or a note that reports results ends with what
was not run or not read.

**Implementation notes.** Something new in `lib/*/src/x.jl` gets or extends `x.md` beside it:
the theory, the implementation, what it does not do, and the difficulties met on the way.
Numbers in a note come from a script in the repository (an example, a benchmark, a test).

**Notes in the vault.** A tag line, a summary in a blockquote, then `> Sources:` (papers with
exact theorem numbers, then `code:`) and `> Theory (CT-ML wiki):`. Display math has its `$$`
fences on their own lines (the build refuses truncating blocks); write `t_{\min}`, not `t_\min`,
for KaTeX. Wikilinks resolve by note name.

**Checking notes.** `python3 docs/site/check_notes.py` lists wikilinks without a target, file paths in
notes that do not exist, and source files without a note. Its wikilink hits on escaped pipes
(`[[Name\|alias]]` in tables), `[[wikilinks]]` and the templates are false.

**Citations.** Add the work to `docs/src/refs.bib` first, with metadata checked against arXiv or
Crossref. Then:

```sh
python3 docs/site/link_sources.py   # adds the "Bibliography:" line under each note's sources
python3 docs/site/bib2vault.py      # vault/Bibliography.md
python3 docs/site/bib2cff.py        # the references of CITATION.cff
python3 docs/site/zenodo.py         # .zenodo.json, from CITATION.cff and contributors.toml
```

`link_sources.py` keeps every existing `> Bibliography:` link whose key is still in `refs.bib`
(LP-004). So when you remove a citation from a note's sources, delete its link from that note's
Bibliography line by hand. Check `git diff` after running the scripts as usual.

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
- Device code is tested with JLArrays and `JLArrays.allowscalar(false)`. The device tests of
  Mycelium, MyceliumLearning and ImplicitLayers run only with `LENTICULUM_DEVICE_TESTS=true`;
  run them before committing device-facing changes. On a machine with an NVIDIA GPU, also run
  them on CUDA: `julia --project=gpu gpu/runtests.jl [Package ...]` (instantiate `gpu/` once; its
  Manifest is not tracked). JLArrays accepts a broadcast that mixes host and device arrays and CUDA
  does not (LP-093). These run locally only; no CI job has a GPU.
- Since Julia 1.12 the main thread is in the `:interactive` pool; a busy task spawned there can
  starve it (`bench/jit_latency.jl`).

**Commits.** A task ends with a commit: the suites of the packages it touched (and of those
that depend on them) pass, the review is triaged, `Status.md` and the backlog are updated, and
then the change is committed, together with those notes. A task whose gates fail is not committed;
say so instead. Helper agents in their own worktrees still commit only when their brief says so.
Write what changed and why; an agent's commit names it as a co-author. The bibliography,
`CITATION.cff` and `.zenodo.json` are generated: commit them together with the `refs.bib` change
that produced them.

## Credit

Contributors are credited as described in CONTRIBUTING.md. Do not add anyone to
`CITATION.cff` or `contributors.toml` without their consent.

<!-- graphicle:start (added by graphicle setup; graphicle disconnect removes it) -->
## Graphicle
Graphicle indexes the code, the notes and their history into a graph that answers questions with
file and line citations. It is optional: `.mcp.json` registers its MCP server, which starts only where
`graphicle` is installed; `graphicle setup` in a checkout indexes it and connects the agents
(`graphicle status` shows the index, `graphicle sync` updates it).
Use the `graphicle` MCP server for where or how something works, for questions across files or history, and before changing unfamiliar code: `recall_project` or `query_graph` with the question, then `get_node` and `get_neighbors` for one symbol; answers cite file and lines. For an exact identifier you already know, grep is enough.
Without the server, run `graphicle context "<question>"` in a terminal. A helper's brief can start from
its output instead of hand-listed starting points.
<!-- graphicle:end -->
