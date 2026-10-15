# Contributing to Lenticulum.jl

Thank you for your interest in Lenticulum.jl. It is an open-source research project on implicit
learning: learning relations instead of functions, with diffusion, equilibrium and algebraic
models, probabilistic inference on factor graphs, energy-based models, and the category theory
that ties them together.

Contributions of every size are welcome, from a corrected typo to a new inference algorithm,
and so are contributions that are not code at all: a derivation, a reference, a review, or a
pointed question.

## The state of things, plainly

I moved fast. A lot of code and a lot of notes were written quickly, much of it with AI
assistance, and I have not checked everything. What is tested is tested against closed forms
and says so; a claim in a note that no test or measurement backs is a claim, not a result. If
you find something wrong, that is a contribution, and a welcome one: open an issue, comment on
the page, or fix it.

Development happens on the branch `dev-danielb`, by one person, and is merged into `master`,
which is what the documentation site publishes. Every push to `dev-danielb` deploys a preview of
the whole site to [mathstruct.org/Lenticulum.jl/preview](https://mathstruct.org/Lenticulum.jl/preview/),
because Quartz renders some things differently from Obsidian and a page is checked on the real
site before it is merged. Pull requests go to `master` and are welcome; once there are more
contributors, `master` will be protected and every change will go through one.

## Ways to contribute

### 1. Code and algorithms

- bug fixes and performance improvements;
- new inference algorithms, models, factors and belief representations;
- extensions and integrations with the Julia ecosystem;
- tests, benchmarks and reproducible examples;
- architectural improvements and refactoring.

For larger changes, please open an issue or a discussion before you start, so that the design
can be agreed on first. The open problems are listed in the vault, for example *Open Problems in
Implicit Diffusion Learning*; picking one up is a good way in.

### 2. Documentation, the vault and tutorials

The documentation is a knowledge base: an Obsidian vault in this repository (`vault/`, and an
implementation note beside each source file), rendered with Quartz beside the API documentation.
You can:

- correct errors or improve explanations;
- add derivations, proofs or references;
- write tutorials and worked examples (`docs/literate/`);
- propose alternative formulations, or point out where a note overstates what is known.

Contributions to the knowledge base count on their own; they do not need to come with code.

### 3. Scientific discussion and review

You do not need to write code or prose to contribute. Every page of the vault has a comment
thread (Giscus, stored in the repository's GitHub Discussions). Questions, corrections,
references, criticism and suggestions for research directions are all welcome there.

**Reviews are contributions.** Checking a derivation, testing a claim, or reading a note
critically is work, and it is credited (see below). Insights from discussions that change a
note or the code are incorporated with attribution.

## Credit and authorship

Community recognition and academic authorship are different things, and both are handled
explicitly.

**Every contribution is recorded.** Code and documentation contributions are visible in the
git history. Contributions that leave no commit (a review, a derivation in a discussion, a
reference that changed a conclusion) are acknowledged in the note or file they shaped and in
the project's acknowledgements.

**Roles are named with CRediT.** Where it matters, contributions are described with the
[CRediT taxonomy](https://credit.niso.org/) of contributor roles (for example *Software*,
*Formal analysis*, *Validation*, *Methodology*, *Writing – review & editing*), the vocabulary
journals use. It says what someone did, rather than ranking how much.

**Zenodo contributors.** Each release is archived on Zenodo with a DOI. People who made
substantial technical, scientific or documentation contributions, reviews included, may be
listed in a release's metadata as contributors, with their role. They are recorded in
`contributors.toml` (name, ORCID, Zenodo contributor type, CRediT roles, and a sentence on the
contribution), from which `.zenodo.json` is generated.

**Zenodo creators (authors of a release).** People who made substantial intellectual
contributions to the methodology, design or development of Lenticulum.jl may be invited to
become creators of a Zenodo release, that is, its authors. This follows the spirit of the usual
criteria for authorship: a substantial contribution to the work, and a share in the
responsibility for what is published.

**Papers are separate.** Authorship of a paper built on Lenticulum.jl is decided for that
paper, with the people who contributed to it. Contributing to the software neither implies nor
excludes it.

Principles:

- Authorship is discussed explicitly, never assigned by counting commits, pull requests or
  lines of code.
- Nobody is listed as a contributor or creator without their consent, and declining has no
  effect on anyone's ability to contribute.
- Contributors keep credit for their individual work, and are welcome to say how they want to
  be acknowledged (name, ORCID, affiliation, or not at all).
- Attribution decisions are made transparently; if you think yours is wrong, say so.

## Papers that build on Lenticulum.jl

If you write a paper that uses or extends Lenticulum.jl, I would be glad if you

- **cite the version you used**, by its Zenodo version DOI (GitHub's "Cite this repository"
  gives the citation; the concept DOI always resolves to the latest version);
- **add a note to the vault** describing the work: what you did, what it shows, what is open,
  with the paper in its sources. It is how the next reader finds it;
- **contribute the code back** where that is reasonable: a method that others can use belongs
  in a package, with its tests and its implementation note, and an experiment's scripts can sit
  beside the note.

The paper then goes into `docs/src/refs.bib` with `relation = {isCitedBy}`, and the next
release's Zenodo record links to it.

**Citing each other is fine.** A paper that cites the software it used, and software whose
documentation cites the papers built on it, are the normal relationship between the two, not
citation manipulation, which is coordinated citing to inflate numbers. Version DOIs keep it
honest in time as well: the paper cites the release it used, which existed before the paper,
and a later release cites the paper.

## Making a contribution

1. **Read** the README and [Start Here](vault/Start%20Here.md), which describes how the
   repository and the vault are organised and the conventions every note follows.
2. **Open an issue or join a discussion**, or go straight to a pull request for small changes.
3. **Code**: each package under `lib/` has its own tests,
   `julia --project=lib/<Package>.jl -e 'using Pkg; Pkg.test()'` (the top-level package: `--project=.`).
   Something new comes with a test against a closed form or a reference where one exists, and
   with an implementation note beside the source file: what it does, what it does not do, and
   the difficulties met on the way.
4. **Notes**: follow the conventions of Start Here (tag line, summary, sources). A new citation
   goes into `docs/src/refs.bib` first; `docs/site/link_sources.py` links it from the note.
   Build the documentation and the vault with `julia --project=docs docs/make.jl` (the vault
   needs Node ≥ 22); the build refuses display math that would render truncated. To see the
   vault as the website will show it while you write, run `docs/site/build.sh --serve` and open
   `http://localhost:8080`.
5. **Claims**: a scientific contribution comes with what supports it, a derivation, a
   reference, a test or a measurement, and says where it stops holding.
6. **Review**: pull requests are reviewed in the open; taking part in reviewing others' work is
   itself a contribution.

Contributions are made under the project's [MIT licence](LICENSE).

**Tools.** Much of this repository was written with AI assistance, and commits say so. The same
is welcome from contributors, on one condition: you have checked what you submit, and you are
responsible for it as if you had written it yourself.

Whether you bring a mathematical idea, an implementation, a tutorial, a review or a
correction: thank you for helping build it.
