#overview

> How this vault is organised, the conventions every note follows, and how to read or extend
> it. The notes are indexed in [[Map of Content]]; the state of the code is in
> [[Refactor 2026-11]].

## The three paths

The library is built around three uses, and the vault is laid out by them.

| path | what it is | notes | code |
|---|---|---|---|
| **A. Gaussian factor graphs** | exact inference on linear-Gaussian graphs (smoothing, state estimation), with nonlinear measurements by linearisation | `vault/Gaussian Graphs/` | `LenticulumCore`, `Mycelium`, `Lenticulum` |
| **B. Relation learners** | a diffusion model as a relation: queried in any direction by root finding, differentiated by the implicit function theorem | `vault/Relation Learners/` | `VariationalDiffusion` |
| **C. Structure and regimes** | switching and regime factors, hierarchical models, expectation maximisation, search over structures | `vault/Structure and Regimes/` | `Lenticulum`, `Mycelium`, `MyceliumLearning` |

`vault/Foundations/` holds what all three share: what a factor is, channels and polarity,
energies. Everything else that was in the vault before November 2026 (the algebraic,
equilibrium and adversarial families, application sketches, some comparisons) is under
`vault/Parked/`: kept, linked, and not revised.

## What lives where

| where | what |
|---|---|
| `vault/Foundations/`, `Gaussian Graphs/`, `Relation Learners/`, `Structure and Regimes/` | concept notes: definitions, derivations, measured results |
| `vault/Design/` | design notes that cut across the paths |
| `vault/Project/` | project records: decisions (`L-ADR-nnnn`), problems (`LP-nnn`), specifications (`L-SPEC-nnn`), the [[Feature Backlog]], [[Refactor 2026-11]] |
| `vault/Research Questions/` | open mathematical questions, self-contained (`RQ-nnn`) |
| `vault/Parked/` | notes off the three paths |
| `lib/*/src/*.md` | implementation notes, one beside each Julia file |
| `parked/*/src/*.md` | implementation notes of the parked packages |

The general category theory these notes build on (Para, lenses, Markov categories, Bayesian
lenses, statistical games, free energy) is in the
[CT-ML wiki](https://mathstruct.org/CategoryTheory-ML-Wiki/); *Track E* of its
[Start Here](https://mathstruct.org/CategoryTheory-ML-Wiki/Start-Here) leads up to every
categorical concept used in the code. This vault keeps what is specific to Lenticulum.

## Conventions

Every note opens the same way:

1. **a tag line**: one or more of `#definition`, `#theorem`, `#derivation`, `#algorithm`,
   `#model`, `#design`, `#implementation`, `#comparison`, `#application`, `#open-problem`,
   `#overview`, `#annotation`, `#example`, `#reference`;
2. **a summary** in a blockquote: the claim of the note in two or three lines;
3. **a sources block**:
   - `> Sources:` the papers with exact definition and theorem numbers, then `code:` the Julia
     files the note describes and `tests:` the tests that check its claims. A note that is our
     own analysis says so ("original to this vault").
   - `> Theory (CT-ML wiki):` links to the general concepts the note uses.

   Every cited work is in [[Bibliography]], generated from `docs/src/refs.bib`; new citations go
   into that file first. `docs/site/link_sources.py` then adds a `> Bibliography:` line under
   each note's sources.

**Claims and evidence.** A result stated in a note names the test or the script that produced
it, in the sentence or in the sources block. A statement without one is labelled a conjecture
or an expectation. A note says what its subject does **not** do as carefully as what it does.

**Tone.** Notes state what is the case, in the present tense. The history of how a result was
reached belongs in the problem record that closed it, not in the concept note.

Mathematics is written in LaTeX (`$…$`, and `$$` fences **on their own lines**); the website
renders it with KaTeX, and the build refuses notes whose display math would be truncated
(`docs/site/README.md`). Commutative diagrams are ```` ```tikz ```` blocks.

**Notation.** A factor's joint space is $Z = X \times Y \times U$: inputs $X$ (clamped), outputs
$Y$ (solved for), latents $U$; $z_0$ is the evidence and $\rho$ the per-coordinate precision.
This is the machine-learning convention, and it is the **reverse** of AutoBayes', where $X$ is
unobserved and $Y$ observed. Notes quoting a paper use its letters and say so once; everything
else follows the table in [[Channels and Polarity]] §"Notation".

Links between notes are `[[wikilinks]]` by note name, so a note can move between folders
without breaking them; links to general concepts go to the CT-ML wiki's published pages.

## Reading it

- **On the web**: the vault is rendered with Quartz as the front page of
  [mathstruct.org/Lenticulum.jl](https://mathstruct.org/Lenticulum.jl/); the API documentation
  is under [api/dev](https://mathstruct.org/Lenticulum.jl/api/dev/).
- **In Obsidian**: open the repository root as a vault. The *Inline TikZ* community plugin
  renders the diagrams.

## Adding to it

Something new in `lib/*/src/x.jl` gets or extends `x.md` beside it: the theory, the
implementation, what it does not do, and the difficulties met. A new general concept goes into
the CT-ML wiki first, and the Lenticulum note links to it.

A design choice that others will build on gets a decision record in `vault/Project/Decisions/`;
something that does not work and needs a decision gets a problem in `vault/Project/Problems/`;
an interface contract gets a specification in `vault/Project/Specifications/`. Each starts from
its template in `vault/templates/` (not published) and gets a line in its index: [[Decisions]],
[[Problem Log]], [[Specifications]]. A decision or specification is `proposed` or `draft` until
Daniel accepts it. A feature that should be implemented gets a row in [[Feature Backlog]].

An open question that is mathematics rather than engineering goes into
`vault/Research Questions/` as an `RQ-nnn` note.

Bringing a parked note or package back: [[Refactor 2026-11]] §"Moving things back".

How contributions are credited is in
[CONTRIBUTING.md](https://github.com/MathStruct/Lenticulum.jl/blob/master/CONTRIBUTING.md).
