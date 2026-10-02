#overview

> How this vault is organised, the conventions every note follows, and how to read or extend
> it. The notes themselves are indexed, in reading order, in [[Map of Content]].

## What lives where

The vault is the whole repository. Its notes sit in three places:

| where | what | example |
|---|---|---|
| `vault/` | **concept and design notes**, grouped by topic | `vault/Factor Graphs/Messages are Inversions.md` |
| `lib/*/src/*.md` | **implementation notes**, one beside each Julia file | `lib/Mycelium.jl/src/messages.md` beside `messages.jl` |
| `src/*.md` | implementation notes for the top-level package | `src/constraint.md` |
| `meta/` | working material: authoring prompts and the PhD proposals (not published) | |

The general category theory these notes build on (Para, lenses, Markov categories, Bayesian
lenses, statistical games, free energy) is **not** repeated here. It lives in the
[CT-ML wiki](https://mathstruct.org/CategoryTheory-ML-Wiki/), which is the root vault this one
extends; *Track E* of its
[Start Here](https://mathstruct.org/CategoryTheory-ML-Wiki/Start-Here) is a reading track that
leads up to every categorical concept used in the code. This vault keeps only what is specific to
Lenticulum: how the theory becomes Julia, which design decisions are ours, and what does not
work yet.

## Conventions

Every note opens the same way:

1. **a tag line** — one or more of `#definition`, `#theorem`, `#derivation`, `#algorithm`,
   `#model`, `#design`, `#implementation`, `#comparison`, `#application`, `#open-problem`,
   `#overview`, `#annotation`, `#example`, `#reference`;
2. **a summary** in a blockquote — the claim of the note in two or three lines;
3. **a sources block**:
   - `> Sources:` the papers with exact definition and theorem numbers, then `code:` the
     Julia files the note describes. A note that is our own analysis says so ("original to
     this vault").
   - `> Theory (CT-ML wiki):` links to the general concepts the note uses.

Implementation notes additionally record, as carefully as what the file does, **what it does
not do** and the difficulties met on the way.

Mathematics is written in LaTeX (`$…$`, and `$$` fences **on their own lines**); the website
renders it with KaTeX, and the build refuses notes whose display math would be truncated
(see `docs/site/README.md`). Commutative diagrams are ```` ```tikz ```` blocks:

```tikz
\usepackage{tikz-cd}
\begin{document}
\begin{tikzcd}
X \arrow[r, "f"] \arrow[dr, "f"'] \arrow[d, "\mathrm{id}_X"'] & Y \arrow[d, "\mathrm{id}_Y"] \\
X \arrow[r, "f"'] & Y
\end{tikzcd}
\end{document}
```

**Notation.** A factor's joint space is $Z = X \times Y \times U$: inputs $X$ (clamped), outputs
$Y$ (solved for), latents $U$; $z_0$ is the evidence and $\rho$ the per-coordinate precision.
This is the machine-learning convention, and it is the **reverse** of AutoBayes', where $X$ is
unobserved and $Y$ observed. Notes quoting a paper use its letters and say so once; everything
else follows the table in [[Channels and Polarity]] §"Notation".

Links between notes are `[[wikilinks]]` by note name; links to general concepts go to the
CT-ML wiki's published pages.

## Reading it

- **On the web**: the vault is rendered with Quartz and deployed beside the API documentation.
- **In Obsidian**: open the repository root as a vault. The *Inline TikZ* community plugin
  renders the diagrams.

## Adding to it

If you implement something, put a markdown file next to the `.jl` file containing the
implementation, with the theory, the implementation, and an extensive description of the
implementation difficulties. A new general concept goes into the CT-ML wiki first, and the
Lenticulum note links to it.
