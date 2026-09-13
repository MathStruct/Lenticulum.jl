# The vault as a website

The Obsidian vault is rendered to a static site with [Quartz v4](https://quartz.jzhao.xyz)
and deployed **inside** the Documenter site, at `<site>/dev/vault/`. The two link to each
other: Documenter's "Theory vault" page points here, and this site's footer points back.

## Building locally

```sh
docs/vault/build.sh            # builds into docs/build/vault/
docs/vault/build.sh --serve    # live preview at http://localhost:8080
```

Needs Node ≥ 22. The first run clones Quartz into `docs/quartz/` (gitignored) and runs
`npm ci`. `docs/make.jl` calls this script automatically when `npx` is on the path, so a
normal Documenter build produces both sites.

## How it is put together

| file | is |
|---|---|
| `quartz.config.ts` | our Quartz configuration, copied over the stock one |
| `quartz.layout.ts` | our layout — stock, plus a footer link back to the API docs |
| `tikz.ts` | a Quartz transformer rendering ```` ```tikz ```` blocks to SVG at build time |
| `.tikz-cache/` | the rendered SVGs, keyed by content hash — **committed** |
| `build.sh` | clones Quartz, injects the above, stages the vault, lints it, builds |

The vault is the whole repository, so the notes live in three places — `markdown/`, the
`*.md` beside each `lib/*/src/*.jl`, and `src/*.md`. Quartz wants one directory, so the script
copies them together. Wikilinks resolve by basename, exactly as in Obsidian, so the staged
directory structure only affects breadcrumbs.

## TikZ

`tikz.ts` is ported from
[MathStruct/CategoryTheory-ML-Wiki](https://github.com/MathStruct/CategoryTheory-ML-Wiki),
where it was written for the same Obsidian plugin format (`\usepackage` lines, then
`\begin{document} … \end{document}`). It renders each block with
[`node-tikzjax`](https://www.npmjs.com/package/node-tikzjax) and caches the SVG under its
content hash in `.tikz-cache/`. Because the cache is committed, a fresh clone — and CI — only
renders diagrams that are new or changed. A block that fails to compile is emitted as its
source in a `<pre class="tikz-error">` and the failure is printed at build time; it does not
break the build.

The stock Quartz checkout has neither the plugin nor the dependency; `build.sh` adds both.

## The one thing to know about `$$`

Obsidian accepts a display-math block written as

```
$$a = b
c = d$$
```

remark-math (what Quartz uses) does not: it reads the first line as an unclosed fence and
**silently drops everything after it from the page**. Thirty notes were being truncated this
way before it was caught. `build.sh` now refuses to build if the pattern is present. Put the
fences on their own lines:

```
$$
a = b
c = d
$$
```

## What does not render

- **`markdown/Prompts/`** is excluded — those are the authoring prompts, not vault content.
- `lib/*/README.md` are excluded to keep `[[README]]` unambiguous (it means the root one).
