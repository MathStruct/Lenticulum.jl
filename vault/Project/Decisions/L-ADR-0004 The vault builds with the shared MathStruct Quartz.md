#design

> **Status: accepted (2026-10-21, Daniel; commit 5a850764).** Related: [MathStruct/quartz](https://github.com/MathStruct/quartz) (its `MATHSTRUCT.md`), `docs/site/build.sh`, `docs/site/README.md`.

## Context

Each MathStruct site carried its own copy of Quartz, and the copies had drifted: dark-mode TikZ, Markdown tabs and the wikilink fix each existed in only some of them. In this vault, `tabs` blocks rendered as raw source and about twenty wikilinks with code labels were broken.

## Options

1. Keep a vendored Quartz per site.
2. One shared engine (MathStruct/quartz), with each site keeping only its configuration.
3. One site for all projects.

## Decision

Option 2. `docs/site/build.sh` fetches MathStruct/quartz into `docs/quartz`, stages the vault and the implementation notes, and builds with `docs/site/quartz.config.ts` and `quartz.layout.ts`. `QUARTZ_DIR` points a build at a local checkout of the engine.

## Consequences

- A plugin or fix lands in MathStruct/quartz once and reaches this vault on its next build.
- The vault's rendering can change without a commit here; a build can be pinned with `QUARTZ_REF`.
- `docs/site/build.sh --serve` previews the vault exactly as the website renders it.
