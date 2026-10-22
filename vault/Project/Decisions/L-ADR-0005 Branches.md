#design

> **Status: accepted (recorded 2026-10-22 from AGENTS.md, where it was already the rule).** Related: `AGENTS.md`, `CONTRIBUTING.md`, `docs/make.jl`.

## Context

The website publishes one branch, contributors need a place for pull requests, and the maintainer wants to see changes rendered before they are published.

## Decision

- `master` is what the site publishes: the vault at the root, the API docs under `api/dev/`.
- `dev` is the shared branch where others' work comes together, by pull request.
- `dev-danielb` is the maintainer's own branch. A push to it deploys a preview: the vault to `preview/`, the API docs to `api/preview/`. `docs/make.jl` chooses from the branch.
- Work for the maintainer goes to `dev-danielb`; work for anyone else goes to a branch of its own, targeting `dev`. Nothing is merged to `master` unless the maintainer asks.

## Consequences

- A preview build has no comment threads (`VAULT_COMMENTS=0`), so discussions stay on the published pages.
- An issue closed by a commit on `dev-danielb` closes only when the commit reaches `master`.
