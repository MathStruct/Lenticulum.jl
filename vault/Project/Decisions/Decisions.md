#overview

> The decisions recorded for Lenticulum.jl, numbered `L-ADR-nnnn`; the `L-` keeps them apart from Moonkale's ADRs and Sophia's `S-ADR`s. A decision is `proposed` until Daniel accepts it. An accepted decision is not reopened without a new ADR that supersedes it. Use the ADR template (`vault/templates/ADR Template.md`) for a new one.

The log starts on 2026-10-22. Decisions taken before then and recorded retrospectively say so in their status line, and point to the notes where they were first written down. Longer design discussions stay in `Design/`; an ADR records the choice and links to them.

| # | Decision | Status |
|---|---|---|
| [[L-ADR-0001 Julia]] | Julia, not PyTorch, JAX, C++, Rust or Mojo | accepted (retrospective) |
| [[L-ADR-0002 Energies are vectors, scalarised late]] | Vector energies in energy spaces, composed by direct sum, scalarised late | accepted (retrospective) |
| [[L-ADR-0003 Pointwise energy and free energy are separate levels]] | Separate spaces and scalarisations for the pointwise and the free energy; σ and polarity as parameters | accepted 2026-10-22 |
| [[L-ADR-0004 The vault builds with the shared MathStruct Quartz]] | The vault is built with MathStruct/quartz, not a vendored copy | accepted 2026-10-21 |
| [[L-ADR-0005 Branches]] | `master` publishes, `dev` collects pull requests, `dev-danielb` is the maintainer's preview | accepted (retrospective) |
| [[L-ADR-0006 Separate likelihood_message from invert]] | `likelihood_message` returns the BP likelihood; `invert` returns the AutoBayes posterior | accepted 2026-10-24 |
| [[L-ADR-0007 Point-dependent scalarisations are localised before use]] | `localise(σ, f, x, a, y, ps, st)` turns a point-dependent scalarisation (Sampson) into an ordinary one before `scalarise` | accepted 2026-10-25 |
| [[L-ADR-0008 A Python wrapper, if any, lives outside the Julia library]] | A Python package, if ever, is separate (`Lenticulum.py`, juliacall), never constrains the Julia library, and waits for a stable interface and demand | proposed 2026-10-29 |
