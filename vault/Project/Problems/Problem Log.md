#overview #open-problem

> Problems found while specifying or building Lenticulum.jl, numbered `LP-nnn` (Moonkale uses `P-nnn`, Sophia `SP-nnn`). Each problem gets a note made from the problem template (`vault/templates/Problem Template.md`) and a line here. Closed problems stay listed, with what closed them.

Implementation notes also record what their code does not do; a problem gets an entry here when it needs a decision, crosses several files, or is waiting for someone to pick it up. GitHub issues are linked from the problem they belong to.

| # | Problem | Status |
|---|---|---|
| [[LP-001 The vault build uses GNU-only sed]] | `sed -i` in `docs/site/build.sh` fails on macOS (issue #9) | closed (`review-argon`, 2026-10-23) |
| [[LP-002 The Jacobian side is not implemented]] | Finite-difference `energy_jacobian` and `gauss_newton_metric` added in `LenticulumCore`; AD extensions for $D_\theta \mathbf F$ remain open | open (partially addressed on `review-argon`) |
| [[LP-003 SquaredNorm does not check its metric]] | An indefinite `M` makes the Jensen gap negative, silently | closed (`review-argon`, 2026-10-23) |
| [[LP-004 link_sources rewrites unrelated notes]] | The bibliography linker drops links in three unrelated notes | closed (`review-argon`, 2026-10-23) |
| [[LP-005 Learning the noise covariance]] | `Q` and `Σ` are fixed; learning them needs a positive-definite parametrisation and the full free energy | open, later |
| [[LP-006 Gaussian entropies are relative to Lebesgue measure]] | Categorical KL entropy of a continuous belief is infinite; ours are relative to Lebesgue | accepted, documented |
| [[LP-007 Missing implementation notes]] | `gaussian.md` and `beliefs.md` are cited but not written | closed (`review-argon`, 2026-10-23) |
| [[LP-008 No conformance test for the factor interface]] | The factor interface is tested factor by factor; issue #8 slipped through | closed (`review-argon`, 2026-10-23) |
| [[LP-009 Game-level free energy is declared but not implemented]] | `entropy` and `free_energy` exist only as declarations; graphs use `local_free_energy` | closed (`review-argon`, 2026-10-23) |
| [[LP-010 Where the graph of a problem comes from]] | Finding a problem's factor graph: from a model, by structure learning, or by exploring with TDA first | open, theory |
| [[LP-011 Gradient couplings and isfrozen are declared but unused]] | Per-edge couplings and `isfrozen` are stored but no code reads them | open |
| [[LP-012 Factor graphs and schemes]] | Links to algebraic geometry: singular models, dual-number points, cellular sheaves | open, theory |
