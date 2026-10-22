#open-problem

> **Status: open (2026-10-22).** Related: `AGENTS.md` §"Citations", `docs/site/link_sources.py`.

**Problem.** `docs/site/link_sources.py` adds the `> Bibliography:` line under each note's sources. It also rewrites the Bibliography lines of a few unrelated notes (`lib/VariationalDiffusion.jl/src/implicit.md`, `sampler.md`, *Backpropagation through Implicit Inference*), dropping links that were there. Contributors are told to check `git diff` and restore those files by hand.

**Options.**
1. Find the matching rule that misfires (probably a citation key that is a prefix of another, or a source line it cannot parse) and fix it, with a test on the three notes.
2. Make the script refuse to remove an existing link, and report instead.

**Leaning.** Option 2 first, as a guard; then option 1.
