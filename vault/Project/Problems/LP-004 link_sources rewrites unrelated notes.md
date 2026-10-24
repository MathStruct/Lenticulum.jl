#open-problem

> **Status: closed (2026-10-23, `review-argon`).** Related: `AGENTS.md` §"Citations", `docs/site/link_sources.py`.

**Problem.** `docs/site/link_sources.py` adds the `> Bibliography:` line under each note's sources. It also rewrites the Bibliography lines of a few unrelated notes (`lib/VariationalDiffusion.jl/src/implicit.md`, `sampler.md`, *Backpropagation through Implicit Inference*), dropping links that were there. Contributors are told to check `git diff` and restore those files by hand.

**Options.**
1. Find the matching rule that misfires (probably a citation key that is a prefix of another, or a source line it cannot parse) and fix it, with a test on the three notes.
2. Make the script refuse to remove an existing link, and report instead.

**Resolution.** Option 1 implemented in `docs/site/link_sources.py`: added unique `(lead_author_lastname, year)` matching anchored to the start of a citation clause (`(?:^|[;:(]|\band)\s*`), so `Author Year` and `Author et al., Venue Year` citations in `implicit.md`, `sampler.md`, and *Backpropagation through Implicit Inference* match directly without false co-author matches.

**Trade-off (2026-10-24).** The script now also keeps every existing `> Bibliography:` link whose key is still in `refs.bib`. That protects links it cannot re-derive, but a citation removed from a note's sources keeps its link until the line is edited by hand. The new author–year matching anchors the lead author at the start of a citation clause, which avoids the false co-author match an earlier draft produced (Kingma, a co-author of Song et al. 2021, briefly gave *The Diffusion Family* a citation of Kingma et al. 2021).
