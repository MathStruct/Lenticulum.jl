#open-problem

> **Status: open (2026-10-22).** Related: [GitHub issue #9](https://github.com/MathStruct/Lenticulum.jl/issues/9), `docs/site/build.sh`, [[L-ADR-0004 The vault builds with the shared MathStruct Quartz]].

**Problem.** `docs/site/build.sh` substitutes the API address into the front page with `sed -i "s#@API@#…#g" file`. BSD sed (macOS) reads the expression as the backup suffix and fails, so the documented vault build does not run on a Mac (reported by docxology in issue #9).

**Options.**
1. `sed -e … file > tmp && mv tmp file`, which both seds accept.
2. Do the substitution in the shell, or write the front page with the address already in it.

**Leaning.** Option 1: one line, no change in behaviour on Linux.
