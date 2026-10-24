#open-problem

> **Status: closed (2026-10-23, `review-argon`; a CI guard added 2026-10-25, ba881303).** Related: [GitHub issue #9](https://github.com/MathStruct/Lenticulum.jl/issues/9), `docs/site/build.sh`, [[L-ADR-0004 The vault builds with the shared MathStruct Quartz]].

**Problem.** `docs/site/build.sh` substitutes the API address into the front page with `sed -i "s#@API@#…#g" file`. BSD sed (macOS) reads the expression as the backup suffix and fails, so the documented vault build does not run on a Mac (reported by docxology in issue #9).

**Options.**
1. `sed -e … file > tmp && mv tmp file`, which both seds accept.
2. Do the substitution in the shell, or write the front page with the address already in it.

**Resolution.** Option 1 implemented in `docs/site/build.sh` (`sed -e ... "$STAGE/index.md" > "$STAGE/index.md.tmp" && mv "$STAGE/index.md.tmp" "$STAGE/index.md"`).

**Follow-up (2026-10-25, Codex + Claude).** `docs/site/check_portability.py`, run by the Docs workflow, fails on GNU-only shell forms and Bash 4 features in the docs tooling, so Linux CI now catches what issue #9 reported; `@API@` is replaced literally, and a failed link rewrite stops the build.
