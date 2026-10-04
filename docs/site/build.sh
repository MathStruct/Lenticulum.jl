#!/usr/bin/env bash
# Build the Obsidian vault into a static site with Quartz, next to the Documenter site.
#
#   docs/site/build.sh               # -> docs/build/vault/
#   docs/site/build.sh --serve       # local preview on http://localhost:8080
#
# Layout:
#   docs/site/quartz.config.ts      our config    (committed)
#   docs/site/quartz.layout.ts      our layout    (committed)
#   docs/quartz/                    Quartz v4 checkout (gitignored; cloned on first run)
#   docs/quartz/content/            staged copy of the vault (regenerated every run)
#   docs/build/vault/               output — deployed by Documenter as <site>/dev/vault/
#
# The vault is the whole repository (see "vault/Start Here.md"), but its notes live in three places:
# vault/, the *.md files beside each lib/*/src/*.jl, and src/*.md. Quartz wants one content
# directory, so this script stages them together. Wikilinks resolve by basename ("shortest"),
# exactly as Obsidian does, so the staged directory structure is only for breadcrumbs.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
QUARTZ="$ROOT/docs/quartz"
OUT="$ROOT/docs/build/vault"
SERVE=0
[[ "${1:-}" == "--serve" ]] && SERVE=1

# --- 1. a Quartz checkout -----------------------------------------------------
if [[ ! -f "$QUARTZ/package.json" ]]; then
  echo ">> cloning Quartz v4 into docs/quartz"
  git clone --depth 1 --branch v4 https://github.com/jackyzha0/quartz.git "$QUARTZ"
fi
if [[ ! -d "$QUARTZ/node_modules" ]]; then
  echo ">> npm ci"
  (cd "$QUARTZ" && npm ci --no-audit --no-fund)
  # npm >= 12 blocks install scripts by default; esbuild and sharp need theirs to fetch
  # their native binaries, so approve them and rebuild.
  (cd "$QUARTZ" && npm install-scripts approve esbuild sharp @parcel/watcher >/dev/null 2>&1 \
     && npm rebuild esbuild sharp @parcel/watcher --no-audit --no-fund) || true
fi

# --- 2. our configuration over the stock one ----------------------------------
cp "$HERE/quartz.config.ts" "$QUARTZ/quartz.config.ts"
cp "$HERE/quartz.layout.ts" "$QUARTZ/quartz.layout.ts"

# --- 2b. the TikZ plugin (renders ```tikz blocks to SVG at build time) -------------
# Ported from MathStruct/CategoryTheory-ML-Wiki. The stock checkout has neither the plugin
# nor its dependency, so both are added here; the render cache lives in this directory and
# is committed, so a fresh clone only renders diagrams that are new.
if [[ ! -d "$QUARTZ/node_modules/node-tikzjax" ]]; then
  echo ">> npm install node-tikzjax"
  (cd "$QUARTZ" && npm install --no-save --no-audit --no-fund node-tikzjax@^1.0.5)
fi
cp "$HERE/tikz.ts" "$QUARTZ/quartz/plugins/transformers/tikz.ts"
grep -q 'from "./tikz"' "$QUARTZ/quartz/plugins/transformers/index.ts" || \
  echo 'export { TikZ } from "./tikz"' >> "$QUARTZ/quartz/plugins/transformers/index.ts"
mkdir -p "$HERE/.tikz-cache"
rm -rf "$QUARTZ/.tikz-cache" && ln -s "$HERE/.tikz-cache" "$QUARTZ/.tikz-cache"

# --- 3. stage the vault ----------------------------------------------------------
CONTENT="$QUARTZ/content"
rm -rf "$CONTENT"
mkdir -p "$CONTENT"

# the theory notes (meta/ — the authoring prompts and the PhD proposals — is not published)
cp -r "$ROOT/vault/." "$CONTENT/"

# the implementation notes that sit beside the Julia source
for pkg in "$ROOT"/lib/*/; do
  name="$(basename "$pkg")"
  if compgen -G "$pkg/src/*.md" > /dev/null; then
    mkdir -p "$CONTENT/lib/$name"
    cp "$pkg"/src/*.md "$CONTENT/lib/$name/"
  fi
done
mkdir -p "$CONTENT/src" && cp "$ROOT"/src/*.md "$CONTENT/src/"

# the root-level note the vault links to as [[README]].
# It is written for GitHub, so its repo-relative links are rewritten for the site:
# any vault/<path>.md link becomes a site-relative link (Quartz slugs spaces to hyphens),
# and links to other files in the repository go to GitHub.
sed -e 's#](vault/\([^)]*\)\.md)#](\1)#' \
    -e 's#](\(docs/[^)]*\|meta/[^)]*\))#](https://github.com/MathStruct/Lenticulum.jl/blob/master/\1)#' \
    -e '/](https:/!s#%20#-#g' \
    "$ROOT/README.md" > "$CONTENT/README.md"

# the site's front page
cat > "$CONTENT/index.md" <<'MD'
---
title: Lenticulum — theory vault
---

[Lenticulum.jl](https://github.com/MathStruct/Lenticulum.jl) learns **relations instead of
functions**: a model of a joint space $Z = X \times Y \times U$ that decides at query time which
coordinates are inputs $X$, outputs $Y$ and latents $U$. This vault is the larger half of the
project: the mathematics, the papers, the design decisions, and an honest record of what does
not work yet. Pick the door that matches what you already know.

## Coming from machine learning

You know neural networks, backpropagation, perhaps diffusion models and deep equilibrium
models. The short version: a trained denoiser defines a vector field whose stable roots are a
relation; inference is root-finding with the inputs clamped, as a DEQ is evaluated; the
backward pass is the implicit function theorem, one adjoint solve, nothing unrolled. The
networks are **small** (a few thousand parameters over a handful of coordinates), and any Lux
model with any AD backend works.

1. [[Implicit Learners]] — what "learning a relation" means, and the three families
2. [[Implicit Diffusion Learners]] — a diffusion model as a relation; the residual field
3. [[Backpropagation through Implicit Inference]] — the adjoint, and a parabola learned from a circle
4. [[DEQ as a Relation]] — the same idea for equilibrium models
5. [[backends]] — Zygote, Enzyme or Reactant; a 5k-parameter MLP end to end

Or start with code: the [tutorials](../tutorials/01_circle/) (also Jupyter notebooks).

## Coming from statistics or robotics

You know Bayesian inference, Gaussian posteriors, perhaps factor graphs and GTSAM. The short
version: every factor is a joint model whose conditioning direction is chosen per query;
message passing on a factor graph is Bayesian inversion of each factor; the exact results are
on the linear-Gaussian fragment, and the learned factors (diffusion, equilibrium, adversarial)
plug into the same graph.

1. [[Factor Graphs]] and [[Everything is a Factor]] — the setting
2. [[Beliefs]] — what flows along the edges: Gaussian, Dirac, samples
3. [[The Linear Gaussian Chain]] — the case where everything is exact and checked
4. [[Messages are Inversions]] and [[Bethe Free Energy]] — inference and its objective
5. [[SLAM and Sensor Fusion]] — the application that motivates the design

## Coming from category theory

You know lenses, Para, Markov categories, perhaps AutoBayes' statistical games. The short
version: a factor is a parameterized statistical game once a polarity is chosen, and a
Bayesian lens after that; acausal composition is a hypergraph category; the code mirrors these
definitions type for type. Notation note: this vault's inputs $X$ are AutoBayes' $Y$ and
vice versa ([[Channels and Polarity]] §"Notation").

1. [[Factors are Parameterized Statistical Games]] — the central correspondence
2. [[Channels and Polarity]] — open models, cups and caps, and why direction is chosen late
3. [[Inversions and Bayesian Lenses]] — what inference is, categorically
4. [[Acausal Composition is a Hypergraph Category]] — how factors compose
5. [[The Implicit Diffusion Factor as a Statistical Game]] — a learned factor, checked against the definitions

The general theory these notes build on is in the
[CT-ML wiki](https://mathstruct.org/CategoryTheory-ML-Wiki/).

## Everything else

- **[[Map of Content]]** — every note, in reading order.
- **[[Start Here]]** — how the vault is organised, its conventions, and how to read it in Obsidian.
- **[[README]]** — the project pitch.
- **[API documentation](../)** — the Julia packages themselves, built with Documenter.

The vault is written for [Obsidian](https://obsidian.md) and rendered here with
[Quartz](https://quartz.jzhao.xyz), including its `tikz` diagrams, which are compiled to SVG
at build time. The graph view is Quartz's rather than Obsidian's.
MD

# --- 3b. lint: display math that Obsidian accepts but remark-math truncates on -----------
# A `$$` block opened with content on the fence line and closed with `$$` at the end of a
# later content line is an *unclosed fence* to remark-math: everything after it vanishes from
# the rendered page, silently. Obsidian renders it fine, so nothing else catches this.
# Fences must sit on their own lines. Refuse to build rather than publish truncated pages.
python3 - "$CONTENT" <<'PY'
import sys, glob, os
root = sys.argv[1]; bad = []
for f in glob.glob(os.path.join(root, "**", "*.md"), recursive=True):
    lines = open(f, encoding="utf-8").read().split("\n"); i = 0
    while i < len(lines):
        # strip blockquote / list prefixes so math inside callouts and list items is seen too
        r = lines[i].rstrip().lstrip("> ").lstrip()
        if r.startswith("$$") and len(r) > 2 and not (r.endswith("$$") and len(r) > 4):
            j = i + 1
            while j < len(lines) and not lines[j].rstrip().endswith("$$"): j += 1
            if j < len(lines) and lines[j].strip() != "$$": bad.append((os.path.relpath(f, root), i + 1))
            i = j
        i += 1
if bad:
    print("!! display-math blocks that would truncate their page (put the $$ fences on their own lines):")
    for f, n in bad: print(f"   {f}:{n}")
    sys.exit(1)
PY

# --- 4. build ---------------------------------------------------------------------
cd "$QUARTZ"
if [[ $SERVE -eq 1 ]]; then
  exec npx quartz build --serve -o "$OUT"
else
  npx quartz build -o "$OUT"
  echo ">> vault built into docs/build/vault ($(find "$OUT" -name '*.html' | wc -l) pages)"
  # KaTeX renders with throwOnError:false, so a bad formula becomes red text rather than a
  # failed build. Surface them here (e.g. \tag in a single-line $$...$$, which remark-math
  # parses as *inline* math — put such fences on their own lines).
  if grep -rl 'katex-error' "$OUT" --include='*.html' >/dev/null; then
    echo "!! KaTeX parse errors in:"
    grep -rho 'katex-error" title="[^"]*"' "$OUT" --include='*.html' | sort | uniq -c
    grep -rl 'katex-error' "$OUT" --include='*.html' | sed "s#^$OUT/#   #"
    exit 1
  fi
fi
