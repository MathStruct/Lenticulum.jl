#!/usr/bin/env bash
# Build the Obsidian vault into a static site with Quartz, next to the Documenter site.
#
#   docs/vault/build.sh              # -> docs/build/vault/
#   docs/vault/build.sh --serve      # local preview on http://localhost:8080
#
# Layout:
#   docs/vault/quartz.config.ts     our config    (committed)
#   docs/vault/quartz.layout.ts     our layout    (committed)
#   docs/quartz/                    Quartz v4 checkout (gitignored; cloned on first run)
#   docs/quartz/content/            staged copy of the vault (regenerated every run)
#   docs/build/vault/               output — deployed by Documenter as <site>/dev/vault/
#
# The vault is the whole repository (see "Start here.md"), but its notes live in three places:
# markdown/, the *.md files beside each lib/*/src/*.jl, and src/*.md. Quartz wants one content
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

# the theory notes, minus the authoring prompts
cp -r "$ROOT/markdown/." "$CONTENT/"
rm -rf "$CONTENT/Prompts"

# the implementation notes that sit beside the Julia source
for pkg in "$ROOT"/lib/*/; do
  name="$(basename "$pkg")"
  if compgen -G "$pkg/src/*.md" > /dev/null; then
    mkdir -p "$CONTENT/lib/$name"
    cp "$pkg"/src/*.md "$CONTENT/lib/$name/"
  fi
done
mkdir -p "$CONTENT/src" && cp "$ROOT"/src/*.md "$CONTENT/src/"

# the two root-level notes the vault links to as [[README]] and [[Start here]].
# The README is written for GitHub, so its repo-relative links are rewritten for the site:
# any markdown/<path>.md link becomes a site-relative link (Quartz slugs spaces to hyphens).
sed -e 's#](markdown/Index\.md)#](Index)#' \
    -e 's#](docs/vault/README\.md)#](https://github.com/MathStruct/Lenticulum.jl/blob/master/docs/vault/README.md)#' \
    -e 's#](markdown/\([^)]*\)\.md)#](\1)#; s#%20#-#g' \
    "$ROOT/README.md" > "$CONTENT/README.md"
cp "$ROOT/Start here.md" "$CONTENT/Start here.md"

# the site's front page
cat > "$CONTENT/index.md" <<'MD'
---
title: Lenticulum — theory vault
---

This is the Obsidian vault of [Lenticulum.jl](https://github.com/MathStruct/Lenticulum.jl):
the mathematics behind the code, the papers it draws on, the design decisions, and an honest
record of what does not work yet.

- **[[Index]]** — the map of content. Start there.
- **[[Start here]]** — how the vault is organised and which Obsidian plugins it assumes.
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
        r = lines[i].rstrip()
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
fi
