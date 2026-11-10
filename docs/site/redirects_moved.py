#!/usr/bin/env python3
"""Redirect stubs for notes that moved inside the vault (docs/site/moved.tsv).

The refactor of November 2026 regrouped the vault by path (`Families/Diffusion/…` became
`Relation Learners/…`) and moved the top-level package's notes from src/ to lib/Lenticulum.jl/.
Other sites link to the old addresses (the CT-ML wiki, about 25 files). For every row
`old<TAB>new` of moved.tsv this writes, in the deployed site:

  <OUT>/<old slug>.html          ->  <SITE>/<PREFIX><new slug>
  <OUT>/dev/vault/<old slug>.html ->  same   (only with --dev: the address until October 2026)

A stub is not written where the new build has a real page, nor when the new note is missing from
the build (a broken redirect is worse than a 404 that says so).

Usage: redirects_moved.py VAULT_BUILD OUT_DIR [--prefix preview/] [--dev]
"""
import sys
from pathlib import Path

SITE = "https://mathstruct.org/Lenticulum.jl"
HERE = Path(__file__).resolve().parent


def slug(path):
    # Quartz's sluggify (quartz/util/path.ts), segment by segment
    def seg(s):
        return (s.replace(" ", "-").replace("\t", "-").replace("&", "-and-")
                 .replace("%", "-percent").replace("?", "").replace("#", ""))
    return "/".join(seg(s) for s in path.split("/"))


def stub(path, target):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Moved</title>
<link rel="canonical" href="{target}">
<meta name="robots" content="noindex">
<meta http-equiv="refresh" content="0; url={target}">
</head><body><p>This page has moved to <a href="{target}">{target}</a>.</p></body></html>
""", encoding="utf-8")


args = sys.argv[1:]
dev = "--dev" in args
prefix = ""
if "--prefix" in args:
    prefix = args[args.index("--prefix") + 1]
pos = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or args[i - 1] != "--prefix")]
vault, out = Path(pos[0]), Path(pos[1])

n = skipped = 0
for line in (HERE / "moved.tsv").read_text(encoding="utf-8").splitlines():
    if not line.strip() or line.startswith("#"):
        continue
    old, new = (slug(s) for s in line.split("\t"))
    if not (vault / f"{new}.html").exists():
        print(f"!! {new} is not in the build; no redirect for {old}", file=sys.stderr)
        skipped += 1
        continue
    target = f"{SITE}/{prefix}{new}"
    for base in ([out, out / "dev" / "vault"] if dev else [out]):
        page = base / f"{old}.html"
        if base == out and (vault / f"{old}.html").exists():
            continue                       # a real page lives at the old address
        stub(page, target)
        n += 1
print(f"wrote {n} redirect stubs for moved notes into {out}" + (f" ({skipped} skipped)" if skipped else ""))
