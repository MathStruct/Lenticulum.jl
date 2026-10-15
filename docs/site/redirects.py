#!/usr/bin/env python3
"""Redirect stubs for the site's old addresses: everything used to live under dev/.

Until October 2026 the API docs were at <site>/dev/ and the vault at <site>/dev/vault/. Now the
vault is the site's front page and the API docs are under api/dev/. For every page of the new
vault build and of the API build this writes a page at its old address that redirects to the
new one, so links into the old site (the Zenodo record, papers, other repositories) keep working.

Usage: redirects.py VAULT_BUILD API_BUILD OUT_DIR    (OUT_DIR is the old dev/, in gh-pages)
"""
import sys
from pathlib import Path

SITE = "https://mathstruct.org/Lenticulum.jl"
vault, api, out = (Path(a) for a in sys.argv[1:4])


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


n = 0
# the vault: dev/vault/<page>.html  ->  <site>/<page>   (Quartz serves pages without .html)
for page in vault.rglob("*.html"):
    rel = page.relative_to(vault).as_posix()
    target = f"{SITE}/" if rel == "index.html" else f"{SITE}/{rel.removesuffix('.html').removesuffix('/index')}"
    stub(out / "vault" / rel, target)
    n += 1
# the API docs: dev/<page>/index.html  ->  <site>/api/dev/<page>/
for page in api.rglob("index.html"):
    rel = page.relative_to(api).parent.as_posix()
    if rel.startswith("vault"):            # an old build that still had the vault inside
        continue
    stub(out / page.relative_to(api), f"{SITE}/api/dev/" + ("" if rel == "." else rel + "/"))
    n += 1
print(f"wrote {n} redirect stubs into {out}")
