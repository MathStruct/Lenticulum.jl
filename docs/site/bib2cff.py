#!/usr/bin/env python3
"""Write docs/src/refs.bib into CITATION.cff as its `references:` section.

The bibliography of the tutorials, notes and vault is the reference list of the software's
citation metadata, so GitHub's "Cite this repository" and the Zenodo archive list the same
works as the documentation. Everything in CITATION.cff from the line `references:` on is
regenerated; everything above it is left alone. docs/site/build.sh runs this before every
vault build.  Usage: bib2cff.py [refs.bib] [CITATION.cff]
"""
import json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bib2vault import entries, clean  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BIB = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs" / "src" / "refs.bib"
CFF = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "CITATION.cff"

TYPES = {"article": "article", "inproceedings": "conference-paper", "book": "book",
         "incollection": "generic", "mastersthesis": "thesis", "phdthesis": "thesis",
         "techreport": "report", "misc": "generic"}


def q(s):
    """A YAML scalar: JSON strings are valid YAML double-quoted strings."""
    return json.dumps(s, ensure_ascii=False)


def persons(field):
    out = []
    for a in field.split(" and "):
        a = a.strip()
        if re.fullmatch(r"\{[^{}]*\}", a):                       # {Organisation Name}
            out.append({"name": clean(a)})
            continue
        last, _, first = a.partition(",")
        person = {"family-names": clean(last)}
        if first.strip():
            person["given-names"] = clean(first)
        out.append(person)
    return out


def reference(typ, key, f):
    software = f.get("howpublished") == "Software"
    lines = [f"  - type: {'software' if software else TYPES.get(typ, 'generic')}",
             f"    title: {q(clean(f['title']))}"]
    if "author" in f:
        who = persons(f["author"])
    elif "editor" in f:
        who = persons(f["editor"])
    else:   # software without a listed author: credit the project, from the repository URL
        m = re.search(r"github\.com/([^/]+)/", f.get("url", ""))
        who = [{"name": (m.group(1) if m else clean(f["title"]).split(":")[0]) + " contributors"}]
    lines.append("    authors:")
    for p in who:
        first = True
        for k, v in p.items():
            lines.append(f"      {'- ' if first else '  '}{k}: {q(v)}")
            first = False
    if "editor" in f and "author" in f:
        lines.append("    editors:")
        for p in persons(f["editor"]):
            first = True
            for k, v in p.items():
                lines.append(f"      {'- ' if first else '  '}{k}: {q(v)}")
                first = False
    if "year" in f and f["year"].isdigit():
        lines.append(f"    year: {int(f['year'])}")
    if "journal" in f:
        lines.append(f"    journal: {q(clean(f['journal']))}")
    if "booktitle" in f:
        lines.append(f"    collection-title: {q(clean(f['booktitle']))}")
    if "volume" in f:
        lines.append(f"    volume: {q(clean(f['volume']))}")
    if "number" in f:
        lines.append(f"    issue: {q(clean(f['number']))}")
    if "pages" in f:
        a, _, b = clean(f["pages"]).replace("–", "-").partition("-")
        lines.append(f"    start: {q(a.strip())}")
        if b.strip():
            lines.append(f"    end: {q(b.strip())}")
    if "publisher" in f:
        lines.append(f"    publisher:\n      name: {q(clean(f['publisher']))}")
    if "school" in f or "institution" in f:
        lines.append(f"    institution:\n      name: {q(clean(f.get('school') or f['institution']))}")
    if typ in ("mastersthesis", "phdthesis"):
        lines.append(f"    thesis-type: {q('Master' if typ == 'mastersthesis' else 'PhD')}")
    if "doi" in f:
        lines.append(f"    doi: {q(f['doi'])}")
    if "eprint" in f:
        lines.append(f"    url: {q('https://arxiv.org/abs/' + f['eprint'])}")
    elif "url" in f:
        lines.append(f"    url: {q(f['url'])}")
    return "\n".join(lines)


def main():
    head = CFF.read_text(encoding="utf-8")
    head = head.split("\nreferences:", 1)[0].rstrip() + "\n"
    refs = [reference(*e) for e in entries(BIB.read_text(encoding="utf-8"))]
    note = ("# Generated from docs/src/refs.bib by docs/site/bib2cff.py: the works cited in the\n"
            "# tutorials, implementation notes and theory vault. Edit the .bib file, not this list.\n")
    CFF.write_text(head + note + "references:\n" + "\n".join(refs) + "\n", encoding="utf-8")
    print(f"wrote {len(refs)} references into {CFF}")


if __name__ == "__main__":
    main()
