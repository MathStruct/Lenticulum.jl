#!/usr/bin/env python3
"""Write .zenodo.json, Zenodo's metadata for a GitHub release, from the files that already hold it.

Zenodo reads .zenodo.json instead of CITATION.cff when both exist, and only .zenodo.json can
list contributors with roles. So that nothing is kept twice, this file is generated:

- title, version, licence, keywords, abstract, the creators (authors) and the repository
  come from the header of CITATION.cff (everything above `references:`);
- contributors, with their Zenodo contributor type and CRediT roles, come from
  contributors.toml (see CONTRIBUTING.md, "Credit and authorship");
- references come from docs/src/refs.bib, as CITATION.cff's do;
- a refs.bib entry with `relation = {isCitedBy}` is a work that uses Lenticulum: it becomes a
  related identifier of the release ("is cited by"), so the archive links to it.

docs/site/build.sh runs this after bib2cff.py.  Usage: zenodo.py [CITATION.cff] [contributors.toml]
"""
import json, re, sys, tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bib2vault import entries, clean  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CFF = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "CITATION.cff"
CONTRIB = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "contributors.toml"
BIB = ROOT / "docs" / "src" / "refs.bib"
OUT = ROOT / ".zenodo.json"

# DataCite's contributor types, the ones Zenodo accepts
TYPES = {"ContactPerson", "DataCollector", "DataCurator", "DataManager", "Distributor", "Editor",
         "HostingInstitution", "Producer", "ProjectLeader", "ProjectManager", "ProjectMember",
         "RegistrationAgency", "RegistrationAuthority", "RelatedPerson", "Researcher",
         "ResearchGroup", "RightsHolder", "Supervisor", "Sponsor", "WorkPackageLeader", "Other"}


def unquote(v):
    v = v.strip()
    return json.loads(v) if v.startswith('"') else v


def header(text):
    """The few fields of CITATION.cff's hand-written header this needs; not a YAML parser."""
    head = text.split("\nreferences:", 1)[0]
    meta = {}
    for key in ("title", "version", "license", "repository-code", "url", "doi"):
        m = re.search(rf"^{key}:\s*(.+)$", head, re.M)
        if m:
            meta[key] = unquote(m.group(1))
    m = re.search(r"^keywords:\n((?:  - .+\n)+)", head, re.M)
    meta["keywords"] = [unquote(k) for k in re.findall(r"^  - (.+)$", m.group(1), re.M)] if m else []
    m = re.search(r"^abstract: >-\n((?:(?:  .*)?\n)+)", head + "\n", re.M)
    if m:   # a folded block: lines join with spaces, blank lines separate paragraphs
        paras, cur = [], []
        for line in m.group(1).split("\n"):
            if line.strip():
                cur.append(line.strip())
            elif cur:
                paras.append(" ".join(cur)); cur = []
        if cur:
            paras.append(" ".join(cur))
        meta["abstract"] = paras
    m = re.search(r"^authors:\n((?:  .+\n)+)", head, re.M)
    meta["authors"] = []
    if m:
        for block in re.split(r"^  - ", m.group(1), flags=re.M)[1:]:
            person = {}
            for line in ("  " + block).splitlines():
                k, _, v = line.strip().lstrip("- ").partition(":")
                if v.strip():
                    person[k.strip()] = unquote(v)
            meta["authors"].append(person)
    return meta


def person(p):
    out = {"name": f"{p['family-names']}, {p['given-names']}" if "family-names" in p else p["name"]}
    if p.get("affiliation"):
        out["affiliation"] = p["affiliation"]
    if p.get("orcid"):
        out["orcid"] = p["orcid"].rstrip("/").rsplit("/", 1)[-1]
    return out


def reference(typ, key, f):
    who = clean(f.get("author") or f.get("editor") or "").replace(" and ", "; ")
    venue = clean(f.get("journal") or f.get("booktitle") or f.get("howpublished") or "")
    link = (f"https://doi.org/{f['doi']}" if "doi" in f else
            f"https://arxiv.org/abs/{f['eprint']}" if "eprint" in f else f.get("url", ""))
    parts = [f"{who} ({f['year']})" if who and "year" in f else who or f.get("year", ""),
             clean(f["title"]), venue, link]
    return ". ".join(p for p in parts if p)


def cited_by(f):
    if "doi" in f:
        return {"identifier": f["doi"], "relation": "isCitedBy", "scheme": "doi"}
    if "eprint" in f:
        return {"identifier": f"arXiv:{f['eprint']}", "relation": "isCitedBy", "scheme": "arxiv"}
    if "url" in f:
        return {"identifier": f["url"], "relation": "isCitedBy", "scheme": "url"}
    return None


def main():
    meta = header(CFF.read_text(encoding="utf-8"))
    contributors = []
    if CONTRIB.exists():
        for c in tomllib.loads(CONTRIB.read_text(encoding="utf-8")).get("contributor", []):
            t = c.get("type", "Other")
            if t not in TYPES:
                raise SystemExit(f"contributors.toml: {c.get('name')}: unknown Zenodo contributor type {t!r}")
            contributors.append(dict(person(c), type=t))
    bib = list(entries(BIB.read_text(encoding="utf-8")))
    related = [{"identifier": meta["repository-code"], "relation": "isSupplementTo", "scheme": "url"}]
    if "url" in meta:
        related.append({"identifier": meta["url"], "relation": "isDocumentedBy", "scheme": "url"})
    related += [r for typ, key, f in bib if f.get("relation", "").strip() == "isCitedBy" and (r := cited_by(f))]
    out = {
        "title": meta["title"],
        "upload_type": "software",
        "version": meta.get("version", ""),
        "description": "".join(f"<p>{p}</p>" for p in meta.get("abstract", [])),
        "creators": [person(a) for a in meta["authors"]],
        "contributors": contributors,
        "license": meta.get("license", ""),
        "keywords": meta["keywords"],
        "related_identifiers": related,
        "references": [reference(*e) for e in bib],
    }
    if not contributors:
        del out["contributors"]
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(out['creators'])} creators, {len(contributors)} contributors, "
          f"{len(out['references'])} references, {len(related)} related identifiers)")


if __name__ == "__main__":
    main()
