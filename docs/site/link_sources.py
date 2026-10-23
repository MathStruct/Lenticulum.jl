#!/usr/bin/env python3
"""Link every note's free-text "Sources:" line to its entries in the Bibliography note.

For each note (vault/, lib/*/src/*.md, src/*.md) with a "> Sources:" line, find the works of
docs/src/refs.bib that the line cites, recognised only by an arXiv identifier or by the work's
title appearing in the line, or an italicised name equal to its title or its short name before
the colon (*MACE*, *PyTorch 2*); no guessing from author names. Write, right after the
Sources line,

    >
    > Bibliography: [[Bibliography#^key|Author et al. 2019]] · ...

The Sources line itself is left as it is. Rerunning replaces the generated line, so new notes
and new .bib entries are picked up; a note whose sources match nothing gets no line. Run it
after adding citations; docs/site/bib2vault.py then lists, for every entry, the notes citing it.

Usage: link_sources.py [--check]   (--check: report what would change, write nothing)
"""
import re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bib2vault import entries, clean  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BIB = ROOT / "docs" / "src" / "refs.bib"
GENERATED = "> Bibliography:"


def norm(s):
    s = clean(s).lower().replace("’", "'")
    s = re.sub(r"[*_`]", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " " + re.sub(r"\s+", " ", s).strip() + " "


def label(f):
    who = clean(f.get("author") or f.get("editor") or "")
    if not who:
        return clean(f["title"])[:40]
    last = [a.split(",")[0].strip() for a in who.split(" and ")]
    lead = last[0] if len(last) == 1 else (f"{last[0]} & {last[1]}" if len(last) == 2 else f"{last[0]} et al.")
    return f"{lead} {f.get('year', '')}".strip()


def matchers():
    out = []
    for _, key, f in entries(BIB.read_text(encoding="utf-8")):
        raw = clean(f.get("title", ""))
        t = norm(raw)
        title = t if len(t.strip()) >= 12 else None
        short = {t.strip()}                                   # names an italic citation may use
        if "stochastic differential equations" in t:
            short.add(t.replace("stochastic differential equations", "sdes").strip())
        if ":" in raw and len(raw.split(":")[0].strip()) >= 3:
            short.add(norm(raw.split(":")[0]).strip())
        who = clean(f.get("author") or f.get("editor") or "")
        lead_last = norm(who.split(" and ")[0].split(",")[0]).strip() if who else ""
        year = clean(f.get("year", "")).strip()
        out.append((key, f.get("eprint"), title, short, t, label(f), lead_last, year))
    return out


def cited(line, ms):
    found = []
    nline = norm(line)
    italics = [(m.start() / max(len(line), 1), norm(m.group(1)).strip()) for m in re.finditer(r"\*([^*]+)\*", line)]
    # count how many entries share (lead_last, year) so only unique Author-Year pairs match
    ay_counts = {}
    for _, _, _, _, _, _, lead_last, year in ms:
        if lead_last and year:
            ay_counts[(lead_last, year)] = ay_counts.get((lead_last, year), 0) + 1
    for key, eprint, title, short, t, lab, lead_last, year in ms:
        pos = []
        if eprint and eprint in line:
            pos.append(line.index(eprint) / max(len(line), 1))
        if title and title in nline:
            pos.append(nline.index(title) / max(len(nline), 1))
        pos += [p for p, it in italics if it in short         # *MACE*, *PyTorch 2*: exact names
                or (len(it.split()) >= 3 and t.strip().startswith(it))]   # or a title's first words
        if not pos and lead_last and year and ay_counts.get((lead_last, year)) == 1:
            m_ay = re.search(
                rf"(?:^|[;:(]|\band)\s*\b{re.escape(lead_last)}\b[^;]*?\b{re.escape(year)}\b",
                line,
                flags=re.IGNORECASE,
            )
            if m_ay:
                pos.append(m_ay.start() / max(len(line), 1))
        if pos:
            found.append((min(pos), key, lab))
    return [(k, l) for _, k, l in sorted(found)]


def notes():
    yield from (ROOT / "vault").rglob("*.md")
    yield from (ROOT / "lib").glob("*/src/*.md")
    yield from (ROOT / "src").glob("*.md")


def main(check=False):
    ms = matchers()
    valid_labels = {key: lab for key, _, _, _, _, lab, _, _ in ms}
    changed, linked, unmatched = 0, 0, []
    for path in sorted(notes()):
        if path.name == "Bibliography.md":
            continue
        lines = path.read_text(encoding="utf-8").split("\n")
        i = next((k for k, l in enumerate(lines) if l.startswith("> Sources:")), None)
        if i is None:
            continue
        # remove a previously generated line (and the blank quote line before it), keeping valid existing keys (LP-004)
        j = i + 1
        existing = []
        if j + 1 < len(lines) and lines[j].strip() == ">" and lines[j + 1].startswith(GENERATED):
            for m_key in re.finditer(r"\[\[Bibliography#\^([^|\]]+)\|([^\]]+)\]\]", lines[j + 1]):
                k = m_key.group(1)
                if k in valid_labels:
                    existing.append((k, valid_labels[k]))
            del lines[j:j + 2]
        refs = cited(lines[i], ms)
        seen = {k for k, _ in refs}
        for k, lab in existing:
            if k not in seen:
                refs.append((k, lab))
                seen.add(k)
        if refs:
            linked += 1
            lines[i + 1:i + 1] = [">", GENERATED + " " + " · ".join(f"[[Bibliography#^{k}|{l}]]" for k, l in refs)]
        elif re.search(r"\*[^*]+\*|arXiv", lines[i]):           # cites a work, none recognised
            unmatched.append(path.relative_to(ROOT))
        new = "\n".join(lines)
        if new != path.read_text(encoding="utf-8"):
            changed += 1
            if not check:
                path.write_text(new, encoding="utf-8")
    print(f"{linked} notes linked, {changed} files {'would change' if check else 'changed'}; "
          f"{len(unmatched)} notes cite works none of which is in the bibliography")
    for u in unmatched:
        print("   unmatched:", u)


if __name__ == "__main__":
    main(check="--check" in sys.argv)
