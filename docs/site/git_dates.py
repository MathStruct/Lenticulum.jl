#!/usr/bin/env python3
"""Give every staged vault page the date its source last changed.

docs/site/build.sh copies the notes into Quartz's content directory, so every copy is as new
as the build, and Quartz (which reads the file date) would show every page as changed today.
This sets each staged file's modification time to that of the last commit touching its source
in the repository, or to the source's own modification time when it has uncommitted changes or
is not tracked. The CI checkout must have the whole history (`fetch-depth: 0`).

Usage: git_dates.py <content dir>   (run from anywhere; docs/site/build.sh runs it after staging)
"""
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTENT = Path(sys.argv[1]).resolve()


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True).stdout


# the latest commit time of every path, from one pass over the history (newest first)
latest = {}
stamp = None
for line in git("log", "--format=%x00%ct", "--name-only", "--no-renames").splitlines():
    if line.startswith("\x00"):
        stamp = int(line[1:])
    elif line and line not in latest:
        latest[line] = stamp

# paths whose working copy differs from the last commit (modified or untracked)
dirty = {line[3:].strip('"') for line in git("status", "--porcelain", "--untracked-files=all").splitlines()}


def source_of(rel: Path):
    """The repository path a staged file was copied from (see build.sh, step 3)."""
    parts = rel.parts
    if parts[0] == "lib" and len(parts) == 3:              # lib/<pkg>/<note>.md <- lib/<pkg>/src/<note>.md
        return Path("lib", parts[1], "src", parts[2])
    if parts[0] == "src":
        return Path("src", *parts[1:])
    if rel == Path("README.md"):
        return rel
    if rel == Path("index.md"):                            # written by build.sh itself
        return Path("docs/site/build.sh")
    return Path("vault", *parts)


done = kept = 0
for f in CONTENT.rglob("*"):
    if not f.is_file():
        continue
    src = source_of(f.relative_to(CONTENT))
    key = src.as_posix()
    if key in latest and key not in dirty:
        t = latest[key]
        done += 1
    elif (ROOT / src).exists():
        t = (ROOT / src).stat().st_mtime
        kept += 1
    else:
        continue
    os.utime(f, (t, t))
print(f"dated {done} pages by their last commit, {kept} by their working copy")
