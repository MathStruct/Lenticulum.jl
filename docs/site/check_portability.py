#!/usr/bin/env python3
"""Cheap static guard for docs tooling shared by Linux and macOS (Bash 3.2).

Run from anywhere: python3 docs/site/check_portability.py [files ...]
Without arguments, scan tracked and unignored tooling under docs/, vault/ and
.github/workflows/. This is a conservative pattern check, not a shell parser;
keep native Python/Julia regexes out of shell-command lines it would flag.
No dependencies, downloads, or build outputs are needed.
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def command_options(command, options):
    # Also match shell commands embedded in Python/Julia strings and YAML run steps.
    return rf"\b{command}\s+[^\n;|]*?(?:{options})(?=[\s\"'=]|$)"


RULES = [
    (command_options('sed', r'-[A-Za-z]*i[^\s]*|-[A-Za-z]*r[A-Za-z]*|--in-place(?:=[^\s]*)?'),
     'use sed output + mv, and -E rather than -r'),
    (command_options('grep', r'-[A-Za-z]*[P][A-Za-z]*'),
     'avoid PCRE (grep -P); BSD grep has no -P'),
    (command_options('find', r'-printf|-fprintf|-regextype'),
     'use portable find predicates and -exec'),
    (command_options('xargs', r'-[A-Za-z]*r[A-Za-z]*|--no-run-if-empty'),
     'avoid xargs -r; use find -exec'),
    (command_options('readlink', r'-[A-Za-z]*[fe][A-Za-z]*|--canonicalize(?:-existing)?'),
     'resolve paths with Python pathlib'),
    (command_options('date', r'-[A-Za-z]*[dI][A-Za-z]*|--date|--iso-8601'),
     'use Python datetime for date arithmetic/formatting'),
    (command_options('stat', r'-[A-Za-z]*[cft][A-Za-z]*|--format|--printf'),
     'stat flags differ across systems; use Python pathlib.stat'),
    (command_options('cp', r'--[a-z][a-z-]*'),
     'BSD cp has no long options (e.g. --parents)'),
    (command_options('mktemp', r'--[a-z][a-z-]*|-[A-Za-z]*[tp][A-Za-z]*'),
     'use mktemp [-d] with an explicit path ending in XXXXXX'),
    (r'\b(?:sed|grep)\s+[^\n]*?\\[|+?]',
     'avoid GNU basic-regex operators; split expressions or use -E'),
    (r'\b(?:mapfile|readarray)\b|\b(?:declare|local|typeset)\s+-[a-zA-Z]*A',
     'avoid Bash 4 arrays/builtins; macOS ships Bash 3.2'),
    (r'\$\{[^}\n]*(?:,,|\^\^)[^}\n]*\}|\|&|&>>|\bshopt\s+-s\s+globstar',
     'avoid Bash 4 case conversion, redirections and globstar'),
]
RULES = [(re.compile(pattern), message) for pattern, message in RULES]


def violations(text):
    # Join continued shell commands so flags on the following line are checked.
    logical = re.sub(r'\\\n[ \t]*', ' ', text)
    return [(match.start(), message) for rule, message in RULES
            for match in rule.finditer(logical)]


def self_test():
    forbidden = [
        'sed -i "s/a/b/" file', 'sed -i.bak -e expr file',
        'sed --in-place expr file', 'sed -r expr file',
        'grep -P expr file',
        'find . -printf "%p"', 'xargs -r echo', 'readlink -f path',
        'date -d yesterday', 'stat -c %s path', 'cp --parents a b',
        'mktemp --tmpdir', 'mktemp -t name',
        r"sed 's/a\|b/c/' file", r"grep 'a\+' file",
        'declare -A items', 'mapfile items', '${name,,}',
        'command |& cat', 'command &>> log', 'shopt -s globstar',
        'sed \\\n  -i expr file',
    ]
    portable = [
        'sed -E "s/a|b/c/" file', "sed -e expr file > tmp",
        'grep -q pattern file', "find . -name '*.md' -exec cat {} +",
        'cp -pR source dest', 'cp -a source dest', 'mktemp -d',
        'grep -rl expr . --include="*.html"', 'mktemp -d "${TMPDIR:-/tmp}/docs.XXXXXX"',
        '[[ "$name" =~ ^[0-9]+-proto$ ]]',
    ]
    for text in forbidden:
        assert violations(text), f'guard missed: {text}'
    for text in portable:
        assert not violations(text), f'guard rejected portable code: {text}'


def main():
    self_test()
    if sys.argv[1:]:
        paths = [Path(arg).resolve() for arg in sys.argv[1:]]
    else:
        names = subprocess.check_output([
            'git', '-C', str(ROOT), 'ls-files', '-z', '--cached', '--others',
            '--exclude-standard', '--', 'docs', 'vault', '.github/workflows',
        ]).decode().split('\0')
        paths = [ROOT / name for name in sorted(set(names)) if name
                 and Path(name).suffix in {'.sh', '.py', '.jl', '.yml', '.yaml'}]
    failed = False
    for path in paths:
        if path == Path(__file__).resolve():
            continue  # This file contains the forbidden patterns and their self-tests.
        text = path.read_text(encoding='utf-8')
        for _, message in violations(text):
            print(f'{path}: {message}', file=sys.stderr)
            failed = True
    if failed:
        return 1
    print('Docs portability patterns and guard self-tests passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
