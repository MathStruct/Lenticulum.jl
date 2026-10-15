#!/usr/bin/env bash
# Deploy the vault build (docs/vault-build/, from docs/site/build.sh) to the gh-pages branch:
#
#   docs/site/deploy_vault.sh root       # master: the vault is the site's front page
#   docs/site/deploy_vault.sh preview    # the development branch: <site>/preview/
#
# Run by the Docs workflow after docs/make.jl, which has already deployed the API docs under
# api/ (deploydocs' dirname). The root belongs to the vault except for what others own: api/
# (Documenter), preview/, the old release folders and versions.js from before the move. A root
# deploy also rewrites dev/, the site's address until October 2026, as redirect stubs to the new
# addresses (docs/site/redirects.py), so links into the old site keep working.
#
# Needs GITHUB_TOKEN and GITHUB_REPOSITORY (set in GitHub Actions); DRYRUN=1 stops before pushing.
set -euo pipefail

TARGET="${1:?usage: deploy_vault.sh root|preview}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
VAULT="$ROOT/docs/vault-build"
API="$ROOT/docs/build"
[[ -f "$VAULT/index.html" ]] || { echo "no vault build in $VAULT" >&2; exit 1; }

WORK="$(mktemp -d)"
[[ "${KEEP_WORK:-0}" == "1" ]] || trap 'rm -rf "$WORK"' EXIT; echo ">> working copy: $WORK"
REMOTE="${PAGES_REMOTE:-https://x-access-token:${GITHUB_TOKEN:-}@github.com/${GITHUB_REPOSITORY:-MathStruct/Lenticulum.jl}.git}"
git clone --quiet --depth 1 --branch gh-pages "$REMOTE" "$WORK"
cd "$WORK"

case "$TARGET" in
  preview)
    rm -rf preview
    cp -a "$VAULT" preview
    ;;
  root)
    # everything at the root is the vault's, except what others own
    shopt -s dotglob nullglob
    for e in *; do
      case "$e" in .git|api|preview|dev|stable|versions.js|CNAME|.nojekyll) continue ;; esac
      [[ "$e" =~ ^[0-9]+-proto$ || "$e" =~ ^v[0-9] ]] && continue
      rm -rf -- "$e"
    done
    cp -a "$VAULT/." ./
    rm -rf dev
    python3 "$HERE/redirects.py" "$VAULT" "$API" dev
    ;;
  *)
    echo "unknown target $TARGET (root or preview)" >&2; exit 1 ;;
esac

git add -A
if git diff --cached --quiet; then
  echo ">> vault: nothing to deploy"
  exit 0
fi
git -c user.name="github-actions[bot]" -c user.email="41898282+github-actions[bot]@users.noreply.github.com" \
  commit --quiet -m "Vault: deploy to ${TARGET} from ${GITHUB_SHA:-local}"
if [[ "${DRYRUN:-0}" == "1" ]]; then
  echo ">> dry run: would push $(git diff --stat HEAD~1 | tail -1)"
  exit 0
fi
for attempt in 1 2 3; do
  git push --quiet origin gh-pages && { echo ">> vault deployed to ${TARGET}"; exit 0; }
  git pull --quiet --rebase origin gh-pages
done
echo "!! could not push the vault" >&2
exit 1
