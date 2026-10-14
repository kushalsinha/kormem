#!/usr/bin/env bash
# Publish kormem to PyPI (engine + CLI) and npm (launcher for npx / bun / bunx).
#
#   scripts/release.sh --dry-run   build, test and check everything; upload nothing
#   scripts/release.sh             publish
#
# Tokens come from the repo-root .env (KEY = value lines) and are never printed:
#   PYPI_TOKEN   a PyPI API token (pypi-…)
#   NPM_TOKEN    an npm automation or granular token with publish rights
#
# PyPI goes first: the npm launcher installs `kormem==<version>` from PyPI on first run.
set -euo pipefail
cd "$(dirname "$0")/.."

DRY=""
[[ "${1:-}" == "--dry-run" ]] && DRY=1

envval() { sed -nE "s/^[[:space:]]*$1[[:space:]]*=[[:space:]]*(.*)$/\1/p" .env 2>/dev/null | tail -1 | sed -E "s/^[\"']//; s/[\"'][[:space:]]*$//; s/[[:space:]]+$//"; }

VERSION=$(sed -nE 's/^__version__ = "(.*)"/\1/p' packages/engine/kormem/__init__.py)
for f in packages/cli/package.json plugin/.claude-plugin/plugin.json .claude-plugin/marketplace.json; do
  v=$(sed -nE 's/.*"version": *"([^"]+)".*/\1/p' "$f" | head -1)
  [[ "$v" == "$VERSION" ]] || { echo "Version mismatch: $f has $v, kormem/__init__.py has $VERSION"; exit 1; }
done
echo "kormem $VERSION"

if [[ -z "$DRY" ]]; then
  PYPI_TOKEN=$(envval PYPI_TOKEN); NPM_TOKEN=$(envval NPM_TOKEN)
  [[ -n "$PYPI_TOKEN" ]] || { echo "PYPI_TOKEN missing from .env"; exit 1; }
  [[ -n "$NPM_TOKEN" ]] || { echo "NPM_TOKEN missing from .env"; exit 1; }
  [[ -z "$(git status --porcelain)" ]] || { echo "Commit or stash changes before publishing."; exit 1; }
fi

echo "== tests"
(cd packages/engine && python -m pytest tests -q)
if command -v claude >/dev/null; then claude plugin validate . >/dev/null && echo "plugin manifest ok"; fi

echo "== build"
rm -rf dist && mkdir -p dist
(cd packages/engine && uv build --out-dir ../../dist)
(cd packages/cli && npm pack --pack-destination ../../dist >/dev/null)
ls -1 dist

if [[ -n "$DRY" ]]; then
  uvx twine check dist/kormem-"$VERSION".tar.gz dist/kormem-"$VERSION"-py3-none-any.whl
  (cd packages/cli && npm publish --dry-run >/dev/null && echo "npm publish --dry-run ok")
  echo "Dry run done. Nothing was uploaded."
  exit 0
fi

echo "== PyPI"
UV_PUBLISH_TOKEN="$PYPI_TOKEN" uv publish --check-url https://pypi.org/simple/ dist/kormem-"$VERSION".tar.gz dist/kormem-"$VERSION"-py3-none-any.whl

echo "== npm"
NPMRC=$(mktemp); trap 'rm -f "$NPMRC"' EXIT
printf '//registry.npmjs.org/:_authToken=%s\n' "$NPM_TOKEN" > "$NPMRC"
# npm tokens can only stage; approve the staged version on npmjs.com (2FA) to publish it.
NPM_CONFIG_USERCONFIG="$NPMRC" npx -y npm@latest stage publish "./dist/kormem-$VERSION.tgz" --access public

git tag -a "v$VERSION" -m "kormem $VERSION"
echo "Published kormem $VERSION to PyPI and staged it on npm (approve it on npmjs.com), and tagged v$VERSION locally (push it with: git push origin v$VERSION)."
