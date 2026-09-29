#!/usr/bin/env bash
# Publica una versión (semver): mueve "Unreleased" del CHANGELOG.md a "[X.Y.Z] - fecha", fija la
# versión en el fichero del proyecto (pom.xml / package.json / pyproject.toml, el que exista),
# hace el commit "release: vX.Y.Z" y crea el tag vX.Y.Z. NO empuja nada: revisa y luego
#   git push origin master --tags
# El tag dispara .github/workflows/release.yml (checks → imagen :vX.Y.Z si el repo la publica →
# GitHub Release con las notas de esa sección del CHANGELOG).
#
# Uso: bash release.sh X.Y.Z        (desde master, árbol limpio, con algo en "Unreleased")
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

version="${1:?Uso: release.sh X.Y.Z}"
[[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "[release] versión no semver: $version" >&2; exit 1; }
tag="v$version"
branch="$(git rev-parse --abbrev-ref HEAD)"
[ "$branch" = "master" ] || { echo "[release] hay que publicar desde master (estás en $branch)" >&2; exit 1; }
[ -z "$(git status --porcelain)" ] || { echo "[release] el árbol de trabajo no está limpio" >&2; exit 1; }
! git rev-parse -q --verify "refs/tags/$tag" >/dev/null || { echo "[release] el tag $tag ya existe" >&2; exit 1; }
[ -f CHANGELOG.md ] || { echo "[release] falta CHANGELOG.md" >&2; exit 1; }
grep -q '^## \[Unreleased\]' CHANGELOG.md || { echo "[release] CHANGELOG.md no tiene sección '## [Unreleased]'" >&2; exit 1; }
! grep -q "^## \[$version\]" CHANGELOG.md || { echo "[release] CHANGELOG.md ya tiene la sección [$version]" >&2; exit 1; }

today="$(date -u +%F)"
python3 - "$version" "$today" <<'PY'
import re, sys
version, today = sys.argv[1], sys.argv[2]
text = open("CHANGELOG.md", encoding="utf-8").read()
head, _, rest = text.partition("## [Unreleased]")
body, sep, tail = rest.partition("\n## [")
if not body.strip():
    sys.exit("[release] la sección Unreleased está vacía: no hay nada que publicar")
new = f"{head}## [Unreleased]\n\n## [{version}] - {today}{body.rstrip()}\n{sep}{tail}"
open("CHANGELOG.md", "w", encoding="utf-8").write(new)
PY

if [ -f pom.xml ]; then
  # Primer <version> tras el <artifactId> del proyecto (no el del parent).
  python3 - "$version" <<'PY'
import re, sys
version = sys.argv[1]
pom = open("pom.xml", encoding="utf-8").read()
end_parent = pom.index("</parent>")
m = re.search(r"<version>[^<]+</version>", pom[end_parent:])
pom = pom[:end_parent + m.start()] + f"<version>{version}</version>" + pom[end_parent + m.end():]
open("pom.xml", "w", encoding="utf-8").write(pom)
PY
fi
if [ -f package.json ]; then
  python3 - "$version" <<'PY'
import re, sys
version = sys.argv[1]
text = open("package.json", encoding="utf-8").read()
text = re.sub(r'("version"\s*:\s*")[^"]+(")', rf'\g<1>{version}\2', text, count=1)
open("package.json", "w", encoding="utf-8").write(text)
PY
fi
if [ -f pyproject.toml ] && grep -q '^version *= *"' pyproject.toml; then
  sed -i -E "s/^version *= *\"[^\"]+\"/version = \"$version\"/" pyproject.toml
fi
if [ -f VERSION ]; then
  echo "$version" > VERSION
fi

git add -A
git commit -q -m "release: $tag"
git tag -a "$tag" -m "$tag"
echo "[release] commit y tag $tag creados en local. Revisa con: git show --stat HEAD"
echo "[release] publicar: git push origin master --tags"
