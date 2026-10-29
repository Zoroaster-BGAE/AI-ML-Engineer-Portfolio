#!/bin/sh
# Fetch upstream open-source code into ./third_party (git-ignored). Nothing here is vendored into this repository.
#
#   TOTEM_REPO_URL=<official repository URL, linked from arXiv:2402.16412> \
#   TOTEM_REF=<commit or tag> \
#   sh scripts/fetch_third_party.sh
set -eu

: "${TOTEM_REPO_URL:?set TOTEM_REPO_URL to the official TOTEM repository (linked from arXiv:2402.16412)}"
: "${TOTEM_REF:?pin a commit or tag in TOTEM_REF so the fetch is reproducible}"

mkdir -p third_party
if [ ! -d third_party/TOTEM/.git ]; then
    git clone "$TOTEM_REPO_URL" third_party/TOTEM
fi
git -C third_party/TOTEM fetch --quiet origin
git -C third_party/TOTEM checkout --quiet "$TOTEM_REF"
echo "TOTEM at $(git -C third_party/TOTEM rev-parse --short HEAD). Read its LICENSE before reusing anything."
