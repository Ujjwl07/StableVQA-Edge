#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
if command -v sha256sum >/dev/null 2>&1; then
  sha256sum -c MODEL_MANIFEST.sha256
else
  shasum -a 256 -c MODEL_MANIFEST.sha256
fi
