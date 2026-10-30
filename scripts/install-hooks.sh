#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
git config --local core.hooksPath .husky
printf 'Installed pre-commit gate: staged backend + database + frontend tests + production build.\n'
