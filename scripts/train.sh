#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec "${FEC1_PYTHON:-.venv/bin/python}" -m fec1 train "$@"
