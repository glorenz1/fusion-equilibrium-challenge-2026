#!/usr/bin/env python3
"""Compatibility entry point; prefer python -m fec1."""
from fec1.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
