"""Compatibility entry point; prefer python -m fec1.training.pca."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fec1.training.pca import fit_pca as fit_pca, main

__all__ = ["fit_pca", "main"]

if __name__ == "__main__":
    main()
