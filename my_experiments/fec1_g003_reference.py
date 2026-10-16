"""Identity of the existing 0.992817 FEC1 ensemble used for repair comparisons."""
from my_experiments.fec1_frozen_baseline import ROOT, _verify

DIRECTORY = ROOT / "my_experiments/candidates/fec1_geometry_aligned_096"
REPORT = DIRECTORY / "fec1_geometry_aligned_g003_blend.json"
PREDICTION = DIRECTORY / "fec1_geometry_aligned_g003_blend.npz"
BRANCH = DIRECTORY / "fec1_geometry_aligned_g003.npz"
REPORT_SHA256 = "bdc7177a071452191e3c6fdf8ad7fb7dc69a7ee876ceaff485609998201a2065"
PREDICTION_SHA256 = "00b110c4f8249973c2865dfdef989d7c615010b492d38d82f85095801a50d47c"


def verify_report():
    _verify(REPORT, REPORT, REPORT_SHA256)


def verify_prediction():
    _verify(PREDICTION, PREDICTION, PREDICTION_SHA256)
