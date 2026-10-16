"""Shared constants and configuration for the Fusion Equilibrium Challenge scorer (metric v2).

The scoring contract (mirrors fusion-equilibrium-challenge-starter/validate_submission.py):
one .npz per machine config, keyed  shot_{i:04d}_{name}  in HF stream order, with
    shot_XXXX_psirz   (T, 65, 65)   poloidal flux map
    shot_XXXX_q95     (T,)          ) the ONLY two separately-predicted scalars -- the ones
    shot_XXXX_betaN   (T,)          ) not recoverable from psi(R,Z) (they need F(psi) / p(psi))

Everything else is DERIVED from the submitted flux map by the scorer itself (derive.py), with
the same functional applied to the ground-truth flux, so a scalar can only be earned by a psi
that actually implies it (no independent scalar head to tune -- the v1 gameability fix):

    S = 0.55 * max(0, R2_psi)
      + 0.15 * max(0, R2_qb)              # mean pooled R2 of {q95, betaN} vs stored EFIT labels
      + 0.10 * (1 - min(1, D_LCFS))
      + 0.20 * Consistency                # mean_j max(0, R2_cons_j) over the seven derived
                                          # scalars, R2_cons_j pooled over valid frames of
                                          # f_j(psi_pred) vs f_j(psi_gt)

v3 (the consistency overhaul): the shipped corpus is now DIVERTED FRAMES ONLY on both machines,
so the per-scalar limited/diverted class mask that v2 baked into the bundle is gone -- every
scored frame is diverted by construction, and R2_psi / R2_qb / Consistency finally share one
frame population. `dsep` is no longer scored at all: on DIII-D the stored label is the a-file
X-point<->limiter clearance rather than a drsep (corr -0.041 against the drsep functional over
207,402 frames), and with the corpus restricted to diverted frames it carries no remaining role.
The only residual population difference is D_LCFS, which still skips frames whose *reference*
contour extraction failed. G_ratio = S_MAST / S_D3D.

v3.1 (sign invariance): DIII-D and MAST ship psi with OPPOSITE sign conventions (see AXIS_SIGN
below), which made G_ratio effectively unwinnable -- a geometrically perfect MAST prediction
expressed in DIII-D's convention scored R2_psi = -5.71, below the constant baseline. The scorer
now normalises the global flux SIGN of each machine's submission (PSI_SIGNS), so the challenge
tests cross-machine physics rather than a guess at an upstream storage convention. Amplitude is
untouched. Only R2_psi and the magnetic-axis derivation were ever sign-sensitive: D_LCFS and
five of the seven consistency scalars were already invariant, because lcfs.py recovers the same
boundary under either sign (measured Hausdorff exactly 0) and li depends only on |grad psi|.
"""
from __future__ import annotations

# The two scalars still predicted as values (submitted channels, scored as pooled R2 vs the
# stored EFIT labels). Order is the storage order of the bundle's `scal`/`smask` arrays.
SCALARS = ["q95", "betaN"]
N_SCALARS = len(SCALARS)

# The seven psi-derived consistency scalars, in the storage order of the bundle's `cons`/`cmask`
# arrays (matches derive.SCALAR_NAMES). `dsep` was the eighth in v2 and is dropped in v3 -- see
# the module docstring.
CONS_SCALARS = ["R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li"]
N_CONS = len(CONS_SCALARS)

# Native flux grid per machine (rows = Z, cols = R). Both machines ship a dense, finite 65x65.
GRID_SHAPE = (65, 65)

# Composite weights (METRIC_V2_PROPOSAL.md sec 1). psi stays dominant; the only genuinely
# independent scalars (q95, betaN) keep a real weight; boundary + interior geometry split the
# rest (D_LCFS and the shape part of Consistency deliberately overlap -- keep W_LCFS modest).
W_PSI, W_QB, W_LCFS, W_CONS = 0.55, 0.15, 0.10, 0.20

# Challenge-2 admissibility gate, on the full DIII-D composite (starter issue #7). The old
# prose-only "R2_psi > 0.6" gate bounded just the W_PSI term, leaving ~45% of the denominator
# free -- sandbagging DIII-D to the floor bought a ~3x G_ratio inflation. Gating the composite
# bounds the whole denominator (cap < 1.2x) and implies R2_psi >= 0.727. Enforced in
# metrics.g_ratio, so ineligible entries show g_ratio = 0 on the leaderboard.
CH2_GATE_D3D_S = 0.85

# The test configs, and their machine tag as stored in the parquet `source` column.
CONFIG_MACHINE = {
    "diii_d_public_test": "DIII-D",
    "mast_public_test": "MAST",
    "diii_d_private_test": "DIII-D",
    "mast_private_test": "MAST",
}

# Per-machine flux sign so the plasma magnetic axis is always a MAXIMUM of phi = SIGN * psi.
# This is the sign convention of each machine's stored GROUND TRUTH, measured on it: DIII-D's
# EFIT puts the axis at the psi MINIMUM (99.98% of 1,559,340 frames, 0 the other way), while
# EFIT++/FAIR-MAST puts it at the psi MAXIMUM (100% of MAST frames). The machines agree on
# current direction -- this is a provenance difference between two EFIT implementations, not
# physics. Submissions are NOT held to it; see PSI_SIGNS.
AXIS_SIGN = {"DIII-D": -1.0, "MAST": +1.0}

# Candidate global flux signs a submission may be written in. Because the two machines' stored
# psi have opposite conventions (above), a model that transfers correctly from DIII-D to MAST
# still lands sign-inverted, and scoring it raw gives R2_psi = -5.71 -- worse than predicting a
# constant. So the scorer is SIGN-INVARIANT: it scores each machine under both signs and keeps
# the better one, symmetrically on both machines. The choice is ONE bit per machine, made once
# over the whole fold and reported in scores.json as `<tag>_psi_sign`; it is never made per shot
# or per frame, so it cannot be farmed for noise. Amplitude is NOT normalised -- only the sign.
PSI_SIGNS = (1.0, -1.0)

# Codabench runtime layout (scoring_program metadata `command: python3 score.py`).
INPUT_DIR = "/app/input"
OUTPUT_DIR = "/app/output"
# Ground-truth reference bundles are resolved in this order (first that exists wins):
REF_SEARCH_PATHS = ["/app/input/ref", "/opt/fusion_ref"]

SCORING_VERSION = "3.2.0"
