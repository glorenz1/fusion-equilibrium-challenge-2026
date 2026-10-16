"""
Fusion Data Challenge Data Provider

This provider handles fusion plasma data from DIII-D and MAST in Parquet format.

The data includes:
- Target: Magnetic flux map (psirz) - 2D images over time
- Targets: EFIT scalar labels (efit_beta_n, efit_li, efit_q95, efit_r_axis, efit_z_axis)
  and the EFIT-derived x-point gap (magnetics_dsep) - one value per efit_times step
- Inputs: Coil currents (magnetics) and Thomson scattering measurements

Due to the multidimensional nature of the data, this provider uses custom graphers
for visualization of 2D/3D data, while scalar 1D signals (coil currents) can be
displayed in default graphs.

Supported Data Sources:
======================

DIII-D (General Atomics, San Diego):
  - Flux grid: 65x65
  - Coils: F1A-F9B (shaping), ECOILA (ohmic), bcoil (toroidal)

MAST (Mega Ampere Spherical Tokamak, UK):
  - Flux grid: 65x65 (corrected dataset; the upstream 65x129 grid interleaves 65 real
    R columns with 64 empty ones, which are dropped to recover the dense 65x65 grid).
    The flux grapher also filters any remaining all-NaN columns defensively.
  - Coils: P2L-P6U (poloidal), Solenoid, TF, Ip

Data Format:
  - Parquet files: parquet_data/d3d_shot_XXXXX.parquet or parquet_data/mast_shot_XXXXX.parquet
  - One row per shot (all data as nested arrays)
  - Columns: efit_times, efit_psirz, efit_grid_R/Z, efit_beta_n/li/q95/r_axis/z_axis,
    magnetics_time, magnetics_*, thomson_core_times, thomson_core_*

File: apps/fusion_data_challenge/fusion_data_provider.py
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

# Check for pandas (required for Parquet)
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None
    print("Warning: pandas not available. Install with: pip install pandas pyarrow")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None


# ===== CONSTANTS FOR DATA SOURCES =====

# Data source identifiers
SOURCE_DIII_D = "DIII-D"
SOURCE_MAST = "MAST"

# Data folder
PARQUET_FOLDER = "parquet_data"


# ===== SIGNAL DEFINITIONS (ORIGINAL NAMING CONVENTIONS) =====
# 
# Each tokamak keeps its original signal names. Signals are prefixed with the
# source name in the UI for clarity: "DIII-D: F1A" or "MAST: p2l_current"

# DIII-D signals (General Atomics, San Diego)
# Signal path -> Display name
DIII_D_SIGNALS = {
    # Main coils
    "magnetics/ECOILA": "ECOILA",      # Ohmic heating (central solenoid)
    "magnetics/bcoil": "bcoil",         # Toroidal field
    "magnetics/plasma_current": "Ip",   # Plasma current
    "magnetics/dsep": "dsep",           # X-point gap (>0 diverted, <0 limited)
    # Shaping coils (F1A-F9B = 18 coils)
    "magnetics/F1A": "F1A", "magnetics/F1B": "F1B",
    "magnetics/F2A": "F2A", "magnetics/F2B": "F2B",
    "magnetics/F3A": "F3A", "magnetics/F3B": "F3B",
    "magnetics/F4A": "F4A", "magnetics/F4B": "F4B",
    "magnetics/F5A": "F5A", "magnetics/F5B": "F5B",
    "magnetics/F6A": "F6A", "magnetics/F6B": "F6B",
    "magnetics/F7A": "F7A", "magnetics/F7B": "F7B",
    "magnetics/F8A": "F8A", "magnetics/F8B": "F8B",
    "magnetics/F9A": "F9A", "magnetics/F9B": "F9B",
}

# MAST signals (Culham Centre for Fusion Energy, UK)
# Signal path -> Display name
MAST_SIGNALS = {
    # Main coils
    "magnetics/plasma_current": "Ip",           # Plasma current
    "magnetics/tf_current": "TF",               # Toroidal field
    "magnetics/sol_current": "Solenoid",        # Central solenoid
    "magnetics/efps_current": "EFPS",           # Error field correction
    # Poloidal field coils (P2-P6, Lower and Upper = 10 coils)
    "magnetics/p2l_current": "P2L", "magnetics/p2u_current": "P2U",
    "magnetics/p3l_current": "P3L", "magnetics/p3u_current": "P3U",
    "magnetics/p4l_current": "P4L", "magnetics/p4u_current": "P4U",
    "magnetics/p5l_current": "P5L", "magnetics/p5u_current": "P5U",
    "magnetics/p6l_current": "P6L", "magnetics/p6u_current": "P6U",
    # X-point gap (>0 diverted, <0 limited) - parity with DIII-D
    "magnetics/dsep": "dsep",
}

# Per-signal units.
#
# Coil currents are harmonized across the two machines: every POLOIDAL SHAPING coil -- DIII-D's
# F1A..F9B and MAST's P2..P6 -- is in kiloampere-turns, so the two machines' shaping coils are
# directly comparable. DIII-D's F-coil turn counts (58 or 55) are folded into the values at build
# time; MAST's P-coils are natively kA*turn upstream. Plasma current is kA on both machines.
#
# Four channels stay in plain kA rather than kA*turn because no trustworthy turn count exists for
# them: DIII-D `ECOILA` (EFIT models that group as 48 single-turn elements and ECOILB is a second
# co-located group not shipped, so its turn convention is ambiguous) and `bcoil`, and MAST's
# `sol_current` / `tf_current` (toroidal coils have no poloidal-geometry entry). `efps_current` is
# not a coil at all. Anything not listed defaults to kA*turn.
_KA = "kA"
SIGNAL_UNITS = {
    "magnetics/dsep": "m",
    # DIII-D -- not kA*turn
    "magnetics/ECOILA": _KA,
    "magnetics/bcoil": _KA,
    "magnetics/plasma_current": _KA,
    # MAST -- not kA*turn
    "magnetics/sol_current": _KA,
    "magnetics/tf_current": _KA,
    "magnetics/efps_current": _KA,
}


def _get_signal_display_name(internal_path: str, source: str) -> str:
    """Get display name with source prefix for a signal."""
    if source == SOURCE_DIII_D and internal_path in DIII_D_SIGNALS:
        return f"DIII-D: {DIII_D_SIGNALS[internal_path]}"
    elif source == SOURCE_MAST and internal_path in MAST_SIGNALS:
        return f"MAST: {MAST_SIGNALS[internal_path]}"
    # Fallback: return path with source prefix
    return f"{source}: {internal_path.split('/')[-1]}"

def _parse_signal_name(signal_name: str) -> tuple:
    """
    Parse a signal name to extract source and internal path.
    
    Args:
        signal_name: Display name like "DIII-D: F1A" or "MAST: P2L"
        
    Returns:
        (source, internal_path) or (None, signal_name) if not parseable
    """
    if signal_name.startswith("DIII-D: "):
        short_name = signal_name[8:]  # Remove "DIII-D: " prefix
        # Find internal path
        for path, name in DIII_D_SIGNALS.items():
            if name == short_name:
                return (SOURCE_DIII_D, path)
    elif signal_name.startswith("MAST: "):
        short_name = signal_name[6:]  # Remove "MAST: " prefix
        # Find internal path
        for path, name in MAST_SIGNALS.items():
            if name == short_name:
                return (SOURCE_MAST, path)
    
    # Not a prefixed name - check if it's a raw path
    if signal_name in DIII_D_SIGNALS:
        return (SOURCE_DIII_D, signal_name)
    if signal_name in MAST_SIGNALS:
        return (SOURCE_MAST, signal_name)
    
    return (None, signal_name)


# ===== PARQUET DATA LOADING =====

def _find_parquet_file(data_folder: Path, shot_number: str, source: Optional[str] = None) -> Optional[Path]:
    """Find Parquet file for a shot. Files are now prefixed with source: d3d_shot_*.parquet or mast_shot_*.parquet."""
    parquet_dir = Path(data_folder) / PARQUET_FOLDER
    
    # Try with source prefix first (new naming convention)
    if source == SOURCE_DIII_D:
        parquet_path = parquet_dir / f"d3d_shot_{shot_number}.parquet"
        if parquet_path.exists():
            return parquet_path
    elif source == SOURCE_MAST:
        parquet_path = parquet_dir / f"mast_shot_{shot_number}.parquet"
        if parquet_path.exists():
            return parquet_path
    
    # Try both prefixes if source not specified (for backward compatibility)
    for prefix in ["d3d_", "mast_", ""]:
        parquet_path = parquet_dir / f"{prefix}shot_{shot_number}.parquet"
        if parquet_path.exists():
            return parquet_path
    
    return None


def _load_parquet_shot(parquet_path: Path) -> Optional[pd.DataFrame]:
    """Load a Parquet shot file into a DataFrame."""
    if not PANDAS_AVAILABLE:
        return None
    try:
        return pd.read_parquet(parquet_path)
    except Exception:
        return None


def _detect_source_from_parquet(df: pd.DataFrame) -> str:
    """Detect data source from Parquet DataFrame."""
    if 'source' in df.columns and len(df) > 0:
        return str(df['source'].iloc[0])
    # Check for MAST-specific columns
    if 'magnetics_sol_current' in df.columns or 'magnetics_p2l_current' in df.columns:
        return SOURCE_MAST
    # Check for DIII-D columns
    if 'magnetics_ECOILA' in df.columns or 'magnetics_F1A' in df.columns:
        return SOURCE_DIII_D
    return SOURCE_DIII_D  # Default


def _get_parquet_magnetics_columns(df: pd.DataFrame) -> List[str]:
    """Get list of magnetics column names from Parquet DataFrame."""
    return [col for col in df.columns if col.startswith('magnetics_')]


class ParquetDataWrapper:
    """
    Wrapper class to provide a consistent interface for Parquet data.
    
    NEW FORMAT (v2.0): Single row per shot with nested arrays:
      - efit_times: list[float] (EFIT timestamps)
      - efit_psirz: list[list[list[float]]] (flux maps: time x R x Z)
      - magnetics_time: list[float] (magnetics timestamps)
      - magnetics_*: list[float] (coil currents at original resolution)
      - thomson_core_times: list[float]
      - thomson_core_Te: list[list[float]] (profiles: time x spatial)
    """
    def __init__(self, df: pd.DataFrame, source: str):
        self.df = df
        self.source = source
        self._closed = False
        # Cache frequently accessed data
        self._efit_times = None
        self._magnetics_time = None
    
    def close(self):
        """Mark as closed (for API compatibility)."""
        self._closed = True
        self.df = None
    
    @property
    def is_parquet(self) -> bool:
        return True
    
    def get_efit_times(self) -> Optional[np.ndarray]:
        """Get EFIT time array."""
        if self._efit_times is not None:
            return self._efit_times
        if self.df is None:
            return None
        if 'efit_times' in self.df.columns:
            self._efit_times = np.array(self.df['efit_times'].iloc[0])
            return self._efit_times
        return None
    
    def get_magnetics_time(self, signal_name: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Get magnetics time array.
        
        Args:
            signal_name: Optional signal name for DIII-D (which has per-signal times)
        
        Returns:
            Time array for the signal (or shared time for MAST)
        """
        if self.df is None:
            return None
        
        # For DIII-D: check for signal-specific time array
        if signal_name:
            time_col = f'magnetics_{signal_name}_times'
            if time_col in self.df.columns:
                return np.array(self.df[time_col].iloc[0])
        
        # MAST or fallback: shared magnetics_time
        if self._magnetics_time is not None:
            return self._magnetics_time
        if 'magnetics_time' in self.df.columns:
            self._magnetics_time = np.array(self.df['magnetics_time'].iloc[0])
            return self._magnetics_time
        
        # Last resort: try to find any magnetics time array
        for col in self.df.columns:
            if col.endswith('_times') and col.startswith('magnetics_'):
                return np.array(self.df[col].iloc[0])
        
        return None
    
    def get_times(self) -> Optional[np.ndarray]:
        """Get time array (EFIT times for flux data)."""
        return self.get_efit_times()
    
    def get_flux_data(self, time_index: int = 0) -> Optional[np.ndarray]:
        """Get flux map at a specific EFIT time index."""
        if self.df is None or 'efit_psirz' not in self.df.columns:
            return None
        try:
            psirz_all = self.df['efit_psirz'].iloc[0]  # Single row with nested array
            n_times = len(psirz_all)
            time_index = max(0, min(time_index, n_times - 1))
            psirz = psirz_all[time_index]
            # Handle nested array structure - need to stack rows into 2D array
            if isinstance(psirz, (list, np.ndarray)) and len(psirz) > 0:
                if isinstance(psirz[0], (list, np.ndarray)):
                    return np.stack([np.array(row) for row in psirz])
            return np.array(psirz)
        except Exception:
            return None

    def get_magnetics_signal(self, signal_name: str) -> Optional[np.ndarray]:
        """Get a magnetics signal as time series (at original resolution)."""
        col_name = f'magnetics_{signal_name}'
        if self.df is None or col_name not in self.df.columns:
            return None
        try:
            return np.array(self.df[col_name].iloc[0])
        except Exception:
            return None
    
    def get_thomson_core_times(self) -> Optional[np.ndarray]:
        """Get core Thomson time array."""
        if self.df is None or 'thomson_core_times' not in self.df.columns:
            return None
        try:
            return np.array(self.df['thomson_core_times'].iloc[0])
        except Exception:
            return None
    
    def get_thomson_edge_times(self) -> Optional[np.ndarray]:
        """Get edge Thomson time array."""
        if self.df is None or 'thomson_edge_times' not in self.df.columns:
            return None
        try:
            return np.array(self.df['thomson_edge_times'].iloc[0])
        except Exception:
            return None
    
    def get_thomson_core(self, measurement: str = 'Te', time_index: int = 0) -> tuple:
        """Get core Thomson profile at a time index. Returns (profile, R_coords)."""
        if self.df is None:
            return None, None
        
        col_name = f'thomson_core_{measurement}'
        if col_name not in self.df.columns:
            col_name = f'thomson_core_{"Te" if measurement == "temperature" else "ne"}'
        
        if col_name not in self.df.columns:
            return None, None
        
        try:
            profiles = self.df[col_name].iloc[0]  # list of profiles
            n_times = len(profiles)
            time_index = max(0, min(time_index, n_times - 1))
            # Each profile is already a 1D array
            profile = np.asarray(profiles[time_index])
            
            r_coords = None
            if 'thomson_core_R' in self.df.columns:
                r_val = self.df['thomson_core_R'].iloc[0]
                if r_val is not None and len(r_val) > 0:
                    r_coords = np.asarray(r_val)
            return profile, r_coords
        except Exception:
            return None, None
    
    def get_thomson_edge(self, measurement: str = 'Te', time_index: int = 0) -> tuple:
        """Get edge Thomson profile at a time index. Returns (profile, spatial_coords)."""
        if self.df is None:
            return None, None
        
        col_name = f'thomson_edge_{measurement}'
        if col_name not in self.df.columns:
            col_name = f'thomson_edge_{"Te" if measurement == "temperature" else "ne"}'
        
        if col_name not in self.df.columns:
            return None, None
        
        try:
            profiles = self.df[col_name].iloc[0]  # list of profiles
            n_times = len(profiles)
            time_index = max(0, min(time_index, n_times - 1))
            # Each profile is already a 1D array
            profile = np.asarray(profiles[time_index])
            
            spatial = None
            if 'thomson_edge_spatial' in self.df.columns:
                s_val = self.df['thomson_edge_spatial'].iloc[0]
                if s_val is not None and len(s_val) > 0:
                    spatial = np.asarray(s_val)
            return profile, spatial
        except Exception:
            return None, None
    
    def get_all_thomson_core(self, measurement: str = 'Te') -> tuple:
        """Get all core Thomson profiles. Returns (profiles_2d, R_coords, times)."""
        if self.df is None:
            return None, None, None
        
        col_name = f'thomson_core_{measurement}'
        if col_name not in self.df.columns:
            col_name = f'thomson_core_{"Te" if measurement == "temperature" else "ne"}'
        
        if col_name not in self.df.columns:
            return None, None, None
        
        try:
            # Get profiles from single row - list of profiles (each profile is an array)
            profiles_list = self.df[col_name].iloc[0]
            if profiles_list is None or len(profiles_list) == 0:
                return None, None, None
            # Stack into 2D array (time, spatial) - np.array() doesn't work on nested arrays
            profiles = np.stack([np.array(p) for p in profiles_list])
            
            # Get R coordinates
            r_coords = None
            if 'thomson_core_R' in self.df.columns:
                r_val = self.df['thomson_core_R'].iloc[0]
                if r_val is not None and len(r_val) > 0:
                    r_coords = np.array(r_val)
            
            times = self.get_thomson_core_times()
            return profiles, r_coords, times
        except Exception:
            return None, None, None
    
    def get_all_thomson_edge(self, measurement: str = 'Te') -> tuple:
        """Get all edge Thomson profiles. Returns (profiles_2d, spatial_coords, times)."""
        if self.df is None:
            return None, None, None
        
        col_name = f'thomson_edge_{measurement}'
        if col_name not in self.df.columns:
            col_name = f'thomson_edge_{"Te" if measurement == "temperature" else "ne"}'
        
        if col_name not in self.df.columns:
            return None, None, None
        
        try:
            # Get profiles from single row - list of profiles (each profile is an array)
            profiles_list = self.df[col_name].iloc[0]
            if profiles_list is None or len(profiles_list) == 0:
                return None, None, None
            # Stack into 2D array (time, spatial) - np.array() doesn't work on nested arrays
            profiles = np.stack([np.array(p) for p in profiles_list])
            
            # Get spatial coordinates
            spatial = None
            if 'thomson_edge_spatial' in self.df.columns:
                s_val = self.df['thomson_edge_spatial'].iloc[0]
                if s_val is not None and len(s_val) > 0:
                    spatial = np.array(s_val)
            
            times = self.get_thomson_edge_times()
            return profiles, spatial, times
        except Exception:
            return None, None, None
    
    def get_dsep(self) -> tuple:
        """
        Get the dsep (x-point gap) signal and its time array.

        dsep > 0 means the plasma is diverted (an x-point lies inside the vessel).
        dsep < 0 means the plasma is limited (no x-point in the vessel).

        Returns (data, times) in meters and milliseconds, or (None, None) if missing.
        """
        if self.df is None or 'magnetics_dsep' not in self.df.columns:
            return None, None
        try:
            data = np.asarray(self.df['magnetics_dsep'].iloc[0])
            # Prefer a dsep-specific time array; otherwise fall back to shared magnetics_time
            if 'magnetics_dsep_times' in self.df.columns:
                times = np.asarray(self.df['magnetics_dsep_times'].iloc[0])
            elif 'magnetics_time' in self.df.columns:
                times = np.asarray(self.df['magnetics_time'].iloc[0])
            else:
                times = np.arange(len(data), dtype=float)
            return data, times
        except Exception:
            return None, None

    def get_efit_scalar(self, column: str) -> tuple:
        """
        Get an EFIT scalar target (one value per efit_times step) and its time array.

        These are the scored equilibrium summaries added in the corrected dataset:
        efit_beta_n, efit_li, efit_q95, efit_r_axis, efit_z_axis. Present in `train`
        and withheld on the test splits.

        Returns (values, efit_times), or (None, None) if the column is absent.
        """
        if self.df is None or column not in self.df.columns:
            return None, None
        try:
            values = np.asarray(self.df[column].iloc[0], dtype=float).ravel()
            times = self.get_efit_times()
            if times is None:
                times = np.arange(len(values), dtype=float)
            return values, times
        except Exception:
            return None, None

    def get_grid_coords(self, flux_shape: tuple) -> tuple:
        """
        Get R and Z coordinate arrays for the flux grid.
        Returns (r_coords, z_coords) in physical units (meters).
        
        Args:
            flux_shape: Shape of flux data (n_z, n_r) to ensure correct dimensions
        
        Returns:
            Tuple of (r_coords, z_coords) as numpy arrays
        """
        if self.df is None:
            # Fallback to normalized coords
            return np.linspace(0, 1, flux_shape[1]), np.linspace(0, 1, flux_shape[0])
        
        n_z, n_r = flux_shape
        
        # Check for stored grid coordinates (new format: efit_grid_R, efit_grid_Z)
        r_coords = None
        for col_name in ['efit_grid_R', 'grid_R']:
            if col_name in self.df.columns:
                grid_r = np.array(self.df[col_name].iloc[0])
                r_min, r_max = grid_r.min(), grid_r.max()
                r_coords = np.linspace(r_min, r_max, n_r)
                break
        if r_coords is None:
            r_coords = np.linspace(0, 1, n_r)
        
        z_coords = None
        for col_name in ['efit_grid_Z', 'grid_Z']:
            if col_name in self.df.columns:
                grid_z = np.array(self.df[col_name].iloc[0])
                z_min, z_max = grid_z.min(), grid_z.max()
                z_coords = np.linspace(z_min, z_max, n_z)
                break
        if z_coords is None:
            z_coords = np.linspace(0, 1, n_z)
        
        return r_coords, z_coords


# ===== REQUIRED API FUNCTIONS =====

def fetch_record_ids_for_dataset_id(data_folder: Path, dataset_id: Optional[str] = None) -> List[str]:
    """
    Return list of available shot numbers (record IDs) from Parquet files.
    
    Record IDs are prefixed to indicate source:
    - DIII-D shots: "DIII-D_" + numeric ID (e.g., "DIII-D_182494")
    - MAST shots: "MAST_" + numeric ID (e.g., "MAST_30110")
    """
    if not PANDAS_AVAILABLE:
        return []
    
    record_ids = []
    data_folder = Path(data_folder)
    
    try:
        parquet_path = data_folder / PARQUET_FOLDER
        if not parquet_path.exists():
            return []
        
        # Look for files with new naming convention (d3d_shot_*.parquet, mast_shot_*.parquet)
        for p in parquet_path.glob("d3d_shot_*.parquet"):
            shot_id = p.stem.replace("d3d_shot_", "")
            record_ids.append(f"DIII-D_{shot_id}")
        
        for p in parquet_path.glob("mast_shot_*.parquet"):
            shot_id = p.stem.replace("mast_shot_", "")
            record_ids.append(f"MAST_{shot_id}")
        
        # Also check for old naming convention (backward compatibility)
        for p in parquet_path.glob("shot_*.parquet"):
            # Skip if already processed (avoid duplicates)
            if "d3d_" in p.name or "mast_" in p.name:
                continue
            shot_id = p.stem.replace("shot_", "")
            # Detect source from file content
            try:
                df = pd.read_parquet(p, columns=['source'])
                source = df['source'].iloc[0] if 'source' in df.columns else None
                if source == SOURCE_MAST:
                    record_ids.append(f"MAST_{shot_id}")
                else:
                    record_ids.append(f"DIII-D_{shot_id}")
            except Exception:
                # Fallback: guess from shot number (MAST shots are typically < 50000)
                try:
                    if int(shot_id) < 50000:
                        record_ids.append(f"MAST_{shot_id}")
                    else:
                        record_ids.append(f"DIII-D_{shot_id}")
                except ValueError:
                    record_ids.append(f"DIII-D_{shot_id}")
        
        # Sort: DIII-D shots first numerically, then MAST shots numerically
        def sort_key(x):
            if x.startswith("MAST_"):
                num = x.replace("MAST_", "")
                return (1, int(num) if num.isdigit() else 0)
            elif x.startswith("DIII-D_"):
                num = x.replace("DIII-D_", "")
                return (0, int(num) if num.isdigit() else 0)
            else:
                return (2, 0)
        
        return sorted(record_ids, key=sort_key)
    except Exception:
        return []


def fetch_data(
    data_folder: Path,
    dataset_id: Optional[str],
    record_id: Optional[str],
    signals: Optional[List[str]],
    global_data_params: Dict[str, Any],
    data_trim_1: Optional[float] = None,
    data_trim_2: Optional[float] = None,
    calling_graph: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load fusion data from Parquet files.
    
    Behavior:
    - calling_graph=None: Returns 1D scalar signals for default graph display
    - calling_graph='grapher_name': Returns data wrapper for custom processing
    
    Args:
        data_folder: Path to data directory containing parquet_data/
        dataset_id: Dataset identifier (unused)
        record_id: Shot number (prefixed with "DIII-D_" or "MAST_")
        signals: Signals to load
        global_data_params: Global parameters (unused)
        data_trim_1: Start time trim (milliseconds)
        data_trim_2: End time trim (milliseconds)
        calling_graph: Name of the custom grapher calling this function
        
    Returns:
        Dictionary with standard signal structure or data wrapper for custom graphers
    """
    
    # Edge case: None record
    if record_id is None:
        return {"id": None, "signals": [], "errored_signals": []}
    
    if not PANDAS_AVAILABLE:
        return {
            "id": record_id,
            "signals": [],
            "errored_signals": signals or [],
            "error_message": "pandas not available. Install with: pip install pandas pyarrow"
        }
    
    # Parse record_id to get shot number and source
    if record_id.startswith("MAST_"):
        shot_number = record_id.replace("MAST_", "")
        source = SOURCE_MAST
    elif record_id.startswith("DIII-D_"):
        shot_number = record_id.replace("DIII-D_", "")
        source = SOURCE_DIII_D
    else:
        shot_number = record_id
        source = None
    
    data_folder = Path(data_folder)
    
    # Load from Parquet (with source hint for faster lookup)
    parquet_path = _find_parquet_file(data_folder, shot_number, source)
    if parquet_path is None:
        return {
            "id": record_id,
            "signals": [],
            "errored_signals": signals or [],
            "error_message": f"Parquet file not found: {PARQUET_FOLDER}/{('d3d_' if source == SOURCE_DIII_D else 'mast_' if source == SOURCE_MAST else '')}shot_{shot_number}.parquet"
        }
    
    df = _load_parquet_shot(parquet_path)
    if df is None:
        return {
            "id": record_id,
            "signals": [],
            "errored_signals": signals or [],
            "error_message": f"Failed to load Parquet file: {parquet_path}"
        }
    
    data_source = _detect_source_from_parquet(df)
    data_wrapper = ParquetDataWrapper(df, data_source)
    
    # Valid graphers for custom processing
    valid_graphers = {
        "magnetic_flux_contour",
        "thomson_scattering_core",
        "thomson_scattering_tangential",
        "diagnostics_overview",
        "dsep_xpoint_gap",
        "efit_scalars",
    }
    
    if calling_graph is not None and calling_graph in valid_graphers:
        return {
            "id": record_id,
            "data_wrapper": data_wrapper,
            "data_source": data_source,
            "trim_1": data_trim_1,
            "trim_2": data_trim_2,
            "signals": [],
            "errored_signals": []
        }
    
    # Default graph mode: Load scalar signals from Parquet
    data_wrapper.close()
    return _load_signals_from_parquet(df, data_source, record_id, signals, data_trim_1, data_trim_2)


def _load_signals_from_parquet(
    df: pd.DataFrame,
    data_source: str,
    record_id: str,
    signals: Optional[List[str]],
    data_trim_1: Optional[float],
    data_trim_2: Optional[float]
) -> Dict[str, Any]:
    """
    Load scalar signals from Parquet DataFrame.
    
    NEW FORMAT (v2.0): Single row with nested arrays.
    - MAST: magnetics_time (shared), magnetics_* (per signal)
    - DIII-D: magnetics_{signal}_times (per signal), magnetics_{signal} (per signal)
    """
    
    # Default signals
    if signals is None or len(signals) == 0:
        if data_source == SOURCE_DIII_D:
            signals = ["DIII-D: ECOILA"]
        else:
            signals = ["MAST: Solenoid"]
    
    data_array = []
    errored_signals = []
    
    # Get shared magnetics time array (MAST format)
    shared_times = None
    if 'magnetics_time' in df.columns:
        shared_times = np.array(df['magnetics_time'].iloc[0])
    
    for sig in signals:
        # Parse signal name
        sig_source, internal_path = _parse_signal_name(sig)
        
        # Check source match
        if sig_source is not None and sig_source != data_source:
            errored_signals.append(f"{sig} (wrong source: expected {data_source})")
            continue
        
        # Map internal path to Parquet column name
        # Internal path like "magnetics/ECOILA" -> column "magnetics_ECOILA"
        if internal_path.startswith("magnetics/"):
            signal_key = internal_path.split('/')[-1]
            col_name = f"magnetics_{signal_key}"
        else:
            signal_key = internal_path.replace("/", "_")
            col_name = signal_key
        
        if col_name not in df.columns:
            errored_signals.append(f"{sig} (not found)")
            continue
        
        # Get data from single row (new format: nested array)
        raw_data = df[col_name].iloc[0]
        data = np.array(raw_data)
        
        # Handle case where data might be a scalar (old format)
        if data.ndim == 0:
            data = df[col_name].values
        
        # Get time array for this signal
        # DIII-D: individual time arrays (magnetics_{signal}_times)
        # MAST: shared time array (magnetics_time)
        times_col = f"{col_name}_times"
        if times_col in df.columns:
            times = np.array(df[times_col].iloc[0])
        elif shared_times is not None:
            times = shared_times
        else:
            # Fallback: create synthetic time array
            times = np.arange(len(data))
        
        # Apply trimming
        if data_trim_1 is not None or data_trim_2 is not None:
            t1 = data_trim_1 if data_trim_1 is not None else times[0]
            t2 = data_trim_2 if data_trim_2 is not None else times[-1]
            time_mask = (times >= t1) & (times <= t2)
            data = data[time_mask]
            signal_times = times[time_mask]
        else:
            signal_times = times
        
        # Get display name
        display_name = _get_signal_display_name(internal_path, data_source)

        data_unit = SIGNAL_UNITS.get(internal_path, "kA*turn")
        record = {
            "data": data,
            "data_name": display_name,
            "times": signal_times,
            "units": {"data": data_unit, "times": "ms"},
            "dims": ("times",),
        }
        data_array.append(record)
    
    return {
        "id": record_id,
        "signals": data_array,
        "errored_signals": errored_signals
    }


def _get_all_signal_display_names() -> List[str]:
    """
    Return list of all available signal display names from both sources.
    
    Returns names in format "DIII-D: F1A" or "MAST: P2L" for clear source identification.
    """
    signals = []
    
    # Add DIII-D signals
    for path, name in DIII_D_SIGNALS.items():
        signals.append(f"DIII-D: {name}")
    
    # Add MAST signals
    for path, name in MAST_SIGNALS.items():
        signals.append(f"MAST: {name}")
    
    return sorted(signals)


# ===== CUSTOM GRAPHERS =====

def _get_theme_colors(theme_value: str) -> Dict[str, str]:
    """Get theme-specific colors matching dFL themes."""
    THEME_COLORS = {
        'sophelio_dark': {
            'background': '#000000',
            'text_primary': '#FFFFFF',
            'text_secondary': 'rgba(255, 255, 255, 0.65)',
            'border': '#569CD6',
            'plot_bgcolor': '#000000'
        },
        'light': {
            'background': '#f1f3f5',
            'text_primary': 'rgba(33, 37, 41, 0.85)',
            'text_secondary': 'rgba(33, 37, 41, 0.65)',
            'border': 'rgba(0, 0, 0, 0.1)',
            'plot_bgcolor': '#f1f3f5'
        },
        'dark': {
            'background': '#222327',
            'text_primary': 'rgba(255, 255, 255, 0.85)',
            'text_secondary': 'rgba(255, 255, 255, 0.65)',
            'border': 'rgba(255, 255, 255, 0.1)',
            'plot_bgcolor': '#222327'
        },
        'default': {
            'background': '#222327',
            'text_primary': 'rgba(255, 255, 255, 0.85)',
            'text_secondary': 'rgba(255, 255, 255, 0.65)',
            'border': 'rgba(255, 255, 255, 0.1)',
            'plot_bgcolor': '#222327'
        },
        'refined_gray': {
            'background': '#3a3b42',
            'text_primary': 'rgba(255, 255, 255, 0.85)',
            'text_secondary': 'rgba(255, 255, 255, 0.65)',
            'border': 'rgba(255, 255, 255, 0.15)',
            'plot_bgcolor': '#3a3b42'
        },
        'forest': {
            'background': '#1a2721',
            'text_primary': 'rgba(240, 248, 240, 0.9)',
            'text_secondary': 'rgba(240, 248, 240, 0.7)',
            'border': 'rgba(240, 248, 240, 0.15)',
            'plot_bgcolor': '#1a2721'
        },
        'publication': {
            'background': '#FFFFFF',
            'text_primary': '#000000',
            'text_secondary': '#333333',
            'border': '#CCCCCC',
            'plot_bgcolor': '#FFFFFF'
        }
    }
    return THEME_COLORS.get(theme_value, THEME_COLORS['light'])


def safe_param_str(parameters: Dict[str, Any], key: str, default: str) -> str:
    """Safely extract string parameter with default."""
    value = parameters.get(key, default)
    if value in (None, '', []):
        return default
    return str(value)


def safe_param_float(parameters: Dict[str, Any], key: str, default: float) -> float:
    """Safely extract float parameter with default."""
    value = parameters.get(key, default)
    if value in (None, '', []):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_param_int(parameters: Dict[str, Any], key: str, default: int) -> int:
    """Safely extract int parameter with default."""
    value = parameters.get(key, default)
    if value in (None, '', []):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def grapher_magnetic_flux_contour(app_control_parameters: Dict[str, Any], parameters: Dict[str, Any]):
    """
    Plot magnetic flux (psirz) as contour plot at a specific time.
    
    This grapher displays the poloidal magnetic flux distribution (psirz) in the
    R-Z plane at a selected time index. This shows the magnetic field geometry
    of the plasma - the target that models are trying to predict.
    
    Uses Parquet data format.
    """
    if not PLOTLY_AVAILABLE:
        return "Plotly not installed. Install with: pip install plotly"
    
    # Extract parameters
    time_index = safe_param_int(parameters, 'magnetic_flux_contour_time_index', 0)
    num_contours = safe_param_int(parameters, 'magnetic_flux_contour_num_contours', 30)
    
    # Get theme colors
    theme_value = app_control_parameters.get('theme_value', 'light')
    theme_colors = _get_theme_colors(theme_value)
    
    # Extract app control parameters
    dc = app_control_parameters.get("data_coordinator")
    record_id = app_control_parameters.get("record_id")
    
    if dc is None or record_id is None:
        return "Error: No data coordinator or record ID available"
    
    # Fetch data
    try:
        data = dc.fetch_data(
            dc.data_folder,
            None,
            record_id,
            None,
            {},
            None,
            None,
            calling_graph="magnetic_flux_contour"
        )
    except TypeError:
        data = dc.fetch_data(dc.data_folder, None, record_id, None, {}, None, None)
    
    if data.get("error_message"):
        return f"Error: {data['error_message']}"
    
    # Load from Parquet via data wrapper
    data_wrapper = data.get("data_wrapper")
    if data_wrapper is None:
        return "Error: No data wrapper available"
    
    # Get data source for MAST-specific handling
    data_source = data.get("data_source", SOURCE_DIII_D)
    
    try:
        flux_data = data_wrapper.get_flux_data(time_index)
        psirz_times = data_wrapper.get_times()
        
        if flux_data is None or psirz_times is None:
            data_wrapper.close()
            return "Error: Could not load flux data"
        
        time_index = max(0, min(time_index, len(psirz_times) - 1))
        flux_data = data_wrapper.get_flux_data(time_index)
        selected_time = psirz_times[time_index]
        
        # Get physical grid coordinates (uses stored grid_R/grid_Z if available)
        r_coords, z_coords = data_wrapper.get_grid_coords(flux_data.shape)
        
        # For MAST, keep only columns with ALL valid data (no NaNs)
        # MAST data has scattered valid columns due to grid conversion
        if data_source == SOURCE_MAST and np.any(np.isnan(flux_data)):
            # Find columns where ALL values are valid (no NaNs in any row)
            valid_cols_mask = ~np.any(np.isnan(flux_data), axis=0)
            if np.any(valid_cols_mask):
                # Extract only valid columns (not a contiguous slice)
                flux_data = flux_data[:, valid_cols_mask]
                r_coords = r_coords[valid_cols_mask]
        
        data_wrapper.close()
        
    except Exception as e:
        return f"Error loading data: {str(e)}"
    
    # Create plot
    fig = go.Figure()
    
    flux_min = np.nanmin(flux_data)
    flux_max = np.nanmax(flux_data)
    flux_range = flux_max - flux_min
    
    # Protect against edge cases
    if not np.isfinite(flux_range) or flux_range <= 0:
        flux_range = 1.0
        flux_min = 0.0
        flux_max = 1.0
    
    fig.add_trace(go.Contour(
        x=r_coords,
        y=z_coords,
        z=flux_data,
        colorscale='rdbu',
        contours=dict(
            start=flux_min,
            end=flux_max,
            size=flux_range / num_contours,
            showlabels=True
        ),
        colorbar=dict(
            title=dict(
                text='Flux<br>(V s / rad)',
                side='right'
            )
        ),
        hovertemplate='R: %{x:.3f} m<br>Z: %{y:.3f} m<br>Flux: %{z:.2e}<extra></extra>'
    ))
    
    # Calculate graph dimensions based on actual data aspect ratio
    r_range = r_coords[-1] - r_coords[0]
    z_range = z_coords[-1] - z_coords[0]
    aspect_ratio = r_range / z_range if z_range > 0 else 1.0
    
    # Base height, width scales with aspect ratio
    base_height = 800
    graph_width = int(base_height * aspect_ratio) + 150  # +150 for colorbar
    
    # Apply theme
    fig.update_layout(
        paper_bgcolor=theme_colors['background'],
        plot_bgcolor=theme_colors['plot_bgcolor'],
        font=dict(
            color=theme_colors['text_primary'],
            size=12,
            family="Cascadia Mono, Cascadia Mono PL, monospace"
        ),
        xaxis=dict(
            title='Major Radius R (m)',
            title_font=dict(color=theme_colors['text_primary']),
            color=theme_colors['text_primary'],
            gridcolor=theme_colors['border'],
            scaleanchor='y',
            scaleratio=1
        ),
        yaxis=dict(
            title='Vertical Position Z (m)',
            title_font=dict(color=theme_colors['text_primary']),
            color=theme_colors['text_primary'],
            gridcolor=theme_colors['border']
        ),
        title=dict(
            text=f'Shot {record_id}: Magnetic Flux at t={selected_time:.1f} ms (index {time_index})',
            font=dict(size=16, color=theme_colors['text_primary'])
        ),
        height=base_height,
        width=graph_width
    )
    
    return fig


def grapher_thomson_scattering_core(app_control_parameters: Dict[str, Any], parameters: Dict[str, Any]):
    """
    Plot Thomson scattering core system profiles as heatmap (temperature and density vs radius over time).
    
    This grapher displays the core vertical Thomson scattering measurements as a heatmap,
    showing electron temperature and density as functions of radial position and time.
    """
    if not PLOTLY_AVAILABLE:
        return "Plotly not installed. Install with: pip install plotly"
    
    # Extract parameters
    measurement_param = safe_param_str(parameters, 'thomson_scattering_core_measurement', 'temperature')
    colorscale_param = safe_param_str(parameters, 'thomson_scattering_core_colorscale', 'viridis')
    
    # Map display values to measurement type keys (UI may pass display values)
    measurement_map = {
        'Temperature (Te)': 'temperature',
        'temperature': 'temperature',
        'Density (ne)': 'density',
        'density': 'density',
    }
    measurement_type = measurement_map.get(measurement_param, measurement_param.lower() if measurement_param else 'temperature')
    
    # Map display values to colorscale names
    colorscale_map = {
        'Viridis': 'viridis',
        'Plasma': 'plasma',
        'Inferno': 'inferno',
        'Red-Blue': 'rdbu',
        'Jet': 'jet'
    }
    colorscale = colorscale_map.get(colorscale_param, colorscale_param.lower())
    
    # Get theme colors
    theme_value = app_control_parameters.get('theme_value', 'light')
    theme_colors = _get_theme_colors(theme_value)
    
    # Extract app control parameters
    dc = app_control_parameters.get("data_coordinator")
    record_id = app_control_parameters.get("record_id")
    
    if dc is None or record_id is None:
        return "Error: No data coordinator or record ID available"
    
    # Fetch data
    try:
        data = dc.fetch_data(
            dc.data_folder,
            None,
            record_id,
            None,
            {},
            None,
            None,
            calling_graph="thomson_scattering_core"
        )
    except TypeError:
        data = dc.fetch_data(dc.data_folder, None, record_id, None, {}, None, None)
    
    if data.get("error_message"):
        return f"Error: {data['error_message']}"
    
    # Get data source for correct axis labeling
    data_source = data.get("data_source", SOURCE_DIII_D)
    
    # Set labels based on measurement type
    units = "eV" if measurement_type == 'temperature' else "m^-3"
    label = "Temperature" if measurement_type == 'temperature' else "Density"
    
    # Load from Parquet via data wrapper
    data_wrapper = data.get("data_wrapper")
    if data_wrapper is None:
        return "Error: No data wrapper available"
    
    try:
        measure_key = 'Te' if measurement_type == 'temperature' else 'ne'
        profile_data, r_coords, times = data_wrapper.get_all_thomson_core(measure_key)
        
        if profile_data is None:
            data_wrapper.close()
            return f"Error: Could not load {measurement_type} data"
        
        # Parquet returns (time, spatial), need (spatial, time) for heatmap
        heatmap_data = profile_data.T
        num_spatial, num_time = heatmap_data.shape
        
        data_wrapper.close()
        
        # Create time array if not available
        if times is None:
            times = np.arange(num_time, dtype=float)
        else:
            times = np.array(times)
            if len(times) != num_time:
                times = times[:num_time] if len(times) > num_time else np.pad(times, (0, num_time - len(times)), mode='edge')
        
        # Handle spatial coordinates
        # DIII-D core Thomson uses a vertical laser - all channels at same R (~1.94m)
        # but different Z positions (which we don't have). Use channel index instead.
        # MAST has proper R coordinates that vary per channel.
        use_channel_index = False
        spatial_coords = None
        spatial_label = "Radial Position R (m)"  # Default for MAST
        
        if r_coords is not None:
            r_coords = np.array(r_coords)
            if len(r_coords.shape) == 2:
                r_coords = r_coords[0, :]
            if len(r_coords) != num_spatial:
                r_coords = r_coords[:num_spatial] if len(r_coords) > num_spatial else np.pad(r_coords, (0, num_spatial - len(r_coords)), mode='edge')
            
            # Check if R coordinates have meaningful variation
            valid_r = r_coords[~np.isnan(r_coords)]
            if len(valid_r) > 0 and len(np.unique(valid_r)) > 1:
                spatial_coords = r_coords
            else:
                use_channel_index = True
        else:
            use_channel_index = True
        
        if use_channel_index:
            # DIII-D core system: use channel index since R is constant
            # The vertical laser means channels are at different Z positions
            spatial_coords = np.arange(num_spatial, dtype=float)
            if data_source == SOURCE_DIII_D:
                spatial_label = "Channel (vertical Z position)"
            else:
                spatial_label = "Channel Index"
        
        # Check for valid data
        valid_data = np.isfinite(heatmap_data)
        if not np.any(valid_data):
                return f"Warning: No valid {measurement_type} data found (all NaN or Inf)"
        
        # Get valid data subset for calculations
        valid_values = heatmap_data[valid_data]
        
        # Check if all values are zero
        if np.all(valid_values == 0):
            return f"Warning: All {measurement_type} values are zero. Data shape: {heatmap_data.shape}"
        
        # Calculate color scale bounds with overflow protection
        with np.errstate(all='ignore'):
            zmin = float(np.nanmin(valid_values))
            zmax = float(np.nanmax(valid_values))
            zmean = float(np.nanmean(valid_values))
            zstd = float(np.nanstd(valid_values, ddof=1)) if len(valid_values) > 1 else 0.0
        
        # Handle case where all values are the same
        if zmax == zmin:
            zmin = zmean - 0.01 * abs(zmean) if zmean != 0 else -0.01
            zmax = zmean + 0.01 * abs(zmean) if zmean != 0 else 0.01
        elif zstd > 0 and (zmax - zmin) / (abs(zmean) + 1e-10) < 0.1:
            z_range = max(abs(zmax - zmean), abs(zmin - zmean), zstd * 3)
            zmin = zmean - z_range
            zmax = zmean + z_range
            if colorscale not in ['rdbu', 'rdylbu', 'spectral']:
                colorscale = 'rdbu'
        
    except Exception as e:
        return f"Error loading data: {str(e)}"
    
    # Create heatmap
    fig = go.Figure()
    
    # Determine hover template based on coordinate type
    if use_channel_index:
        hover_template = 'Time: %{x:.1f} ms<br>Channel: %{y:.0f}<br>Value: %{z:.4g}<extra></extra>'
    else:
        hover_template = 'Time: %{x:.1f} ms<br>R: %{y:.3f} m<br>Value: %{z:.4g}<extra></extra>'
    
    fig.add_trace(go.Heatmap(
        x=times,
        y=spatial_coords,
        z=heatmap_data,
        colorscale=colorscale,
        colorbar=dict(
            title=dict(
                text=f'{label}<br>({units})' if units else label,
                side='right'
            )
        ),
        hovertemplate=hover_template,
        zmin=zmin,
        zmax=zmax
    ))
    
    # Apply theme
    fig.update_layout(
        paper_bgcolor=theme_colors['background'],
        plot_bgcolor=theme_colors['plot_bgcolor'],
        font=dict(
            color=theme_colors['text_primary'],
            size=12,
            family="Cascadia Mono, Cascadia Mono PL, monospace"
        ),
        xaxis=dict(
            title='Time (ms)',
            title_font=dict(color=theme_colors['text_primary']),
            color=theme_colors['text_primary'],
            gridcolor=theme_colors['border']
        ),
        yaxis=dict(
            title=spatial_label,
            title_font=dict(color=theme_colors['text_primary']),
            color=theme_colors['text_primary'],
            gridcolor=theme_colors['border']
        ),
        title=dict(
            text=f'Shot {record_id}: Core Thomson Scattering {label}',
            font=dict(size=16, color=theme_colors['text_primary'])
        ),
        height=700
    )
    
    return fig


def grapher_dsep_xpoint_gap(app_control_parameters: Dict[str, Any], parameters: Dict[str, Any]):
    """
    Plot dsep (x-point gap) versus time, with shading that distinguishes
    diverted (dsep > 0) from limited (dsep < 0) phases of the discharge.

    dsep is an EFIT-derived geometric quantity: the radial distance from
    the primary separatrix to the secondary separatrix at the outboard
    midplane. Diverted plasmas have an x-point inside the vessel; limited
    plasmas touch a material limiter and have no in-vessel x-point.

    Available for both DIII-D (signal: magnetics_dsep) and MAST (same
    column name, derived from esm/dr_sep_out in the upstream MAST API).
    """
    if not PLOTLY_AVAILABLE:
        return "Plotly not installed. Install with: pip install plotly"

    theme_value = app_control_parameters.get('theme_value', 'light')
    theme_colors = _get_theme_colors(theme_value)

    dc = app_control_parameters.get("data_coordinator")
    record_id = app_control_parameters.get("record_id")
    if dc is None or record_id is None:
        return "Error: No data coordinator or record ID available"

    try:
        data = dc.fetch_data(
            dc.data_folder, None, record_id, None, {}, None, None,
            calling_graph="dsep_xpoint_gap"
        )
    except TypeError:
        data = dc.fetch_data(dc.data_folder, None, record_id, None, {}, None, None)

    if data.get("error_message"):
        return f"Error: {data['error_message']}"

    data_wrapper = data.get("data_wrapper")
    if data_wrapper is None:
        return "Error: No data wrapper available"

    dsep, times = data_wrapper.get_dsep()
    data_wrapper.close()

    if dsep is None or times is None:
        return ("Error: dsep signal not present in this shot. "
                "Expected column magnetics_dsep (with optional magnetics_dsep_times).")

    # Drop NaN samples - dsep is only defined inside the plasma window
    finite = np.isfinite(dsep) & np.isfinite(times)
    if not np.any(finite):
        return "Warning: dsep is all NaN for this shot (no valid plasma window)."
    dsep = np.asarray(dsep, dtype=float)[finite]
    times = np.asarray(times, dtype=float)[finite]

    # MAST stores times in seconds, DIII-D stores them in ms. Infer from
    # the data so the x-axis label and tick magnitudes line up.
    time_unit = "s" if times.max() < 10.0 else "ms"

    diverted_frac = float(np.mean(dsep > 0))

    fig = go.Figure()

    # Shade diverted region (dsep > 0) and limited region (dsep < 0).
    # Use fill='tozeroy' against a y=0 baseline.
    diverted = np.where(dsep > 0, dsep, 0.0)
    limited = np.where(dsep < 0, dsep, 0.0)

    fig.add_trace(go.Scatter(
        x=times, y=diverted, mode='lines',
        line=dict(width=0),
        fill='tozeroy',
        fillcolor='rgba(46, 160, 67, 0.35)',
        name='Diverted (dsep > 0)',
        hoverinfo='skip',
        showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=times, y=limited, mode='lines',
        line=dict(width=0),
        fill='tozeroy',
        fillcolor='rgba(218, 54, 51, 0.35)',
        name='Limited (dsep < 0)',
        hoverinfo='skip',
        showlegend=True,
    ))

    # The dsep trace itself
    hover_t_fmt = '%{x:.4f}' if time_unit == 's' else '%{x:.1f}'
    fig.add_trace(go.Scatter(
        x=times, y=dsep, mode='lines',
        line=dict(color=theme_colors['text_primary'], width=1.5),
        name='dsep',
        hovertemplate=f't = {hover_t_fmt} {time_unit}<br>dsep = %{{y:.4f}} m<extra></extra>',
    ))

    # y=0 reference line marking the diverted/limited boundary
    fig.add_hline(y=0, line=dict(color=theme_colors['text_secondary'],
                                  width=1, dash='dash'))

    fig.update_layout(
        paper_bgcolor=theme_colors['background'],
        plot_bgcolor=theme_colors['plot_bgcolor'],
        font=dict(color=theme_colors['text_primary'], size=12,
                  family="Cascadia Mono, Cascadia Mono PL, monospace"),
        xaxis=dict(
            title=f'Time ({time_unit})',
            title_font=dict(color=theme_colors['text_primary']),
            color=theme_colors['text_primary'],
            gridcolor=theme_colors['border'],
        ),
        yaxis=dict(
            title='dsep (m)',
            title_font=dict(color=theme_colors['text_primary']),
            color=theme_colors['text_primary'],
            gridcolor=theme_colors['border'],
        ),
        title=dict(
            text=f"Shot {record_id}: x-point gap (dsep) — "
                 f"{100*diverted_frac:.0f}% of the discharge is diverted",
            font=dict(size=16, color=theme_colors['text_primary']),
        ),
        height=500,
        legend=dict(
            orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
            bgcolor='rgba(0,0,0,0)',
            font=dict(color=theme_colors['text_primary']),
        ),
    )

    return fig


def grapher_efit_scalars(app_control_parameters: Dict[str, Any], parameters: Dict[str, Any]):
    """
    Plot the scored EFIT scalar targets vs time.

    These are the equilibrium-summary labels added in the corrected dataset — normalised
    beta, internal inductance, edge safety factor, and the magnetic-axis R/Z position.
    They are EFIT-derived prediction targets (present in `train`, withheld on test),
    shown here on the demo shots for context.
    """
    if not PLOTLY_AVAILABLE:
        return "Plotly not installed. Install with: pip install plotly"

    theme_value = app_control_parameters.get('theme_value', 'light')
    theme_colors = _get_theme_colors(theme_value)

    dc = app_control_parameters.get("data_coordinator")
    record_id = app_control_parameters.get("record_id")
    if dc is None or record_id is None:
        return "Error: No data coordinator or record ID available"

    try:
        data = dc.fetch_data(
            dc.data_folder, None, record_id, None, {}, None, None,
            calling_graph="efit_scalars"
        )
    except TypeError:
        data = dc.fetch_data(dc.data_folder, None, record_id, None, {}, None, None)

    if data.get("error_message"):
        return f"Error: {data['error_message']}"

    data_wrapper = data.get("data_wrapper")
    if data_wrapper is None:
        return "Error: No data wrapper available"

    # (column, label, unit)
    scalars = [
        ("efit_beta_n", "β_N (normalised beta)", ""),
        ("efit_li", "ℓi (internal inductance)", ""),
        ("efit_q95", "q95 (safety factor)", ""),
        ("efit_r_axis", "R_axis (magnetic axis)", "m"),
        ("efit_z_axis", "Z_axis (magnetic axis)", "m"),
    ]
    available = []
    for col, label, unit in scalars:
        values, times = data_wrapper.get_efit_scalar(col)
        if values is not None and times is not None:
            available.append((label, unit, np.asarray(times, dtype=float), np.asarray(values, dtype=float)))
    data_wrapper.close()

    if not available:
        return ("Error: no EFIT scalar targets found in this shot. Expected columns like "
                "efit_beta_n, efit_li, efit_q95, efit_r_axis, efit_z_axis (present in train).")

    n = len(available)
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True,
                        subplot_titles=[a[0] for a in available], vertical_spacing=0.06)
    color = theme_colors['text_primary']
    time_unit = None
    for i, (label, unit, times, values) in enumerate(available, start=1):
        finite = np.isfinite(times) & np.isfinite(values)
        t, v = times[finite], values[finite]
        if time_unit is None and t.size:
            time_unit = "s" if t.max() < 10.0 else "ms"
        fig.add_trace(
            go.Scatter(x=t, y=v, mode='lines', line=dict(color=color, width=1.5),
                       name=label, showlegend=False,
                       hovertemplate='t=%{x:.1f}<br>%{y:.4g}<extra>' + label + '</extra>'),
            row=i, col=1,
        )
        ytitle = unit if unit else ""
        fig.update_yaxes(title_text=ytitle, row=i, col=1,
                         color=color, gridcolor=theme_colors['border'])
    fig.update_xaxes(title_text=f"Time ({time_unit or 'ms'})", row=n, col=1,
                     color=color, gridcolor=theme_colors['border'])
    fig.update_layout(
        paper_bgcolor=theme_colors['background'],
        plot_bgcolor=theme_colors['plot_bgcolor'],
        font=dict(color=color, size=12, family="Cascadia Mono, Cascadia Mono PL, monospace"),
        title=dict(text=f"Shot {record_id}: EFIT scalar targets",
                   font=dict(size=16, color=color)),
        height=max(500, 180 * n),
    )
    return fig


def grapher_thomson_scattering_tangential(app_control_parameters: Dict[str, Any], parameters: Dict[str, Any]):
    """
    Plot Thomson scattering tangential system profiles as heatmap (temperature and density vs vertical position over time).
    
    This grapher displays the edge tangential Thomson scattering measurements as a heatmap,
    showing electron temperature and density as functions of vertical position and time.
    """
    if not PLOTLY_AVAILABLE:
        return "Plotly not installed. Install with: pip install plotly"
    
    # Extract parameters
    measurement_param = safe_param_str(parameters, 'thomson_scattering_tangential_measurement', 'temperature')
    colorscale_param = safe_param_str(parameters, 'thomson_scattering_tangential_colorscale', 'viridis')
    
    # Map display values to measurement type keys (UI may pass display values)
    measurement_map = {
        'Temperature (Te)': 'temperature',
        'temperature': 'temperature',
        'Density (ne)': 'density',
        'density': 'density',
    }
    measurement_type = measurement_map.get(measurement_param, measurement_param.lower() if measurement_param else 'temperature')
    
    # Map display values to colorscale names
    colorscale_map = {
        'Viridis': 'viridis',
        'Plasma': 'plasma',
        'Inferno': 'inferno',
        'Red-Blue': 'rdbu',
        'Jet': 'jet'
    }
    colorscale = colorscale_map.get(colorscale_param, colorscale_param.lower())
    
    # Get theme colors
    theme_value = app_control_parameters.get('theme_value', 'light')
    theme_colors = _get_theme_colors(theme_value)
    
    # Extract app control parameters
    dc = app_control_parameters.get("data_coordinator")
    record_id = app_control_parameters.get("record_id")
    
    if dc is None or record_id is None:
        return "Error: No data coordinator or record ID available"
    
    # Fetch data
    try:
        data = dc.fetch_data(
            dc.data_folder,
            None,
            record_id,
            None,
            {},
            None,
            None,
            calling_graph="thomson_scattering_tangential"
        )
    except TypeError:
        data = dc.fetch_data(dc.data_folder, None, record_id, None, {}, None, None)
    
    if data.get("error_message"):
        return f"Error: {data['error_message']}"
    
    # Get data source
    data_source = data.get("data_source", SOURCE_DIII_D)
    
    # Set labels
    units = "eV" if measurement_type == 'temperature' else "m^-3"
    label = "Temperature" if measurement_type == 'temperature' else "Density"
    spatial_label = "Vertical Position Z (m)" if data_source == SOURCE_DIII_D else "Radial Position R (m)"
    
    # Load from Parquet via data wrapper
    data_wrapper = data.get("data_wrapper")
    if data_wrapper is None:
        return "Error: No data wrapper available"
    
    try:
        measure_key = 'Te' if measurement_type == 'temperature' else 'ne'
        profile_data, tan_z, times = data_wrapper.get_all_thomson_edge(measure_key)
        
        if profile_data is None:
            data_wrapper.close()
            return f"Error: Could not load edge {measurement_type} data"
        
        # Parquet returns (time, spatial), need (spatial, time) for heatmap
        heatmap_data = profile_data.T
        num_spatial, num_time = heatmap_data.shape
        
        data_wrapper.close()
        
        # Create time array if not available
        if times is None:
            times = np.arange(num_time, dtype=float)
        else:
            times = np.array(times)
            if len(times) != num_time:
                times = times[:num_time] if len(times) > num_time else np.pad(times, (0, num_time - len(times)), mode='edge')
        
        # Handle spatial coordinates
        if tan_z is None:
            tan_z = np.arange(num_spatial, dtype=float)
        else:
            tan_z = np.array(tan_z)
            if len(tan_z.shape) == 2:
                tan_z = tan_z[0, :]
            if len(tan_z) != num_spatial:
                tan_z = tan_z[:num_spatial] if len(tan_z) > num_spatial else np.pad(tan_z, (0, num_spatial - len(tan_z)), mode='edge')
        
        # Check for valid data
        valid_data = np.isfinite(heatmap_data)
        if not np.any(valid_data):
            return f"Warning: No valid {measurement_type} data found (all NaN or Inf)"
        
        valid_values = heatmap_data[valid_data]
        
        if np.all(valid_values == 0):
            return f"Warning: All edge {measurement_type} values are zero"
        
        # Calculate color scale bounds
        with np.errstate(all='ignore'):
            zmin = float(np.nanmin(valid_values))
            zmax = float(np.nanmax(valid_values))
            zmean = float(np.nanmean(valid_values))
            zstd = float(np.nanstd(valid_values, ddof=1)) if len(valid_values) > 1 else 0.0
        
        if zmax == zmin:
            zmin = zmean - 0.01 * abs(zmean) if zmean != 0 else -0.01
            zmax = zmean + 0.01 * abs(zmean) if zmean != 0 else 0.01
        elif zstd > 0 and (zmax - zmin) / (abs(zmean) + 1e-10) < 0.1:
            z_range = max(abs(zmax - zmean), abs(zmin - zmean), zstd * 3)
            zmin = zmean - z_range
            zmax = zmean + z_range
            if colorscale not in ['rdbu', 'rdylbu', 'spectral']:
                colorscale = 'rdbu'
        
    except Exception as e:
        return f"Error loading data: {str(e)}"
    
    # Create heatmap
    fig = go.Figure()
    
    fig.add_trace(go.Heatmap(
        x=times,
        y=tan_z,
        z=heatmap_data,
        colorscale=colorscale,
        colorbar=dict(
            title=dict(
                text=f'{label}<br>({units})' if units else label,
                side='right'
            )
        ),
        hovertemplate='Time: %{x:.1f} ms<br>Z: %{y:.3f} m<br>Value: %{z:.6f}<extra></extra>',
        zmin=zmin,
        zmax=zmax
    ))
    
    # Apply theme
    fig.update_layout(
        paper_bgcolor=theme_colors['background'],
        plot_bgcolor=theme_colors['plot_bgcolor'],
        font=dict(
            color=theme_colors['text_primary'],
            size=12,
            family="Cascadia Mono, Cascadia Mono PL, monospace"
        ),
        xaxis=dict(
            title='Time (ms)',
            title_font=dict(color=theme_colors['text_primary']),
            color=theme_colors['text_primary'],
            gridcolor=theme_colors['border']
        ),
        yaxis=dict(
            title=spatial_label,
            title_font=dict(color=theme_colors['text_primary']),
            color=theme_colors['text_primary'],
            gridcolor=theme_colors['border']
        ),
        title=dict(
            text=f'Shot {record_id}: Edge Thomson Scattering {label}',
            font=dict(size=16, color=theme_colors['text_primary'])
        ),
        height=700
    )
    
    return fig

# ===== PROVIDER CONFIGURATION =====

def get_provider(_: Any) -> Dict[str, Any]:
    """
    Return provider configuration for fusion data challenge.
    
    This provider handles fusion plasma data from DIII-D and MAST where standard 1D signal
    representation is used for coil currents, while multidimensional data (flux maps,
    Thomson scattering profiles) requires custom graphers.
    
    Data sources:
    - DIII-D: General Atomics tokamak (San Diego, USA)
    - MAST: Mega Ampere Spherical Tokamak (Culham, UK)
    """
    
    # Set data folder path (parent folder containing parquet_data/)
    data_folder = Path(__file__).parent
    
    # Discover records
    record_ids = fetch_record_ids_for_dataset_id(data_folder)
    
    # Get all scalar (1D) signals that can be displayed in default graphs
    # Multidimensional signals (2D, 3D) require custom graphers
    # Signal names are prefixed with source: "DIII-D: F1A" or "MAST: P2L"
    all_possible_signals = _get_all_signal_display_names()
    
    # Define custom graphers
    custom_grapher_dictionary = {
        "magnetic_flux_contour": {
            "display_name": "Magnetic Flux Contour (Target)",
            "parameters": {
                "time_index": {
                    "display_name": "Time Index",
                    "default": 0,
                    "min": 0,
                    "max": 100  # Adjust based on actual data
                },
                "num_contours": {
                    "display_name": "Number of Contours",
                    "default": 30,
                    "min": 10,
                    "max": 100
                }
            },
            "function": grapher_magnetic_flux_contour
        },
        "thomson_scattering_core": {
            "display_name": "Thomson Scattering (Core)",
            "parameters": {
                "measurement": {
                    "display_name": "Measurement Type",
                    "default": "temperature",
                    "options": {
                        "temperature": "Temperature (Te)",
                        "density": "Density (ne)"
                    }
                },
                "colorscale": {
                    "display_name": "Color Scale",
                    "default": "viridis",
                    "options": {
                        "viridis": "Viridis",
                        "plasma": "Plasma",
                        "inferno": "Inferno",
                        "rdbu": "Red-Blue",
                        "jet": "Jet"
                    }
                }
            },
            "function": grapher_thomson_scattering_core
        },
        "dsep_xpoint_gap": {
            "display_name": "X-Point Gap (dsep)",
            "parameters": {},
            "function": grapher_dsep_xpoint_gap
        },
        "efit_scalars": {
            "display_name": "EFIT Scalar Targets",
            "parameters": {},
            "function": grapher_efit_scalars
        },
        "thomson_scattering_tangential": {
            "display_name": "Thomson Scattering (Tangential)",
            "parameters": {
                "measurement": {
                    "display_name": "Measurement Type",
                    "default": "temperature",
                    "options": {
                        "temperature": "Temperature (Te)",
                        "density": "Density (ne)"
                    }
                },
                "colorscale": {
                    "display_name": "Color Scale",
                    "default": "viridis",
                    "options": {
                        "viridis": "Viridis",
                        "plasma": "Plasma",
                        "inferno": "Inferno",
                        "rdbu": "Red-Blue",
                        "jet": "Jet"
                    }
                }
            },
            "function": grapher_thomson_scattering_tangential
        }
    }
    
    # Build provider configuration
    provider = {
        # Required keys
        "fetch_data": fetch_data,
        "dataset_id": "fusion_data_challenge_v1",
        "fetch_record_ids_for_dataset_id": fetch_record_ids_for_dataset_id,
        "all_possible_signals": all_possible_signals,
        "is_date": False,  # Numeric time in milliseconds
        "data_folder": data_folder,
        
        # Optional features
        "custom_smoothing_options": {},
        "custom_normalizing_options": {},
        "auto_label_function_dictionary": {},
        "all_labels": ["Event", "Disruption", "H-mode", "L-mode"],
        "custom_grapher_dictionary": custom_grapher_dictionary,
        "layout_options": {},
        "trim_data": None,
        
        # Hooks
        "label_export_hook": None,
        "data_export_hook": None,
        "manual_label_hook": None,
        
        # Defaults
        "default_trim_1": None,
        "default_trim_2": None,
    }
    
    return provider


# ===== TEST (run independently) =====
if __name__ == "__main__":
    provider = get_provider(None)
    print(f"Dataset ID: {provider['dataset_id']}")
    print(f"Data folder: {provider['data_folder']}")
    print(f"Number of scalar signals: {len(provider['all_possible_signals'])}")
    print(f"Sample signals: {provider['all_possible_signals'][:10]}")
    
    records = provider["fetch_record_ids_for_dataset_id"](provider["data_folder"])
    print(f"\nFound {len(records)} shots")
    if records:
        print(f"First shot: {records[0]}")
        print(f"Last shot: {records[-1]}")
        
        # Test 1: Load scalar signal with default graph (calling_graph=None)
        print("\n=== Test 1: Default graph mode (scalar signal) ===")
        data = provider["fetch_data"](
            provider["data_folder"],
            None,
            records[0],
            ["magnetics/ECOILA", "magnetics/F1A"],  # Scalar signals
            {},
            None,
            None,
            calling_graph=None
        )
        print(f"Signals loaded: {len(data.get('signals', []))}")
        if data.get('signals'):
            sig = data['signals'][0]
            print(f"  Signal: {sig['data_name']}")
            print(f"  Data shape: {sig['data'].shape}")
            print(f"  Times shape: {sig['times'].shape}")
            print(f"  Units: {sig.get('units', {})}")
        print(f"Errored signals: {data.get('errored_signals', [])}")
        
        # Test 2: Custom grapher mode
        print("\n=== Test 2: Custom grapher mode ===")
        data = provider["fetch_data"](
            provider["data_folder"],
            None,
            records[0],
            None,
            {},
            None,
            None,
            calling_graph="magnetic_flux_contour"
        )
        print(f"Data wrapper loaded: {data.get('data_wrapper') is not None}")
        if data.get('data_wrapper'):
            dw = data['data_wrapper']
            print(f"Data source: {data.get('data_source')}")
            print(f"Times available: {len(dw.get_times())}")
            flux = dw.get_flux_data(0)
            if flux is not None:
                print(f"Flux shape: {flux.shape}")
            dw.close()
