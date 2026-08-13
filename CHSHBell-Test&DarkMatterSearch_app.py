"""
🔬 CHSH Bell-Test & Dark Matter Search - Complete Pipeline
===========================================================
Supports VBI_Coincidence dataset format with proper phase extraction.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.fft import fft, fftfreq
from scipy.optimize import curve_fit
from scipy.stats import chi2, norm, pearsonr
from scipy.signal import correlate, correlation_lags
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
import sys
import os
from pathlib import Path
import json
import base64
from io import BytesIO

warnings.filterwarnings('ignore')

# ============================================================================
# CONSTANTS
# ============================================================================

HBAR = 6.582119569e-16  # eV·s
C = 2.99792458e8  # m/s
PI = np.pi

# ============================================================================
# VBI DATA FORMAT PARSER (Fixed)
# ============================================================================

def parse_vbi_data(file) -> pd.DataFrame:
    """
    Parse the VBI_Coincidence dataset format.
    
    The dataset has 29 columns:
    0: alpha (motorized stage position - Alice)
    1: beta (motorized stage position - Bob)  
    2: piezo_a (piezo position - Alice)
    3: piezo_b (piezo position - Bob)
    4-7: Single counts channels 1-4
    8: Unknown/trigger
    9: N_AB (coincidence channels 1&2 = A+,B+)
    10: N_CD (coincidence channels 3&4 = A-,B-)
    11-12: Other coincidence channels
    13-28: Additional data
    """
    try:
        # Read the space-separated file
        df = pd.read_csv(file, sep=r'\s+', header=None)
        
        # Extract relevant columns
        data = {
            'alpha': df[0],
            'beta': df[1],
            'piezo_a': df[2],
            'piezo_b': df[3],
            'singles_1': df[4],
            'singles_2': df[5],
            'singles_3': df[6],
            'singles_4': df[7],
            'N_AB': df[9],   # Coincidence A+B+ (channels 1&2)
            'N_CD': df[10],  # Coincidence A-B- (channels 3&4)
        }
        
        # Try to get cross-coincidences if available
        if df.shape[1] > 12:
            # Column 11 might be N_AC (A+B-)
            # Column 12 might be N_BD (A-B+)
            data['N_AC'] = df[11] if df.shape[1] > 11 else 0
            data['N_BD'] = df[12] if df.shape[1] > 12 else 0
        else:
            data['N_AC'] = 0
            data['N_BD'] = 0
        
        df_processed = pd.DataFrame(data)
        
        # Filter out rows with zero coincidences
        df_processed = df_processed[(df_processed['N_AB'] > 0) & (df_processed['N_CD'] > 0)]
        
        # The phase is encoded in the piezo positions
        # In this dataset, alpha and beta are fixed, piezo scans the phase
        # We'll use the piezo_b position as the phase indicator
        # But we need to handle the fact that the values are very close
        # We'll scale and shift to get meaningful phase values
        
        # For this dataset, the phase is proportional to the piezo position
        # We'll use the relative piezo position within the scan range
        piezo_min = df_processed['piezo_b'].min()
        piezo_max = df_processed['piezo_b'].max()
        piezo_range = piezo_max - piezo_min
        
        if piezo_range > 0:
            # Normalize piezo position to [0, 2*PI] phase range
            df_processed['phase'] = 2 * PI * (df_processed['piezo_b'] - piezo_min) / piezo_range
        else:
            # If no range, use the raw value
            df_processed['phase'] = df_processed['piezo_b']
        
        # Also compute the phase from piezo_a as a check
        piezo_a_min = df_processed['piezo_a'].min()
        piezo_a_max = df_processed['piezo_a'].max()
        piezo_a_range = piezo_a_max - piezo_a_min
        if piezo_a_range > 0:
            df_processed['phase_a'] = 2 * PI * (df_processed['piezo_a'] - piezo_a_min) / piezo_a_range
        else:
            df_processed['phase_a'] = df_processed['piezo_a']
        
        return df_processed
    except Exception as e:
        st.error(f"Error parsing VBI data: {str(e)}")
        return None


def compute_CHSH_from_vbi(df: pd.DataFrame) -> dict:
    """
    Compute CHSH S-parameter from VBI data.
    Uses the phase encoded in piezo positions to find the optimal settings.
    """
    if df is None or len(df) == 0:
        return {"error": "No data to analyze"}
    
    st.info(f"Analyzing {len(df)} data points")
    
    # Use the phase from piezo_b (or phase_a if phase_b is not varying)
    phase_col = 'phase' if df['phase'].nunique() > 1 else 'phase_a'
    df['phase_used'] = df[phase_col]
    
    # Group by phase to get unique phase settings
    # Use a small bin size to capture the variation
    n_bins = min(50, len(df) // 5)
    if n_bins < 4:
        n_bins = len(df)
    
    # Bin the data by phase
    phase_bins = np.linspace(df['phase_used'].min(), df['phase_used'].max(), n_bins + 1)
    df['phase_bin'] = np.digitize(df['phase_used'], phase_bins)
    
    # Average the data in each phase bin
    grouped = df.groupby('phase_bin').agg({
        'N_AB': 'mean',
        'N_CD': 'mean',
        'N_AC': 'mean',
        'N_BD': 'mean',
        'phase_used': 'mean',
        'piezo_a': 'mean',
        'piezo_b': 'mean',
        'alpha': 'first',
        'beta': 'first'
    }).reset_index()
    
    # Compute E for each phase setting
    results = []
    for _, row in grouped.iterrows():
        N_AB = max(0, row['N_AB'])
        N_CD = max(0, row['N_CD'])
        N_AC = max(0, row['N_AC']) if not np.isnan(row['N_AC']) else 0
        N_BD = max(0, row['N_BD']) if not np.isnan(row['N_BD']) else 0
        
        total = N_AB + N_CD + N_AC + N_BD
        if total < 1:
            continue
        
        E = (N_AB + N_CD - N_AC - N_BD) / total
        sigma_E = np.sqrt(N_AB + N_CD + N_AC + N_BD) / total
        
        results.append({
            'phase': row['phase_used'],
            'piezo_a': row['piezo_a'],
            'piezo_b': row['piezo_b'],
            'alpha': row['alpha'],
            'beta': row['beta'],
            'E': E,
            'sigma_E': sigma_E,
            'N_AB': N_AB,
            'N_CD': N_CD,
            'N_AC': N_AC,
            'N_BD': N_BD,
            'total': total
        })
    
    if len(results) < 4:
        return {"error": f"Need at least 4 phase settings, found {len(results)}"}
    
    df_results = pd.DataFrame(results)
    
    # Sort by phase
    df_results = df_results.sort_values('phase').reset_index(drop=True)
    
    # Find the 4 settings that maximize S
    # For CHSH, we need: (a,b), (a,b'), (a',b), (a',b')
    # In a swept-phase experiment, these correspond to 4 specific phase values
    
    # Use the standard method: find the extrema of E vs phase
    # The CHSH settings are at the peaks and valleys of the correlation
    E_values = df_results['E'].values
    phase_values = df_results['phase'].values
    sigma_values = df_results['sigma_E'].values
    n = len(E_values)
    
    # Find peaks and valleys using simple derivative
    peaks = []
    valleys = []
    
    for i in range(1, n - 1):
        if E_values[i] > E_values[i-1] and E_values[i] > E_values[i+1]:
            peaks.append(i)
        if E_values[i] < E_values[i-1] and E_values[i] < E_values[i+1]:
            valleys.append(i)
    
    # If we don't have enough peaks/valleys, use the extrema of the data
    if len(peaks) < 2 or len(valleys) < 2:
        # Find the global extrema
        max_idx = np.argmax(E_values)
        min_idx = np.argmin(E_values)
        
        # Find the next extrema by searching away from the global ones
        # For peaks: search in the remaining data
        remaining = [i for i in range(n) if i != max_idx and i != min_idx]
        if len(remaining) > 0:
            # Find the next max and min
            second_max_idx = max(remaining, key=lambda i: E_values[i])
            second_min_idx = min(remaining, key=lambda i: E_values[i])
        else:
            # If not enough points, use the edges
            second_max_idx = min(1, n-1)
            second_min_idx = max(1, n-2)
        
        # Use these as our peak/valley positions
        peaks = [max_idx, second_max_idx]
        valleys = [min_idx, second_min_idx]
    
    # Sort peaks and valleys by phase
    peaks = sorted(peaks)
    valleys = sorted(valleys)
    
    # Select the 4 settings
    # E(a,b) = first peak (max correlation)
    # E(a,b') = first valley (min correlation)
    # E(a',b) = second peak
    # E(a',b') = second valley
    selected_indices = []
    
    if len(peaks) >= 1:
        selected_indices.append(peaks[0])
    if len(valleys) >= 1:
        selected_indices.append(valleys[0])
    if len(peaks) >= 2:
        selected_indices.append(peaks[1])
    elif len(peaks) == 1:
        # If only one peak, use the other side
        idx = min(peaks[0] + n//4, n-1)
        selected_indices.append(idx)
    else:
        selected_indices.append(min(1, n-1))
    
    if len(valleys) >= 2:
        selected_indices.append(valleys[1])
    elif len(valleys) == 1:
        idx = max(valleys[0] - n//4, 0)
        selected_indices.append(idx)
    else:
        selected_indices.append(max(n-2, 0))
    
    # Ensure we have exactly 4 unique indices
    selected_indices = list(dict.fromkeys(selected_indices))
    while len(selected_indices) < 4:
        idx = (selected_indices[-1] + 1) % n
        if idx not in selected_indices:
            selected_indices.append(idx)
    
    selected_indices = sorted(selected_indices)[:4]
    
    # Get the selected rows
    selected = [df_results.iloc[idx] for idx in selected_indices]
    
    if len(selected) < 4:
        return {"error": f"Could not find 4 CHSH settings, found {len(selected)}"}
    
    # Compute S from the selected settings
    # Order: E(a,b), E(a,b'), E(a',b), E(a',b')
    E_ab = selected[0]['E']
    E_abp = selected[1]['E']
    E_apb = selected[2]['E']
    E_apbp = selected[3]['E']
    
    sig_ab = selected[0]['sigma_E']
    sig_abp = selected[1]['sigma_E']
    sig_apb = selected[2]['sigma_E']
    sig_apbp = selected[3]['sigma_E']
    
    S = E_ab - E_abp + E_apb + E_apbp
    sigma_S = np.sqrt(sig_ab**2 + sig_abp**2 + sig_apb**2 + sig_apbp**2)
    sigma_above = (abs(S) - 2.0) / sigma_S if sigma_S > 0 else 0
    
    return {
        "S": S,
        "sigma_S": sigma_S,
        "sigma_above_classical": sigma_above,
        "violates_classical_bound": abs(S) > 2.0,
        "within_tsirelson_bound": abs(S) <= 2 * np.sqrt(2) + 1e-9,
        "selected_settings": selected,
        "E_values": {
            "E(a,b)": (E_ab, sig_ab),
            "E(a,b')": (E_abp, sig_abp),
            "E(a',b)": (E_apb, sig_apb),
            "E(a',b')": (E_apbp, sig_apbp),
        },
        "all_results": df_results,
        "n_points": len(df_results),
        "peaks": peaks,
        "valleys": valleys
    }

# ============================================================================
# DATA LOADING & PROCESSING (for timestamp data)
# ============================================================================

def load_events(file) -> pd.DataFrame:
    """Load timestamped detection-event CSV."""
    try:
        df = pd.read_csv(file)
        required = {"timestamp_ns", "channel"}
        missing = required - set(df.columns)
        if missing:
            st.error(f"Events file missing required column(s): {missing}")
            return None
        df = df.sort_values("timestamp_ns").reset_index(drop=True)
        if df["timestamp_ns"].isna().any():
            st.error("Events file contains NaN timestamps")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading events: {str(e)}")
        return None


def load_settings(file) -> pd.DataFrame:
    """Load run/angle-setting log CSV."""
    try:
        df = pd.read_csv(file)
        required = {"run_id", "start_ns", "end_ns", "angle_a_deg", "angle_b_deg"}
        missing = required - set(df.columns)
        if missing:
            st.error(f"Settings file missing required column(s): {missing}")
            return None
        df["run_key"] = df["run_id"].astype(str).str.lower()
        return df
    except Exception as e:
        st.error(f"Error loading settings: {str(e)}")
        return None


def parse_channel_map(spec: str) -> dict:
    """Parse channel map string into dictionary."""
    mapping = {}
    try:
        for part in spec.split(","):
            ch_str, label = part.split(":")
            ch = int(ch_str.strip())
            label = label.strip()
            if label not in ("A+", "A-", "B+", "B-"):
                st.error(f"Channel label '{label}' invalid")
                return None
            mapping[ch] = label
        required_labels = {"A+", "A-", "B+", "B-"}
        if set(mapping.values()) != required_labels:
            st.error(f"Channel map must cover exactly {required_labels}")
            return None
        return mapping
    except Exception as e:
        st.error(f"Error parsing channel map: {str(e)}")
        return None


def generate_sample_data() -> tuple:
    """Generate synthetic data for testing."""
    np.random.seed(42)
    n_pairs = 20000
    window_ns = 2.0
    
    angles = {"ab": (0, 22.5), "abp": (0, 67.5), "apb": (45, 22.5), "apbp": (45, 67.5)}
    events_rows = []
    settings_rows = []
    t_cursor = 0.0
    channel_map = {1: "A+", 2: "A-", 3: "B+", 4: "B-"}
    inv_map = {v: k for k, v in channel_map.items()}

    for run_id, (a, b) in angles.items():
        run_duration_ns = n_pairs * 1000.0
        start_ns = t_cursor
        E_true = -np.cos(np.radians(2 * (a - b)))
        p_correlated = (1 + E_true) / 2
        pair_times = np.sort(np.random.uniform(0, run_duration_ns, n_pairs)) + start_ns
        
        for pt in pair_times:
            correlated = np.random.random() < p_correlated
            a_outcome = np.random.choice(["+", "-"])
            b_outcome = a_outcome if correlated else ("-" if a_outcome == "+" else "+")
            jitter_a = np.random.normal(0, 0.3)
            jitter_b = np.random.normal(0, 0.3)
            events_rows.append((pt + jitter_a, inv_map[f"A{a_outcome}"]))
            events_rows.append((pt + jitter_b, inv_map[f"B{b_outcome}"]))
        
        n_acc = int(n_pairs * 0.02)
        for _ in range(n_acc):
            ch = np.random.choice([1, 2, 3, 4])
            events_rows.append((start_ns + np.random.uniform(0, run_duration_ns), ch))
        
        end_ns = start_ns + run_duration_ns
        settings_rows.append((run_id, start_ns, end_ns, a, b))
        t_cursor = end_ns + 1e6

    events_df = pd.DataFrame(events_rows, columns=["timestamp_ns", "channel"])
    events_df = events_df.sort_values("timestamp_ns").reset_index(drop=True)
    settings_df = pd.DataFrame(settings_rows, columns=["run_id", "start_ns", "end_ns", "angle_a_deg", "angle_b_deg"])
    settings_df["run_key"] = settings_df["run_id"].astype(str).str.lower()
    return events_df, settings_df, channel_map

# ============================================================================
# COINCIDENCE COUNTING (for timestamp data)
# ============================================================================

@dataclass
class CoincidenceResult:
    window_ns: float
    counts: dict
    singles: dict
    accidental_rate: dict
    run_duration_s: float


def find_coincidences(times_a: np.ndarray, times_b: np.ndarray, window_ns: float) -> int:
    """Count coincidences between two timestamp arrays."""
    i, j = 0, 0
    n, m = len(times_a), len(times_b)
    count = 0
    while i < n and j < m:
        dt = times_a[i] - times_b[j]
        if abs(dt) <= window_ns:
            count += 1
            if times_a[i] <= times_b[j]:
                i += 1
            else:
                j += 1
        elif dt < -window_ns:
            i += 1
        else:
            j += 1
    return count


def estimate_accidentals(times_a: np.ndarray, times_b: np.ndarray, window_ns: float, duration_s: float) -> float:
    """Estimate accidental coincidences."""
    rate_a = len(times_a) / duration_s if duration_s > 0 else 0.0
    rate_b = len(times_b) / duration_s if duration_s > 0 else 0.0
    window_s = (2 * window_ns) * 1e-9
    return rate_a * rate_b * window_s * duration_s


def analyze_coincidences(events: pd.DataFrame, channel_map: dict, window_ns: float,
                          start_ns: float = None, end_ns: float = None) -> CoincidenceResult:
    """Run pairwise coincidence analysis."""
    df = events.copy()
    if start_ns is not None:
        df = df[df["timestamp_ns"] >= start_ns]
    if end_ns is not None:
        df = df[df["timestamp_ns"] <= end_ns]
    if len(df) == 0:
        raise ValueError("No events found in time range")

    duration_s = (df["timestamp_ns"].max() - df["timestamp_ns"].min()) * 1e-9
    df["label"] = df["channel"].map(channel_map)
    if df["label"].isna().any():
        unmapped = sorted(df.loc[df["label"].isna(), "channel"].unique())
        raise ValueError(f"Unmapped channels: {unmapped}")

    times = {lbl: df.loc[df["label"] == lbl, "timestamp_ns"].to_numpy() for lbl in ("A+", "A-", "B+", "B-")}
    singles = {lbl: len(t) for lbl, t in times.items()}

    counts = {}
    accidentals = {}
    for a_lbl in ("A+", "A-"):
        for b_lbl in ("B+", "B-"):
            c = find_coincidences(times[a_lbl], times[b_lbl], window_ns)
            counts[(a_lbl, b_lbl)] = c
            accidentals[(a_lbl, b_lbl)] = estimate_accidentals(times[a_lbl], times[b_lbl], window_ns, duration_s)

    return CoincidenceResult(window_ns=window_ns, counts=counts, singles=singles,
                              accidental_rate=accidentals, run_duration_s=duration_s)

# ============================================================================
# CHSH ANALYSIS (for timestamp data)
# ============================================================================

def correlation_E(coinc: CoincidenceResult, subtract_accidentals: bool = True) -> tuple:
    """Compute polarization correlation E(a,b)."""
    def net(key):
        raw = coinc.counts[key]
        acc = coinc.accidental_rate[key] if subtract_accidentals else 0.0
        return max(0.0, raw - acc)

    Npp = net(("A+", "B+"))
    Nmm = net(("A-", "B-"))
    Npm = net(("A+", "B-"))
    Nmp = net(("A-", "B+"))
    total = Npp + Nmm + Npm + Nmp
    if total == 0:
        return 0.0, 1.0, {"N++": 0, "N--": 0, "N+-": 0, "N-+": 0, "total": 0}

    E = (Npp + Nmm - Npm - Nmp) / total
    X, Y = Npp + Nmm, Npm + Nmp
    sigma_X = np.sqrt(max(X, 1e-9))
    sigma_Y = np.sqrt(max(Y, 1e-9))
    dEdX = 2 * Y / total**2
    dEdY = -2 * X / total**2
    sigma_E = np.sqrt((dEdX * sigma_X)**2 + (dEdY * sigma_Y)**2)

    return E, sigma_E, {"N++": Npp, "N--": Nmm, "N+-": Npm, "N-+": Nmp, "total": total}


def analyze_run(events: pd.DataFrame, settings: pd.DataFrame, channel_map: dict,
                 window_ns: float, subtract_accidentals: bool = True) -> dict:
    """Full CHSH analysis pipeline."""
    settings = settings.copy()
    if "run_key" not in settings.columns:
        settings["run_key"] = settings["run_id"].astype(str).str.lower()
    
    per_run = {}
    for _, row in settings.iterrows():
        key = row["run_key"]
        if key not in ["ab", "abp", "apb", "apbp"]:
            continue
        
        try:
            coinc = analyze_coincidences(events, channel_map, window_ns,
                                          start_ns=row["start_ns"], end_ns=row["end_ns"])
            E, sigE, raw = correlation_E(coinc, subtract_accidentals)
            
            per_run[key] = {
                "E": E, "sigma_E": sigE, "raw_counts": raw,
                "angle_a_deg": row["angle_a_deg"], "angle_b_deg": row["angle_b_deg"],
                "singles": coinc.singles, "duration_s": coinc.run_duration_s,
            }
        except Exception as e:
            st.warning(f"Error analyzing run {key}: {str(e)}")
            continue
    
    required = ["ab", "abp", "apb", "apbp"]
    missing = [r for r in required if r not in per_run]
    if missing:
        return {"error": f"Missing required runs: {missing}", "per_run": per_run, "chsh": None}
    
    S = per_run["ab"]["E"] - per_run["abp"]["E"] + per_run["apb"]["E"] + per_run["apbp"]["E"]
    sigma_S = np.sqrt(per_run["ab"]["sigma_E"]**2 + per_run["abp"]["sigma_E"]**2 + 
                      per_run["apb"]["sigma_E"]**2 + per_run["apbp"]["sigma_E"]**2)
    sigma_above = (abs(S) - 2.0) / sigma_S if sigma_S > 0 else float("inf")
    
    chsh = {
        "S": S, "sigma_S": sigma_S, "sigma_above_classical": sigma_above,
        "violates_classical_bound": abs(S) > 2.0,
        "within_tsirelson_bound": abs(S) <= 2 * np.sqrt(2) + 1e-9
    }
    
    return {"per_run": per_run, "chsh": chsh}

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_chsh_plot(results):
    """Create CHSH visualization."""
    if not results or "S" not in results:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name="S-Parameter",
        x=["CHSH S"],
        y=[results["S"]],
        error_y=dict(type='data', array=[results["sigma_S"]]),
        text=[f"{results['S']:.4f}"],
        textposition='auto'
    ))
    
    fig.add_hline(y=2.0, line_dash="dash", line_color="red", 
                  annotation_text="Classical Bound (2)")
    fig.add_hline(y=-2.0, line_dash="dash", line_color="red")
    fig.add_hline(y=2*np.sqrt(2), line_dash="dot", line_color="green",
                  annotation_text=f"Tsirelson Bound ({2*np.sqrt(2):.3f})")
    fig.add_hline(y=-2*np.sqrt(2), line_dash="dot", line_color="green")
    
    fig.update_layout(
        title="CHSH S-Parameter",
        yaxis_title="S",
        height=400,
        showlegend=False
    )
    return fig


def create_vbi_scatter_plot(results):
    """Create scatter plot of E vs phase for VBI data."""
    if not results or "all_results" not in results:
        return None
    
    df = results["all_results"]
    fig = go.Figure()
    
    # Scatter plot of E vs phase
    fig.add_trace(go.Scatter(
        x=df['phase'],
        y=df['E'],
        mode='markers+lines',
        name='E(a,b)',
        marker=dict(size=8, color=df['E'], colorscale='RdBu', showscale=True),
        error_y=dict(type='data', array=df['sigma_E'], visible=True),
        text=[f"phase: {p:.3f}<br>E: {e:.4f}±{s:.4f}" 
              for p, e, s in zip(df['phase'], df['E'], df['sigma_E'])],
        hoverinfo='text'
    ))
    
    # Mark peaks and valleys
    if "peaks" in results:
        for idx in results["peaks"]:
            if idx < len(df):
                fig.add_trace(go.Scatter(
                    x=[df.iloc[idx]['phase']],
                    y=[df.iloc[idx]['E']],
                    mode='markers',
                    marker=dict(size=12, color='red', symbol='triangle-up'),
                    name=f'Peak {idx}',
                    showlegend=False
                ))
    
    if "valleys" in results:
        for idx in results["valleys"]:
            if idx < len(df):
                fig.add_trace(go.Scatter(
                    x=[df.iloc[idx]['phase']],
                    y=[df.iloc[idx]['E']],
                    mode='markers',
                    marker=dict(size=12, color='blue', symbol='triangle-down'),
                    name=f'Valley {idx}',
                    showlegend=False
                ))
    
    # Highlight selected settings
    if "selected_settings" in results:
        selected = results["selected_settings"]
        labels = ["E(a,b)", "E(a,b')", "E(a',b)", "E(a',b')"]
        colors = ['red', 'blue', 'green', 'orange']
        for i, s in enumerate(selected):
            if i < len(labels):
                fig.add_trace(go.Scatter(
                    x=[s['phase']],
                    y=[s['E']],
                    mode='markers',
                    marker=dict(size=15, color=colors[i % len(colors)], symbol='star'),
                    name=labels[i],
                    showlegend=True
                ))
    
    fig.update_layout(
        title="Correlation E vs Phase",
        xaxis_title="Phase (radians)",
        yaxis_title="E(a,b)",
        height=500,
        hovermode='closest'
    )
    return fig


def create_confidence_plot(results):
    """Create confidence visualization."""
    if not results or "sigma_above_classical" not in results:
        return None
    
    sigma = results["sigma_above_classical"]
    
    fig = go.Figure()
    x = np.linspace(-3, 3, 100)
    y = np.exp(-x**2/2) / np.sqrt(2*np.pi)
    
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='lines',
        name='Normal Distribution',
        line=dict(color='gray', dash='dash')
    ))
    
    fig.add_vline(x=min(sigma, 3), line_dash="solid", line_color="red",
                  annotation_text=f"Measured: {sigma:.1f}σ")
    
    fig.add_vrect(x0=5, x1=10, fillcolor="green", opacity=0.2,
                  annotation_text="5σ Discovery", annotation_position="top")
    fig.add_vrect(x0=3, x1=5, fillcolor="yellow", opacity=0.2,
                  annotation_text="3σ Evidence", annotation_position="top")
    
    fig.update_layout(
        title="Statistical Significance",
        xaxis_title="Sigma (σ)",
        yaxis_title="Probability Density",
        height=400
    )
    return fig

# ============================================================================
# STREAMLIT APPLICATION
# ============================================================================

st.set_page_config(
    page_title="🔬 CHSH Bell-Test & Dark Matter Search",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f0f2f6, #e8ecf1);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .violation-strong {
        color: #27ae60;
        font-weight: bold;
    }
    .violation-moderate {
        color: #f39c12;
        font-weight: bold;
    }
    .violation-weak {
        color: #e74c3c;
        font-weight: bold;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        color: white;
        font-weight: bold;
    }
    .stButton > button:hover {
        opacity: 0.8;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'events_df' not in st.session_state:
    st.session_state.events_df = None
if 'settings_df' not in st.session_state:
    st.session_state.settings_df = None
if 'channel_map' not in st.session_state:
    st.session_state.channel_map = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'vbi_results' not in st.session_state:
    st.session_state.vbi_results = None
if 'vbi_data' not in st.session_state:
    st.session_state.vbi_data = None
if 'data_type' not in st.session_state:
    st.session_state.data_type = None


def load_sample_data():
    """Load sample data into session state."""
    with st.spinner("Generating sample data..."):
        events_df, settings_df, channel_map = generate_sample_data()
        st.session_state.events_df = events_df
        st.session_state.settings_df = settings_df
        st.session_state.channel_map = channel_map
        st.session_state.data_type = "timestamp"
    st.success("✅ Sample data loaded successfully!")


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("## 🧪 CHSH Bell-Test")
    st.markdown("### Dark Matter Search")
    
    # Data Loading Section
    st.markdown("---")
    st.markdown("### 📁 Data Loading")
    
    data_type = st.radio(
        "Data format:",
        ["VBI Coincidence (.dat)", "Timestamp CSV", "Use Sample Data"]
    )
    
    if data_type == "VBI Coincidence (.dat)":
        vbi_file = st.file_uploader("Upload VBI_Coincidence.dat", type=['dat', 'txt', 'csv'])
        
        if vbi_file and st.button("Load VBI Data", type="primary"):
            with st.spinner("Parsing VBI data..."):
                df = parse_vbi_data(vbi_file)
                if df is not None and len(df) > 0:
                    st.session_state.vbi_data = df
                    st.session_state.data_type = "vbi"
                    st.success(f"✅ Loaded {len(df)} data points from VBI file!")
                else:
                    st.error("Failed to parse VBI data. Please check the file format.")
        
        if st.session_state.get('vbi_data') is not None and len(st.session_state.vbi_data) > 0:
            if st.button("Run CHSH from VBI Data", type="primary"):
                with st.spinner("Computing CHSH S-parameter..."):
                    results = compute_CHSH_from_vbi(st.session_state.vbi_data)
                    if "error" in results:
                        st.error(f"Error: {results['error']}")
                    else:
                        st.session_state.vbi_results = results
                        st.success(f"✅ CHSH S = {results['S']:.4f} ± {results['sigma_S']:.4f}")
                        st.info(f"Data points analyzed: {results.get('n_points', 0)}")
    
    elif data_type == "Timestamp CSV":
        events_file = st.file_uploader("Upload Events CSV", type=['csv'])
        settings_file = st.file_uploader("Upload Settings CSV", type=['csv'])
        channel_map_str = st.text_input(
            "Channel Map:",
            value="1:A+,2:A-,3:B+,4:B-",
            help="Format: channel:label,channel:label,..."
        )
        
        if st.button("Load Timestamp Data", type="primary"):
            if events_file and settings_file:
                events_df = load_events(events_file)
                settings_df = load_settings(settings_file)
                channel_map = parse_channel_map(channel_map_str)
                
                if events_df is not None and settings_df is not None and channel_map is not None:
                    st.session_state.events_df = events_df
                    st.session_state.settings_df = settings_df
                    st.session_state.channel_map = channel_map
                    st.session_state.data_type = "timestamp"
                    st.success("✅ Data loaded successfully!")
        
        if st.session_state.data_type == "timestamp" and st.session_state.events_df is not None:
            st.markdown("---")
            st.markdown("### ⚙️ Analysis Parameters")
            
            window_ns = st.slider(
                "Coincidence Window (ns):",
                min_value=0.5,
                max_value=10.0,
                value=2.0,
                step=0.5
            )
            
            subtract_accidentals = st.checkbox(
                "Subtract Accidentals",
                value=True
            )
            
            if st.button("Run CHSH Analysis", type="primary"):
                with st.spinner("Running CHSH analysis..."):
                    try:
                        results = analyze_run(
                            st.session_state.events_df,
                            st.session_state.settings_df,
                            st.session_state.channel_map,
                            window_ns,
                            subtract_accidentals
                        )
                        st.session_state.analysis_results = results
                        if "error" in results:
                            st.error(f"Error: {results['error']}")
                        else:
                            st.success(f"✅ S = {results['chsh']['S']:.4f} ± {results['chsh']['sigma_S']:.4f}")
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")
    
    else:  # Sample Data
        if st.button("Load Sample Data", type="primary"):
            load_sample_data()
        
        if st.session_state.data_type == "timestamp" and st.session_state.events_df is not None:
            st.markdown("---")
            st.markdown("### ⚙️ Analysis Parameters")
            
            window_ns = st.slider(
                "Coincidence Window (ns):",
                min_value=0.5,
                max_value=10.0,
                value=2.0,
                step=0.5
            )
            
            subtract_accidentals = st.checkbox(
                "Subtract Accidentals",
                value=True
            )
            
            if st.button("Run CHSH Analysis", type="primary"):
                with st.spinner("Running CHSH analysis..."):
                    try:
                        results = analyze_run(
                            st.session_state.events_df,
                            st.session_state.settings_df,
                            st.session_state.channel_map,
                            window_ns,
                            subtract_accidentals
                        )
                        st.session_state.analysis_results = results
                        if "error" in results:
                            st.error(f"Error: {results['error']}")
                        else:
                            st.success(f"✅ S = {results['chsh']['S']:.4f} ± {results['chsh']['sigma_S']:.4f}")
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")

# ============================================================================
# MAIN CONTENT
# ============================================================================

# Header
st.markdown('<p class="main-header">🔬 CHSH Bell-Test & Dark Matter Search</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Quantum Entanglement Analysis with Dark Matter Interference Detection</p>', unsafe_allow_html=True)

# Display VBI results
if st.session_state.data_type == "vbi" and st.session_state.get('vbi_results'):
    results = st.session_state.vbi_results
    
    if "error" in results:
        st.error(f"Error: {results['error']}")
    else:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sigma = results["sigma_above_classical"]
            color = "strong" if sigma > 5 else "moderate" if sigma > 3 else "weak"
            st.markdown(f"""
            <div class="metric-card">
                <h3>S-Parameter</h3>
                <h2 class="violation-{color}">{results["S"]:.4f}</h2>
                <p>± {results["sigma_S"]:.4f}</p>
                <p>Classical bound: 2.0</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Significance</h3>
                <h2 class="violation-{color}">{results["sigma_above_classical"]:.2f} σ</h2>
                <p>{'✅' if results["violates_classical_bound"] else '❌'} {'Violates' if results["violates_classical_bound"] else 'Does not violate'} classical bound</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Tsirelson Bound</h3>
                <p>|S| ≤ {2*np.sqrt(2):.3f}</p>
                <p>{'✅' if results["within_tsirelson_bound"] else '⚠️'} {'Within' if results["within_tsirelson_bound"] else 'Exceeds'} quantum bound</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Data overview
        st.markdown(f"**Data points analyzed:** {results.get('n_points', 0)} phase settings")
        
        # Selected settings
        st.markdown("### 📋 Selected CHSH Settings")
        selected_data = []
        labels = ["E(a,b)", "E(a,b')", "E(a',b)", "E(a',b')"]
        for i, s in enumerate(results["selected_settings"]):
            if i < len(labels):
                selected_data.append({
                    "Setting": labels[i],
                    "Phase": f"{s['phase']:.3f}",
                    "E": f"{s['E']:+.4f}",
                    "σ_E": f"{s['sigma_E']:.4f}",
                    "N_AB": int(s['N_AB']),
                    "N_CD": int(s['N_CD'])
                })
        st.dataframe(pd.DataFrame(selected_data), use_container_width=True)
        
        # Plots
        col1, col2 = st.columns(2)
        with col1:
            fig = create_chsh_plot(results)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = create_vbi_scatter_plot(results)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        # Confidence plot
        fig = create_confidence_plot(results)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

# Display timestamp results
elif st.session_state.data_type == "timestamp" and st.session_state.analysis_results:
    results = st.session_state.analysis_results
    
    if "error" in results:
        st.error(f"Error: {results['error']}")
    else:
        chsh = results["chsh"]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sigma = chsh["sigma_above_classical"]
            color = "strong" if sigma > 5 else "moderate" if sigma > 3 else "weak"
            st.markdown(f"""
            <div class="metric-card">
                <h3>S-Parameter</h3>
                <h2 class="violation-{color}">{chsh["S"]:.4f}</h2>
                <p>± {chsh["sigma_S"]:.4f}</p>
                <p>Classical bound: 2.0</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Significance</h3>
                <h2 class="violation-{color}">{chsh["sigma_above_classical"]:.2f} σ</h2>
                <p>{'✅' if chsh["violates_classical_bound"] else '❌'} {'Violates' if chsh["violates_classical_bound"] else 'Does not violate'} classical bound</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Tsirelson Bound</h3>
                <p>|S| ≤ {2*np.sqrt(2):.3f}</p>
                <p>{'✅' if chsh["within_tsirelson_bound"] else '⚠️'} {'Within' if chsh["within_tsirelson_bound"] else 'Exceeds'} quantum bound</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Per-setting results
        st.markdown("### 📋 Per-Setting Results")
        data = []
        for key in ["ab", "abp", "apb", "apbp"]:
            if key not in results["per_run"]:
                continue
            r = results["per_run"][key]
            data.append({
                "Setting": key.upper(),
                "A (deg)": r["angle_a_deg"],
                "B (deg)": r["angle_b_deg"],
                "E(a,b)": f"{r['E']:+.4f}",
                "σ_E": f"{r['sigma_E']:.4f}",
                "Coincidences": r["raw_counts"]["total"]
            })
        st.dataframe(pd.DataFrame(data), use_container_width=True)
        
        # Plots
        col1, col2 = st.columns(2)
        with col1:
            fig = create_chsh_plot(chsh)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

else:
    # Welcome screen
    st.markdown("""
    ### 🚀 Getting Started
    
    Choose your data format from the sidebar:
    
    #### 1. VBI Coincidence (.dat)
    - Upload the `VBI_Coincidence_20230707.dat` file
    - The system will automatically parse the phase settings and compute CHSH
    - Works with swept-phase data where piezo position encodes the phase
    
    #### 2. Timestamp CSV
    - Upload `events.csv` and `settings.csv`
    - Configure channel mapping
    - Run CHSH analysis
    
    #### 3. Sample Data
    - Click "Load Sample Data" to test the pipeline
    - Synthetic entangled photon data will be generated
    
    ### 📁 Understanding VBI Data
    
    The VBI dataset contains:
    - **Alpha, Beta**: Motorized stage positions (Alice and Bob)
    - **Piezo positions**: Fine phase scanning
    - **N_AB**: Coincidence count for A+,B+
    - **N_CD**: Coincidence count for A-,B-
    - **N_AC, N_BD**: Cross-coincidence counts
    
    The system finds the optimal CHSH settings by analyzing the correlation as a function of phase!
    """)
