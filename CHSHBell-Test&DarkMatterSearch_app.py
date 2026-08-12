"""
🔬 CHSH Bell-Test & Dark Matter Search - Complete Pipeline
===========================================================
Single-file Streamlit application for analyzing photon coincidence data,
computing CHSH Bell-test statistics, and searching for dark matter
interference signatures.

Author: Tony E. Ford | QCAUS Lab Analysis Tools
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
PC_TO_M = 3.08567758e16
KPC_TO_M = PC_TO_M * 1000
SOLAR_MASS = 1.98847e30  # kg
G_NEWTON = 6.67430e-11
SIDEREAL_DAY_S = 86164.0905
SOLAR_DAY_S = 86400.0

# ============================================================================
# DATA LOADING & PROCESSING
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
    return events_df, settings_df, channel_map

# ============================================================================
# COINCIDENCE COUNTING
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
# CHSH ANALYSIS
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
    settings["run_key"] = settings["run_id"].astype(str).str.lower()
    
    per_run = {}
    for _, row in settings.iterrows():
        key = row["run_key"]
        if key not in ["ab", "abp", "apb", "apbp"]:
            continue
        
        coinc = analyze_coincidences(events, channel_map, window_ns,
                                      start_ns=row["start_ns"], end_ns=row["end_ns"])
        E, sigE, raw = correlation_E(coinc, subtract_accidentals)
        
        per_run[key] = {
            "E": E, "sigma_E": sigE, "raw_counts": raw,
            "angle_a_deg": row["angle_a_deg"], "angle_b_deg": row["angle_b_deg"],
            "singles": coinc.singles, "duration_s": coinc.run_duration_s,
        }
    
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
# DARK MATTER SEARCH
# ============================================================================

def bin_coincidences_by_time(events: pd.DataFrame, channel_map: dict,
                             window_ns: float, bin_width_s: float,
                             start_ns: float = None, end_ns: float = None,
                             min_coincidences: int = 10) -> pd.DataFrame:
    """Bin coincidence counts into time bins."""
    df = events.copy()
    if start_ns is not None:
        df = df[df["timestamp_ns"] >= start_ns]
    if end_ns is not None:
        df = df[df["timestamp_ns"] <= end_ns]
    
    if len(df) == 0:
        return pd.DataFrame(columns=["time_s", "pair", "counts"])
    
    df["label"] = df["channel"].map(channel_map)
    if df["label"].isna().any():
        return pd.DataFrame()
    
    df["time_s"] = df["timestamp_ns"] * 1e-9
    start_time = df["time_s"].min()
    end_time = df["time_s"].max()
    
    bins = np.arange(start_time, end_time, bin_width_s)
    bin_centers = bins[:-1] + bin_width_s/2
    
    results = []
    for a_lbl in ("A+", "A-"):
        for b_lbl in ("B+", "B-"):
            times_a = df[df["label"] == a_lbl]["time_s"].values
            times_b = df[df["label"] == b_lbl]["time_s"].values
            
            hist_a, _ = np.histogram(times_a, bins=bins)
            hist_b, _ = np.histogram(times_b, bins=bins)
            
            for i, (na, nb) in enumerate(zip(hist_a, hist_b)):
                if na > 0 and nb > 0:
                    expected_coinc = na * nb * (2 * window_ns * 1e-9) / bin_width_s
                    if expected_coinc > 0.1:
                        results.append({
                            "time_s": bin_centers[i],
                            "pair": f"{a_lbl}_{b_lbl}",
                            "counts": expected_coinc
                        })
    
    if len(results) == 0:
        return pd.DataFrame()
    
    df_results = pd.DataFrame(results)
    df_results = df_results.groupby("time_s").filter(
        lambda x: x["counts"].sum() >= min_coincidences
    )
    return df_results


def fit_oscillatory_modulation(times: np.ndarray, counts: np.ndarray,
                               omega_range: tuple, n_frequencies: int = 100) -> dict:
    """Fit sinusoidal modulation to time-binned counts."""
    times = np.array(times)
    counts = np.array(counts)
    
    counts_norm = counts / np.mean(counts)
    errors = np.sqrt(counts) / np.mean(counts)
    
    omega_min, omega_max = omega_range
    omega_grid = np.logspace(np.log10(omega_min), np.log10(omega_max), n_frequencies)
    
    best_freq = None
    best_amplitude = None
    best_phase = None
    best_p_value = 1.0
    
    for omega in omega_grid:
        try:
            cos_terms = np.cos(omega * times)
            sin_terms = np.sin(omega * times)
            
            X = np.column_stack([np.ones_like(times), cos_terms, sin_terms])
            weights = 1.0 / (errors**2 + 1e-10)
            W = np.diag(weights)
            
            beta = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ (counts_norm))
            
            A = np.sqrt(beta[1]**2 + beta[2]**2)
            phi = np.arctan2(-beta[2], beta[1])
            
            predicted = 1 + A * np.cos(omega * times + phi)
            chi2_val = np.sum(((counts_norm - predicted)**2) / (errors**2 + 1e-10))
            dof = len(times) - 3
            
            if dof > 0:
                p_value = 1 - chi2.cdf(chi2_val, dof)
                if p_value < best_p_value:
                    best_freq = omega
                    best_amplitude = A
                    best_phase = phi
                    best_p_value = p_value
        except:
            continue
    
    return {
        "best_frequency_hz": best_freq/(2*np.pi) if best_freq else None,
        "best_amplitude": best_amplitude,
        "best_phase": best_phase,
        "p_value": best_p_value,
        "significance_sigma": norm.ppf(1 - best_p_value/2) if best_p_value > 0 else 0,
        "detected": best_freq is not None and best_p_value < 0.05
    }


def analyze_dark_matter(events: pd.DataFrame, settings: pd.DataFrame, 
                        channel_map: dict, window_ns: float,
                        dm_mass_eV: float = 1e-6, epsilon: float = 0.01) -> dict:
    """Search for dark matter interference signatures."""
    if events is None or settings is None or channel_map is None:
        return {"error": "Data not loaded"}
    
    try:
        # Get base results
        base_result = analyze_run(events, settings, channel_map, window_ns)
        
        # Time-binned analysis
        time_series_data = {}
        for run_key in ["ab", "abp", "apb", "apbp"]:
            row = settings[settings["run_key"] == run_key]
            if len(row) == 0:
                continue
            
            start_ns = row["start_ns"].values[0]
            end_ns = row["end_ns"].values[0]
            
            binned = bin_coincidences_by_time(
                events, channel_map, window_ns, 1.0,
                start_ns=start_ns, end_ns=end_ns
            )
            
            if len(binned) == 0:
                continue
            
            times = []
            E_values = []
            for time, group in binned.groupby("time_s"):
                counts = group.set_index("pair")["counts"].to_dict()
                if len(counts) < 4:
                    continue
                    
                Npp = counts.get("A+_B+", 0)
                Nmm = counts.get("A-_B-", 0)
                Npm = counts.get("A+_B-", 0)
                Nmp = counts.get("A-_B+", 0)
                
                total = Npp + Nmm + Npm + Nmp
                if total == 0:
                    continue
                    
                E = (Npp + Nmm - Npm - Nmp) / total
                times.append(time)
                E_values.append(E)
            
            if len(times) > 10:
                time_series_data[run_key] = {
                    "times": np.array(times),
                    "E": np.array(E_values)
                }
        
        # Search for oscillations
        if len(time_series_data) >= 4:
            # Combine all runs
            all_times = []
            all_E = []
            for key in ["ab", "abp", "apb", "apbp"]:
                if key in time_series_data:
                    all_times.extend(time_series_data[key]["times"])
                    all_E.extend(time_series_data[key]["E"])
            
            if len(all_times) > 20:
                omega_min = 0.5 * epsilon * dm_mass_eV * C**2 / HBAR
                omega_max = 2.0 * epsilon * dm_mass_eV * C**2 / HBAR
                
                fit_result = fit_oscillatory_modulation(
                    np.array(all_times), np.array(all_E),
                    (omega_min, omega_max)
                )
                
                return {
                    "base_result": base_result,
                    "time_series": time_series_data,
                    "oscillation": fit_result,
                    "omega_beat": fit_result["best_frequency_hz"] * 2 * np.pi if fit_result["best_frequency_hz"] else None,
                    "coupling_strength": fit_result["best_amplitude"] / 2 if fit_result["best_amplitude"] else None,
                    "detection_significance": fit_result["significance_sigma"]
                }
        
        return {"base_result": base_result, "no_oscillation": True}
    
    except Exception as e:
        return {"error": str(e)}

# ============================================================================
# SIDEREAL TIME ANALYSIS
# ============================================================================

def convert_to_sidereal_time(timestamps_ns: np.ndarray, 
                             longitude_deg: float = -105.0) -> np.ndarray:
    """Convert UTC timestamps to Local Sidereal Time."""
    timestamps_s = timestamps_ns / 1e-9
    # Simplified sidereal time calculation
    # For a more accurate calculation, use astropy
    jd = timestamps_s / 86400.0 + 2440587.5  # Approximate JD
    gmst = 280.46061837 + 360.98564736629 * (jd - 2451545.0)
    gmst = gmst % 360
    lst = (gmst + longitude_deg) % 360
    return lst


def analyze_sidereal(events: pd.DataFrame, channel_map: dict, window_ns: float,
                     longitude: float = -105.0) -> dict:
    """Analyze sidereal time modulation."""
    if events is None or channel_map is None:
        return {"error": "Data not loaded"}
    
    try:
        # Convert timestamps to sidereal time
        timestamps = events['timestamp_ns'].values
        sidereal_times = convert_to_sidereal_time(timestamps, longitude)
        
        events_copy = events.copy()
        events_copy['sidereal_time_deg'] = sidereal_times
        events_copy['sidereal_hour'] = (events_copy['sidereal_time_deg'] / 15.0).astype(int) % 24
        
        # Bin by sidereal hour
        results = []
        for hour in range(24):
            mask = events_copy['sidereal_hour'] == hour
            hour_events = events_copy[mask]
            
            if len(hour_events) > 100:
                try:
                    coinc = analyze_coincidences(hour_events, channel_map, window_ns)
                    E, sigma_E, _ = correlation_E(coinc, subtract_accidentals=True)
                    results.append({
                        'sidereal_hour': hour,
                        'sidereal_angle_deg': hour * 15.0,
                        'E': E,
                        'sigma_E': sigma_E,
                        'counts': len(hour_events)
                    })
                except:
                    pass
        
        if len(results) < 10:
            return {"error": "Insufficient data for sidereal analysis"}
        
        df = pd.DataFrame(results)
        
        # Fit sinusoidal modulation
        hours = df['sidereal_hour'].values
        E_values = df['E'].values
        errors = df['sigma_E'].values
        
        def model(t, E0, A, phi):
            return E0 + A * np.cos(2 * np.pi * t / 24 + phi)
        
        try:
            popt, pcov = curve_fit(model, hours, E_values, sigma=errors,
                                   p0=[np.mean(E_values), 0.01, 0.0])
            E0, A, phi = popt
            perr = np.sqrt(np.diag(pcov))
            sigma_A = A / perr[1] if perr[1] > 0 else 0
            
            return {
                "data": df,
                "E0": E0,
                "amplitude": A,
                "amplitude_error": perr[1],
                "phase": phi,
                "significance": sigma_A,
                "detected": abs(sigma_A) > 3.0
            }
        except:
            return {"data": df, "error": "Fit failed"}
    
    except Exception as e:
        return {"error": str(e)}

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_chsh_plot(results):
    """Create CHSH visualization."""
    if not results or "chsh" not in results:
        return None
    
    chsh = results["chsh"]
    fig = go.Figure()
    
    # S parameter
    fig.add_trace(go.Bar(
        name="S-Parameter",
        x=["CHSH S"],
        y=[chsh["S"]],
        error_y=dict(type='data', array=[chsh["sigma_S"]]),
        text=[f"{chsh['S']:.4f}"],
        textposition='auto'
    ))
    
    # Bounds
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


def create_correlation_plot(results):
    """Create correlation values plot."""
    if not results or "per_run" not in results:
        return None
    
    fig = go.Figure()
    for key in ["ab", "abp", "apb", "apbp"]:
        r = results["per_run"][key]
        fig.add_trace(go.Bar(
            name=key.upper(),
            x=[f"a={r['angle_a_deg']}°\nb={r['angle_b_deg']}°"],
            y=[r["E"]],
            error_y=dict(type='data', array=[r["sigma_E"]]),
            text=[f"{r['E']:.4f}"],
            textposition='auto'
        ))
    
    fig.update_layout(
        title="Correlation E(a,b) for Each Setting",
        xaxis_title="Angle Setting",
        yaxis_title="E(a,b)",
        showlegend=True,
        height=400
    )
    return fig


def create_dm_oscillation_plot(time_series, oscillation):
    """Create dark matter oscillation plot."""
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=("Sidereal Modulation", "Residuals"))
    
    if not time_series or not oscillation:
        return fig
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, (key, data) in enumerate(time_series.items()):
        if i < len(colors):
            fig.add_trace(
                go.Scatter(
                    x=data['times'],
                    y=data['E'],
                    mode='markers+lines',
                    name=f"E({key})",
                    marker=dict(color=colors[i])
                ),
                row=1, col=1
            )
    
    # Add fitted oscillation
    if oscillation.get("best_frequency_hz") is not None:
        first_key = list(time_series.keys())[0]
        if first_key in time_series:
            t = np.linspace(time_series[first_key]['times'][0],
                           time_series[first_key]['times'][-1], 1000)
            A = oscillation["best_amplitude"]
            omega = oscillation["best_frequency_hz"] * 2 * np.pi
            phi = oscillation["best_phase"] or 0
            y = 1 + A * np.cos(omega * t + phi)
            
            fig.add_trace(
                go.Scatter(
                    x=t, y=y,
                    mode='lines',
                    name='Fitted Oscillation',
                    line=dict(color='red', dash='dash')
                ),
                row=1, col=1
            )
    
    fig.update_layout(height=400, showlegend=True)
    return fig


def create_sidereal_plot(sidereal_data):
    """Create sidereal time plot."""
    if not sidereal_data or "data" not in sidereal_data:
        return None
    
    df = sidereal_data["data"]
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['sidereal_hour'],
        y=df['E'],
        mode='markers+lines',
        name='E(a,b)',
        error_y=dict(type='data', array=df['sigma_E'])
    ))
    
    # Add fit if available
    if sidereal_data.get("detected"):
        hours = np.linspace(0, 24, 100)
        E0 = sidereal_data["E0"]
        A = sidereal_data["amplitude"]
        phi = sidereal_data["phase"]
        y_fit = E0 + A * np.cos(2*np.pi*hours/24 + phi)
        
        fig.add_trace(go.Scatter(
            x=hours, y=y_fit,
            mode='lines',
            name='Sinusoidal Fit',
            line=dict(color='red', dash='dash')
        ))
    
    fig.update_layout(
        title="Sidereal Time Modulation",
        xaxis_title="Sidereal Hour",
        yaxis_title="E(a,b)",
        height=400,
        showlegend=True
    )
    return fig


def create_coincidence_heatmap(results):
    """Create coincidence count heatmap."""
    if not results or "per_run" not in results:
        return None
    
    data = []
    for key in ["ab", "abp", "apb", "apbp"]:
        r = results["per_run"][key]
        data.append({
            'Setting': key.upper(),
            'N++': r['raw_counts']['N++'],
            'N--': r['raw_counts']['N--'],
            'N+-': r['raw_counts']['N+-'],
            'N-+': r['raw_counts']['N-+']
        })
    
    df = pd.DataFrame(data)
    df_matrix = df.set_index('Setting')[['N++', 'N--', 'N+-', 'N-+']]
    
    fig = px.imshow(
        df_matrix.values,
        labels=dict(x="Outcome", y="Setting", color="Counts"),
        x=['N++', 'N--', 'N+-', 'N-+'],
        y=df['Setting'],
        title="Coincidence Counts Heatmap",
        color_continuous_scale="Viridis"
    )
    fig.update_layout(height=400)
    return fig


def create_confidence_plot(results):
    """Create confidence visualization."""
    if not results or "chsh" not in results:
        return None
    
    sigma = results["chsh"]["sigma_above_classical"]
    
    fig = go.Figure()
    x = np.linspace(-3, 3, 100)
    y = np.exp(-x**2/2) / np.sqrt(2*np.pi)
    
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='lines',
        name='Normal Distribution',
        line=dict(color='gray', dash='dash')
    ))
    
    fig.add_vline(x=sigma, line_dash="solid", line_color="red",
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

# Page configuration
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
if 'dm_results' not in st.session_state:
    st.session_state.dm_results = None
if 'sidereal_results' not in st.session_state:
    st.session_state.sidereal_results = None
if 'data_source' not in st.session_state:
    st.session_state.data_source = None


def load_sample_data():
    """Load sample data into session state."""
    with st.spinner("Generating sample data..."):
        events_df, settings_df, channel_map = generate_sample_data()
        st.session_state.events_df = events_df
        st.session_state.settings_df = settings_df
        st.session_state.channel_map = channel_map
        st.session_state.data_source = "sample"
    st.success("✅ Sample data loaded successfully!")


def handle_file_upload(events_file, settings_file, channel_map_str):
    """Handle file upload and data loading."""
    if events_file and settings_file and channel_map_str:
        events_df = load_events(events_file)
        if events_df is None:
            return False
        
        settings_df = load_settings(settings_file)
        if settings_df is None:
            return False
        
        channel_map = parse_channel_map(channel_map_str)
        if channel_map is None:
            return False
        
        st.session_state.events_df = events_df
        st.session_state.settings_df = settings_df
        st.session_state.channel_map = channel_map
        st.session_state.data_source = "upload"
        st.success("✅ Data loaded successfully!")
        return True
    return False


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("## 🧪 CHSH Bell-Test")
    st.markdown("### Dark Matter Search")
    
    # Data Loading Section
    st.markdown("---")
    st.markdown("### 📁 Data Loading")
    
    data_source = st.radio(
        "Select data source:",
        ["Upload Files", "Use Sample Data"],
        index=1
    )
    
    if data_source == "Upload Files":
        events_file = st.file_uploader("Upload Events CSV", type=['csv'])
        settings_file = st.file_uploader("Upload Settings CSV", type=['csv'])
        channel_map_str = st.text_input(
            "Channel Map:",
            value="1:A+,2:A-,3:B+,4:B-",
            help="Format: channel:label,channel:label,..."
        )
        
        if st.button("Load Data", type="primary"):
            handle_file_upload(events_file, settings_file, channel_map_str)
    
    else:  # Sample Data
        if st.button("Load Sample Data", type="primary"):
            load_sample_data()
    
    # Analysis Parameters
    if st.session_state.events_df is not None:
        st.markdown("---")
        st.markdown("### ⚙️ Analysis Parameters")
        
        window_ns = st.slider(
            "Coincidence Window (ns):",
            min_value=0.5,
            max_value=10.0,
            value=2.0,
            step=0.5,
            help="Timing window for coincidence detection"
        )
        
        subtract_accidentals = st.checkbox(
            "Subtract Accidentals",
            value=True,
            help="Subtract accidental coincidence background"
        )
        
        col1, col2 = st.columns(2)
        with col1:
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
                        st.success("✅ Analysis complete!")
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")
        
        with col2:
            if st.button("Run DM Search", type="primary"):
                with st.spinner("Searching for dark matter..."):
                    try:
                        dm_results = analyze_dark_matter(
                            st.session_state.events_df,
                            st.session_state.settings_df,
                            st.session_state.channel_map,
                            window_ns
                        )
                        st.session_state.dm_results = dm_results
                        if "error" not in dm_results:
                            st.success("✅ DM search complete!")
                        else:
                            st.warning(f"DM search issue: {dm_results['error']}")
                    except Exception as e:
                        st.error(f"DM search failed: {str(e)}")
        
        if st.button("Run Sidereal Analysis", type="primary"):
            with st.spinner("Analyzing sidereal modulation..."):
                try:
                    sidereal_results = analyze_sidereal(
                        st.session_state.events_df,
                        st.session_state.channel_map,
                        window_ns
                    )
                    st.session_state.sidereal_results = sidereal_results
                    if "error" not in sidereal_results:
                        st.success("✅ Sidereal analysis complete!")
                    else:
                        st.warning(f"Sidereal issue: {sidereal_results['error']}")
                except Exception as e:
                    st.error(f"Sidereal analysis failed: {str(e)}")
    
    # Data Info
    if st.session_state.events_df is not None:
        st.markdown("---")
        st.markdown("### 📊 Data Info")
        st.metric("Events", f"{len(st.session_state.events_df):,}")
        if st.session_state.settings_df is not None:
            st.metric("Settings Runs", len(st.session_state.settings_df))


# ============================================================================
# MAIN CONTENT
# ============================================================================

# Header
st.markdown('<p class="main-header">🔬 CHSH Bell-Test & Dark Matter Search</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Quantum Entanglement Analysis with Dark Matter Interference Detection</p>', unsafe_allow_html=True)

# Status indicators
if st.session_state.events_df is not None:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "📊 Events",
            f"{len(st.session_state.events_df):,}",
            help="Total detection events"
        )
    with col2:
        if st.session_state.analysis_results:
            S = st.session_state.analysis_results["chsh"]["S"]
            sigma = st.session_state.analysis_results["chsh"]["sigma_above_classical"]
            color = "green" if sigma > 5 else "orange" if sigma > 3 else "red"
            st.metric(
                "🎯 S-Parameter",
                f"{S:.4f}",
                delta=f"{S-2:.4f}",
                delta_color="normal"
            )
    with col3:
        if st.session_state.analysis_results:
            sigma = st.session_state.analysis_results["chsh"]["sigma_above_classical"]
            st.metric(
                "📈 Significance",
                f"{sigma:.2f} σ",
                help="Sigma above classical bound"
            )

# Results Tabs
if st.session_state.analysis_results:
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 CHSH Results",
        "🔬 Dark Matter Search",
        "🌙 Sidereal Analysis",
        "📈 Visualizations",
        "📋 Report"
    ])
    
    with tab1:
        st.markdown("## 📊 CHSH Bell-Test Results")
        
        results = st.session_state.analysis_results
        chsh = results["chsh"]
        
        # Summary cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            violation = chsh["violates_classical_bound"]
            sigma = chsh["sigma_above_classical"]
            color = "strong" if sigma > 5 else "moderate" if sigma > 3 else "weak"
            status = "✅ Strong" if sigma > 5 else "⚠️ Moderate" if sigma > 3 else "❌ Weak"
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>S-Parameter</h3>
                <h2>{chsh["S"]:.4f}</h2>
                <p>± {chsh["sigma_S"]:.4f}</p>
                <p>Classical bound: 2.0</p>
                <p class="violation-{color}">{status} violation</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Statistical Significance</h3>
                <h2 class="violation-{color}">{chsh["sigma_above_classical"]:.2f} σ</h2>
                <p>Above classical bound</p>
                <p>{'✅' if chsh["within_tsirelson_bound"] else '⚠️'} Within Tsirelson bound</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>CHSH Interpretation</h3>
                <p><b>|S| ≤ 2</b>: Classical (Local Realism)</p>
                <p><b>|S| ≤ {2*np.sqrt(2):.3f}</b>: Quantum</p>
                <p><b>Current: |S| = {abs(chsh["S"]):.4f}</b></p>
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed results
        st.markdown("### 📋 Per-Setting Results")
        
        data = []
        for key in ["ab", "abp", "apb", "apbp"]:
            r = results["per_run"][key]
            data.append({
                "Setting": key.upper(),
                "A (deg)": r["angle_a_deg"],
                "B (deg)": r["angle_b_deg"],
                "E(a,b)": f"{r['E']:+.4f}",
                "σ_E": f"{r['sigma_E']:.4f}",
                "Duration (s)": f"{r['duration_s']:.1f}",
                "Coincidences": r["raw_counts"]["total"]
            })
        
        st.dataframe(pd.DataFrame(data), use_container_width=True)
        
        # Plots
        col1, col2 = st.columns(2)
        with col1:
            fig = create_chsh_plot(results)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = create_correlation_plot(results)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("## 🔬 Dark Matter Interference Search")
        
        if st.session_state.dm_results:
            dm = st.session_state.dm_results
            
            if "error" in dm:
                st.warning(f"Dark matter search issue: {dm['error']}")
            elif "no_oscillation" in dm:
                st.info("No significant oscillation detected in the data")
                
                # Show base CHSH results
                if "base_result" in dm:
                    base = dm["base_result"]
                    st.metric("S-Parameter", f"{base['chsh']['S']:.4f}")
                    st.metric("Significance", f"{base['chsh']['sigma_above_classical']:.2f} σ")
            else:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("""
                    <div class="metric-card">
                        <h3>Oscillation Detection</h3>
                    """, unsafe_allow_html=True)
                    
                    osc = dm["oscillation"]
                    if osc.get("detected"):
                        st.success("✅ Modulation detected!")
                        st.metric("Frequency", f"{osc['best_frequency_hz']:.6f} Hz")
                        st.metric("Significance", f"{osc['significance_sigma']:.2f} σ")
                    else:
                        st.warning("❌ No significant modulation")
                
                with col2:
                    st.markdown("""
                    <div class="metric-card">
                        <h3>Dark Matter Parameters</h3>
                    """, unsafe_allow_html=True)
                    
                    if dm.get("omega_beat"):
                        st.metric("ω_beat", f"{dm['omega_beat']:.4e} rad/s")
                        st.metric("Coupling Strength", f"{dm['coupling_strength']:.4f}")
                        
                        # Estimate mass
                        if dm["coupling_strength"]:
                            mass_est = dm["omega_beat"] * HBAR / (dm["coupling_strength"] * C**2)
                            st.metric("Mass Estimate", f"{mass_est:.2e} eV")
                
                with col3:
                    st.markdown("""
                    <div class="metric-card">
                        <h3>Physics Interpretation</h3>
                    """, unsafe_allow_html=True)
                    
                    if dm.get("detection_significance", 0) > 5:
                        st.success("🔬 Strong dark matter signal!")
                    elif dm.get("detection_significance", 0) > 3:
                        st.warning("⚠️ Moderate dark matter evidence")
                    else:
                        st.info("No dark matter signal")
                
                # Oscillation plot
                if "time_series" in dm and "oscillation" in dm:
                    fig = create_dm_oscillation_plot(dm["time_series"], dm["oscillation"])
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("Run the dark matter search from the sidebar")
    
    with tab3:
        st.markdown("## 🌙 Sidereal Time Analysis")
        
        if st.session_state.sidereal_results:
            sidereal = st.session_state.sidereal_results
            
            if "error" in sidereal:
                st.warning(f"Sidereal analysis issue: {sidereal['error']}")
            elif "data" in sidereal:
                df = sidereal["data"]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("""
                    <div class="metric-card">
                        <h3>Modulation Detection</h3>
                    """, unsafe_allow_html=True)
                    
                    if sidereal.get("detected"):
                        st.success("✅ Sidereal modulation detected!")
                        st.metric("Amplitude", f"{sidereal['amplitude']:.4f}")
                        st.metric("Significance", f"{sidereal['significance']:.2f} σ")
                    else:
                        st.warning("❌ No sidereal modulation detected")
                
                with col2:
                    st.markdown("""
                    <div class="metric-card">
                        <h3>Fit Parameters</h3>
                    """, unsafe_allow_html=True)
                    
                    if sidereal.get("detected"):
                        st.metric("E₀", f"{sidereal['E0']:.4f}")
                        st.metric("Phase", f"{sidereal['phase']:.2f} rad")
                
                with col3:
                    st.markdown("""
                    <div class="metric-card">
                        <h3>Data Overview</h3>
                    """, unsafe_allow_html=True)
                    
                    st.metric("Sidereal Bins", len(df))
                    st.metric("Mean E", f"{df['E'].mean():+.4f}")
                
                # Sidereal plot
                fig = create_sidereal_plot(sidereal)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Data table
                st.markdown("### 📊 Sidereal Data")
                st.dataframe(df, use_container_width=True)
        else:
            st.info("Run the sidereal analysis from the sidebar")
    
    with tab4:
        st.markdown("## 📈 Advanced Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = create_coincidence_heatmap(st.session_state.analysis_results)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = create_confidence_plot(st.session_state.analysis_results)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        # 3D visualization
        st.markdown("### 🎨 3D Parameter Space")
        results = st.session_state.analysis_results
        
        angles_a = []
        angles_b = []
        E_values = []
        labels = []
        
        for key in ["ab", "abp", "apb", "apbp"]:
            r = results["per_run"][key]
            angles_a.append(r["angle_a_deg"])
            angles_b.append(r["angle_b_deg"])
            E_values.append(r["E"])
            labels.append(key.upper())
        
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=angles_a,
            y=angles_b,
            z=E_values,
            mode='markers+text',
            marker=dict(size=15, color=E_values, colorscale='Viridis'),
            text=labels,
            textposition="top center"
        ))
        
        fig.update_layout(
            scene=dict(
                xaxis_title="Angle A (deg)",
                yaxis_title="Angle B (deg)",
                zaxis_title="E(a,b)"
            ),
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.markdown("## 📋 Full Analysis Report")
        
        results = st.session_state.analysis_results
        chsh = results["chsh"]
        
        # Generate report
        report = f"""# CHSH Bell-Test Analysis Report

## Summary

- **S-Parameter**: {chsh["S"]:.4f} ± {chsh["sigma_S"]:.4f}
- **Classical Bound Violation**: {chsh["sigma_above_classical"]:.2f} σ
- **Within Tsirelson Bound**: {'Yes' if chsh["within_tsirelson_bound"] else 'No'}
- **Conclusion**: {'Quantum entanglement detected' if chsh["sigma_above_classical"] > 5 else 'Insufficient evidence for entanglement'}

## Per-Setting Results

| Setting | A (deg) | B (deg) | E(a,b) | σ_E | Duration (s) | Coincidences |
|---------|---------|---------|--------|-----|--------------|--------------|
"""
        
        for key in ["ab", "abp", "apb", "apbp"]:
            r = results["per_run"][key]
            report += f"| {key.upper()} | {r['angle_a_deg']} | {r['angle_b_deg']} | {r['E']:+.4f} | {r['sigma_E']:.4f} | {r['duration_s']:.1f} | {r['raw_counts']['total']} |\n"
        
        # Add dark matter results
        if st.session_state.dm_results and "error" not in st.session_state.dm_results:
            dm = st.session_state.dm_results
            if "oscillation" in dm and dm["oscillation"].get("detected"):
                osc = dm["oscillation"]
                report += f"""
                
## Dark Matter Search Results

- **Oscillation Detected**: Yes
- **Frequency**: {osc['best_frequency_hz']:.6f} Hz
- **Amplitude**: {osc['best_amplitude']:.4f}
- **Significance**: {osc['significance_sigma']:.2f} σ
- **Coupling Strength**: {dm.get('coupling_strength', 'N/A')}
"""
        
        # Add sidereal results
        if st.session_state.sidereal_results and "data" in st.session_state.sidereal_results:
            sidereal = st.session_state.sidereal_results
            if sidereal.get("detected"):
                report += f"""
                
## Sidereal Time Analysis

- **Sidereal Modulation**: Detected
- **Amplitude**: {sidereal['amplitude']:.4f} ± {sidereal['amplitude_error']:.4f}
- **Significance**: {sidereal['significance']:.2f} σ
- **Phase**: {sidereal['phase']:.2f} rad
"""
        
        # Physics interpretation
        report += f"""
        
## Physics Interpretation

The CHSH S-parameter of {chsh["S"]:.4f} {'violates' if chsh["violates_classical_bound"] else 'does not violate'} the classical bound of 2.

- **Local Realism**: {'Excluded' if chsh["violates_classical_bound"] else 'Not excluded'} ({chsh["sigma_above_classical"]:.2f} σ)
- **Quantum Mechanics**: {'Consistent' if chsh["within_tsirelson_bound"] else 'Inconsistent'} (within Tsirelson bound)
"""
        
        st.markdown(report)
        
        # Download button
        st.download_button(
            label="📥 Download Full Report",
            data=report,
            file_name="chsh_analysis_report.md",
            mime="text/markdown"
        )

else:
    # Welcome screen
    st.markdown("""
    ### 🚀 Getting Started
    
    1. **Load your data** using the sidebar
    2. **Configure the channel mapping** for your detectors
    3. **Set analysis parameters** (coincidence window, etc.)
    4. **Run the analysis** to compute CHSH S-parameter
    5. **Explore the results** in the tabs above
    
    ### 📁 Data Format Requirements
    
    **Events CSV:**
    - Columns: `timestamp_ns`, `channel`
    - `timestamp_ns`: detection time in nanoseconds
    - `channel`: detector channel ID (integer)
    
    **Settings CSV:**
    - Columns: `run_id`, `start_ns`, `end_ns`, `angle_a_deg`, `angle_b_deg`
    - `run_id`: one of `ab`, `abp`, `apb`, `apbp`
    - `start_ns`/`end_ns`: time range for each run
    - `angle_a_deg`/`angle_b_deg`: analyzer angles in degrees
    
    ### 🔬 Features
    
    - **CHSH Bell Test**: Compute S-parameter and violation significance
    - **Dark Matter Search**: Detect oscillatory modulations in coincidence counts
    - **Sidereal Analysis**: Search for daily modulation patterns
    - **Interactive Visualizations**: Plotly-based dynamic plots
    - **Comprehensive Reporting**: Generate full analysis reports
    """)
    
    # Quick example
    if st.button("🚀 Load Example & Run Analysis"):
        load_sample_data()
        with st.spinner("Running analysis..."):
            try:
                results = analyze_run(
                    st.session_state.events_df,
                    st.session_state.settings_df,
                    st.session_state.channel_map,
                    2.0,
                    True
                )
                st.session_state.analysis_results = results
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")
