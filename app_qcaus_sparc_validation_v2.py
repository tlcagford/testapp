#!/usr/bin/env python3
"""
QCAUS SPARC Rotation-Curve Validation Lab
------------------------------------------

Validation-focused replacement for the previous four-parameter auto-fit.

Design:
  1. Baryons-only baseline
  2. NFW baseline
  3. FDM + NFW
  4. QCAUS two-field, Omega = 0 (nested null)
  5. QCAUS two-field, Omega free

Scientific guardrails:
  * epsilon is NOT fitted to rotation curves: rotation data do not independently
    identify a photon/dark-photon optical residual parameter.
  * Omega is tested as a nested extension: Omega=0 is the null.
  * QCAUS density is renormalized to a fixed target halo mass so changing Omega
    cannot silently create or destroy mass.
  * AIC/BIC and Delta-chi2 are reported; no "detection" label is fabricated.
  * The app accepts standard SPARC-style per-galaxy files or CSV uploads.

Run:
    streamlit run app.py

Self-test:
    python app.py --test
"""

from __future__ import annotations

import io
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.optimize import least_squares, minimize
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

try:
    import streamlit as st
    HAS_STREAMLIT = True
except Exception:
    HAS_STREAMLIT = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

G_KPC_KMS_MSUN = 4.30091e-6  # kpc (km/s)^2 / Msun


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FitResult:
    name: str
    params: Dict[str, float]
    chi2: float
    dof: int
    aic: float
    bic: float
    n_params: int
    success: bool
    message: str


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def finite_arrays(*arrays: np.ndarray) -> bool:
    return all(np.all(np.isfinite(np.asarray(a, dtype=float))) for a in arrays)


def safe_log10(x: float) -> float:
    return float(np.log10(max(float(x), 1e-30)))


def weighted_chi2(obs: np.ndarray, model: np.ndarray, err: np.ndarray) -> float:
    e = np.maximum(np.asarray(err, dtype=float), 1e-6)
    r = (np.asarray(obs, dtype=float) - np.asarray(model, dtype=float)) / e
    return float(np.sum(r * r))


def information_criteria(chi2: float, n_params: int, n_obs: int) -> Tuple[float, float]:
    n = max(int(n_obs), 1)
    k = max(int(n_params), 0)
    aic = chi2 + 2.0 * k
    bic = chi2 + k * np.log(n)
    return float(aic), float(bic)


def deduplicate_sorted(
    r: np.ndarray,
    *ys: np.ndarray
) -> Tuple[np.ndarray, Tuple[np.ndarray, ...]]:
    idx = np.argsort(r)
    rs = np.asarray(r, dtype=float)[idx]
    ys_sorted = [np.asarray(y, dtype=float)[idx] for y in ys]

    keep = np.concatenate(([True], np.diff(rs) > 0))
    rs = rs[keep]
    ys_sorted = [y[keep] for y in ys_sorted]
    return rs, tuple(ys_sorted)


# ---------------------------------------------------------------------------
# SPARC data loading
# ---------------------------------------------------------------------------

SPARC_CANONICAL = [
    "R",
    "Vobs",
    "e_Vobs",
    "Vgas",
    "Vdisk",
    "Vbul",
    "SBdisk",
    "SBbul",
    "Q",
]


def _clean_columns(cols):
    out = []
    for c in cols:
        s = str(c).strip()
        s = s.replace("[", "").replace("]", "").replace("(", "").replace(")", "")
        out.append(s)
    return out


def load_sparc_dataframe(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """
    Accept:
      * CSV with named columns
      * whitespace-delimited SPARC-like .dat/.txt files
      * SPARC rotmod files with 9 numeric columns
    """
    name = filename.lower()

    # First attempt: normal CSV.
    try:
        text = file_bytes.decode("utf-8", errors="ignore")
    except Exception:
        text = ""

    # Drop comment-only lines for easier parsing.
    lines = []
    for line in text.splitlines():
        if not line.strip():
            continue
        if line.lstrip().startswith("#"):
            continue
        lines.append(line)

    cleaned = "\n".join(lines)

    df: Optional[pd.DataFrame] = None

    # CSV / comma-separated.
    if "," in cleaned:
        try:
            df = pd.read_csv(io.StringIO(cleaned))
        except Exception:
            df = None

    # Whitespace-separated.
    if df is None:
        try:
            raw = pd.read_csv(io.StringIO(cleaned), sep=r"\s+", header=None, engine="python")
            df = raw
        except Exception as exc:
            raise ValueError(f"Could not parse {filename}: {exc}") from exc

    if df is None or df.empty:
        raise ValueError(f"No tabular data found in {filename}")

    # Named-column normalization.
    lower = {str(c).strip().lower(): c for c in df.columns}

    aliases = {
        "R": ["r", "radius", "radius_kpc", "r_kpc", "r(kpc)"],
        "Vobs": ["vobs", "v_obs", "vobs_kms", "vobs(km/s)", "velocity"],
        "e_Vobs": ["evobs", "e_vobs", "errvobs", "sigma_v", "error", "e_vobs(km/s)"],
        "Vgas": ["vgas", "v_gas", "gas"],
        "Vdisk": ["vdisk", "v_disk", "disk"],
        "Vbul": ["vbul", "v_bul", "bulge"],
    }

    mapped = {}
    for canonical, choices in aliases.items():
        for key in choices:
            if key in lower:
                mapped[canonical] = lower[key]
                break

    if all(k in mapped for k in ["R", "Vobs", "e_Vobs"]):
        out = pd.DataFrame({k: pd.to_numeric(df[mapped[k]], errors="coerce") for k in mapped})
    else:
        # SPARC standard numeric layout:
        # R, Vobs, e_Vobs, Vgas, Vdisk, Vbul, SBdisk, SBbul, Q
        if df.shape[1] < 6:
            raise ValueError(
                "Need named columns R/Vobs/e_Vobs or a SPARC-style table with "
                "at least 6 numeric columns: R Vobs e_Vobs Vgas Vdisk Vbul"
            )
        num = df.apply(pd.to_numeric, errors="coerce")
        out = pd.DataFrame(
            {
                "R": num.iloc[:, 0],
                "Vobs": num.iloc[:, 1],
                "e_Vobs": num.iloc[:, 2],
                "Vgas": num.iloc[:, 3] if num.shape[1] > 3 else 0.0,
                "Vdisk": num.iloc[:, 4] if num.shape[1] > 4 else 0.0,
                "Vbul": num.iloc[:, 5] if num.shape[1] > 5 else 0.0,
            }
        )

    for c in ["Vgas", "Vdisk", "Vbul"]:
        if c not in out:
            out[c] = 0.0

    out = out.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["R", "Vobs", "e_Vobs"]
    )

    # Physical / numerical guards.
    out = out[
        (out["R"] > 0)
        & (out["e_Vobs"] > 0)
        & (out["Vobs"] >= 0)
    ].copy()

    if len(out) < 6:
        raise ValueError("At least 6 valid rotation-curve points are required.")

    out = out.sort_values("R").reset_index(drop=True)

    out["Vgas"] = pd.to_numeric(out["Vgas"], errors="coerce").fillna(0.0)
    out["Vdisk"] = pd.to_numeric(out["Vdisk"], errors="coerce").fillna(0.0)
    out["Vbul"] = pd.to_numeric(out["Vbul"], errors="coerce").fillna(0.0)

    return out


def demo_sparc_like() -> pd.DataFrame:
    """Deterministic synthetic sanity-check dataset."""
    r = np.geomspace(0.2, 45.0, 42)
    vgas = 28.0 * (1.0 - np.exp(-r / 3.0))
    vdisk = 85.0 * (1.0 - np.exp(-r / 5.5))
    vbul = 45.0 * np.exp(-r / 1.5)

    # NFW-like synthetic halo.
    rho0 = 6.0e7
    rs = 5.5
    rho = rho0 / ((r / rs) * (1.0 + r / rs) ** 2)
    shell = 4.0 * np.pi * r * r * rho
    dr = np.gradient(r)
    m_enc = np.cumsum(shell * dr)
    vhalo = np.sqrt(np.maximum(G_KPC_KMS_MSUN * m_enc / r, 0.0))

    vtrue = np.sqrt(vgas**2 + 0.65 * vdisk**2 + vbul**2 + vhalo**2)
    rng = np.random.default_rng(42)
    err = np.full_like(r, 4.0)
    vobs = np.maximum(vtrue + rng.normal(0.0, err), 0.0)

    return pd.DataFrame(
        {
            "R": r,
            "Vobs": vobs,
            "e_Vobs": err,
            "Vgas": vgas,
            "Vdisk": vdisk,
            "Vbul": vbul,
        }
    )


# ---------------------------------------------------------------------------
# Baryons
# ---------------------------------------------------------------------------

def baryonic_velocity(
    vgas: np.ndarray,
    vdisk: np.ndarray,
    vbul: np.ndarray,
    upsilon_disk: float,
    upsilon_bulge: float,
) -> np.ndarray:
    """
    Standard rotation-curve decomposition:
      V_bar^2 = V_gas^2 + Υ_d V_disk^2 + Υ_b V_bul^2
    """
    v2 = (
        np.asarray(vgas, dtype=float) ** 2
        + float(upsilon_disk) * np.asarray(vdisk, dtype=float) ** 2
        + float(upsilon_bulge) * np.asarray(vbul, dtype=float) ** 2
    )
    return np.sqrt(np.maximum(v2, 0.0))


# Backwards-compatible helper name.
def baryonic_mass_from_components(
    R_kpc: np.ndarray,
    Vgas: np.ndarray,
    Vdisk: np.ndarray,
    Vbul: np.ndarray,
    upsilon_disk: float = 0.5,
    upsilon_bulge: float = 0.7,
) -> np.ndarray:
    """
    Returns a spherical-equivalent enclosed baryonic mass from the
    baryonic rotation contribution. This is a convenience diagnostic;
    the model fits below use component velocities directly.
    """
    vb = baryonic_velocity(Vgas, Vdisk, Vbul, upsilon_disk, upsilon_bulge)
    r = np.maximum(np.asarray(R_kpc, dtype=float), 1e-6)
    return r * vb**2 / G_KPC_KMS_MSUN


# ---------------------------------------------------------------------------
# Halo / FDM models
# ---------------------------------------------------------------------------

def nfw_enclosed_mass(r_kpc: np.ndarray, rho_s: float, r_s: float) -> np.ndarray:
    x = np.maximum(np.asarray(r_kpc, dtype=float), 1e-8) / max(float(r_s), 1e-8)
    return 4.0 * np.pi * float(rho_s) * max(float(r_s), 1e-8) ** 3 * (
        np.log1p(x) - x / (1.0 + x)
    )


def nfw_velocity(r_kpc: np.ndarray, rho_s: float, r_s: float) -> np.ndarray:
    r = np.maximum(np.asarray(r_kpc, dtype=float), 1e-8)
    m = nfw_enclosed_mass(r, rho_s, r_s)
    return np.sqrt(np.maximum(G_KPC_KMS_MSUN * m / r, 0.0))


def fdm_soliton_density(
    r_kpc: np.ndarray,
    m22: float,
    rho_c_scale: float = 1.0
) -> np.ndarray:
    """
    Schive-type normalized core profile:
      rho = rho_c / [1 + 0.091 (r/r_c)^2]^8
      r_c = 1.6 / m22 kpc

    The amplitude is supplied as a scale parameter and can be renormalized
    to a target enclosed mass by the caller.
    """
    r = np.asarray(r_kpc, dtype=float)
    m22 = max(float(m22), 1e-4)
    rc = 1.6 / m22
    rho_c = max(float(rho_c_scale), 1e-20)
    return rho_c / np.power(1.0 + 0.091 * (r / rc) ** 2, 8.0)


def enclosed_mass_from_density(r_kpc: np.ndarray, rho: np.ndarray) -> np.ndarray:
    r = np.asarray(r_kpc, dtype=float)
    rho = np.asarray(rho, dtype=float)
    order = np.argsort(r)
    rr = r[order]
    rh = rho[order]
    if len(rr) < 2:
        return np.array([0.0])
    shell = 4.0 * np.pi * rr**2 * rh
    m = np.zeros_like(rr)
    m[1:] = np.cumsum(0.5 * (shell[1:] + shell[:-1]) * np.diff(rr))
    out = np.empty_like(m)
    out[order] = m
    return out


def normalized_profile_to_mass(
    r_kpc: np.ndarray,
    rho_shape: np.ndarray,
    target_mass_msun: float,
) -> np.ndarray:
    """
    Renormalize a positive density shape so its enclosed mass at the largest
    supplied radius equals target_mass_msun.
    """
    rr = np.asarray(r_kpc, dtype=float)
    shape = np.maximum(np.asarray(rho_shape, dtype=float), 0.0)

    # Normalize with a sorted numerical integral.
    order = np.argsort(rr)
    rsort = rr[order]
    psort = shape[order]
    shell = 4.0 * np.pi * rsort**2 * psort
    mass_shape = np.trapz(shell, rsort)

    if not np.isfinite(mass_shape) or mass_shape <= 0:
        return np.zeros_like(shape)

    scale = max(float(target_mass_msun), 0.0) / mass_shape
    return shape * scale


def two_field_density(
    r_kpc: np.ndarray,
    m22: float,
    omega: float,
    target_mass_msun: float,
    r_s_halo: Optional[float] = None,
) -> np.ndarray:
    """
    Conservative, mass-preserving QCAUS two-field proxy for rotation-curve
    testing.

    The interference factor is:
      1 + Omega * cos(r / r_beat)

    and the entire density is renormalized to the supplied target mass.
    This keeps Omega from changing the total halo mass by construction.

    IMPORTANT:
      This is an explicitly phenomenological QCAUS proxy, not a derived
      solution of a coupled Schrödinger-Poisson field theory.
    """
    r = np.asarray(r_kpc, dtype=float)
    m22 = max(float(m22), 1e-4)
    omega = float(np.clip(omega, -0.95, 0.95))

    base = fdm_soliton_density(r, m22, rho_c_scale=1.0)

    # Optional weak outer envelope so the model can be compared over extended
    # SPARC radii without pretending that a pure soliton is the whole halo.
    if r_s_halo is None:
        r_s_halo = max(3.0, float(np.nanmedian(r)) / 2.0)
    envelope = 1.0 / (1.0 + (r / max(float(r_s_halo), 1e-6)) ** 2)

    r_beat = max(0.15, 1.0 / m22)
    interference = 1.0 + omega * np.cos(2.0 * np.pi * r / r_beat)

    rho_shape = np.maximum(base * envelope * interference, 0.0)
    return normalized_profile_to_mass(r, rho_shape, target_mass_msun)


def qcaus_halo_velocity(
    r_kpc: np.ndarray,
    m22: float,
    omega: float,
    log10_target_mass: float,
    r_s_halo: float,
) -> np.ndarray:
    target_mass = 10.0 ** float(log10_target_mass)
    rho = two_field_density(
        r_kpc=r_kpc,
        m22=m22,
        omega=omega,
        target_mass_msun=target_mass,
        r_s_halo=r_s_halo,
    )
    m_enc = enclosed_mass_from_density(r_kpc, rho)
    r = np.maximum(np.asarray(r_kpc, dtype=float), 1e-8)
    return np.sqrt(np.maximum(G_KPC_KMS_MSUN * m_enc / r, 0.0))


# ---------------------------------------------------------------------------
# Model prediction helpers
# ---------------------------------------------------------------------------

def model_baryons(
    df: pd.DataFrame,
    upsilon_disk: float,
    upsilon_bulge: float,
) -> np.ndarray:
    return baryonic_velocity(
        df["Vgas"].to_numpy(float),
        df["Vdisk"].to_numpy(float),
        df["Vbul"].to_numpy(float),
        upsilon_disk,
        upsilon_bulge,
    )


def model_nfw(
    df: pd.DataFrame,
    log10_rho_s: float,
    r_s: float,
    upsilon_disk: float,
    upsilon_bulge: float,
) -> np.ndarray:
    vb = model_baryons(df, upsilon_disk, upsilon_bulge)
    vh = nfw_velocity(df["R"].to_numpy(float), 10.0**log10_rho_s, r_s)
    return np.sqrt(vb**2 + vh**2)


def model_fdm_nfw(
    df: pd.DataFrame,
    log10_m22: float,
    log10_rho_s: float,
    r_s: float,
    log10_mhalo: float,
    upsilon_disk: float,
    upsilon_bulge: float,
) -> np.ndarray:
    vb = model_baryons(df, upsilon_disk, upsilon_bulge)
    r = df["R"].to_numpy(float)

    m22 = 10.0 ** float(log10_m22)
    # FDM core mass set as a fraction of the halo mass; this is intentionally
    # simple and documented as a phenomenological fitting model.
    mhalo = 10.0 ** float(log10_mhalo)
    mcore = 0.25 * mhalo

    # Use a finite-grid density normalized to mcore.
    rho_shape = fdm_soliton_density(r, m22, 1.0)
    rho = normalized_profile_to_mass(r, rho_shape, mcore)
    m_enc_core = enclosed_mass_from_density(r, rho)
    vcore = np.sqrt(np.maximum(G_KPC_KMS_MSUN * m_enc_core / np.maximum(r, 1e-8), 0.0))

    vhalo = nfw_velocity(r, 10.0**log10_rho_s, r_s)
    return np.sqrt(vb**2 + vcore**2 + vhalo**2)


def model_qcaus(
    df: pd.DataFrame,
    log10_m22: float,
    log10_mhalo: float,
    r_s_halo: float,
    omega: float,
    upsilon_disk: float,
    upsilon_bulge: float,
) -> np.ndarray:
    vb = model_baryons(df, upsilon_disk, upsilon_bulge)
    r = df["R"].to_numpy(float)

    m22 = 10.0 ** float(log10_m22)
    vq = qcaus_halo_velocity(
        r_kpc=r,
        m22=m22,
        omega=omega,
        log10_target_mass=log10_mhalo,
        r_s_halo=r_s_halo,
    )
    return np.sqrt(vb**2 + vq**2)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def _fit_least_squares(
    residual_fn,
    x0: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> Tuple[np.ndarray, bool, str]:
    if not HAS_SCIPY:
        raise RuntimeError("SciPy is required for auto-fitting.")
    res = least_squares(
        residual_fn,
        x0=np.asarray(x0, dtype=float),
        bounds=(np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)),
        max_nfev=3000,
    )
    return np.asarray(res.x, dtype=float), bool(res.success), str(res.message)


def fit_baryons_only(
    df: pd.DataFrame,
    bounds_ups=(0.0, 2.0),
) -> FitResult:
    r = df
    obs = r["Vobs"].to_numpy(float)
    err = r["e_Vobs"].to_numpy(float)

    x0 = np.array([0.5, 0.7], dtype=float)
    lo = np.array([bounds_ups[0], 0.0], dtype=float)
    hi = np.array([bounds_ups[1], 2.0], dtype=float)

    def res(x):
        pred = model_baryons(r, x[0], x[1])
        return (obs - pred) / np.maximum(err, 1e-6)

    x, ok, msg = _fit_least_squares(res, x0, lo, hi)
    chi2 = float(np.sum(res(x) ** 2))
    aic, bic = information_criteria(chi2, 2, len(df))
    return FitResult(
        "Baryons only",
        {"upsilon_disk": x[0], "upsilon_bulge": x[1]},
        chi2,
        max(len(df) - 2, 1),
        aic,
        bic,
        2,
        ok,
        msg,
    )


def fit_nfw(df: pd.DataFrame) -> FitResult:
    obs = df["Vobs"].to_numpy(float)
    err = df["e_Vobs"].to_numpy(float)

    x0 = np.array([7.0, 8.0, 0.5, 0.7], dtype=float)
    lo = np.array([4.0, 0.5, 0.0, 0.0], dtype=float)
    hi = np.array([10.5, 50.0, 2.0, 2.0], dtype=float)

    def res(x):
        pred = model_nfw(df, x[0], x[1], x[2], x[3])
        return (obs - pred) / np.maximum(err, 1e-6)

    x, ok, msg = _fit_least_squares(res, x0, lo, hi)
    rr = res(x)
    chi2 = float(np.sum(rr**2))
    aic, bic = information_criteria(chi2, 4, len(df))
    return FitResult(
        "NFW",
        {
            "log10_rho_s": x[0],
            "r_s": x[1],
            "upsilon_disk": x[2],
            "upsilon_bulge": x[3],
        },
        chi2,
        max(len(df) - 4, 1),
        aic,
        bic,
        4,
        ok,
        msg,
    )


def fit_fdm_nfw(df: pd.DataFrame) -> FitResult:
    obs = df["Vobs"].to_numpy(float)
    err = df["e_Vobs"].to_numpy(float)

    x0 = np.array([-0.2, 7.0, 8.0, 10.5, 0.5, 0.7], dtype=float)
    lo = np.array([-2.0, 4.0, 0.5, 8.0, 0.0, 0.0], dtype=float)
    hi = np.array([1.3, 10.5, 50.0, 12.5, 2.0, 2.0], dtype=float)

    def res(x):
        pred = model_fdm_nfw(df, x[0], x[1], x[2], x[3], x[4], x[5])
        return (obs - pred) / np.maximum(err, 1e-6)

    x, ok, msg = _fit_least_squares(res, x0, lo, hi)
    rr = res(x)
    chi2 = float(np.sum(rr**2))
    aic, bic = information_criteria(chi2, 6, len(df))
    return FitResult(
        "FDM + NFW",
        {
            "log10_m22": x[0],
            "log10_rho_s": x[1],
            "r_s": x[2],
            "log10_mhalo": x[3],
            "upsilon_disk": x[4],
            "upsilon_bulge": x[5],
        },
        chi2,
        max(len(df) - 6, 1),
        aic,
        bic,
        6,
        ok,
        msg,
    )


def fit_qcaus(df: pd.DataFrame, omega_free: bool) -> FitResult:
    obs = df["Vobs"].to_numpy(float)
    err = df["e_Vobs"].to_numpy(float)

    # QCAUS rotation-curve fit intentionally excludes epsilon.
    # epsilon belongs to an independent photon/optical observable.
    if omega_free:
        x0 = np.array([-0.2, 10.5, 8.0, 0.35, 0.5, 0.7], dtype=float)
        lo = np.array([-2.0, 8.0, 0.3, 0.0, 0.0, 0.0], dtype=float)
        hi = np.array([1.3, 12.5, 80.0, 0.95, 2.0, 2.0], dtype=float)
        n_params = 6
        name = "QCAUS two-field (Omega free)"
    else:
        x0 = np.array([-0.2, 10.5, 8.0, 0.5, 0.7], dtype=float)
        lo = np.array([-2.0, 8.0, 0.3, 0.0, 0.0], dtype=float)
        hi = np.array([1.3, 12.5, 80.0, 2.0, 2.0], dtype=float)
        n_params = 5
        name = "QCAUS two-field (Omega=0)"

    def res(x):
        if omega_free:
            log_m22, log_mhalo, r_s, omega, ud, ub = x
        else:
            log_m22, log_mhalo, r_s, ud, ub = x
            omega = 0.0
        pred = model_qcaus(
            df,
            log10_m22=log_m22,
            log10_mhalo=log_mhalo,
            r_s_halo=r_s,
            omega=omega,
            upsilon_disk=ud,
            upsilon_bulge=ub,
        )
        return (obs - pred) / np.maximum(err, 1e-6)

    x, ok, msg = _fit_least_squares(res, x0, lo, hi)
    rr = res(x)
    chi2 = float(np.sum(rr**2))
    aic, bic = information_criteria(chi2, n_params, len(df))

    if omega_free:
        params = {
            "m22": 10.0 ** x[0],
            "log10_mhalo": x[1],
            "r_s": x[2],
            "Omega": x[3],
            "upsilon_disk": x[4],
            "upsilon_bulge": x[5],
        }
    else:
        params = {
            "m22": 10.0 ** x[0],
            "log10_mhalo": x[1],
            "r_s": x[2],
            "Omega": 0.0,
            "upsilon_disk": x[3],
            "upsilon_bulge": x[4],
        }

    return FitResult(
        name,
        params,
        chi2,
        max(len(df) - n_params, 1),
        aic,
        bic,
        n_params,
        ok,
        msg,
    )


def predict_from_fit(df: pd.DataFrame, fit: FitResult) -> np.ndarray:
    p = fit.params
    if fit.name == "Baryons only":
        return model_baryons(df, p["upsilon_disk"], p["upsilon_bulge"])
    if fit.name == "NFW":
        return model_nfw(
            df,
            p["log10_rho_s"],
            p["r_s"],
            p["upsilon_disk"],
            p["upsilon_bulge"],
        )
    if fit.name == "FDM + NFW":
        return model_fdm_nfw(
            df,
            np.log10(p["log10_m22"]) if "m22" not in p else p["m22"],
            p["log10_rho_s"],
            p["r_s"],
            p["log10_mhalo"],
            p["upsilon_disk"],
            p["upsilon_bulge"],
        )

    # QCAUS parameterization.
    return model_qcaus(
        df,
        np.log10(p["m22"]),
        p["log10_mhalo"],
        p["r_s"],
        p["Omega"],
        p["upsilon_disk"],
        p["upsilon_bulge"],
    )


# ---------------------------------------------------------------------------
# Comparison / diagnostics
# ---------------------------------------------------------------------------

def results_table(results) -> pd.DataFrame:
    rows = []
    best_aic = min(r.aic for r in results)
    best_bic = min(r.bic for r in results)

    for r in results:
        rows.append(
            {
                "Model": r.name,
                "chi2": r.chi2,
                "reduced_chi2": r.chi2 / max(r.dof, 1),
                "AIC": r.aic,
                "ΔAIC": r.aic - best_aic,
                "BIC": r.bic,
                "ΔBIC": r.bic - best_bic,
                "k": r.n_params,
                "Success": r.success,
            }
        )
    return pd.DataFrame(rows)


def qcaus_nested_delta_chi2(
    qcaus_null: FitResult,
    qcaus_free: FitResult,
) -> float:
    return float(qcaus_null.chi2 - qcaus_free.chi2)


def rms_velocity_residual(
    obs: np.ndarray,
    model: np.ndarray,
) -> float:
    return float(np.sqrt(np.mean((np.asarray(obs) - np.asarray(model)) ** 2)))


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def run_self_tests() -> bool:
    if not HAS_SCIPY:
        print("[FAIL] SciPy unavailable; required for fitting tests.")
        return False

    ok = True

    def chk(name, cond):
        nonlocal ok
        passed = bool(cond)
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
        ok = ok and passed

    df = demo_sparc_like()
    chk("demo dataset has >= 6 rows", len(df) >= 6)
    chk("finite data", finite_arrays(*[df[c].to_numpy(float) for c in df.columns]))
    chk("positive radii", bool(np.all(df["R"] > 0)))

    vb = model_baryons(df, 0.5, 0.7)
    chk("baryonic velocity finite", bool(np.all(np.isfinite(vb))))

    rho = two_field_density(
        df["R"].to_numpy(float),
        m22=2.0,
        omega=0.4,
        target_mass_msun=1e10,
        r_s_halo=8.0,
    )
    m_enc = enclosed_mass_from_density(df["R"].to_numpy(float), rho)
    chk("QCAUS density non-negative", bool(np.all(rho >= 0)))
    chk("QCAUS mass finite", bool(np.all(np.isfinite(m_enc))))
    chk("QCAUS mass non-decreasing", bool(np.all(np.diff(m_enc) >= -1e-6)))

    # Same target mass should be preserved when Omega changes on the grid.
    rho0 = two_field_density(
        df["R"].to_numpy(float), 2.0, 0.0, 1e10, 8.0
    )
    rho4 = two_field_density(
        df["R"].to_numpy(float), 2.0, 0.4, 1e10, 8.0
    )
    m0 = enclosed_mass_from_density(df["R"].to_numpy(float), rho0)[-1]
    m4 = enclosed_mass_from_density(df["R"].to_numpy(float), rho4)[-1]
    chk("Omega preserves target mass approximately", abs(m0 - m4) / 1e10 < 1e-8)

    b = fit_baryons_only(df)
    n = fit_nfw(df)
    f = fit_fdm_nfw(df)
    q0 = fit_qcaus(df, omega_free=False)
    q = fit_qcaus(df, omega_free=True)

    chk("baryons fit completes", b.success)
    chk("NFW fit completes", n.success)
    chk("FDM+NFW fit completes", f.success)
    chk("QCAUS Omega=0 fit completes", q0.success)
    chk("QCAUS Omega-free fit completes", q.success)
    chk("Omega=0 nested null", q0.params["Omega"] == 0.0)
    chk("rotation fit contains no epsilon", "epsilon" not in q.params)

    print(f"\nQCAUS Δchi2 (Omega=0 minus free) = {qcaus_nested_delta_chi2(q0,q):.4f}")
    print("\nModel comparison:")
    print(results_table([b, n, f, q0, q]).to_string(index=False))

    return ok


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def plot_rotation_curves(df: pd.DataFrame, fits) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5.4))
    r = df["R"].to_numpy(float)
    obs = df["Vobs"].to_numpy(float)
    err = df["e_Vobs"].to_numpy(float)

    ax.errorbar(
        r, obs, yerr=err, fmt="o", ms=4, capsize=2,
        label="SPARC / input observations"
    )

    for fit in fits:
        pred = predict_from_fit(df, fit)
        ax.plot(r, pred, lw=2, label=fit.name)

    ax.set_xscale("log")
    ax.set_xlabel("Radius (kpc)")
    ax.set_ylabel("Circular velocity (km/s)")
    ax.set_title("Rotation-curve model comparison")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def main():
    st.set_page_config(
        page_title="QCAUS SPARC Validation Lab",
        page_icon="🧪",
        layout="wide",
    )

    st.title("🧪 QCAUS SPARC Rotation-Curve Validation Lab")
    st.caption(
        "Nested model comparison: baryons → NFW → FDM+NFW → "
        "QCAUS two-field. Rotation data fit m and Ω; ε is intentionally "
        "reserved for an independent photon/optical observable."
    )

    with st.sidebar:
        st.header("Dataset")
        source = st.radio(
            "Data source",
            ["Synthetic sanity-check", "Upload SPARC galaxy file"],
            index=1,
        )

        uploaded = None
        if source == "Upload SPARC galaxy file":
            uploaded = st.file_uploader(
                "Upload CSV / TXT / DAT",
                type=["csv", "txt", "dat"],
                help=(
                    "Named columns or standard SPARC-style numeric layout. "
                    "Expected core fields: R, Vobs, e_Vobs, Vgas, Vdisk, Vbul."
                ),
            )

        st.divider()
        st.header("Baryonic priors")
        disk_prior = st.slider("Initial disk M/L", 0.0, 2.0, 0.5, 0.05)
        bulge_prior = st.slider("Initial bulge M/L", 0.0, 2.0, 0.7, 0.05)

        st.divider()
        st.header("Run")
        do_fit = st.button("🚀 Run full model comparison", use_container_width=True)

    if source == "Synthetic sanity-check":
        df = demo_sparc_like()
        source_name = "Synthetic deterministic test galaxy"
    else:
        if uploaded is None:
            st.info(
                "Upload a SPARC galaxy rotation-curve file. "
                "No observational result is calculated until data are supplied."
            )
            return
        try:
            df = load_sparc_dataframe(uploaded.getvalue(), uploaded.name)
            source_name = uploaded.name
        except Exception as exc:
            st.error(f"Dataset parsing failed: {exc}")
            return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Data points", f"{len(df)}")
    c2.metric("R range", f"{df['R'].min():.2f}–{df['R'].max():.2f} kpc")
    c3.metric("V max", f"{df['Vobs'].max():.1f} km/s")
    c4.metric("Input", source_name)

    st.subheader("Observed rotation curve")
    fig0, ax0 = plt.subplots(figsize=(10, 4))
    ax0.errorbar(
        df["R"], df["Vobs"], yerr=df["e_Vobs"],
        fmt="o", ms=4, capsize=2
    )
    ax0.set_xscale("log")
    ax0.set_xlabel("Radius (kpc)")
    ax0.set_ylabel("Vobs (km/s)")
    ax0.grid(alpha=0.25)
    fig0.tight_layout()
    st.pyplot(fig0)
    plt.close(fig0)

    if "fit_results" not in st.session_state:
        st.session_state.fit_results = None

    if do_fit:
        if not HAS_SCIPY:
            st.error("SciPy is required for fitting. Add scipy to requirements.txt.")
            return

        with st.spinner("Running nested model comparison..."):
            try:
                # Actual fitting:
                fit_b = fit_baryons_only(df)
                fit_n = fit_nfw(df)
                fit_f = fit_fdm_nfw(df)
                fit_q0 = fit_qcaus(df, omega_free=False)
                fit_q = fit_qcaus(df, omega_free=True)
                fits = [fit_b, fit_n, fit_f, fit_q0, fit_q]

                st.session_state.fit_results = fits
            except Exception as exc:
                st.error(f"Fit failed: {exc}")
                return

    fits = st.session_state.fit_results
    if fits is None:
        st.info("Run the full model comparison to generate fit statistics.")
        return

    st.subheader("Model comparison")
    table = results_table(fits)
    st.dataframe(
        table.style.format(
            {
                "chi2": "{:.2f}",
                "reduced_chi2": "{:.3f}",
                "AIC": "{:.2f}",
                "ΔAIC": "{:.2f}",
                "BIC": "{:.2f}",
                "ΔBIC": "{:.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    fig = plot_rotation_curves(df, fits)
    st.pyplot(fig)
    plt.close(fig)

    q0 = next(x for x in fits if x.name == "QCAUS two-field (Omega=0)")
    q = next(x for x in fits if x.name == "QCAUS two-field (Omega free)")
    delta_chi2 = qcaus_nested_delta_chi2(q0, q)

    st.subheader("QCAUS nested test")
    a, b, c, d = st.columns(4)
    a.metric("Ω=0 χ²", f"{q0.chi2:.2f}")
    b.metric("Ω-free χ²", f"{q.chi2:.2f}")
    c.metric("Δχ²", f"{delta_chi2:.2f}")
    d.metric("Best-fit Ω", f"{q.params['Omega']:.4f}")

    st.warning(
        "A lower χ² with Ω free is not, by itself, evidence for a two-field "
        "effect. Compare ΔAIC/ΔBIC, parameter boundaries, residual structure, "
        "and independent validation galaxies."
    )

    st.subheader("Best-fit parameters")
    for fit in fits:
        with st.expander(fit.name, expanded=(fit.name == "QCAUS two-field (Omega free)")):
            st.json({k: float(v) for k, v in fit.params.items()})

    # Residuals for the QCAUS pair.
    st.subheader("QCAUS residuals")
    r = df["R"].to_numpy(float)
    obs = df["Vobs"].to_numpy(float)
    err = np.maximum(df["e_Vobs"].to_numpy(float), 1e-6)
    pred0 = predict_from_fit(df, q0)
    predq = predict_from_fit(df, q)

    fig_r, ax_r = plt.subplots(figsize=(10, 4.3))
    ax_r.axhline(0, lw=1)
    ax_r.errorbar(
        r, obs - pred0, yerr=err, fmt="o", ms=3.8,
        label="QCAUS Ω=0 residual"
    )
    ax_r.plot(r, obs - predq, "x-", ms=4, lw=1.2, label="QCAUS Ω-free residual")
    ax_r.set_xscale("log")
    ax_r.set_xlabel("Radius (kpc)")
    ax_r.set_ylabel("Vobs − Vmodel (km/s)")
    ax_r.grid(alpha=0.25)
    ax_r.legend(fontsize=8)
    fig_r.tight_layout()
    st.pyplot(fig_r)
    plt.close(fig_r)

    rms0 = rms_velocity_residual(obs, pred0)
    rmsq = rms_velocity_residual(obs, predq)
    st.write(
        f"Velocity RMS residual: **Ω=0: {rms0:.2f} km/s** | "
        f"**Ω-free: {rmsq:.2f} km/s**"
    )

    st.subheader("Scientific interpretation")
    st.markdown(
        """
**What this app can test**

- Whether adding a QCAUS-shaped halo component improves a rotation-curve fit.
- Whether freeing Ω produces enough improvement to justify the extra parameter.
- Whether the inferred QCAUS parameters are stable across independent galaxies.

**What this app cannot establish from rotation curves alone**

- A photon/dark-photon mixing measurement ε.
- A dark-photon detection.
- A confirmation of the underlying two-field field theory.
- A cosmological constraint on the same m22.

**Required next step:** fit a held-out galaxy sample, then test the same m22
against an independent cosmological observable. The optical/photon sector
requires a separate observable and likelihood rather than fitting ε to V(r).
"""
    )

    with st.expander("Validation boundary"):
        st.markdown(
            """
1. **Mathematical:** deterministic equations and numerical invariants.
2. **Fit:** nested models on observed rotation curves.
3. **Model selection:** χ², AIC, BIC.
4. **Cross-validation:** unseen galaxies.
5. **Cross-domain:** cosmology / lensing / photon-sector observables.
6. **Claim:** no detection claim unless independent data support it.
"""
        )


if __name__ == "__main__":
    if "--test" in sys.argv:
        good = run_self_tests()
        raise SystemExit(0 if good else 1)
    if HAS_STREAMLIT:
        main()
    else:
        print("Streamlit is not installed. Run: pip install streamlit scipy pandas matplotlib")
        raise SystemExit(1)
