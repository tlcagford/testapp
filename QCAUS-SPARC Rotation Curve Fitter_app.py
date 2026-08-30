"""
QCAUS Two-Field Model SPARC Rotation Curve Fitter
Test the QCAUS model against real galaxy rotation curve data.
Includes auto-fit for 2 and 4 parameters, plot color configurations,
and custom diagnostic metrics.
"""

import json
import os
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.optimize import minimize
from scipy.integrate import quad

# ============================================================================
# 1. DATA LOADING
# ============================================================================

@st.cache_data
def load_corpus():
    """Download and load the unified HI rotation curve corpus from Zenodo."""
    filename = "rotation_curve_corpus_v7.json"
    url = "https://zenodo.org"
    
    if not os.path.exists(filename):
        with st.spinner(f"Downloading {filename} from Zenodo..."):
            urllib.request.urlretrieve(url, filename)
        st.success(f"Downloaded {filename}")
    
    with open(filename) as f:
        corpus = json.load(f)
    return corpus

# ============================================================================
# 2. QCAUS TWO-FIELD MODEL
# ============================================================================

G = 4.301e-9  # kpc (km/s)^2 / M_sun

def fdm_density(r, rho0, r_s):
    if r == 0:
        return rho0
    k = np.pi / r_s
    kr = k * r
    return rho0 * (np.sin(kr) / kr) ** 2

def fdm_mass_enclosed(r, rho0, r_s):
    if r == 0:
        return 0.0
    def integrand(rp):
        return 4 * np.pi * rp**2 * fdm_density(rp, rho0, r_s)
    # limit=200 prevents truncation errors inside optimization routines
    return quad(integrand, 0, r, limit=200)[0]

def qcaus_rotation_curve(r, rho0, r_s, M_baryon, epsilon=0.0, Omega=0.0):
    if r <= 0:
        return 0.0
    M_fdm = fdm_mass_enclosed(r, rho0, r_s)
    interference_factor = 1.0 + epsilon * Omega * np.exp(-(r / r_s)**2) * 0.1
    M_total = M_baryon + M_fdm * interference_factor
    return np.sqrt(G * M_total / r)

# ============================================================================
# 3. BARYONIC MASS
# ============================================================================

def baryonic_mass_from_components(r, Vgas, Vdisk, Vbul):
    V_baryon = np.sqrt(Vgas**2 + Vdisk**2 + Vbul**2)
    return r * V_baryon**2 / G

# ============================================================================
# 4. AUTO-FIT FUNCTIONS
# ============================================================================

def fit_qcaus_2param(R, Vobs, errV, M_baryon, epsilon, Omega, initial_guess):
    """Fit rho0 and r_s only."""
    def chi2(params):
        rho0, r_s = params
        V_model = np.array([
            qcaus_rotation_curve(r, rho0, r_s, M_baryon[i], epsilon, Omega)
            for i, r in enumerate(R)
        ])
        return np.sum(((Vobs - V_model) / errV)**2)
    bounds = [(1e2, 1e10), (0.1, 50.0)]
    result = minimize(chi2, initial_guess, bounds=bounds, method='L-BFGS-B')
    return result.x

def fit_qcaus_all(R, Vobs, errV, M_baryon, initial_guess):
    """Fit all four parameters: rho0, r_s, epsilon, Omega."""
    def chi2(params):
        rho0, r_s, eps, Omega = params
        V_model = np.array([
            qcaus_rotation_curve(r, rho0, r_s, M_baryon[i], eps, Omega)
            for i, r in enumerate(R)
        ])
        return np.sum(((Vobs - V_model) / errV)**2)
    bounds = [(1e2, 1e10), (0.1, 50.0), (0.0, 1.0), (0.0, 1.0)]
    result = minimize(chi2, initial_guess, bounds=bounds, method='L-BFGS-B')
    return result.x

# ============================================================================
# 5. STREAMLIT APP
# ============================================================================

def main():
    st.set_page_config(page_title="QCAUS SPARC Fitter", layout="wide")
    
    st.title("🌌 QCAUS Two-Field Model: SPARC Rotation Curve Fitter")
    st.markdown("""
    Test the QCAUS two-field FDM model against real galaxy rotation curve data 
    from the SPARC survey. Customize plot cosmetics and optimize weights on the fly.
    """)
    
    # Load data
    with st.spinner("Loading SPARC rotation curve corpus..."):
        corpus = load_corpus()
    
    galaxies = [g['galaxy'] for g in corpus['galaxies']]
    
    # --- Sidebar Controls ---
    st.sidebar.header("Data Selection")
    
    selected_galaxy = st.sidebar.selectbox(
        "Select Galaxy",
        galaxies,
        index=galaxies.index("DDO161") if "DDO161" in galaxies else 0
    )
    
    galaxy_data = next(g for g in corpus['galaxies'] if g['galaxy'] == selected_galaxy)
    data_points = galaxy_data['data']
    
    R = np.array([p['Rad'] for p in data_points])
    Vobs = np.array([p['Vobs'] for p in data_points])
    errV = np.array([p.get('errV', np.nan) for p in data_points])
    
    Vgas = np.array([p.get('Vgas', 0.0) for p in data_points])
    Vdisk = np.array([p.get('Vdisk', 0.0) for p in data_points])
    Vbul = np.array([p.get('Vbul', 0.0) for p in data_points])
    
    valid = ~np.isnan(errV) & (errV > 0) & (R > 0)
    R_fit = R[valid]
    Vobs_fit = Vobs[valid]
    errV_fit = errV[valid]
    Vgas_fit = Vgas[valid] if len(Vgas) == len(R) else np.zeros_like(R_fit)
    Vdisk_fit = Vdisk[valid] if len(Vdisk) == len(R) else np.zeros_like(R_fit)
    Vbul_fit = Vbul[valid] if len(Vbul) == len(R) else np.zeros_like(R_fit)
    
    # --- QCAUS Parameters with Session State ---
    st.sidebar.markdown("---")
    st.sidebar.header("Model Hyperparameters")
    
    if 'log_rho0' not in st.session_state:
        st.session_state.log_rho0 = 5.0
    if 'r_s' not in st.session_state:
        st.session_state.r_s = 1.0
    if 'epsilon' not in st.session_state:
        st.session_state.epsilon = 0.01
    if 'Omega' not in st.session_state:
        st.session_state.Omega = 0.5
    
    log_rho0 = st.sidebar.slider(
        "log10(rho0) [M_sun/kpc^3]",
        min_value=2.0, max_value=10.0, step=0.1,
        value=st.session_state.log_rho0
    )
    st.session_state.log_rho0 = log_rho0
    
    r_s = st.sidebar.slider(
        "r_s [kpc]",
        min_value=0.1, max_value=50.0, step=0.1,
        value=st.session_state.r_s
    )
    st.session_state.r_s = r_s
    
    epsilon = st.sidebar.slider(
        "epsilon (kinetic mixing)",
        min_value=0.0, max_value=1.0, step=0.01,
        value=st.session_state.epsilon
    )
    st.session_state.epsilon = epsilon
    
    Omega = st.sidebar.slider(
        "Omega (interference coherence)",
        min_value=0.0, max_value=1.0, step=0.05,
        value=st.session_state.Omega
    )
    st.session_state.Omega = Omega
    
    # --- Auto-Fit Actions ---
    st.sidebar.markdown("---")
    st.sidebar.header("Optimization Engines")
    
    if st.sidebar.button("🚀 Auto-Fit Core (rho0 & r_s)", use_container_width=True):
        if len(R_fit) > 2:
            with st.spinner("Executing L-BFGS-B parameter sweep..."):
                M_baryon_fit = baryonic_mass_from_components(R_fit, Vgas_fit, Vdisk_fit, Vbul_fit)
                initial_guess = [10**st.session_state.log_rho0, st.session_state.r_s]
                try:
                    best_rho0, best_r_s = fit_qcaus_2param(
                        R_fit, Vobs_fit, errV_fit, M_baryon_fit,
                        st.session_state.epsilon, st.session_state.Omega,
                        initial_guess
                    )
                    st.session_state.log_rho0 = float(np.log10(best_rho0))
                    st.session_state.r_s = float(best_r_s)
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"Fit failure occurred: {e}")
        else:
            st.sidebar.warning("Insufficient structural data variance to execute core fit.")
    
    if st.sidebar.button("🚀 Auto-Fit Full (4 Parameters)", use_container_width=True):
        if len(R_fit) > 4:
            with st.spinner("Executing multi-dimensional minimization..."):
                M_baryon_fit = baryonic_mass_from_components(R_fit, Vgas_fit, Vdisk_fit, Vbul_fit)
                initial_guess = [
                    10**st.session_state.log_rho0,
                    st.session_state.r_s,
                    st.session_state.epsilon,
                    st.session_state.Omega
                ]
                try:
                    best_rho0, best_r_s, best_eps, best_Omega = fit_qcaus_all(
                        R_fit, Vobs_fit, errV_fit, M_baryon_fit, initial_guess
                    )
                    st.session_state.log_rho0 = float(np.log10(best_rho0))
                    st.session_state.r_s = float(best_r_s)
                    st.session_state.epsilon = float(best_eps)
                    st.session_state.Omega = float(best_Omega)
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"Fit failure occurred: {e}")
        else:
            st.sidebar.warning("Degrees of freedom exceed total filtering vector metrics.")

    # --- Plot Design Controls ---
    st.sidebar.markdown("---")
    st.sidebar.header("Plot Presentation Styles")
    fit_line_color = st.sidebar.color_picker("Model Prediction Line", "#DC143C")
    baryon_line_color = st.sidebar.color_picker("Baryonic Profile Base", "#1E90FF")
    data_points_color = st.sidebar.color_picker("SPARC Observables", "#000000")

    # ============================================================================
    # 6. DASHBOARD METRICS AND PLOT COMPUTATION
    # ============================================================================
    
    # Calculate model predictions based on parameters
    M_baryon_fit = baryonic_mass_from_components(R_fit, Vgas_fit, Vdisk_fit, Vbul_fit)
    V_model = np.array([
        qcaus_rotation_curve(r, 10**log_rho0, r_s, M_baryon_fit[i], epsilon, Omega)
        for i, r in enumerate(R_fit)
    ])
    
