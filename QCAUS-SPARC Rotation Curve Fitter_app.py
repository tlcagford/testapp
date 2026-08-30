"""
QCAUS Two-Field Model SPARC Rotation Curve Fitter
Test the QCAUS model against real galaxy rotation curve data.
Includes auto-fit for 2 and 4 parameters.
Compatible with NumPy 2.0 (uses scipy.integrate.quad).
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
    url = "https://zenodo.org/records/19563417/files/rotation_curve_corpus_v7.json"
    
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
    from the SPARC survey.
    """)
    
    # Load data
    with st.spinner("Loading SPARC rotation curve corpus..."):
        corpus = load_corpus()
    
    galaxies = [g['galaxy'] for g in corpus['galaxies']]
    
    # Sidebar controls
    st.sidebar.header("Controls")
    
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
    st.sidebar.header("QCAUS Parameters")
    
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
    
    # --- Auto-Fit Buttons ---
    st.sidebar.markdown("---")
    
    if st.sidebar.button("🚀 Auto-Fit (rho0 & r_s)", use_container_width=True):
        if len(R_fit) > 3:
            with st.spinner("Fitting rho0 and r_s..."):
                M_baryon_fit = baryonic_mass_from_components(R_fit, Vgas_fit, Vdisk_fit, Vbul_fit)
                initial_guess = [10**st.session_state.log_rho0, st.session_state.r_s]
                try:
                    best_rho0, best_r_s = fit_qcaus_2param(
                        R_fit, Vobs_fit, errV_fit, M_baryon_fit,
                        st.session_state.epsilon, st.session_state.Omega,
                        initial_guess
                    )
                    st.session_state.log_rho0 = np.log10(best_rho0)
                    st.session_state.r_s = best_r_s
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"Fit failed: {e}")
        else:
            st.sidebar.warning("Not enough valid data points to fit.")
    
    if st.sidebar.button("🚀 Auto-Fit (All 4 Parameters)", use_container_width=True):
        if len(R_fit) > 5:
            with st.spinner("Fitting all parameters..."):
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
                    st.session_state.log_rho0 = np.log10(best_rho0)
                    st.session_state.r_s = best_r_s
                    st.session_state.epsilon = best_eps
                    st.session_state.Omega = best_Omega
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"Fit failed: {e}")
        else:
            st.sidebar.warning("Not enough valid data points to fit all 4 parameters.")
    
    # Get current parameters
    rho0 = 10**st.session_state.log_rho0
    r_s = st.session_state.r_s
    epsilon = st.session_state.epsilon
    Omega = st.session_state.Omega
    
    # Compute model
    M_baryon = baryonic_mass_from_components(R, Vgas, Vdisk, Vbul)
    V_qcaus = np.array([
        qcaus_rotation_curve(r, rho0, r_s, M_baryon[i], epsilon, Omega)
        for i, r in enumerate(R)
    ])
    
    # ========================================================================
    # 6. DISPLAY RESULTS
    # ========================================================================
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Rotation Curve")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar(R, Vobs, yerr=errV, fmt='o', color='black',
                   label=f'{selected_galaxy} (SPARC)', capsize=3, markersize=4)
        V_baryon = np.sqrt(Vgas**2 + Vdisk**2 + Vbul**2)
        ax.plot(R, V_baryon, '--', color='orange', label='Baryonic only')
        ax.plot(R, V_qcaus, '-', color='red', linewidth=2,
               label=f'QCAUS (rho0={rho0:.1e}, r_s={r_s:.2f}, eps={epsilon:.2f})')
        ax.set_xlabel('Radius [kpc]', fontsize=12)
        ax.set_ylabel('Circular Velocity [km/s]', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{selected_galaxy} — QCAUS Two-Field Model Fit')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Model Parameters")
        st.metric("Galaxy", selected_galaxy)
        st.metric("Data Points (total)", len(R))
        st.metric("Fit Points", len(R_fit))
        st.metric("rho0", f"{rho0:.2e} M_sun/kpc^3")
        st.metric("r_s", f"{r_s:.2f} kpc")
        st.metric("epsilon", f"{epsilon:.3f}")
        st.metric("Omega", f"{Omega:.3f}")
        
        if len(R_fit) > 0:
            M_baryon_fit = baryonic_mass_from_components(R_fit, Vgas_fit, Vdisk_fit, Vbul_fit)
            V_qcaus_fit = np.array([
                qcaus_rotation_curve(r, rho0, r_s, M_baryon_fit[i], epsilon, Omega)
                for i, r in enumerate(R_fit)
            ])
            residuals = Vobs_fit - V_qcaus_fit
            chi2 = np.sum((residuals / errV_fit)**2)
            dof = len(R_fit) - 4  # now fitting 4 parameters
            if dof > 0:
                reduced_chi2 = chi2 / dof
                st.metric("chi^2/dof", f"{reduced_chi2:.2f}")
                if reduced_chi2 < 2.0:
                    st.success("Excellent fit!")
                elif reduced_chi2 < 5.0:
                    st.info("Good fit.")
                elif reduced_chi2 < 20.0:
                    st.warning("Moderate fit. Try auto-fit all 4 parameters.")
                else:
                    st.error("Poor fit. Use auto-fit all 4 parameters.")
        
        st.info("""
        **About this fit**
        
        The QCAUS two-field model combines:
        - FDM soliton core: rho(r) = rho0 * [sin(kr)/(kr)]^2
        - PDP interference: eps * Omega * exp(-Omega * r^2)
        
        **Auto-fit (2 param)** optimizes rho0 and r_s.
        **Auto-fit (4 param)** optimizes all four.
        """)

if __name__ == "__main__":
    main()
