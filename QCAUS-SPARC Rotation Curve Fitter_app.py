"""
QCAUS Two-Field Model SPARC Rotation Curve Fitter
Test the QCAUS model against real galaxy rotation curve data.
"""

import json
import os
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.optimize import curve_fit
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

# Physical constants
G = 4.301e-9  # kpc (km/s)^2 / M_sun (gravitational constant in SPARC units)
M_sun = 1.0   # solar mass in solar masses

def fdm_density(r, rho0, r_s):
    """
    FDM soliton density profile.
    ρ(r) = ρ₀ [sin(kr)/(kr)]², where k = π/r_s
    """
    if r == 0:
        return rho0
    k = np.pi / r_s
    kr = k * r
    return rho0 * (np.sin(kr) / kr) ** 2

def fdm_mass_enclosed(r, rho0, r_s):
    """
    Enclosed mass of FDM soliton within radius r.
    M(r) = 4π ∫₀ʳ ρ(r') r'² dr'
    """
    def integrand(rp):
        return 4 * np.pi * rp**2 * fdm_density(rp, rho0, r_s)
    return quad(integrand, 0, r, limit=200)[0]

def qcaus_rotation_curve(r, rho0, r_s, M_baryon, epsilon=0.0, Omega=0.0):
    """
    QCAUS two-field model rotation curve.
    
    V_circ²(r) = G * [M_baryon(r) + M_FDM(r)] / r
    
    Parameters:
    - r: radius in kpc
    - rho0: central density of FDM soliton (M_sun/kpc³)
    - r_s: soliton scale radius (kpc)
    - M_baryon: enclosed baryonic mass at radius r (M_sun)
    - epsilon: kinetic mixing parameter (dimensionless)
    - Omega: interference coherence parameter (dimensionless)
    
    Note: epsilon and Omega affect the interference term in the density,
    which manifests as small-scale fluctuations in the rotation curve.
    For a first-order fit, we primarily use the soliton component.
    """
    # FDM contribution
    M_fdm = fdm_mass_enclosed(r, rho0, r_s)
    
    # Interference correction (small effect from epsilon and Omega)
    # For simplicity in this demonstration, we include it as a perturbation
    interference_factor = 1.0 + epsilon * Omega * np.exp(-(r / r_s)**2) * 0.1
    
    # Total enclosed mass
    M_total = M_baryon + M_fdm * interference_factor
    
    # Circular velocity
    V_circ = np.sqrt(G * M_total / r) if r > 0 else 0.0
    return V_circ

# ============================================================================
# 3. BARYONIC MASS MODEL (simplified)
# ============================================================================

def baryonic_mass_from_components(r, Vgas, Vdisk, Vbul):
    """
    Estimate enclosed baryonic mass from rotation curve components.
    M_baryon(r) = r * V_baryon² / G
    where V_baryon² = Vgas² + Vdisk² + Vbul²
    """
    V_baryon = np.sqrt(Vgas**2 + Vdisk**2 + Vbul**2)
    return r * V_baryon**2 / G

# ============================================================================
# 4. STREAMLIT APP
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
    
    # Galaxy selection
    selected_galaxy = st.sidebar.selectbox(
        "Select Galaxy",
        galaxies,
        index=galaxies.index("DDO161") if "DDO161" in galaxies else 0
    )
    
    # Get galaxy data
    galaxy_data = next(g for g in corpus['galaxies'] if g['galaxy'] == selected_galaxy)
    data_points = galaxy_data['data']
    
    # Extract observables
    R = np.array([p['Rad'] for p in data_points])
    Vobs = np.array([p['Vobs'] for p in data_points])
    errV = np.array([p.get('errV', np.nan) for p in data_points])
    
    # Extract baryonic components (if available)
    Vgas = np.array([p.get('Vgas', 0.0) for p in data_points])
    Vdisk = np.array([p.get('Vdisk', 0.0) for p in data_points])
    Vbul = np.array([p.get('Vbul', 0.0) for p in data_points])
    
    # Filter out points with missing errors
    valid = ~np.isnan(errV) & (errV > 0)
    R_fit = R[valid]
    Vobs_fit = Vobs[valid]
    errV_fit = errV[valid]
    Vgas_fit = Vgas[valid] if len(Vgas) == len(R) else np.zeros_like(R_fit)
    
    # QCAUS parameters
    st.sidebar.header("QCAUS Parameters")
    
    log_rho0 = st.sidebar.slider(
        "log₁₀(ρ₀) [M☉/kpc³]",
        min_value=2.0, max_value=8.0, value=5.0, step=0.1
    )
    rho0 = 10**log_rho0
    
    r_s = st.sidebar.slider(
        "rₛ [kpc]",
        min_value=0.1, max_value=10.0, value=1.0, step=0.1
    )
    
    epsilon = st.sidebar.slider(
        "ε (kinetic mixing)",
        min_value=0.0, max_value=1.0, value=0.01, step=0.01
    )
    
    Omega = st.sidebar.slider(
        "Ω (interference coherence)",
        min_value=0.0, max_value=1.0, value=0.5, step=0.05
    )
    
    # Compute baryonic mass and model prediction
    M_baryon = baryonic_mass_from_components(R, Vgas, Vdisk, Vbul)
    
    # Compute QCAUS prediction for each radius
    V_qcaus = np.array([
        qcaus_rotation_curve(r, rho0, r_s, M_baryon[i], epsilon, Omega)
        for i, r in enumerate(R)
    ])
    
    # ========================================================================
    # 5. DISPLAY RESULTS
    # ========================================================================
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Rotation Curve")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Observational data
        ax.errorbar(R, Vobs, yerr=errV, fmt='o', color='black', 
                   label=f'{selected_galaxy} (SPARC)', capsize=3, markersize=4)
        
        # Baryonic contribution
        V_baryon = np.sqrt(Vgas**2 + Vdisk**2 + Vbul**2)
        ax.plot(R, V_baryon, '--', color='orange', label='Baryonic only')
        
        # QCAUS prediction
        ax.plot(R, V_qcaus, '-', color='red', linewidth=2, 
               label=f'QCAUS (ρ₀={rho0:.1e}, rₛ={r_s:.2f}, ε={epsilon:.2f})')
        
        ax.set_xlabel('Radius [kpc]', fontsize=12)
        ax.set_ylabel('Circular Velocity [km/s]', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{selected_galaxy} — QCAUS Two-Field Model Fit')
        
        st.pyplot(fig)
    
    with col2:
        st.subheader("Model Parameters")
        
        st.metric("Galaxy", selected_galaxy)
        st.metric("Data Points", len(R))
        st.metric("ρ₀", f"{rho0:.2e} M☉/kpc³")
        st.metric("rₛ", f"{r_s:.2f} kpc")
        st.metric("ε", f"{epsilon:.3f}")
        st.metric("Ω", f"{Omega:.3f}")
        
        # Compute residuals
        residuals = Vobs_fit - qcaus_rotation_curve(R_fit, rho0, r_s, 
            baryonic_mass_from_components(R_fit, Vgas_fit, np.zeros_like(R_fit), np.zeros_like(R_fit)),
            epsilon, Omega
        )
        
        # Simple chi-squared
        chi2 = np.sum((residuals / errV_fit)**2)
        dof = len(R_fit) - 3  # rho0, r_s, epsilon
        if dof > 0:
            reduced_chi2 = chi2 / dof
            st.metric("χ²/dof", f"{reduced_chi2:.2f}")
        
        st.info(f"""
        **ℹ️ About this fit**
        
        The QCAUS two-field model combines:
        - **FDM soliton core**: ρ(r) = ρ₀ [sin(kr)/(kr)]²
        - **PDP interference**: ε · Ω · e^{-Ωr²}
        
        Adjust the sliders to find the best fit to the data.
        """)

if __name__ == "__main__":
    main()
