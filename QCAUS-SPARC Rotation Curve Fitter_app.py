import json
import os
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.optimize import minimize
from scipy.integrate import quad

# ============================================================================ #
# 1. DATA LOADING
# ============================================================================ #
@st.cache_data
def load_corpus():
    """Download and load the unified HI rotation curve corpus from Zenodo."""
    filename = "rotation_curve_corpus_v7.json"
    url = "https://zenodo.org" 
    
    if not os.path.exists(filename):
        with st.spinner(f"Downloading {filename} from Zenodo..."):
            try:
                urllib.request.urlretrieve(url, filename)
                st.success(f"Downloaded {filename}")
            except Exception as e:
                st.error(f"Failed to download corpus: {e}")
                mock_data = {
                    "galaxies": [
                        {
                            "galaxy": "DDO161",
                            "data": [
                                {"Rad": 1.0, "Vobs": 50.0, "errV": 5.0, "Vgas": 20.0, "Vdisk": 30.0, "Vbul": 0.0},
                                {"Rad": 2.0, "Vobs": 75.0, "errV": 6.0, "Vgas": 25.0, "Vdisk": 45.0, "Vbul": 0.0},
                                {"Rad": 3.0, "Vobs": 90.0, "errV": 7.0, "Vgas": 30.0, "Vdisk": 55.0, "Vbul": 0.0},
                                {"Rad": 5.0, "Vobs": 100.0, "errV": 8.0, "Vgas": 35.0, "Vdisk": 60.0, "Vbul": 0.0},
                                {"Rad": 8.0, "Vobs": 105.0, "errV": 10.0, "Vgas": 35.0, "Vdisk": 62.0, "Vbul": 0.0}
                            ]
                        }
                    ]
                }
                return mock_data

    with open(filename) as f:
        corpus = json.load(f)
    return corpus

# ============================================================================ #
# 2. QCAUS TWO-FIELD MODEL
# ============================================================================ #
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
    if r > r_s:
        r = r_s
    def integrand(rp):
        return 4 * np.pi * rp**2 * fdm_density(rp, rho0, r_s)
    return quad(integrand, 0, r, limit=200)[0]

def qcaus_rotation_curve(r, rho0, r_s, M_baryon, epsilon=0.0, Omega=0.0):
    if r <= 0:
        return 0.0
    M_fdm = fdm_mass_enclosed(r, rho0, r_s)
    interference_factor = 1.0 + epsilon * Omega * np.exp(-(r / r_s)**2) * 0.1
    M_total = M_baryon + M_fdm * interference_factor
    return np.sqrt(max(0.0, G * M_total / r))

# ============================================================================ #
# 3. BARYONIC MASS
# ============================================================================ #
def baryonic_mass_from_components(r, Vgas, Vdisk, Vbul):
    V_baryon = np.sqrt(Vgas**2 + Vdisk**2 + Vbul**2)
    return r * V_baryon**2 / G

# ============================================================================ #
# 4. AUTO-FIT FUNCTIONS
# ============================================================================ #
def fit_qcaus_2param(R, Vobs, errV, M_baryon, epsilon, Omega, initial_guess):
    def chi2(params):
        rho0, r_s = params
        V_model = np.array([
            qcaus_rotation_curve(r, rho0, r_s, M_baryon[i], epsilon, Omega) 
            for i, r in enumerate(R)
        ])
        return np.sum(((Vobs - V_model) / errV)**2)
        
    bounds = [(1e2, 1e10), (0.1, 50.0)]
    result = minimize(chi2, initial_guess, bounds=bounds, method="L-BFGS-B")
    return result.x

def fit_qcaus_all(R, Vobs, errV, M_baryon, initial_guess):
    def chi2(params):
        rho0, r_s, eps, Omega = params
        V_model = np.array([
            qcaus_rotation_curve(r, rho0, r_s, M_baryon[i], eps, Omega) 
            for i, r in enumerate(R)
        ])
        return np.sum(((Vobs - V_model) / errV)**2)
        
    bounds = [(1e2, 1e10), (0.1, 50.0), (0.0, 1.0), (0.0, 1.0)]
    result = minimize(chi2, initial_guess, bounds=bounds, method="L-BFGS-B")
    return result.x

# ============================================================================ #
# 5. STREAMLIT APP
# ============================================================================ #
def main():
    st.set_page_config(page_title="QCAUS SPARC Fitter", layout="wide")
    st.title("🌌 QCAUS Two-Field Model: SPARC Rotation Curve Fitter")
    st.markdown(
        "Test the QCAUS two-field FDM model against real galaxy rotation curve data from the SPARC survey. "
        "Customize plot cosmetics and optimize weights on the fly."
    )

    corpus = load_corpus()
    galaxies = [g["galaxy"] for g in corpus["galaxies"]]

    # --- Sidebar Controls ---
    st.sidebar.header("Data Selection")
    selected_galaxy = st.sidebar.selectbox(
        "Select Galaxy", 
        galaxies, 
        index=galaxies.index("DDO161") if "DDO161" in galaxies else 0
    )
    
    galaxy_data = next(g for g in corpus["galaxies"] if g["galaxy"] == selected_galaxy)
    data_points = galaxy_data["data"]
    
    R = np.array([p["Rad"] for p in data_points])
    Vobs = np.array([p["Vobs"] for p in data_points])
    errV = np.array([p.get("errV", np.nan) for p in data_points])
    Vgas = np.array([p.get("Vgas", 0.0) for p in data_points])
    Vdisk = np.array([p.get("Vdisk", 0.0) for p in data_points])
    Vbul = np.array([p.get("Vbul", 0.0) for p in data_points])

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
    
    if "log_rho0" not in st.session_state:
        st.session_state.log_rho0 = 5.0
    if "r_s" not in st.session_state:
        st.session_state.r_s = 1.0
    if "epsilon" not in st.session_state:
        st.session_state.epsilon = 0.01
    if "Omega" not in st.session_state:
        st.session_state.Omega = 0.5

    log_rho0 = st.sidebar.slider("log10(rho0) [M_sun/kpc^3]", min_value=2.0, max_value=10.0, step=0.1, value=st.session_state.log_rho0)
    st.session_state.log_rho0 = log_rho0
    
    r_s = st.sidebar.slider("r_s [kpc]", min_value=0.1, max_value=50.0, step=0.1, value=st.session_state.r_s)
    st.session_state.r_s = r_s
    
    epsilon = st.sidebar.slider("epsilon (kinetic mixing)", min_value=0.0, max_value=1.0, step=0.01, value=st.session_state.epsilon)
    st.session_state.epsilon = epsilon
    
    Omega = st.sidebar.slider("Omega (interference coherence)", min_value=0.0, max_value=1.0, step=0.05, value=st.session_state.Omega)
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
                        st.session_state.epsilon, st.session_state.Omega, initial_guess
                    )
                    st.session_state.log_rho0 = float(np.log10(max(1e2, best_rho0)))
                    st.session_state.r_s = float(best_r_s)
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"Fit failure occurred: {e}")
        else:
            st.sidebar.warning("Insufficient data variance to execute core fit.")

    if st.sidebar.button("🚀 Auto-Fit Full (4 Parameters)", use_container_width=True):
        if len(R_fit) > 4:
            with st.spinner("Executing multi-dimensional minimization..."):
                M_baryon_fit = baryonic_mass_from_components(R_fit, Vgas_fit, Vdisk_fit, Vbul_fit)
                initial_guess = [
                    10**st.session_state.log_rho0, st.session_state.r_s, 
                    st.session_state.epsilon, st.session_state.Omega
                ]
                try:
                    best_rho0, best_r_s, best_eps, best_Omega = fit_qcaus_all(
                        R_fit, Vobs_fit, errV_fit, M_baryon_fit, initial_guess
                    )
                    st.session_state.log_rho0 = float(np.log10(max(1e2, best_rho0)))
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

    # ============================================================================ #
    # 6. DASHBOARD METRICS AND PLOT COMPUTATION
    # ============================================================================ #
    M_baryon_fit = baryonic_mass_from_components(R_fit, Vgas_fit, Vdisk_fit, Vbul_fit)
    V_model = np.array([
        qcaus_rotation_curve(r, 10**log_rho0, r_s, M_baryon_fit[i], epsilon, Omega) 
        for i, r in enumerate(R_fit)
    ])
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.errorbar(R_fit, Vobs_fit, yerr=errV_fit, fmt='o', color=data_points_color, label='Observed Data (SPARC)')
        ax.plot(R_fit, V_model, label='QCAUS Model Fit', color=fit_line_color, lw=2)
        
        V_baryon = np.sqrt(Vgas_fit**2 + Vdisk_fit**2 + Vbul_fit**2)
        ax.plot(R_fit, V_baryon, label='Baryonic Component', color=baryon_line_color, linestyle='--')
        
        ax.set_xlabel('Radius (kpc)')
        ax.set_ylabel('Velocity (km/s)')
        ax.set_title(f"Galaxy Rotation Curve Fit: {selected_galaxy}")
        ax.legend()
        st.pyplot(fig)
        
    with col2:
        st.subheader("Diagnostic Metrics")
        chi2_val = np.sum(((Vobs_fit - V_model) / errV_fit) ** 2)
        dof = len(R_fit) - 4
        red_chi2 = chi2_val / max(1, dof)
        
        st.metric("Total $\chi^2$", f"{chi2_val:.2f}")
        st.metric("Reduced $\chi^2$", f"{red_chi2:.2f}")

if __name__ == "__main__":
    main()
