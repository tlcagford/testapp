# app.py## Dark Matter Field Explorer# --------------------------# Prototype scientific dashboard for:#   1. Two-field dark-matter interference#   2. Schrödinger-Poisson evolution#   3. FDM soliton density profiles#   4. Basic gravitational potential calculation## Run:#   pip install streamlit numpy scipy plotly#   streamlit run app.py
import numpy as npimport streamlit as stimport plotly.graph_objects as gofrom scipy.fft import fft2, ifft2, fftfreqfrom scipy.ndimage import gaussian_filter
st.set_page_config(    page_title="Dark Matter Field Explorer",    page_icon=" ",    layout="wide",)
# ============================================================# Constants / defaults# ============================================================
G = 1.0HBAR = 1.0C = 1.0
st.title(" Dark Matter Field Explorer")st.caption(    "Prototype simulator for ultralight dark matter, "    "two-field interference, FDM solitons, and gravitational potentials.")
# ============================================================# Sidebar# ============================================================
st.sidebar.header("Simulation Parameters")
N = st.sidebar.slider(    "Grid resolution",    min_value=64,    max_value=256,    value=128,    step=32,)
box_size = st.sidebar.slider(    "Box size",    min_value=10.0,    max_value=100.0,    value=40.0,    step=5.0,)
mass = st.sidebar.slider(    "Field mass",    min_value=0.05,    max_value=5.0,    value=1.0,    step=0.05,)
field_ratio = st.sidebar.slider(    "Dark-field amplitude ratio",    min_value=0.0,    max_value=2.0,    value=0.8,    step=0.05,)
delta_phase = st.sidebar.slider(    "Relative phase Δφ",    min_value=0.0,    max_value=2.0 * np.pi,    value=0.0,    step=0.1,)
momentum = st.sidebar.slider(    "Relative wave number",    min_value=0.0,    max_value=2.0,    value=0.5,    step=0.05,)
noise = st.sidebar.slider(    "Initial field noise",    min_value=0.0,    max_value=0.5,    value=0.03,    step=0.01,)
smooth = st.sidebar.slider(    "Density smoothing",    min_value=0.0,    max_value=3.0,    value=0.5,    step=0.1,)
# ============================================================# Grid# ============================================================
x = np.linspace(-box_size / 2, box_size / 2, N, endpoint=False)y = np.linspace(-box_size / 2, box_size / 2, N, endpoint=False)
X, Y = np.meshgrid(x, y)
dx = box_size / N
R = np.sqrt(X**2 + Y**2)
# ============================================================# Two-field model# ============================================================
# Localized luminous/dark field envelopesigma = box_size / 7.0
envelope = np.exp(    -(X**2 + Y**2) / (2.0 * sigma**2))
# Field Lpsi_L = envelope.astype(complex)
# Field D with relative momentum and phasepsi_D = (    field_ratio    * envelope    * np.exp(1j * (momentum * X + delta_phase)))
# Complex random perturbationsrng = np.random.default_rng(42)
noise_L = noise * (    rng.normal(size=(N, N))    + 1j * rng.normal(size=(N, N)))
noise_D = noise * (    rng.normal(size=(N, N))    + 1j * rng.normal(size=(N, N)))
psi_L *= 1.0 + noise_Lpsi_D *= 1.0 + noise_D
# ============================================================# Density equations# ============================================================
rho_L = np.abs(psi_L) ** 2rho_D = np.abs(psi_D) ** 2
interference = 2.0 * np.real(    np.conjugate(psi_L)    * psi_D)
rho_total = rho_L + rho_D + interference
# Physical densities cannot be negative.rho_total = np.maximum(rho_total, 0.0)
if smooth > 0:    rho_plot = gaussian_filter(rho_total, smooth)else:    rho_plot = rho_total
# ============================================================# Schrödinger-Poisson potential# ============================================================
def poisson_potential(density, box):    """    Solve ∇² Φ = 4πGρ using a periodic Fourier-space solver.    The k=0 mode is removed because a periodic Poisson problem    requires the mean density to be subtracted.    """
    n = density.shape[0]
    kx = 2.0 * np.pi * fftfreq(n, d=box / n)    ky = 2.0 * np.pi * fftfreq(n, d=box / n)
    KX, KY = np.meshgrid(kx, ky)
    k2 = KX**2 + KY**2
    rho_k = fft2(density - np.mean(density))
    phi_k = np.zeros_like(rho_k, dtype=complex)
    mask = k2 > 0
    phi_k[mask] = (        -4.0 * np.pi * G * rho_k[mask] / k2[mask]    )
    phi = np.real(ifft2(phi_k))
    return phi

phi = poisson_potential(rho_plot, box_size)
# Gravitational accelerationdphi_dy, dphi_dx = np.gradient(phi, dx)
gx = -dphi_dxgy = -dphi_dy
gmag = np.sqrt(gx**2 + gy**2)
# ============================================================# FDM Soliton# ============================================================
st.header("FDM Soliton Profile")
m22 = st.slider(    "m₂₂ — particle mass / 10⁻²² eV",    min_value=0.1,    max_value=10.0,    value=1.0,    step=0.1,)
rho_c = st.slider(    "Central density",    min_value=0.1,    max_value=10.0,    value=1.0,    step=0.1,)
# Schive et al. style characteristic radiusrc = 1.6 / m22
r_profile = np.linspace(0.0, box_size / 2.0, 500)
rho_sol = rho_c / (    1.0 + 0.091 * (r_profile / rc) ** 2) ** 8
fig_soliton = go.Figure()
fig_soliton.add_trace(    go.Scatter(        x=r_profile,        y=rho_sol,        mode="lines",        name="Soliton density",        line=dict(color="#7dd3fc", width=3),    ))
fig_soliton.update_layout(    template="plotly_dark",    xaxis_title="r",    yaxis_title="ρ(r)",    yaxis_type="log",    height=450,)
st.plotly_chart(fig_soliton, use_container_width=True)
st.info(    f"Characteristic soliton radius in this simplified model: "    f"r_c ≈ {rc:.3f} simulation units.")
# ============================================================# Visualization helpers# ============================================================
def heatmap(data, title, colorscale="Viridis"):    fig = go.Figure(        data=go.Heatmap(            x=x,            y=y,            z=data,            colorscale=colorscale,            colorbar=dict(title="Amplitude"),        )    )
    fig.update_layout(        title=title,        template="plotly_dark",        xaxis_title="x",        yaxis_title="y",        height=500,    )
    return fig

# ============================================================# Main field visualizations# ============================================================
st.header("Two-Field Dark-Matter Configuration")
col1, col2 = st.columns(2)
with col1:    st.plotly_chart(        heatmap(            rho_L,            "Field L density |ψ_L|²",            "Blues",        ),        use_container_width=True,    )
with col2:    st.plotly_chart(        heatmap(            rho_D,            "Field D density |ψ_D|²",            "Reds",        ),        use_container_width=True,    )
st.subheader("Interference")
st.plotly_chart(    heatmap(        interference,        "2 Re(ψ_L* ψ_D)",        "RdBu",    ),    use_container_width=True,)
st.subheader("Combined Density")
st.plotly_chart(    heatmap(        rho_plot,        "ρ = |ψ_L|² + |ψ_D|² + interference",        "Viridis",    ),    use_container_width=True,)
# ============================================================# Gravitational potential# ============================================================
st.header("Schrödinger–Poisson Gravity")
col1, col2 = st.columns(2)
with col1:    st.plotly_chart(        heatmap(            phi,            "Gravitational potential Φ",            "Magma",        ),        use_container_width=True,    )
with col2:    st.plotly_chart(        heatmap(            gmag,            "Gravitational field |∇Φ|",            "Plasma",        ),        use_container_width=True,    )
# ============================================================# Radial density profile# ============================================================
st.header("Radial Density Profile")
r_bins = np.linspace(    0.0,    box_size / 2.0,    80,)
r_values = []rho_values = []
for i in range(len(r_bins) - 1):
    mask = (        (R >= r_bins[i])        & (R < r_bins[i + 1])    )
    if np.any(mask):        r_values.append(            0.5 * (r_bins[i] + r_bins[i + 1])        )        rho_values.append(            np.mean(rho_plot[mask])        )
fig_radial = go.Figure()
fig_radial.add_trace(    go.Scatter(        x=r_values,        y=rho_values,        mode="lines+markers",        name="Simulation",        line=dict(            color="#a78bfa",            width=3,        ),    ))
fig_radial.update_layout(    template="plotly_dark",    xaxis_title="Radius",    yaxis_title="Mean density",    yaxis_type="log",    height=450,)
st.plotly_chart(    fig_radial,    use_container_width=True,)
# ============================================================# Summary metrics# ============================================================
st.header("System Diagnostics")
total_mass = np.sum(rho_total) * dx * dxpeak_density = float(np.max(rho_total))mean_density = float(np.mean(rho_total))potential_min = float(np.min(phi))potential_max = float(np.max(phi))
c1, c2, c3, c4 = st.columns(4)
c1.metric(    "Total field mass",    f"{total_mass:.4e}",)
c2.metric(    "Peak density",    f"{peak_density:.4e}",)
c3.metric(    "Mean density",    f"{mean_density:.4e}",)
c4.metric(    "Potential range",    f"{potential_min:.2e} → {potential_max:.2e}",)
# ============================================================# Scientific interpretation# ============================================================
st.header("Interpretation")
st.markdown(    f"""### Current configuration
- **Field mass:** `{mass:.3f}`- **Relative dark-field amplitude:** `{field_ratio:.3f}`- **Relative phase:** `{delta_phase:.3f}` rad- **Relative wave number:** `{momentum:.3f}`- **Soliton m₂₂:** `{m22:.2f}`- **Soliton radius:** `{rc:.3f}`
The interference term is
`2 Re(ψ_L* ψ_D)`,
which can create constructive and destructive density structures.
The gravitational potential is obtained from the simplifiedperiodic Poisson equation
`∇²Φ = 4πGρ`.
This is a research prototype rather than a validated cosmologicalsimulation. A production system would need relativistic/cosmologicalevolution, calibrated physical units, baryonic matter, expansion,validated initial conditions, and comparison against observationaldatasets.""")
# ============================================================# Export# ============================================================
st.header("Export")
if st.button("Generate density CSV"):
    import io
    output = io.StringIO()
    output.write("x,y,density,potential\\n")
    for iy in range(N):        for ix in range(N):            output.write(                f"{X[iy, ix]},"                f"{Y[iy, ix]},"                f"{rho_plot[iy, ix]},"                f"{phi[iy, ix]}\\n"            )
    st.download_button(        label=" Download simulation data",        data=output.getvalue(),        file_name="dark_matter_simulation.csv",        mime="text/csv",    )
st.caption(    "Dark Matter Field Explorer — experimental scientific software.")
