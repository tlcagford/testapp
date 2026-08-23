"""
QCAUS v4.0 — Quantum Cosmology & Astrophysics Unified Suite
Tony E. Ford | ORCID: 0009-0005-3715-8711
Primary:   tlcagford@protonmail.com
Alternate: tlcagford@gmail.com
GitHub: tlcagford | SourceForge: peaceandjustice | X: @TonyFor76801259

QCAUS Dark Matter Field Explorer — Cosmological / Astronomical Bridge

KEY FEATURES:
• Dark Matter Field Explorer: field, density, interference and static Poisson diagnostics
• PSF Improvement tab: upload, enhance, ALL downstream astronomical tabs use the improved image
• WITH/WITHOUT comparison on every tab
• Casimir vacuum-physics tab, kept independent from the cosmological and astronomical bridges
• About tab first — no image required to read theory
• Inline sliders above every image for live per-tab control
• Phase scrubber + auto-animate for wave tabs
• Full self-test suite: python app.py --test

Run:  streamlit run app.py
Test: python app.py --test
"""

import sys, math, io, base64, time, zipfile
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime

try:
    from scipy.ndimage import gaussian_filter
    HAS_SCIPY = True
except ImportError:
    def gaussian_filter(a, sigma=1): return a
    HAS_SCIPY = False

try:
    from astropy.io import fits
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False

try:
    import streamlit as st
    HAS_ST = True
except ImportError:
    HAS_ST = False

# ═══════════════════════════════════════════════════════════════════════════
#  PHYSICS CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════
B_CRIT  = 4.414e13          # G        — Schwinger critical field
ALPHA   = 1.0 / 137.036     # —        — QED fine-structure constant
G_GAL   = 4.30091e-3 * 1e-3 # (km/s)² kpc M☉⁻¹
H0      = 67.4              # km/s/Mpc — Planck 2018
OMEGA_M = 0.315             # —        — matter density
HBAR    = 1.0545718e-34     # J·s
C_LIGHT = 2.99792458e8      # m/s

# ═══════════════════════════════════════════════════════════════════════════
#  SYNTHETIC PRESETS (immediate output, no upload required)
# ═══════════════════════════════════════════════════════════════════════════

def _make_presets() -> dict:
    size = 256
    x = np.linspace(-5, 5, size)
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X**2 + Y**2)

    mag = np.exp(-R/1.5) * (1 + 0.4 * np.cos(5 * np.arctan2(Y, X) - 2 * R))
    mag = (mag - mag.min()) / (mag.max() - mag.min())

    cls = 2.0 * np.exp(-((X-0.2)**2 + (Y-0.1)**2)/0.3)
    for cx, cy, a in [(-1.2,-0.8,0.8),(1.0,0.5,0.6),(0.5,1.2,0.5)]:
        cls += a * np.exp(-((X-cx)**2 + (Y-cy)**2)/0.25)
    cls = (cls - cls.min()) / (cls.max() - cls.min())

    bh = 1.0 - np.exp(-R**2 / 0.6)
    bh = (bh - bh.min()) / (bh.max() - bh.min())

    return {
        'magnetar': (mag * 255).astype(np.uint8),
        'cluster':  (cls * 255).astype(np.uint8),
        'eht':      (bh  * 255).astype(np.uint8),
    }

PRESETS = _make_presets()

# ═══════════════════════════════════════════════════════════════════════════
#  STANDARD MODELS
# ═══════════════════════════════════════════════════════════════════════════

def classical_plane_wave(shape=(200, 200)) -> np.ndarray:
    h, w = shape
    x = np.linspace(-np.pi, np.pi, w)
    y = np.linspace(-np.pi, np.pi, h)
    X, Y = np.meshgrid(x, y)
    z = np.cos(2*X)*np.cos(Y)
    return (z-z.min())/(z.max()-z.min())

def classical_keplerian(r_kpc: np.ndarray) -> np.ndarray:
    return 150.0 / np.sqrt(r_kpc + 0.5)

def lcdm_flat(r_kpc: np.ndarray) -> np.ndarray:
    return 220.0 * np.ones_like(r_kpc)

def nfw_density(shape: tuple, r_s: float = 20.0) -> np.ndarray:
    h, w = shape
    y_i, x_i = np.ogrid[:h, :w]
    r = np.sqrt((x_i-w/2)**2 + (y_i-h/2)**2)
    x = r / r_s
    nfw = 1.0 / (x*(1+x)**2 + 0.01)
    return nfw / nfw.max()

def bbks_transfer(k: np.ndarray) -> np.ndarray:
    q  = k / (OMEGA_M * (H0/100)**2)
    t1 = np.log(1+2.34*q) / (2.34*q)
    t2 = (1+3.89*q+(16.1*q)**2+(5.46*q)**3+(6.71*q)**4)**(-0.25)
    return t1 * t2

def lcdm_power_spectrum(k: np.ndarray) -> np.ndarray:
    P = k * bbks_transfer(k)**2
    return P / (P.max()+1e-9)

# ═══════════════════════════════════════════════════════════════════════════
#  FORD / QCAUS PHYSICS MODELS
# ═══════════════════════════════════════════════════════════════════════════

def fdm_soliton(shape: tuple, m_22: float) -> np.ndarray:
    """
    P1: FDM Soliton — Schive+2014 / Ford 2026
    ρ_sol(r) = ρ_c / [1 + 0.091·(r/r_c)²]⁸
    r_c = min(H,W) / (8·m₂₂) [px]
    """
    h, w = shape
    y_i, x_i = np.ogrid[:h, :w]
    r = np.sqrt((x_i-w/2)**2 + (y_i-h/2)**2)
    r_c = max(min(h,w)/(8.0*m_22), 3.0)
    rho = 1.0 / (1+0.091*(r/r_c)**2)**8
    return rho / rho.max()

def two_field_wave(shape: tuple, epsilon: float, m_22: float,
                   t: float=0.0, omega_pd: float=0.2,
                   fringe_scale: float=40.0) -> dict:
    """
    P2: Ford Two-Field FDM Wave — Ford 2026 / Holdom 1986
    ψ_L = e^{−r²/8}·e^{i(r·cos t+π/4)}
    ψ_D = e^{−r²/8}·e^{i(r·ω_D·sin t+ω_beat)}
    ω_D = 1+ε·10¹⁰·m₂₂·Ω_PD
    ω_beat = ε·10¹⁰·t·m₂₂·Ω_PD·2π
    ρ = |ψ_L|²+|ψ_D|²+2·Ω_PD·Re(ψ_L·ψ_D*)
    """
    h, w = shape
    y_i, x_i = np.ogrid[:h, :w]
    r = np.sqrt((x_i-w/2)**2+(y_i-h/2)**2) / (min(h,w)/fringe_scale)
    env = np.exp(-r**2/8.0)
    omega_D    = 1.0 + epsilon*1e10*m_22*omega_pd
    beat_phase = epsilon*1e10*t*m_22*omega_pd*2*np.pi
    psi_L = env * np.exp(1j*(r*np.cos(t)+np.pi/4))
    psi_D = env * np.exp(1j*(r*omega_D*np.sin(t)+beat_phase))
    rho_L     = np.abs(psi_L)**2
    rho_D     = np.abs(psi_D)**2
    rho_cross = 2.0*omega_pd*np.real(psi_L*np.conj(psi_D))
    rho_total = rho_L+rho_D+rho_cross
    return {'psi_light':np.real(psi_L),'psi_dark':np.real(psi_D),
            'rho':rho_total,'rho_cross':rho_cross,
            'rho_L':rho_L,'rho_D':rho_D,
            'rho_peak':float(rho_total.max()),
            'beat_phase':float(beat_phase),
            'epsilon_eff':float(epsilon*1e10*m_22*omega_pd)}

def pdp_spectral_duality(image: np.ndarray, omega_pd: float,
                          fringe_scale: float, m_22: float) -> np.ndarray:
    """
    P3: PDP Spectral Duality — Ford 2026
    dark_mask(R) = e^{−Ω_PD·R²/f²}·|sin(2πRm₂₂/f)|·(1−e^{−R²/f²})
    """
    h, w = image.shape
    y_i, x_i = np.ogrid[:h, :w]
    R = np.sqrt((x_i-w/2)**2+(y_i-h/2)**2)
    f = float(fringe_scale)
    mask = (np.exp(-omega_pd*R**2/f**2)
            *np.abs(np.sin(2*np.pi*R*m_22/f))
            *(1-np.exp(-R**2/f**2)))
    return mask/(mask.max()+1e-9)

def entanglement_residuals(original: np.ndarray, ord_mode: np.ndarray,
                           dark_mode: np.ndarray, omega_pd: float) -> np.ndarray:
    """
    P4: Entanglement Residuals — Ford 2026
    S_res = −ρ·ln ρ + Ω_PD·[|ψ_ord+ψ_dark|²−ψ_ord²−ψ_dark²]
    """
    cross = np.abs(ord_mode+dark_mode)**2-ord_mode**2-dark_mode**2
    rho = original/(original.max()+1e-9)
    ent = -rho*np.log(rho+1e-10)
    res = ent+omega_pd*cross
    return res/(res.max()+1e-9)

def dark_photon_detection(dark_mode: np.ndarray, residuals: np.ndarray,
                          prior: float=0.5, strength: float=0.06) -> np.ndarray:
    """
    P5: Bayesian Dark Photon Detection — Ford 2026
    P_dark = π·L/(π·L+(1−π))  L = dark_mask+κ·S_res
    """
    L = dark_mode+strength*residuals; L = L/(L.max()+1e-9)
    return (prior*L)/(prior*L+(1-prior)+1e-9)

def blue_halo_fusion(img_gray: np.ndarray, residuals: np.ndarray,
                     dark_mode: np.ndarray, gamma: float=0.45) -> np.ndarray:
    """
    P6: Blue-Halo Fusion — Ford 2026
    R=source^γ  G=S_res^γ  B=dark_mask^γ
    """
    r = np.clip(img_gray/(img_gray.max()+1e-9),0,1)**gamma
    g = np.clip(residuals/(residuals.max()+1e-9),0,1)**gamma
    b = np.clip(dark_mode/(dark_mode.max()+1e-9),0,1)**gamma
    return (np.clip(np.stack([r,g,b],axis=2),0,1)*255).astype(np.uint8)

def rgb_full_overlay(img_gray: np.ndarray, soliton: np.ndarray,
                     interference: np.ndarray, p_dark: np.ndarray,
                     omega_pd: float=0.2) -> np.ndarray:
    """
    P7: Full-Frame RGB Quantum Overlay — Ford 2026
    R = source·(1−Ω·0.3)+P_dark·Ω·0.4
    G = source·(1−Ω·0.5)+soliton·Ω·0.8
    B = source·(1−Ω·0.5)+interference·Ω·0.8
    """
    orig   = np.clip(img_gray/(img_gray.max()+1e-9),0,1)
    sol_n  = np.clip(soliton/(soliton.max()+1e-9),0,1)
    int_n  = np.clip(interference/(interference.max()+1e-9),0,1)
    pd_n   = np.clip(p_dark/(p_dark.max()+1e-9),0,1)
    R_ch = np.clip(orig*(1-omega_pd*0.3)+pd_n*omega_pd*0.4,0,1)
    G_ch = np.clip(orig*(1-omega_pd*0.5)+sol_n*omega_pd*0.8,0,1)
    B_ch = np.clip(orig*(1-omega_pd*0.5)+int_n*omega_pd*0.8,0,1)
    return (np.stack([R_ch,G_ch,B_ch],axis=2)*255).astype(np.uint8)

def before_after_composite(img_gray: np.ndarray,
                            rgb_overlay: np.ndarray) -> np.ndarray:
    """Split-screen: LEFT=original  RIGHT=QCAUS  gold seam at w/2."""
    h, w = img_gray.shape
    orig = np.clip((img_gray/(img_gray.max()+1e-9)*255),0,255).astype(np.uint8)
    orig_rgb = np.stack([orig,orig,orig],axis=2)
    comp = np.zeros((h,w,3),dtype=np.uint8)
    comp[:,:w//2,:] = orig_rgb[:,:w//2,:]
    comp[:,w//2:,:] = rgb_overlay[:,w//2:,:]
    comp[:,w//2-1:w//2+1,:] = [255,215,0]
    return comp

def fdm_rotation_curve(m_22: float, r_kpc: np.ndarray) -> np.ndarray:
    """
    P8: FDM Rotation Curve — Ford 2026 / Schive+2014
    r_c=1.6/m₂₂ kpc  ρ_c=1.9e7·m₂₂⁻²·r_c⁻⁴  V_c=√(G·M(<r)/r)
    """
    r_c   = 1.6/m_22
    rho_c = 1.9e7*m_22**(-2)*(1/r_c)**4
    rho_s = rho_c/(1+0.091*(r_kpc/r_c)**2)**8
    rho_h = rho_c*0.1/(1+(r_kpc/(5*r_c))**3)
    m_enc = np.cumsum(4*np.pi*r_kpc**2*(rho_s+rho_h)*(r_kpc[1]-r_kpc[0]))
    return np.sqrt(G_GAL*m_enc/r_kpc)

def qcis_power_spectrum(k: np.ndarray, f_nl: float,
                         n_q: float) -> tuple:
    """
    P9: QCIS Power Spectrum — Ford 2026
    P_QCAUS(k) = P_ΛCDM(k)·[1+f_NL·sin(n_q·ln k)]
    """
    P = lcdm_power_spectrum(k)
    return P, P*(1+f_nl*np.sin(n_q*np.log(k+1e-9)))

def von_neumann_primordial(omega_pd: float, dark_mass_eV: float,
                           mixing_angle: float,
                           t_max: float=10.0, n_steps: int=200) -> dict:
    """
    P10: Von Neumann Primordial Entanglement — VN 1932 / Ford 2026
    iħ∂ρ/∂t=[H_eff,ρ]−iΓρ  H=[[0,θ],[θ,m_D]]  S=−Tr(ρ·ln ρ)
    """
    t   = np.linspace(0,t_max,n_steps)
    H   = np.array([[0.0,mixing_angle],[mixing_angle,dark_mass_eV]])
    rgg = np.zeros(n_steps,dtype=complex); rgg[0]=1.0
    rdd = np.zeros(n_steps,dtype=complex)
    rgd = np.zeros(n_steps,dtype=complex)
    S   = np.zeros(n_steps)
    dt  = t[1]-t[0]
    for i in range(1,n_steps):
        g    = omega_pd*0.1
        drgg = 1j*(H[0,1]*rgd[i-1]-H[1,0]*np.conj(rgd[i-1]))-g*rgg[i-1]
        drdd = -1j*(H[0,1]*rgd[i-1]-H[1,0]*np.conj(rgd[i-1]))-g*rdd[i-1]
        drgd = (-1j*(H[0,0]*rgd[i-1]-rgd[i-1]*H[1,1]+H[0,1]*(rdd[i-1]-rgg[i-1]))-g*rgd[i-1])
        rgg[i]=rgg[i-1]+dt*drgg; rdd[i]=rdd[i-1]+dt*drdd; rgd[i]=rgd[i-1]+dt*drgd
        ev = np.clip(np.real(np.linalg.eigvalsh(
            [[rgg[i],rgd[i]],[np.conj(rgd[i]),rdd[i]]])),1e-10,1)
        S[i] = -np.sum(ev*np.log(ev))
    return {'t':t,'entropy':S,'P_mix':np.real(rdd)}

def primordial_pdp_overlay(shape: tuple, m_22: float, omega_pd: float,
                           fringe_scale: float, t: float=0.0) -> np.ndarray:
    """
    P11: Primordial PDP Overlay — Ford 2026
    R=ρ_sol  G=fringe (anchored to soliton)  B=ρ_cross
    """
    sol  = fdm_soliton(shape, m_22)
    h, w = shape
    y_i, x_i = np.ogrid[:h, :w]
    r_c = np.sqrt((x_i-w/2)**2+(y_i-h/2)**2)
    f   = float(fringe_scale)
    fringe = (np.exp(-omega_pd*r_c**2/f**2)
              *np.abs(np.sin(2*np.pi*r_c*m_22/f))
              *(1-np.exp(-r_c**2/f**2)))
    fringe = fringe/(fringe.max()+1e-9)
    wd    = two_field_wave(shape,1e-10,m_22,t=t,omega_pd=omega_pd,fringe_scale=fringe_scale)
    cross = wd['rho_cross']-wd['rho_cross'].min()
    cross = cross/(cross.max()+1e-9)
    return (np.clip(np.stack([sol,fringe,cross],axis=2),0,1)*255).astype(np.uint8)

def psf_restoration(image: np.ndarray, sigma: float=2.0,
                    strength: float=1.5) -> dict:
    """
    P12: PSF Restoration — Wiener deconvolution proxy
    sharp = image + strength·(image − blur_σ)
    """
    blurred  = gaussian_filter(image, sigma=sigma)
    sharpened= np.clip(image+strength*(image-blurred),0,1)
    return {'blurred':blurred,'sharpened':sharpened}

def weak_lensing_kappa(soliton_map: np.ndarray) -> np.ndarray:
    """P13: Weak Lensing κ = Σ_sol/Σ_crit"""
    return soliton_map/(soliton_map.max()+1e-9)*0.3

def power_21cm(k: np.ndarray, m_22: float) -> np.ndarray:
    """P14: 21cm EoR — FDM suppression k>k_J=10·m₂₂"""
    return lcdm_power_spectrum(k)/(1+(k/(10*m_22))**2)

def cmb_lensing_phi(soliton_map: np.ndarray) -> np.ndarray:
    """P15: CMB Lensing φ — Gaussian-smoothed soliton proxy"""
    phi = gaussian_filter(soliton_map, sigma=5)
    return phi/(phi.max()+1e-9)

def bh_shadow(shape: tuple, epsilon: float) -> np.ndarray:
    """P16: Black Hole Shadow — Kerr + dark-photon ring at 1.5R_s"""
    h, w = shape
    y_i, x_i = np.ogrid[:h, :w]
    r  = np.sqrt((x_i-w/2)**2+(y_i-h/2)**2)
    Rs = min(h,w)/8
    phi= np.arctan2(y_i-h//2, x_i-w//2)
    shad = 1.0-np.exp(-(r/Rs)**4)
    ring = epsilon*1e10*np.exp(-((r-Rs*1.5)**2)/Rs**2)*(1+0.1*np.cos(2*phi))
    return np.clip(shad+ring,0,1)

def stellar_kinematics(soliton_map: np.ndarray) -> np.ndarray:
    """P17: Stellar Kinematics σ(r)∝√(GM(<r)/r) — flat core proxy"""
    return soliton_map/(soliton_map.max()+1e-9)

def dark_matter_field_explorer(shape: tuple=(256, 256), box_size: float=40.0,
                               mass_scale: float=1.0, field_ratio: float=0.8,
                               relative_k: float=0.5, phase: float=0.0,
                               noise: float=0.0) -> dict:
    """Static two-field FDM diagnostic in dimensionless simulation units.

    This is a field-level visualization/diagnostic, not a time-evolving
    Schrödinger–Poisson integration. The two complex fields are combined
    coherently, and the periodic Poisson potential is solved from the
    resulting density.
    """
    h, w = shape
    x = np.linspace(-box_size/2.0, box_size/2.0, w, endpoint=False)
    y = np.linspace(-box_size/2.0, box_size/2.0, h, endpoint=False)
    X, Y = np.meshgrid(x, y)
    dx = box_size / w
    sigma = box_size / 7.0
    envelope = np.exp(-(X**2 + Y**2)/(2.0*sigma**2))
    rng = np.random.default_rng(42)
    nL = noise*(rng.normal(size=(h,w)) + 1j*rng.normal(size=(h,w)))
    nD = noise*(rng.normal(size=(h,w)) + 1j*rng.normal(size=(h,w)))
    psi_L = envelope*(1.0+nL)
    psi_D = field_ratio*envelope*np.exp(1j*(relative_k*X+phase))*(1.0+nD)
    rho_L = np.abs(psi_L)**2
    rho_D = np.abs(psi_D)**2
    cross = 2.0*np.real(np.conjugate(psi_L)*psi_D)
    rho_total = np.abs(psi_L+psi_D)**2

    kx = 2.0*np.pi*np.fft.fftfreq(w, d=dx)
    ky = 2.0*np.pi*np.fft.fftfreq(h, d=box_size/h)
    KX, KY = np.meshgrid(kx, ky)
    k2 = KX**2 + KY**2
    rho_k = np.fft.fft2(rho_total-rho_total.mean())
    phi_k = np.zeros_like(rho_k, dtype=np.complex128)
    mask = k2 > 0
    phi_k[mask] = -4.0*np.pi*rho_k[mask]/k2[mask]
    phi = np.real(np.fft.ifft2(phi_k))
    dphi_dy, dphi_dx = np.gradient(phi, dx, dx)
    return {
        'x': x, 'y': y, 'dx': dx, 'mass_scale': mass_scale,
        'psi_L': psi_L, 'psi_D': psi_D,
        'rho_L': rho_L, 'rho_D': rho_D, 'cross': cross,
        'rho_total': rho_total, 'phi': phi,
        'gmag': np.sqrt(dphi_dx**2+dphi_dy**2),
    }


def casimir_visualization(shape: tuple, cavity_nm: float) -> dict:
    """Create a normalized visualization associated with the parallel-plate Casimir scale.

    The numerical Casimir energy density and pressure are calculated independently by
    _casimir_values(). The image returned here is a visualization only and is not a
    spatial solution of the full Casimir boundary-value problem.
    """
    h, w = shape
    y_i, x_i = np.ogrid[:h, :w]
    r = np.sqrt((x_i - w / 2.0) ** 2 + (y_i - h / 2.0) ** 2)
    r_n = r / (min(h, w) / 2.0)
    cas_map = (1.0 + 0.5 * np.cos(2.0 * np.pi * r_n)) * np.exp(-r_n**2)
    cas_map = cas_map / (np.max(np.abs(cas_map)) + 1e-12)
    return {
        "cas_map": cas_map,
        "L_nm": float(cavity_nm),
    }

def magnetar_qed_4panel(b0_log: float, mag_eps: float) -> plt.Figure:
    """Magnetar QED 4-panel — Jackson 1998 / H&E 1936 / Ford 2026"""
    DARK='#0f172a'; fig=plt.figure(figsize=(16,12),facecolor=DARK)
    B0=10.0**b0_log
    xx=np.linspace(-3,3,100); X,Y=np.meshgrid(xx,xx)
    Rg=np.maximum(np.sqrt(X**2+Y**2),0.4); Th=np.arctan2(Y,X)
    Bx=(B0*Rg**-3)*(2*np.cos(Th)**2-np.sin(Th)**2)
    By=(B0*Rg**-3)*(2*np.cos(Th)*np.sin(Th)+np.sin(Th)*np.cos(Th))
    Bmag=np.sqrt(Bx**2+By**2)
    def _cb(mp,ax,lbl):
        cb=fig.colorbar(mp,ax=ax,fraction=0.046,pad=0.04)
        cb.set_label(lbl,color='white'); cb.ax.yaxis.set_tick_params(color='white')
        plt.setp(cb.ax.yaxis.get_ticklabels(),color='white')
    ax1=fig.add_subplot(2,2,1,facecolor=DARK)
    strm=ax1.streamplot(X,Y,Bx,By,color=np.log10(Bmag+1e-30),cmap='plasma',linewidth=1.5,density=1.5)
    ax1.set_title('Panel 1: Dipole |B| log',color='white'); ax1.set_xlim(-2.5,2.5); ax1.set_ylim(-2.5,2.5)
    ax1.tick_params(colors='white'); _cb(strm.lines,ax1,'log₁₀|B| (G)')
    xi=Bmag/B_CRIT; dL=(ALPHA/(45*np.pi))*xi**2
    ax2=fig.add_subplot(2,2,2,facecolor=DARK)
    im2=ax2.imshow(dL/(dL.max()+1e-30),extent=[-3,3,-3,3],origin='lower',cmap='inferno')
    ax2.set_title('Panel 2: E-H ΔL/ΔL_max',color='white'); ax2.tick_params(colors='white'); _cb(im2,ax2,'ΔL/ΔL_max')
    m_d=1e-9; Pc=mag_eps**2*(1-np.exp(-xi**2/(m_d**2+1e-60)))
    ax3=fig.add_subplot(2,2,3,facecolor=DARK)
    im3=ax3.imshow(Pc/(Pc.max()+1e-30),extent=[-3,3,-3,3],origin='lower',cmap='hot')
    ax3.set_title(f'Panel 3: γ→γ\' P_conv ε={mag_eps:.2f}',color='white'); ax3.tick_params(colors='white'); _cb(im3,ax3,'P_conv/P_max')
    r_r=np.linspace(0.4,3,120); B_r=B0*r_r**(-3); xi_r=B_r/B_CRIT
    dL_r=(ALPHA/(45*np.pi))*xi_r**2; Pc_r=mag_eps**2*(1-np.exp(-xi_r**2/(m_d**2+1e-60)))
    ax4=fig.add_subplot(2,2,4,facecolor=DARK); ax4b=ax4.twinx()
    l1,=ax4.plot(r_r,B_r/B_r.max(),'b-',lw=2.5,label='|B| norm.')
    l2,=ax4.plot(r_r,dL_r/dL_r.max(),'g--',lw=2.5,label='ΔL norm.')
    l3,=ax4b.plot(r_r,Pc_r/(Pc_r.max()+1e-30),'r-.',lw=2.5,label='P_conv norm.')
    ax4.set_xlabel('R/R₀',color='white'); ax4.set_ylabel('|B|,ΔL',color='white')
    ax4b.set_ylabel('P_conv',color='white'); ax4.set_title('Panel 4: Radial Profiles',color='white')
    ax4.tick_params(colors='white'); ax4b.tick_params(colors='white'); ax4.grid(alpha=0.25)
    ax4.legend([l1,l2,l3],[l.get_label() for l in [l1,l2,l3]],
               loc='upper right',facecolor=DARK,labelcolor='white',fontsize=9)
    fig.suptitle(f'Magnetar QED  B₀=10^{b0_log:.2f}G  ξ={B0/B_CRIT:.4f}  ε={mag_eps:.2f}',
                 color='white',fontsize=13,y=0.99)
    plt.tight_layout(); return fig

# ═══════════════════════════════════════════════════════════════════════════
#  SELF-TEST SUITE
# ═══════════════════════════════════════════════════════════════════════════

def _run_tests() -> bool:
    """
    QCAUS v4.1 comprehensive self-test suite.
    Tests every pipeline function, formula, edge case, and helper.
    Run: python app.py --test
    """
    results = []
    def chk(name, cond, detail=""):
        results.append(bool(cond))
        tag = 'PASS' if cond else 'FAIL'
        print(f"  [{tag}] {name}" + (f"  ({detail})" if detail else ""))

    print("\n" + "="*50)
    print("  QCAUS v4.1 — Comprehensive Self-Test Suite")
    print("="*50)
    S = (64, 64)
    rng = np.random.default_rng(42)
    ig = rng.random(S).astype(np.float32)   # reproducible test image

    # ── T1: FDM Soliton (Schive+2014) ─────────────────────────────
    print("\n[T1] FDM Soliton  ρ∝[1+0.091(r/r_c)²]⁻⁸")
    for m in [0.1, 1.0, 5.0, 10.0]:
        sol = fdm_soliton(S, m)
        chk(f"m={m} peak normalised to 1",   abs(sol.max()-1.0) < 1e-6)
        chk(f"m={m} centre is peak",         float(sol[32,32]) == float(sol.max()))
        chk(f"m={m} edge < centre",          float(sol[0,0]) < float(sol[32,32]))
        chk(f"m={m} no NaN/Inf",             bool(np.all(np.isfinite(sol))))
        chk(f"m={m} non-negative",           float(sol.min()) >= 0)
    # Core radius scales as 1/m — higher mass → smaller core
    def half_max_r(s):
        row = s[s.shape[0]//2, s.shape[1]//2:]
        idx = np.where(row < 0.5)[0]
        return int(idx[0]) if len(idx) else len(row)
    s1=fdm_soliton((128,128),1.0); s2=fdm_soliton((128,128),2.0); s3=fdm_soliton((128,128),4.0)
    r1,r2,r3 = half_max_r(s1), half_max_r(s2), half_max_r(s3)
    chk("Core r1 > r2 (m=1 vs m=2)",  r1 > r2, f"r1={r1} r2={r2}")
    chk("Core r2 > r3 (m=2 vs m=4)",  r2 > r3, f"r2={r2} r3={r3}")
    chk("Core scales ≈ 1/m",          1.4 < r1/r2 < 2.6, f"ratio={r1/r2:.2f}")

    # ── T2: Ford Two-Field Wave ────────────────────────────────────
    print("\n[T2] Ford Two-Field Wave  ρ=|ψ_L|²+|ψ_D|²+2Ω Re(ψ_L·ψ_D*)")
    for eps in [1e-12, 1e-10, 1e-8]:
        d = two_field_wave(S, eps, 1.0, t=0.0)
        chk(f"ε={eps:.0e} shape",           d['rho'].shape == S)
        chk(f"ε={eps:.0e} rho_peak>0",      d['rho_peak'] > 0)
        chk(f"ε={eps:.0e} psi_L real arr",  np.isrealobj(d['psi_light']))
        chk(f"ε={eps:.0e} psi_D real arr",  np.isrealobj(d['psi_dark']))
        chk(f"ε={eps:.0e} rho finite",      bool(np.all(np.isfinite(d['rho']))))

    # ── T3: PDP Spectral Duality ───────────────────────────────────
    print("\n[T3] PDP Spectral Duality  mask=e^{-ΩR²/f²}·|sin(2πRm/f)|·(1-e^{-R²/f²})")
    for m, om, f in [(1.0,0.2,40),(2.0,0.3,30),(0.5,0.1,60)]:
        mask = pdp_spectral_duality(ig, om, f, m)
        chk(f"m={m} mask∈[0,1]",     0.0 <= float(mask.min()) and float(mask.max()) <= 1.001)
        chk(f"m={m} peak=1",         abs(mask.max()-1.0) < 1e-6)
        chk(f"m={m} no NaN",         bool(np.all(np.isfinite(mask))))
    mask_lo = pdp_spectral_duality(ig, 0.2, 40, 1.0)
    chk("Mask visible (max≥0.9)", float(mask_lo.max()) >= 0.9)

    # ── T4: Entanglement Residuals ─────────────────────────────────
    print("\n[T4] Entanglement Residuals  S_res=-ρ·ln ρ+Ω·[|ψ_ord+ψ_dark|²-ψ_ord²-ψ_dark²]")
    wd = two_field_wave(S, 1e-10, 1.0)
    mask = pdp_spectral_duality(ig, 0.2, 40, 1.0)
    for om in [0.1, 0.2, 0.4]:
        res = entanglement_residuals(ig, wd['psi_light'], mask, om)
        chk(f"Ω={om} residuals≥-0.1", float(res.min()) >= -0.1)
        chk(f"Ω={om} peak=1",           abs(res.max()-1.0) < 1e-6)
        chk(f"Ω={om} finite",           bool(np.all(np.isfinite(res))))

    # ── T5: Bayesian Dark Photon Detection ────────────────────────
    print("\n[T5] Dark Photon Detection  P=π·L/(π·L+(1-π))")
    res = entanglement_residuals(ig, wd['psi_light'], mask, 0.2)
    for prior in [0.1, 0.5, 0.9]:
        pd = dark_photon_detection(mask, res, prior=prior)
        chk(f"π={prior} P_dark∈[0,1]", 0.0 <= float(pd.min()) and float(pd.max()) <= 1.0)
        chk(f"π={prior} finite",        bool(np.all(np.isfinite(pd))))
    pd_lo = dark_photon_detection(mask, res, prior=0.1)
    pd_hi = dark_photon_detection(mask, res, prior=0.9)
    chk("Higher prior → higher P_dark", float(pd_hi.mean()) > float(pd_lo.mean()))

    # ── T6: Blue-Halo Fusion ──────────────────────────────────────
    print("\n[T6] Blue-Halo Fusion  R=src^γ G=S_res^γ B=dark^γ")
    for gamma in [0.3, 0.45, 0.8]:
        bh = blue_halo_fusion(ig, res, mask, gamma=gamma)
        chk(f"γ={gamma} shape (h,w,3)", bh.shape == (S[0],S[1],3))
        chk(f"γ={gamma} dtype uint8",   bh.dtype == np.uint8)
        chk(f"γ={gamma} max≤255",       int(bh.max()) <= 255)
    bh_lo = blue_halo_fusion(ig, res, mask, gamma=0.3)
    bh_hi = blue_halo_fusion(ig, res, mask, gamma=0.9)
    chk("Lower γ → brighter output",    float(bh_lo.mean()) > float(bh_hi.mean()))

    # ── T7: RGB Full Overlay ──────────────────────────────────────
    print("\n[T7] RGB Full Overlay  R=src+P_dark G=src+sol B=src+interf")
    sol = fdm_soliton(S, 1.0)
    pd  = dark_photon_detection(mask, res)
    for om in [0.1, 0.2, 0.5]:
        rgb = rgb_full_overlay(ig, sol, wd['rho'], pd, omega_pd=om)
        chk(f"Ω={om} shape (h,w,3)",    rgb.shape == (S[0],S[1],3))
        chk(f"Ω={om} dtype uint8",       rgb.dtype == np.uint8)
        chk(f"Ω={om} max≤255",           int(rgb.max()) <= 255)
    rgb = rgb_full_overlay(ig, sol, wd['rho'], pd)
    ba  = before_after_composite(ig, rgb)
    chk("BA shape (h,w,3)",              ba.shape == (S[0],S[1],3))
    chk("BA dtype uint8",                ba.dtype == np.uint8)

    # ── T8: FDM Rotation Curve ────────────────────────────────────
    print("\n[T8] FDM Rotation Curve  V_c=√(G·M(<r)/r)")
    r_kpc = np.linspace(0.01, 20.0, 200)
    for m in [0.3, 1.0, 3.0, 8.0]:
        vc = fdm_rotation_curve(m, r_kpc)
        chk(f"m={m} V_c≥0",         float(vc.min()) >= 0)
        chk(f"m={m} V_c<600 km/s",  float(vc.max()) < 600, f"{vc.max():.1f}")
        chk(f"m={m} all finite",     bool(np.all(np.isfinite(vc))))

    # ── T9: QCIS Power Spectrum ───────────────────────────────────
    print("\n[T9] QCIS Spectrum  P_QCAUS=P_ΛCDM·[1+f_NL·sin(n_q·ln k)]")
    k = np.logspace(-3, 1, 200)
    p0, q0 = qcis_power_spectrum(k, 0.0, 1.0)
    chk("f_NL=0 → QCAUS = ΛCDM",        np.allclose(p0, q0))
    chk("P_ΛCDM > 0 everywhere",         bool(np.all(p0 > 0)))
    for fnl, nq in [(1.0,0.5),(2.0,1.0),(5.0,2.0)]:
        ps, pq = qcis_power_spectrum(k, fnl, nq)
        chk(f"f_NL={fnl} differs from ΛCDM", not np.allclose(ps, pq))
        chk(f"f_NL={fnl} P_QCAUS finite",    bool(np.all(np.isfinite(pq))))

    # ── T10: Von Neumann Primordial Entanglement ──────────────────
    print("\n[T10] Von Neumann  iħ∂ρ/∂t=[H,ρ]-iΓρ  S=-Tr(ρ ln ρ)")
    for om, dm, mx in [(0.2,1.0,0.1),(0.3,0.5,0.3),(0.4,0.2,0.2)]:
        p = von_neumann_primordial(om, dm, mx, n_steps=100)
        chk(f"Ω={om} S(0)=0",           abs(float(p['entropy'][0])) < 1e-9)
        chk(f"Ω={om} S≥0 always",       bool(np.all(p['entropy'] >= -1e-9)))
        chk(f"Ω={om} S finite",         bool(np.all(np.isfinite(p['entropy']))))
        chk(f"Ω={om} entropy grows",    float(p['entropy'].max()) > 0.001)
        chk(f"Ω={om} P_mix finite",     bool(np.all(np.isfinite(p['P_mix']))))
    prim = von_neumann_primordial(0.2, 1.0, 0.1)
    chk("VN S_max > 0.05 (complex dtype working)", float(prim['entropy'].max()) > 0.05)

    # ── T11: Primordial PDP Overlay ───────────────────────────────
    print("\n[T11] Primordial PDP Overlay  R=ρ_sol G=fringe B=ρ_cross")
    for m, t in [(0.5,0.0),(1.0,1.5),(3.0,5.0)]:
        ov = primordial_pdp_overlay(S, m, 0.2, 40.0, t=t)
        chk(f"m={m} t={t} shape",       ov.shape == (S[0],S[1],3))
        chk(f"m={m} t={t} uint8",       ov.dtype == np.uint8)
        chk(f"m={m} t={t} non-zero",    bool(ov.max() > 0))

    # ── T12: PSF Restoration ──────────────────────────────────────
    print("\n[T12] PSF Restoration  sharp=img+strength·(img-blur_σ)")
    for sig, strength in [(1.0,1.0),(2.0,1.5),(3.0,2.0)]:
        psf = psf_restoration(ig, sigma=sig, strength=strength)
        chk(f"σ={sig} sharpened shape", psf['sharpened'].shape == S)
        chk(f"σ={sig} blurred shape",   psf['blurred'].shape == S)
        chk(f"σ={sig} blurred finite",  bool(np.all(np.isfinite(psf['blurred']))))

    # ── T13: Weak Lensing ─────────────────────────────────────────
    print("\n[T13] Weak Lensing  κ=Σ_sol/Σ_crit (normalised 0.3)")
    for m in [0.5, 1.0, 5.0]:
        sol = fdm_soliton(S, m)
        kap = weak_lensing_kappa(sol)
        chk(f"m={m} κ shape",     kap.shape == S)
        chk(f"m={m} κ∈[0,0.3]",  0.0 <= float(kap.min()) and float(kap.max()) <= 0.301)
        chk(f"m={m} κ finite",   bool(np.all(np.isfinite(kap))))

    # ── T14: 21cm EoR Power Spectrum ─────────────────────────────
    print("\n[T14] 21cm EoR  P_21=P_ΛCDM/(1+(k/k_J)²)  k_J=10·m₂₂")
    k_ext = np.logspace(-3, 3, 600)
    for m in [0.5, 1.0, 3.0]:
        p21 = power_21cm(k_ext, m)
        chk(f"m={m} P_21 finite",           bool(np.all(np.isfinite(p21))))
        chk(f"m={m} P_21>0",               bool(np.all(p21 > 0)))

    # ── T15: CMB Lensing ──────────────────────────────────────────
    print("\n[T15] CMB Lensing  φ∝Gaussian-smoothed soliton")
    sol_lg = fdm_soliton((256,256), 1.0)
    phi_lg = cmb_lensing_phi(sol_lg)
    chk("φ shape",       phi_lg.shape == (256,256))
    chk("φ∈[0,1]",      0.0 <= float(phi_lg.min()) and float(phi_lg.max()) <= 1.001)
    chk("φ peak=1",      abs(phi_lg.max()-1.0) < 1e-6)
    chk("φ finite",      bool(np.all(np.isfinite(phi_lg))))

    # ── T16: Black Hole Shadow ────────────────────────────────────
    print("\n[T16] BH Shadow  shadow=1-e^{-(r/R_s)⁴} + dark ring")
    for eps in [0.0, 1e-10, 1e-8]:
        bhs = bh_shadow(S, eps)
        chk(f"ε={eps:.0e} shape",        bhs.shape == S)
        chk(f"ε={eps:.0e} ∈[0,1]",      0.0 <= float(bhs.min()) and float(bhs.max()) <= 1.0)
        chk(f"ε={eps:.0e} finite",       bool(np.all(np.isfinite(bhs))))

    # ── T17: Stellar Kinematics ───────────────────────────────────
    print("\n[T17] Stellar Kinematics  σ(r)∝√(GM(<r)/r)")
    for m in [0.5, 1.0, 5.0]:
        sol = fdm_soliton(S, m)
        kin = stellar_kinematics(sol)
        chk(f"m={m} shape",        kin.shape == S)
        chk(f"m={m} ∈[0,1]",      0.0 <= float(kin.min()) and float(kin.max()) <= 1.001)
        chk(f"m={m} peak=1",       abs(kin.max()-1.0) < 1e-6)

    # ── T18: Casimir vacuum physics ───────────────────────────────
    print("\n[T18] Casimir  E=-π²ħc/(720L⁴), P=-π²ħc/(240L⁴)")
    for L_nm in (100, 50):
        e_cas, pressure = _casimir_values(L_nm)
        chk(f"L={L_nm} nm E_cas<0", e_cas < 0)
        chk(f"L={L_nm} nm pressure<0", pressure < 0)
        chk(f"L={L_nm} nm ratio P/E=3", abs(pressure / e_cas - 3.0) < 1e-12)

    # ── T19: Dark Matter Field Explorer ───────────────────────────
    print("\n[T19] Dark Matter Field Explorer  static two-field + Poisson")
    for ratio in (0.0, 0.8, 1.5):
        dm = dark_matter_field_explorer((64,64), 40.0, 1.0, ratio, 0.5, 0.3, 0.0)
        chk(f"ratio={ratio} density shape", dm['rho_total'].shape == (64,64))
        chk(f"ratio={ratio} density finite", bool(np.all(np.isfinite(dm['rho_total']))))
        chk(f"ratio={ratio} density non-negative", bool(np.all(dm['rho_total'] >= 0)))
        chk(f"ratio={ratio} potential finite", bool(np.all(np.isfinite(dm['phi']))))
        chk(f"ratio={ratio} periodic Poisson mean≈0", abs(float(np.mean(dm['phi']))) < 1e-9)

    # ── T20: Helper functions ─────────────────────────────────────
    print("\n[T20] Helpers  arr_to_pil / blue_halo / before_after")
    try:
        pil = arr_to_pil(ig, None)
        chk("arr_to_pil(None cmap) no crash",  isinstance(pil, Image.Image))
    except Exception as e:
        chk("arr_to_pil(None cmap) no crash",  False, str(e))
    pil2 = arr_to_pil(ig, "")
    chk("arr_to_pil('') no crash",  isinstance(pil2, Image.Image))
    for cmap in ['plasma','viridis','inferno','hot']:
        pil_c = arr_to_pil(ig, cmap)
        chk(f"arr_to_pil({cmap})",  pil_c.mode == 'RGB')

    # ── T21: Physics consistency ──────────────────────────────────
    print("\n[T21] Physics Consistency")
    r_kpc = np.linspace(0.01, 20.0, 200)
    vc = fdm_rotation_curve(1.0, r_kpc)
    chk("V_c rises near core",       bool(np.all(np.diff(vc[:15]) > 0)))
    chk("V_c finite everywhere",     bool(np.all(np.isfinite(vc))))
    chk("V_c < 600 km/s",           float(vc.max()) < 600)
    s1=fdm_soliton((200,200),1.0); s2=fdm_soliton((200,200),2.0)
    r1,r2 = half_max_r(s1), half_max_r(s2)
    chk("Core ∝ 1/m", 1.4 < r1/r2 < 2.6, f"{r1/r2:.2f}")
    prim = von_neumann_primordial(0.2, 1.0, 0.2, n_steps=200)
    chk("VN entropy S(end) > S(start)", float(prim['entropy'][-1]) > float(prim['entropy'][0]))
    k = np.logspace(-3, 1, 50)
    ps, pq = qcis_power_spectrum(k, 0.0, 1.0)
    chk("QCIS f_NL=0 exact identity", np.allclose(ps, pq))

    # ── Summary ───────────────────────────────────────────────────
    n_p = sum(results); n_f = len(results) - n_p
    print(f"\n{'='*70}")
    print(f"  {n_p}/{len(results)} passed  |  {n_f} failed")
    print(f"{'='*70}\n")
    return n_f == 0

# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def arr_to_pil(arr, cmap: str="gray") -> Image.Image:
    """Convert 2D float array OR uint8 RGB array to PIL."""
    # Already an RGB uint8 array
    if isinstance(arr, np.ndarray) and arr.ndim == 3:
        return Image.fromarray(arr.astype(np.uint8))
    # Already a PIL Image
    if isinstance(arr, Image.Image):
        return arr
    # 2D float array
    lo, hi = arr.min(), arr.max()
    n = np.clip((arr-lo)/(hi-lo+1e-9), 0, 1)
    # Guard: None, empty string, or "gray"
    if not cmap or cmap == "gray":
        return Image.fromarray((n*255).astype(np.uint8))
    cm = getattr(plt.cm, cmap, plt.cm.viridis)
    rgba = cm(n)
    if rgba.ndim == 3 and rgba.shape[2] == 4:
        rgba = rgba[:,:,:3]
    return Image.fromarray((np.clip(rgba,0,1)*255).astype(np.uint8))

def dl_link(img: Image.Image, fname: str, text: str) -> str:
    buf = io.BytesIO(); img.save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode()
    return (f'<a href="data:image/png;base64,{b64}" download="{fname}" '
            f'style="background:linear-gradient(135deg,#2196F3,#1976D2);color:white;'
            f'padding:5px 10px;border-radius:6px;text-decoration:none;'
            f'display:inline-block;margin:3px 0;font-size:12px;">📥 {text}</a>')

def dl_button(fig, fname: str, label: str="📥 Download"):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='#0f172a')
    buf.seek(0)
    st.download_button(label=label, data=buf, file_name=fname,
                       mime='image/png', width='stretch')

def load_file(f):
    if f.name.lower().endswith('.fits'):
        if not HAS_ASTROPY: st.error("astropy not installed"); return None
        try:
            with fits.open(f) as h:
                d = h[0].data
                if d is None and len(h)>1: d=h[1].data
                d = np.nan_to_num(d.astype(np.float32))
                d = np.log1p(d-d.min()+1)
                d = (d-d.min())/(d.max()-d.min()+1e-9)
                return d
        except Exception as e: st.error(f"FITS: {e}"); return None
    else:
        try:
            return np.array(Image.open(f).convert("L"), dtype=np.float32)/255.0
        except Exception as e: st.error(f"Image error: {e}"); return None

def sci_box(title, physics, significance, interp):
    st.markdown(f"""
<div style="background:linear-gradient(135deg,#0f172a,#1e293b);border-left:4px solid #06b6d4;
border-radius:8px;padding:10px;margin:5px 0 12px 0;">
<h4 style="color:#06b6d4;margin:0 0 5px 0;font-size:13px;">🔬 {title}</h4>
<p style="color:#cbd5e1;margin:2px 0;font-size:11px;"><b>Physics:</b> {physics}</p>
<p style="color:#cbd5e1;margin:2px 0;font-size:11px;"><b>Significance:</b> {significance}</p>
<p style="color:#94a3b8;margin:2px 0;font-size:10px;font-style:italic;">📖 {interp}</p>
</div>""", unsafe_allow_html=True)

def fbox(s): st.markdown(f'<div class="formula-box">{s}</div>', unsafe_allow_html=True)

def with_without(original, processed, cap_orig, cap_proc,
                 cmap_orig='gray', cmap_proc='gray',
                 fname_orig=None, fname_proc=None):
    """
    Side-by-side WITH/WITHOUT comparison.
    Handles: 2D float arrays, uint8 (H,W,3) RGB arrays, PIL Images.
    """
    fname_orig = fname_orig or "without.png"
    fname_proc = fname_proc or "with.png"
    col1, col2 = st.columns(2)
    with col1:
        pil_o = arr_to_pil(original, cmap_orig)
        st.image(pil_o, caption=f"WITHOUT: {cap_orig}", width='stretch')
        st.markdown(dl_link(pil_o, fname_orig, "Download Without"), unsafe_allow_html=True)
    with col2:
        pil_p = arr_to_pil(processed, cmap_proc)
        st.image(pil_p, caption=f"WITH: {cap_proc}", width='stretch')
        st.markdown(dl_link(pil_p, fname_proc, "Download With"), unsafe_allow_html=True)

DARK = '#0f172a'

# ═══════════════════════════════════════════════════════════════════════════
#  STREAMLIT APPLICATION
# ═══════════════════════════════════════════════════════════════════════════


def _plot_line(title, x, series, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=DARK)
    ax.set_facecolor(DARK)
    for xdata, ydata, label, style in series:
        ax.plot(xdata, ydata, style, lw=2, label=label)
    ax.set_title(title, color='white')
    ax.set_xlabel(xlabel, color='white')
    ax.set_ylabel(ylabel, color='white')
    ax.tick_params(colors='white')
    ax.grid(alpha=0.25)
    ax.legend(facecolor=DARK, labelcolor='white')
    fig.tight_layout()
    return fig


def _casimir_values(cavity_nm: float) -> tuple[float, float]:
    """Parallel-plate Casimir energy density and pressure.

    E/V = -pi^2 hbar c / (720 L^4)
    P   = -pi^2 hbar c / (240 L^4)
    """
    L = cavity_nm * 1e-9
    energy_density = -np.pi**2 * HBAR * C_LIGHT / (720.0 * L**4)
    pressure = -np.pi**2 * HBAR * C_LIGHT / (240.0 * L**4)
    return float(energy_density), float(pressure)


def _render_bridge_header():
    st.markdown('<div class="title">🌑 QCAUS — Dark Matter Field Explorer & Cosmological / Astronomical Bridge</div>', unsafe_allow_html=True)
    st.caption(
        "QCAUS research suite reorganized into two observational bridges, with Casimir vacuum physics "
        "with Casimir vacuum physics kept as an independent laboratory module."
    )
    st.warning(
        "Scientific status: established equations and phenomenological model outputs are explicitly separated. "
        "A plotted QCAUS signal is not evidence of detection or validation by itself."
    )


def run_app():
    st.set_page_config(page_title="QCAUS — Dark Matter Field Explorer & Astronomical Bridge", layout="wide",
                       initial_sidebar_state="expanded")
    st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"]{gap:3px;flex-wrap:wrap;}
    .stTabs [data-baseweb="tab"]{border-radius:7px;padding:6px 11px;white-space:nowrap;font-size:12px;}
    .formula-box{background:#1e293b;border-radius:8px;padding:8px;font-family:monospace;font-size:.75rem;color:#a5f3fc;margin:5px 0;}
    .title{font-size:1.8rem;font-weight:900;background:linear-gradient(135deg,#667eea,#06b6d4);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
    .status{background:#0f172a;border-left:4px solid #06b6d4;border-radius:8px;padding:10px;margin:7px 0;color:#cbd5e1;}
    </style>
    """, unsafe_allow_html=True)
    _render_bridge_header()

    if 'img_gray' not in st.session_state:
        st.session_state.img_gray = PRESETS['cluster'].astype(np.float32) / 255.0
    if 'img_improved' not in st.session_state:
        st.session_state.img_improved = st.session_state.img_gray.copy()
    if 'psf_applied' not in st.session_state:
        st.session_state.psf_applied = False
    if 'anim_t' not in st.session_state:
        st.session_state.anim_t = 0.0

    with st.sidebar:
        st.header("⚙️ Global controls")
        source = st.radio("Image source", ["Preset", "Upload FITS / image"])
        if source == "Upload FITS / image":
            up = st.file_uploader("FITS / JPG / PNG", type=["fits", "jpg", "jpeg", "png"])
            if up is not None:
                d = load_file(up)
                if d is not None:
                    st.session_state.img_gray = d
                    st.session_state.img_improved = d.copy()
                    st.session_state.psf_applied = False
        else:
            preset = st.selectbox("Preset", ["Galaxy Cluster (synthetic)", "SGR 1806-20 (synthetic)", "EHT Black Hole (synthetic)"])
            key = {"Galaxy Cluster (synthetic)":"cluster", "SGR 1806-20 (synthetic)":"magnetar", "EHT Black Hole (synthetic)":"eht"}[preset]
            pa = PRESETS[key].astype(np.float32) / 255.0
            if st.session_state.get('_preset_key') != key:
                st.session_state.img_gray = pa
                st.session_state.img_improved = pa.copy()
                st.session_state.psf_applied = False
                st.session_state._preset_key = key
            st.image(PRESETS[key], caption=preset, width=180)

        st.divider()
        st.subheader("Two-field / FDM")
        epsilon = st.select_slider("Kinetic mixing ε", [1e-12,1e-11,1e-10,1e-9,1e-8], value=1e-10, format_func=lambda x:f"{x:.0e}")
        fdm_m = st.slider("FDM mass m₂₂", 0.10, 10.0, 1.0, 0.05)
        omega_pd = st.slider("Phenomenological Ω_PD", 0.0, 0.50, 0.20, 0.01)
        fringe = st.slider("Fringe scale (px)", 10, 80, 45)
        t_now = st.slider("Phase t", 0.0, 20.0, float(st.session_state.anim_t), 0.05)

        st.divider()
        st.subheader("Primordial / cosmological")
        dark_mass = st.slider("Dark mass (×10⁻⁹ eV)", 0.1, 10.0, 1.0, 0.1)
        mixing_angle = st.slider("Effective mixing angle θ", 0.01, 1.0, 0.10, 0.01)
        f_nl = st.slider("Phenomenological f_NL", 0.0, 1.0, 0.10, 0.01)
        n_q = st.slider("Oscillation n_q", 0.0, 2.0, 0.50, 0.05)

        st.divider()
        st.subheader("Astronomical / QED")
        b0_log = st.slider("Magnetar B₀ log₁₀(G)", 13.0, 16.0, 15.0, 0.05)
        mag_eps = st.slider("Magnetar mixing parameter ε", 0.01, 0.50, 0.10, 0.01)

        st.divider()
        st.subheader("Casimir")
        cavity_nm = st.slider("Parallel-plate separation (nm)", 10, 500, 100, 10)


    img_original = st.session_state.img_gray
    img_current = st.session_state.img_improved if st.session_state.psf_applied else img_original

    # Core computations shared by bridge tabs.
    sol = fdm_soliton(img_current.shape, fdm_m)
    wd = two_field_wave(img_current.shape, epsilon, fdm_m, t=t_now, omega_pd=omega_pd, fringe_scale=fringe)
    dark_mask = pdp_spectral_duality(img_current, omega_pd, fringe, fdm_m)
    residuals = entanglement_residuals(img_current, wd['psi_light'], dark_mask, omega_pd)
    p_dark = dark_photon_detection(dark_mask, residuals)
    b_halo = blue_halo_fusion(img_current, residuals, dark_mask)
    rgb_full = rgb_full_overlay(img_current, sol, wd['rho'], p_dark, omega_pd)
    ba_comp = before_after_composite(img_current, rgb_full)
    r_kpc = np.linspace(0.01, 20.0, 200)
    vc_q = fdm_rotation_curve(fdm_m, r_kpc)
    k_arr = np.logspace(-3, 1, 200)
    P_std, P_q = qcis_power_spectrum(k_arr, f_nl, n_q)
    P_21 = power_21cm(k_arr, fdm_m)
    prim = von_neumann_primordial(omega_pd, dark_mass, mixing_angle)
    pdp_ov = primordial_pdp_overlay(img_current.shape, fdm_m, omega_pd, fringe, t_now)
    kappa = weak_lensing_kappa(sol)
    cmb_phi = cmb_lensing_phi(sol)
    bhs = bh_shadow(img_current.shape, epsilon)
    kin = stellar_kinematics(sol)
    vac = casimir_visualization(img_current.shape, cavity_nm)
    nfw = nfw_density(img_current.shape)
    cl_wave = classical_plane_wave(img_current.shape)

    tabs = st.tabs([
        "🌑 Dark Matter Field Explorer",
        "🌉 Bridge overview",
        "🌌 Cosmological bridge",
        "🔭 Astronomical bridge",
        "🔬 QCI Astro / PSF",
        "⚛️ Casimir vacuum",
        "🧪 Validation / tests",
    ])

    with tabs[0]:
        st.subheader("🌑 Dark Matter Field Explorer")
        sci_box("Field-level FDM diagnostic",
                "Two complex field amplitudes → density, interference, periodic Poisson potential and field acceleration",
                "Provides the field-level bridge used by the downstream cosmological and astronomical model layers.",
                "Static dimensionless diagnostic only; this panel is not a time-evolving Schrödinger–Poisson solver and does not by itself validate photon–dark-photon physics.")
        c1,c2,c3 = st.columns(3)
        with c1:
            field_box = st.slider("Field box size",10.0,100.0,40.0,5.0,key="dm_box")
            field_ratio_box = st.slider("Dark-field amplitude ratio",0.0,2.0,0.8,0.05,key="dm_ratio")
        with c2:
            rel_k_box = st.slider("Relative wave number",0.0,2.0,0.5,0.05,key="dm_k")
            phase_box = st.slider("Relative phase Δφ",0.0,2*np.pi,0.0,0.1,key="dm_phase")
        with c3:
            noise_box = st.slider("Initial field noise",0.0,0.10,0.0,0.01,key="dm_noise")
            mass_box = st.slider("Field mass scale",0.05,5.0,1.0,0.05,key="dm_mass")
        dm = dark_matter_field_explorer(img_current.shape, field_box, mass_box, field_ratio_box, rel_k_box, phase_box, noise_box)
        c1,c2 = st.columns(2)
        with c1: with_without(dm["rho_L"], dm["rho_D"], "L-field density |ψ_L|²", "D-field density |ψ_D|²", 'viridis','magma')
        with c2: with_without(dm["cross"], dm["rho_total"], "Interference cross term", "Combined density |ψ_L+ψ_D|²", 'coolwarm','plasma')
        c1,c2 = st.columns(2)
        with c1: st.image(arr_to_pil(dm["phi"],'magma'), caption="Periodic Poisson potential Φ — simulation units", width='stretch')
        with c2: st.image(arr_to_pil(dm["gmag"],'viridis'), caption="Field acceleration |∇Φ| — simulation units", width='stretch')
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Mean density",f"{dm['rho_total'].mean():.3e}")
        c2.metric("Peak density",f"{dm['rho_total'].max():.3e}")
        c3.metric("Potential min",f"{dm['phi'].min():.3e}")
        c4.metric("Potential max",f"{dm['phi'].max():.3e}")
        st.caption("Mass is retained as a traceable model parameter; without time evolution it does not alter this static dimensionless solution.")

    with tabs[1]:
        st.subheader("QCAUS architecture: one bridge, two domains, two independent physics modules")
        st.markdown("""
        **Cosmological bridge:** FDM soliton → two-field phenomenology → primordial evolution → power-spectrum / 21-cm / CMB / lensing observables.

        **Astronomical bridge:** astronomical image → PSF treatment → spectral/PDP maps → dark-sector diagnostic maps → magnetar / black-hole / stellar observables.

        **Casimir:** isolated laboratory vacuum-physics module. It is not used to manufacture an astronomical detection claim.

        """)
        st.info("This organization makes the inferential bridge explicit: model → observable → comparison. It does not treat synthetic overlays as empirical validation.")
        st.metric("Current input", "PSF-enhanced" if st.session_state.psf_applied else "Original")
        st.metric("FDM m₂₂", f"{fdm_m:.2f}")
        st.metric("ε", f"{epsilon:.0e}")

    with tabs[2]:
        st.subheader("🌌 Cosmological bridge")
        sci_box("FDM / two-field cosmological layer",
                "Soliton profile + phenomenological two-field interference + projected cosmological observables",
                "Bridge from field-level assumptions to quantities that can ultimately be compared with cosmological data.",
                "The PDP/Ω_PD terms are model assumptions and require external validation.")
        c1,c2 = st.columns(2)
        with c1:
            with_without(nfw, sol, "Reference: NFW-like profile", f"FDM soliton m₂₂={fdm_m:.2f}", 'viridis','plasma')
        with c2:
            with_without(cl_wave, wd['rho'], "Reference wave", "Two-field model ρ", 'gray','plasma')
        st.markdown("### Rotation curve")
        fig = _plot_line("Reference vs FDM rotation curve", r_kpc, [
            (r_kpc, classical_keplerian(r_kpc), "Classical Keplerian", "--"),
            (r_kpc, lcdm_flat(r_kpc), "ΛCDM reference", "-"),
            (r_kpc, vc_q, f"FDM m₂₂={fdm_m:.2f}", "-"),], "Radius (kpc)", "V (km/s)")
        st.pyplot(fig, width='stretch'); plt.close(fig)
        st.markdown("### Power-spectrum bridge")
        fig = _plot_line("ΛCDM reference vs phenomenological QCAUS correction", k_arr, [
            (k_arr,P_std,"ΛCDM reference","-"),(k_arr,P_q,f"QCAUS phenomenology f_NL={f_nl:.2f}","--"),], "k (h/Mpc)", "Normalized P(k)")
        ax = fig.axes[0]; ax.set_xscale('log'); ax.set_yscale('log')
        st.pyplot(fig, width='stretch'); plt.close(fig)
        st.markdown("### 21-cm / CMB / lensing")
        c1,c2,c3 = st.columns(3)
        with c1: st.image(arr_to_pil(P_21.reshape(1,-1),'viridis'), caption="21-cm model spectrum")
        with c2: st.image(arr_to_pil(kappa,'viridis'), caption="FDM lensing convergence proxy")
        with c3: st.image(arr_to_pil(cmb_phi,'viridis'), caption="CMB lensing proxy")
        st.image(pdp_ov, caption="Primordial PDP overlay — model output only", width='stretch')
        st.markdown("### Primordial density-matrix calculation")
        fig = _plot_line("Primordial model entropy / conversion", prim['t'], [
            (prim['t'], prim['entropy'], "von Neumann entropy", "-"),
            (prim['t'], prim['P_mix'], "model P_mix", "--"),], "Model time", "Value")
        st.pyplot(fig, width='stretch'); plt.close(fig)

    with tabs[3]:
        st.subheader("🔭 Astronomical bridge")
        sci_box("Observation-facing layer",
                "Image → PSF state → model maps → astronomical diagnostics",
                "Provides the bridge into QCI Astro-style image analysis while keeping model outputs distinguishable from measured quantities.",
                "A detection requires calibrated data, a null model, uncertainties and statistical testing.")
        with_without(img_current, rgb_full, "Input image", "Model overlay")
        c1,c2 = st.columns(2)
        with c1:
            st.image(arr_to_pil(dark_mask,'plasma'), caption="PDP spectral-duality model mask")
            st.image(arr_to_pil(residuals,'inferno'), caption="Entanglement residual model")
        with c2:
            st.image(arr_to_pil(p_dark,'hot'), caption="Model dark-photon posterior")
            st.image(Image.fromarray(b_halo), caption="Blue-halo model composite")
        st.subheader("Magnetar QED")
        figm = magnetar_qed_4panel(b0_log, mag_eps)
        st.pyplot(figm, width='stretch'); plt.close(figm)
        st.subheader("Black-hole / stellar diagnostics")
        c1,c2 = st.columns(2)
        with c1: with_without(np.zeros_like(img_current), bhs, "Reference", "Dark-sector ring model")
        with c2: with_without(np.zeros_like(img_current), kin, "Reference", "FDM kinematic proxy")

    with tabs[3]:
        st.subheader("🔬 QCI Astro / PSF bridge")
        sci_box("PSF state management",
                "Current image is either original input or explicitly applied PSF-restored image",
                "Keeps image preprocessing separate from downstream physics so a claimed residual can be traced to the input state.",
                "The current restoration is a sharpening proxy, not a calibrated instrument PSF inversion.")
        psf_sig = st.slider("Blur σ (px)",0.5,5.0,1.8,0.1)
        psf_str = st.slider("Sharpening strength",0.5,3.0,1.3,0.1)
        prev = psf_restoration(img_original, sigma=psf_sig, strength=psf_str)['sharpened']
        with_without(img_original, prev, "Original", "PSF preview")
        c1,c2 = st.columns(2)
        with c1:
            if st.button("Apply PSF state to bridges", use_container_width=True):
                st.session_state.img_improved = prev
                st.session_state.psf_applied = True
                st.rerun()
        with c2:
            if st.button("Reset to original", use_container_width=True):
                st.session_state.img_improved = img_original.copy()
                st.session_state.psf_applied = False
                st.rerun()
        st.subheader("Traceability")
        st.write({"input_state":"PSF-enhanced" if st.session_state.psf_applied else "original",
                  "shape":img_current.shape,
                  "epsilon":epsilon,
                  "m22":fdm_m,
                  "omega_pd":omega_pd,
                  "fringe_scale_px":fringe})

    with tabs[5]:
        st.subheader("⚛️ Casimir vacuum physics — isolated module")
        sci_box("Parallel-plate Casimir effect",
                "E/V = −π²ħc/(720L⁴); pressure = −π²ħc/(240L⁴)",
                "Established quantum-vacuum effect; displayed independently from the cosmological and astronomical bridges.",
                "No dark-photon or propulsion term is inserted into the standard Casimir calculation.")
        e_density, pressure = _casimir_values(cavity_nm)
        c1,c2,c3 = st.columns(3)
        c1.metric("Energy density", f"{e_density:.3e} J/m³")
        c2.metric("Casimir pressure", f"{pressure:.3e} Pa")
        c3.metric("Plate separation", f"{cavity_nm} nm")
        vac_map = np.abs(vac['cas_map'])
        st.image(arr_to_pil(vac_map,'inferno'), caption="Normalized spatial visualization only — not a measured Casimir field", width='stretch')
        st.caption("The spatial map is a visualization. The numerical Casimir values above come from the parallel-plate formula.")

    with tabs[6]:
        st.subheader("🧪 Validation / reproducibility")
        st.write("Run the built-in deterministic test suite with `python app.py --test`.")
        st.write("The application deliberately labels model-derived maps as model outputs and does not display a fabricated confidence percentage.")
        if st.button("Run numerical self-tests now"):
            ok = _run_tests()
            if ok: st.success("All built-in tests passed.")
            else: st.error("One or more built-in tests failed; inspect the server console for details.")
        st.markdown("### Validation boundary")
        st.markdown("""
        1. **Mathematical check:** equations and numerical invariants.
        2. **Simulation check:** convergence, limiting cases, conservation where applicable.
        3. **Instrument check:** calibrated PSF/FITS processing and uncertainty propagation.
        4. **Observational check:** real astronomical datasets and pre-registered null models.
        5. **Claim level:** no detection claim unless the data comparison supports it.
        """)

    st.divider()
    st.caption("QCAUS Dark Matter Field Explorer · Cosmological / Astronomical Bridge · Casimir isolated")


if __name__ == "__main__":
    if "--test" in sys.argv:
        ok = _run_tests()
        sys.exit(0 if ok else 1)
    elif HAS_ST:
        run_app()
    else:
        print("Streamlit not found. Run: pip install streamlit")
        sys.exit(1)
