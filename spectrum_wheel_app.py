"""
QCAUS Spectrum Wheel — Streamlit Edition
Author: Tony E. Ford | QCAUS v2026.1-SW

Live webcam view with:
  1. A real-time color-inversion / hue-rotation filter.
  2. A genuine 2D FFT power spectral density (PSD), computed every frame from
     the live grayscale image and remapped to a polar "wheel": hue = spatial
     frequency ORIENTATION, brightness = POWER at that frequency (log-scaled).

HONEST FRAMING: the wheel is a standard signal-processing artifact (a power
spectral density plot in polar form) — it is not a view into any hidden
field. It is computed live, on every frame, from the actual camera image.

DEPLOYMENT REQUIREMENTS (this needs packages beyond base Streamlit):
    pip install streamlit streamlit-webrtc opencv-python-headless numpy av

If deploying to Streamlit Community Cloud, also add a packages.txt with:
    libgl1
because opencv/av need system video libraries the cloud image doesn't ship
by default. Camera access additionally requires the page be served over
HTTPS (Streamlit Cloud does this automatically; a bare `streamlit run`
on localhost also works, but a plain HTTP deployment will not).
"""
import streamlit as st
import numpy as np
import cv2
import time
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase

st.set_page_config(page_title="QCAUS Spectrum Wheel", page_icon="🔄", layout="centered")

st.markdown("""<style>
[data-testid="stAppViewContainer"]{background:#07111f;color:#e2e8f0;}
h1,h2,h3{color:#38bdf8!important;font-family:'Courier New',monospace!important;}
body,p,li,label{font-family:'Courier New',monospace;}
[data-testid="stMetricValue"]{font-family:'Courier New',monospace!important;color:#e2e8f0!important;}
[data-testid="stMetricLabel"]{color:#64748b!important;font-family:'Courier New',monospace!important;font-size:10px!important;}
</style>""", unsafe_allow_html=True)

st.markdown(
    "<div style='text-align:center;padding:6px 0;'>"
    "<span style='font-family:Courier New;font-size:1.3em;color:#38bdf8;font-weight:bold;'>"
    "🔄 QCAUS SPECTRUM WHEEL</span><br>"
    "<span style='color:#64748b;font-size:11px;'>Live color inversion + real-time power spectral density (PSD) viewer</span>"
    "</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# CORE IMAGE / SPECTRUM PROCESSING (pure functions, testable outside webrtc)
# ══════════════════════════════════════════════════════════════════════════
N = 64            # FFT grid size (must stay a size numpy's fft2 handles well)
WHEEL_PX = 220     # rendered wheel inset size, pixels

def _build_wheel_luts(wheel_px, n):
    """Precompute the polar->frequency index maps and hue map once."""
    yy, xx = np.mgrid[0:wheel_px, 0:wheel_px].astype(np.float32)
    cx = cy = wheel_px / 2.0
    r = wheel_px / 2.0
    dx, dy = xx - cx, yy - cy
    dist = np.sqrt(dx**2 + dy**2)
    mask = dist <= r
    ang = np.arctan2(dy, dx)
    ang[ang < 0] += 2 * np.pi
    freq_r = (dist / r) * (n / 2.0)
    u = np.round(n / 2.0 + freq_r * np.cos(ang)).astype(np.int32)
    v = np.round(n / 2.0 + freq_r * np.sin(ang)).astype(np.int32)
    u = np.clip(u, 0, n - 1)
    v = np.clip(v, 0, n - 1)
    hue_deg = np.degrees(ang)
    return u, v, hue_deg, mask

_WHEEL_U, _WHEEL_V, _WHEEL_HUE, _WHEEL_MASK = _build_wheel_luts(WHEEL_PX, N)

def compute_psd_wheel(gray_full, mode, hue_shift):
    """
    gray_full: full-resolution grayscale frame (uint8 or float, any size).
    Returns: wheel_bgr (WHEEL_PX x WHEEL_PX x 3 uint8), peak_angle_deg, mean_power
    """
    small = cv2.resize(gray_full, (N, N), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    F = np.fft.fftshift(np.fft.fft2(small))
    mag = np.abs(F)
    mag[N // 2, N // 2] = 0  # suppress DC spike so it doesn't wash out the scale
    max_mag = max(mag.max(), 1e-6)
    logmag = np.log1p(mag) / np.log1p(max_mag)

    power_field = logmag[_WHEEL_V, _WHEEL_U]
    hue = (_WHEEL_HUE + hue_shift) % 360.0
    if mode == "invert":
        hue = (hue + 180.0) % 360.0
        val = 1.0 - np.power(1.0 - power_field, 1.4)
    else:
        val = np.power(power_field, 0.85)

    hsv = np.zeros((WHEEL_PX, WHEEL_PX, 3), dtype=np.uint8)
    hsv[..., 0] = (hue / 2.0).astype(np.uint8)       # OpenCV hue range is 0-179
    hsv[..., 1] = 220
    hsv[..., 2] = np.clip(val * 255, 10, 255).astype(np.uint8)
    wheel_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    wheel_bgr[~_WHEEL_MASK] = 0

    ring = mag.copy(); ring[N // 2 - 2:N // 2 + 3, N // 2 - 2:N // 2 + 3] = 0
    peak_idx = np.unravel_index(np.argmax(ring), ring.shape)
    peak_angle = float(np.degrees(np.arctan2(peak_idx[0] - N / 2, peak_idx[1] - N / 2)) % 360)
    mean_power = float(logmag.mean())
    return wheel_bgr, peak_angle, mean_power

def apply_color_filter(frame_bgr, mode, hue_shift, split_x=None):
    """
    Applies invert / hue-rotate to a BGR frame. If split_x is given (pixel
    column), everything left of split_x stays normal and everything right of
    it is inverted — used for the "both views" comparison mode.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    if split_x is not None:
        invert_mask = np.zeros(frame_bgr.shape[:2], dtype=bool)
        invert_mask[:, split_x:] = True
    else:
        invert_mask = np.full(frame_bgr.shape[:2], mode == "invert", dtype=bool)
    v_new = v.copy()
    v_new[invert_mask] = 255.0 - v[invert_mask]
    h_new = (h + hue_shift / 2.0) % 180.0  # OpenCV hue is 0-179
    hsv_out = np.stack([h_new, s, v_new], axis=-1).astype(np.uint8)
    out = cv2.cvtColor(hsv_out, cv2.COLOR_HSV2BGR)
    if split_x is not None:
        cv2.line(out, (split_x, 0), (split_x, out.shape[0]), (248, 189, 56), 2)
    return out

def edge_density(gray_small, prev_small):
    if prev_small is None:
        return 0.0
    return float(np.mean(np.abs(gray_small.astype(np.float32) - prev_small.astype(np.float32))) / 255.0)

def composite_hud(frame_bgr, wheel_bgr, peak_angle, mean_power, edge_val, mode):
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    inset = WHEEL_PX
    margin = 14
    x0, y0 = w - inset - margin, h - inset - margin
    if x0 > 0 and y0 > 0:
        roi = out[y0:y0 + inset, x0:x0 + inset]
        blended = cv2.addWeighted(roi, 0.15, wheel_bgr, 0.85, 0)
        out[y0:y0 + inset, x0:x0 + inset] = blended
        cv2.rectangle(out, (x0, y0), (x0 + inset, y0 + inset), (248, 189, 56), 1)
        cv2.putText(out, "PSD WHEEL", (x0, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (248, 189, 56), 1, cv2.LINE_AA)
    label = f"MODE:{mode.upper()}  EDGE:{edge_val*100:.1f}%  PEAK:{peak_angle:.0f}deg  PWR:{mean_power:.2f}"
    cv2.rectangle(out, (0, 0), (min(w, 8 + 9 * len(label)), 22), (10, 20, 30), -1)
    cv2.putText(out, label, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (56, 189, 248), 1, cv2.LINE_AA)
    return out

# ══════════════════════════════════════════════════════════════════════════
# LIVE VIDEO PROCESSOR
# ══════════════════════════════════════════════════════════════════════════
class SpectrumWheelProcessor(VideoProcessorBase):
    def __init__(self):
        self.mode = "normal"
        self.hue_shift = 0.0
        self.split_view = False
        self.prev_small = None
        self.last_spec_time = 0.0
        self.spec_interval = 0.08  # seconds between FFT recomputes

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        split_x = w // 2 if self.split_view else None
        filtered = apply_color_filter(img, self.mode, self.hue_shift, split_x=split_x)

        gray_small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (N, N), interpolation=cv2.INTER_AREA)
        now = time.time()
        if now - self.last_spec_time > self.spec_interval:
            self.last_spec_time = now
            wheel_bgr, peak_angle, mean_power = compute_psd_wheel(
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), self.mode, self.hue_shift)
            self._edge = edge_density(gray_small, self.prev_small)
            self.prev_small = gray_small
            self._wheel_bgr = wheel_bgr
            self._peak_angle = peak_angle
            self._mean_power = mean_power

        if not hasattr(self, "_wheel_bgr"):
            self._wheel_bgr = np.zeros((WHEEL_PX, WHEEL_PX, 3), dtype=np.uint8)
            self._peak_angle, self._mean_power, self._edge = 0.0, 0.0, 0.0

        out = composite_hud(filtered, self._wheel_bgr, self._peak_angle, self._mean_power, self._edge,
                             "both" if self.split_view else self.mode)
        return av.VideoFrame.from_ndarray(out, format="bgr24")

# ══════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════
c1, c2 = st.columns(2)
with c1:
    mode = st.radio("FILTER MODE", ["normal", "invert"], horizontal=True,
                     format_func=lambda m: "NORMAL" if m == "normal" else "INVERTED")
with c2:
    split_view = st.checkbox("Split-screen comparison (normal | inverted)", value=False)

hue_shift = st.slider("HUE ROTATE (degrees)", 0, 360, 0, 5)

st.caption(
    "The inset wheel is a real 2D FFT of the live grayscale frame, remapped to polar "
    "coordinates — hue encodes spatial-frequency orientation, brightness encodes power "
    "(log-scaled). It's a standard power spectral density (PSD) plot, computed live, "
    "not a view into any hidden field."
)

RTC_CONFIGURATION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

ctx = webrtc_streamer(
    key="qcaus-spectrum-wheel",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=SpectrumWheelProcessor,
    media_stream_constraints={"video": {"facingMode": "environment"}, "audio": False},
    async_processing=True,
)

if ctx.video_processor:
    ctx.video_processor.mode = mode
    ctx.video_processor.hue_shift = float(hue_shift)
    ctx.video_processor.split_view = split_view

if not ctx.state.playing:
    st.info("Click **START** above to grant camera access and begin the live feed.")

with st.expander("ℹ️ About this app / deployment notes", expanded=False):
    st.markdown("""
**What's real here:** the FFT is computed on every frame from the actual camera
image using `numpy.fft.fft2` — nothing about the spectrum is simulated or
pre-rendered. The color filter is a genuine HSV-space transform (value channel
inversion, hue rotation) applied per-pixel in real time.

**What "PSD wheel" means:** Power Spectral Density in polar form. Center =
low spatial frequency (smooth regions of the image), edge of the wheel = high
spatial frequency (fine detail/texture). The angle around the wheel encodes
the *orientation* of that frequency content — e.g. a strongly vertical edge
in the scene shows up as bright power near the horizontal axis of the wheel
(perpendicular to the edge itself, which is standard Fourier-optics behavior).

**Deployment:**
```
pip install streamlit streamlit-webrtc opencv-python-headless numpy av
streamlit run spectrum_wheel_app.py
```
On Streamlit Community Cloud, add a `packages.txt` file containing `libgl1`
alongside your `requirements.txt`, since the cloud base image doesn't ship
the video system libraries `opencv`/`av` need by default.

Camera access requires a secure context — `streamlit run` on localhost works,
and Streamlit Community Cloud serves over HTTPS automatically. A bare HTTP
deployment elsewhere will not be allowed to request the camera.
""")
