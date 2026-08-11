"""
QCAUS Spectrum Wheel — Streamlit Edition
Author: Tony E. Ford | QCAUS v2026.8-SW

Live webcam view with real-time color-inversion, hue-rotation filter,
and genuine 2D FFT power spectral density wheel computed from the
actual camera image every ~80ms.

Capture system saves composited frames to session gallery with
individual and bulk ZIP download. Display adjustments (brightness,
contrast, saturation, gamma) applied pre-filter in the processor.

Keyboard shortcuts: Space=Capture, F=Fullscreen, D=Darkroom, G=Gallery

DEPLOYMENT:
    pip install streamlit streamlit-webrtc opencv-python-headless numpy av Pillow
    # packages.txt must contain: libgl1
"""
import streamlit as st
import numpy as np
import cv2
import time
import av
import io
import base64
import zipfile
from datetime import datetime
from PIL import Image
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase

# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════
N = 64
WHEEL_PX = 220
HSV_HALF = 2.0
MAX_GALLERY = 12

# ═══════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════
if "captures" not in st.session_state:
    st.session_state.captures = []
if "darkroom_mode" not in st.session_state:
    st.session_state.darkroom_mode = False
if "show_gallery" not in st.session_state:
    st.session_state.show_gallery = False
if "capture_trigger" not in st.session_state:
    st.session_state.capture_trigger = False

# ═══════════════════════════════════════════════════════════════════════
# PAGE CONFIG & CSS
# ═══════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="QCAUS Spectrum Wheel", page_icon="🔄", layout="wide")

if st.session_state.darkroom_mode:
    DARKROOM_STYLE = """
    [data-testid="stAppViewContainer"]{background:#000000!important;}
    [data-testid="stHeader"]{display:none!important;}
    [data-testid="stToolbar"]{display:none!important;}
    footer{display:none!important;}
    div[data-testid="stVerticalBlock"] > div:has(> div > iframe){padding:0!important;margin:0!important;}
    """
else:
    DARKROOM_STYLE = """
    [data-testid="stAppViewContainer"]{background:#07111f;color:#e2e8f0;}
    """

BASE_STYLE = """
h1,h2,h3{color:#38bdf8!important;font-family:'Courier New',monospace!important;}
body,p,li,label{font-family:'Courier New',monospace;}
[data-testid="stMetricValue"]{font-family:'Courier New',monospace!important;color:#e2e8f0!important;}
[data-testid="stMetricLabel"]{color:#64748b!important;font-family:'Courier New',monospace!important;font-size:10px!important;}
.stButton>button{font-family:'Courier New',monospace!important;}
.capture-card{border:1px solid #1e3a5f;border-radius:8px;padding:8px;background:#0a1628;margin:4px;}
.gallery-img{border-radius:4px;cursor:pointer;transition:transform 0.2s;}
.gallery-img:hover{transform:scale(1.05);}
"""

st.markdown(
    "<style>" + DARKROOM_STYLE + BASE_STYLE + "</style>",
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════
# WHEEL LOOKUP TABLES (precomputed once at module level)
# ═══════════════════════════════════════════════════════════════════════
def _build_wheel_luts(wheel_px, n):
    """Precompute polar-to-frequency index maps and orientation hue map."""
    yy, xx = np.mgrid[0:wheel_px, 0:wheel_px].astype(np.float32)
    cx = cy = wheel_px / 2.0
    r = wheel_px / 2.0
    dx = xx - cx
    dy = yy - cy
    dist = np.sqrt(dx * dx + dy * dy)
    mask = dist <= r
    ang = np.arctan2(dy, dx)
    ang[ang < 0] += 2.0 * np.pi
    freq_r = (dist / r) * (n / 2.0)
    u = np.round(n / 2.0 + freq_r * np.cos(ang)).astype(np.int32)
    v = np.round(n / 2.0 + freq_r * np.sin(ang)).astype(np.int32)
    u = np.clip(u, 0, n - 1)
    v = np.clip(v, 0, n - 1)
    hue_deg = np.degrees(ang)
    return u, v, hue_deg, mask

_WHEEL_U, _WHEEL_V, _WHEEL_HUE, _WHEEL_MASK = _build_wheel_luts(WHEEL_PX, N)

# ═══════════════════════════════════════════════════════════════════════
# IMAGE PROCESSING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def apply_display_adjustments(frame_bgr, brightness=0.0, contrast=1.0, saturation=1.0, gamma=1.0):
    """Apply brightness, contrast, saturation, and gamma to BGR frame.
    
    Brightness: additive offset in normalized units [-0.5, 0.5]
    Contrast: multiplicative scale around mid-gray 127.5
    Saturation: multiplicative scale on HSV saturation channel
    Gamma: power-law encoding, gamma < 1 lifts shadows, > 1 darkens
    """
    f = frame_bgr.astype(np.float32)
    f = f + brightness * 255.0
    f = (f - 127.5) * contrast + 127.5
    f = np.clip(f / 255.0, 0.0, 1.0)
    f = np.power(f, 1.0 / max(gamma, 0.1))
    f = np.clip(f * 255.0, 0.0, 255.0).astype(np.uint8)
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0.0, 255.0)
    hsv = np.clip(hsv, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def apply_dynamic_range_remap(field01, clahe_clip, blend, gamma):
    """CLAHE-based dynamic range compression for PSD power field.
    
    field01: 2D float array in [0, 1]
    clahe_clip: CLAHE clip limit (higher = more contrast enhancement)
    blend: linear blend factor between raw and equalized [0, 1]
    gamma: final gamma correction
    Returns: 2D float array in [0, 1]
    """
    u8 = np.clip(field01 * 255.0, 0.0, 255.0).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=max(0.1, clahe_clip), tileGridSize=(8, 8))
    eq = clahe.apply(u8).astype(np.float32) / 255.0
    mixed = (1.0 - blend) * field01 + blend * eq
    return np.power(np.clip(mixed, 0.0, 1.0), gamma)


def compute_psd_wheel(gray_full, mode, hue_shift, wheel_color_mode="orientation",
                       primordial_clip=3.0, primordial_blend=0.6, primordial_gamma=0.45):
    """Compute 2D FFT power spectral density and render as polar wheel.
    
    The FFT is computed on the resized grayscale frame. Magnitude is
    log-scaled, DC component suppressed, then remapped to polar
    coordinates where:
      - Radius = spatial frequency (center = low freq, edge = high freq)
      - Angle = spatial frequency orientation
      - Brightness = log power at that frequency
      - Hue = orientation angle (or power level in primordial mode)
    
    Returns: wheel_bgr (uint8 BGR image), peak_angle_deg (float),
             mean_power (float)
    """
    small = cv2.resize(gray_full, (N, N), interpolation=cv2.INTER_AREA)
    small = small.astype(np.float32) / 255.0
    F = np.fft.fftshift(np.fft.fft2(small))
    mag = np.abs(F)
    mag[N // 2, N // 2] = 0.0  # Suppress DC
    max_mag = max(mag.max(), 1e-6)
    logmag = np.log1p(mag) / np.log1p(max_mag)
    power_field = logmag[_WHEEL_V, _WHEEL_U]

    if wheel_color_mode == "primordial":
        remapped = apply_dynamic_range_remap(
            power_field, primordial_clip, primordial_blend, primordial_gamma
        )
        hue = (remapped * 360.0 + hue_shift) % 360.0
        val = np.power(np.clip(remapped, 0.05, 1.0), 0.7)
        if mode == "invert":
            hue = (hue + 180.0) % 360.0
    else:
        hue = (_WHEEL_HUE + hue_shift) % 360.0
        if mode == "invert":
            hue = (hue + 180.0) % 360.0
            val = 1.0 - np.power(1.0 - power_field, 1.4)
        else:
            val = np.power(power_field, 0.85)

    hsv = np.zeros((WHEEL_PX, WHEEL_PX, 3), dtype=np.uint8)
    hsv[..., 0] = (hue / HSV_HALF).astype(np.uint8)
    hsv[..., 1] = 220
    hsv[..., 2] = np.clip(val * 255.0, 10.0, 255.0).astype(np.uint8)
    wheel_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    wheel_bgr[~_WHEEL_MASK] = 0

    ring = mag.copy()
    ring[N // 2 - 2 : N // 2 + 3, N // 2 - 2 : N // 2 + 3] = 0.0
    peak_idx = np.unravel_index(np.argmax(ring), ring.shape)
    peak_angle = float(
        np.degrees(np.arctan2(peak_idx[0] - N / 2, peak_idx[1] - N / 2)) % 360.0
    )
    mean_power = float(logmag.mean())
    return wheel_bgr, peak_angle, mean_power


def apply_color_filter(frame_bgr, mode, hue_shift, split_x=None):
    """Apply HSV-space inversion and/or hue rotation to BGR frame.
    
    mode: 'normal' or 'invert' (inverts V channel)
    hue_shift: degrees to rotate hue (0-360)
    split_x: if provided, pixel column for split-screen (left normal, right filtered)
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
    h_new = (h + hue_shift / HSV_HALF) % 180.0
    hsv_out = np.stack([h_new, s, v_new], axis=-1).astype(np.uint8)
    out = cv2.cvtColor(hsv_out, cv2.COLOR_HSV2BGR)
    if split_x is not None:
        cv2.line(out, (split_x, 0), (split_x, out.shape[0]), (248, 189, 56), 2)
    return out


def edge_density(gray_small, prev_small):
    """Frame-to-frame change metric on already-resized grayscale arrays."""
    if prev_small is None:
        return 0.0
    diff = np.abs(gray_small.astype(np.float32) - prev_small.astype(np.float32))
    return float(np.mean(diff) / 255.0)


def composite_hud(frame_bgr, wheel_bgr, peak_angle, mean_power, edge_val,
                  mode, darkroom=False):
    """Overlay PSD wheel and metadata bar onto frame.
    
    In darkroom mode, only the wheel is overlaid with no text or borders.
    """
    out = frame_bgr.copy()
    if darkroom:
        h, w = out.shape[:2]
        inset = WHEEL_PX
        margin = 14
        x0 = w - inset - margin
        y0 = h - inset - margin
        if x0 > 0 and y0 > 0:
            roi = out[y0:y0 + inset, x0:x0 + inset]
            blended = cv2.addWeighted(roi, 0.1, wheel_bgr, 0.9, 0)
            out[y0:y0 + inset, x0:x0 + inset] = blended
        return out

    h, w = out.shape[:2]
    inset = WHEEL_PX
    margin = 14
    x0 = w - inset - margin
    y0 = h - inset - margin
    if x0 > 0 and y0 > 0:
        roi = out[y0:y0 + inset, x0:x0 + inset]
        blended = cv2.addWeighted(roi, 0.15, wheel_bgr, 0.85, 0)
        out[y0:y0 + inset, x0:x0 + inset] = blended
        cv2.rectangle(out, (x0, y0), (x0 + inset, y0 + inset), (248, 189, 56), 1)
        cv2.putText(out, "PSD WHEEL", (x0, y0 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (248, 189, 56), 1, cv2.LINE_AA)

    label = (
        f"MODE:{mode.upper()}  EDGE:{edge_val*100:.1f}%  "
        f"PEAK:{peak_angle:.0f}deg  PWR:{mean_power:.2f}"
    )
    cv2.rectangle(out, (0, 0), (min(w, 8 + 9 * len(label)), 22), (10, 20, 30), -1)
    cv2.putText(out, label, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (56, 189, 248), 1, cv2.LINE_AA)
    return out


def create_capture_card(frame_bgr, wheel_bgr, metadata):
    """Create a thumbnail card with frame, wheel overlay, and metadata bar."""
    h, w = frame_bgr.shape[:2]
    card_w = 400
    card_h = int(card_w * h / w) + 60
    card = np.zeros((card_h, card_w, 3), dtype=np.uint8)
    card[:] = (10, 22, 40)
    frame_h = card_h - 60
    frame_w = card_w
    frame_resized = cv2.resize(frame_bgr, (frame_w, frame_h))
    card[0:frame_h, 0:frame_w] = frame_resized
    wheel_small = cv2.resize(wheel_bgr, (80, 80))
    wx, wy = card_w - 90, 10
    roi = card[wy:wy + 80, wx:wx + 80]
    blended = cv2.addWeighted(roi, 0.2, wheel_small, 0.8, 0)
    card[wy:wy + 80, wx:wx + 80] = blended
    cv2.rectangle(card, (0, frame_h), (card_w, card_h), (5, 15, 30), -1)
    ts = metadata.get("timestamp", "")
    mode_str = metadata.get("mode", "")
    cv2.putText(card, f"{ts} | {mode_str}", (8, frame_h + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (56, 189, 248), 1, cv2.LINE_AA)
    peak = metadata.get("peak", 0.0)
    power = metadata.get("power", 0.0)
    cv2.putText(card, f"PEAK:{peak:.0f}deg PWR:{power:.2f}",
                (8, frame_h + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                (148, 189, 220), 1, cv2.LINE_AA)
    return card


def frame_to_pil(frame_bgr):
    """Convert BGR numpy array to PIL RGB Image."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def pil_to_bytes(img, fmt="PNG"):
    """Convert PIL Image to bytes buffer."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def get_image_download_link(img_bytes, filename, label="Download"):
    """Generate an HTML download link for base64-encoded image bytes."""
    b64 = base64.b64encode(img_bytes).decode()
    return (
        f'<a href="data:image/png;base64,{b64}" download="{filename}" '
        f'style="font-family:Courier New;color:#38bdf8;text-decoration:none;font-size:12px;">'
        f'⬇ {label}</a>'
    )


# ═══════════════════════════════════════════════════════════════════════
# VIDEO PROCESSOR
# ═══════════════════════════════════════════════════════════════════════
class SpectrumWheelProcessor(VideoProcessorBase):
    """WebRTC video processor with real-time FFT PSD wheel.
    
    Receives frames from webcam, applies display adjustments and color
    filter, computes 2D FFT power spectral density, and composites the
    PSD wheel overlay. Supports capture-on-demand to session gallery.
    """

    def __init__(self):
        self.mode = "normal"
        self.hue_shift = 0.0
        self.split_view = False
        self.wheel_color_mode = "orientation"
        self.darkroom = False
        self.brightness = 0.0
        self.contrast = 1.0
        self.saturation = 1.0
        self.gamma = 1.0
        self.prev_small = None
        self.last_spec_time = 0.0
        self.spec_interval = 0.08
        self.capture_requested = False

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = apply_display_adjustments(
            img, self.brightness, self.contrast, self.saturation, self.gamma
        )
        h, w = img.shape[:2]
        split_x = w // 2 if self.split_view else None
        filtered = apply_color_filter(img, self.mode, self.hue_shift, split_x=split_x)

        gray_small = cv2.resize(
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (N, N), interpolation=cv2.INTER_AREA
        )
        now = time.time()
        if now - self.last_spec_time > self.spec_interval:
            self.last_spec_time = now
            wheel_bgr, peak_angle, mean_power = compute_psd_wheel(
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                self.mode,
                self.hue_shift,
                wheel_color_mode=self.wheel_color_mode,
            )
            self._edge = edge_density(gray_small, self.prev_small)
            self.prev_small = gray_small.copy()
            self._wheel_bgr = wheel_bgr
            self._peak_angle = peak_angle
            self._mean_power = mean_power
            self._current_frame = filtered.copy()

        if not hasattr(self, "_wheel_bgr"):
            self._wheel_bgr = np.zeros((WHEEL_PX, WHEEL_PX, 3), dtype=np.uint8)
            self._peak_angle = 0.0
            self._mean_power = 0.0
            self._edge = 0.0
            self._current_frame = filtered.copy()

        if self.capture_requested:
            self._captured_data = {
                "frame": self._current_frame.copy(),
                "wheel": self._wheel_bgr.copy(),
                "peak": self._peak_angle,
                "power": self._mean_power,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            self.capture_requested = False

        mode_label = "both" if self.split_view else self.mode
        out = composite_hud(
            filtered, self._wheel_bgr, self._peak_angle,
            self._mean_power, self._edge, mode_label, darkroom=self.darkroom,
        )
        return av.VideoFrame.from_ndarray(out, format="bgr24")


# ═══════════════════════════════════════════════════════════════════════
# UI: HEADER
# ═══════════════════════════════════════════════════════════════════════
if not st.session_state.darkroom_mode:
    st.markdown(
        "<div style='text-align:center;padding:6px 0;'>"
        "<span style='font-family:Courier New;font-size:1.3em;color:#38bdf8;font-weight:bold;'>"
        "&#128260; QCAUS SPECTRUM WHEEL</span><br>"
        "<span style='color:#64748b;font-size:11px;'>"
        "Live color inversion + real-time PSD viewer | "
        "Space=Capture F=Fullscreen D=Darkroom G=Gallery"
        "</span></div>",
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════════════
# UI: CONTROLS
# ═══════════════════════════════════════════════════════════════════════
if not st.session_state.darkroom_mode:
    with st.expander("CONTROLS", expanded=True):
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1.5])

        with col1:
            mode = st.radio(
                "FILTER MODE", ["normal", "invert"], horizontal=True,
                format_func=lambda m: "NORMAL" if m == "normal" else "INVERTED",
            )
        with col2:
            wheel_color_mode = st.radio(
                "WHEEL COLOR", ["orientation", "primordial"], horizontal=True,
                format_func=lambda m: "ORIENTATION" if m == "orientation" else "PRIMORDIAL",
            )
        with col3:
            split_view = st.checkbox("Split-screen", value=False)
            darkroom = st.checkbox(
                "Darkroom mode", value=False,
                help="Minimal overlay for immersive viewing",
            )
        with col4:
            st.markdown("**DISPLAY ADJUSTMENTS**")
            brightness = st.slider("Brightness", -0.5, 0.5, 0.0, 0.05, key="brightness")
            contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.05, key="contrast")
            saturation = st.slider("Saturation", 0.0, 2.0, 1.0, 0.05, key="saturation")
            gamma = st.slider(
                "Gamma", 0.3, 3.0, 1.0, 0.05, key="gamma",
                help="Values below 1.0 brighten shadows, above 1.0 darken",
            )

        hue_shift = st.slider("HUE ROTATE (degrees)", 0, 360, 0, 5)
else:
    mode = "normal"
    wheel_color_mode = "orientation"
    split_view = False
    darkroom = True
    brightness = 0.0
    contrast = 1.0
    saturation = 1.0
    gamma = 1.0
    hue_shift = 0

# ═══════════════════════════════════════════════════════════════════════
# UI: CAPTURE & GALLERY BUTTONS
# ═══════════════════════════════════════════════════════════════════════
if not st.session_state.darkroom_mode:
    cap_col1, cap_col2, cap_col3 = st.columns([1, 1, 4])
    with cap_col1:
        if st.button(
            "CAPTURE FRAME", use_container_width=True,
            help="Save current frame + spectrum wheel to gallery",
        ):
            st.session_state.capture_trigger = True
    with cap_col2:
        label = f"GALLERY ({len(st.session_state.captures)})"
        if st.button(label, use_container_width=True,
                     help="Toggle capture gallery view"):
            st.session_state.show_gallery = not st.session_state.show_gallery
    with cap_col3:
        if len(st.session_state.captures) > 0:
            st.markdown(
                f"**{len(st.session_state.captures)} captures saved** | "
                "Scroll down to view gallery",
            )

# ═══════════════════════════════════════════════════════════════════════
# VIDEO STREAM
# ═══════════════════════════════════════════════════════════════════════
RTC_CONFIGURATION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

ctx = webrtc_streamer(
    key="qcaus-spectrum-wheel",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=SpectrumWheelProcessor,
    media_stream_constraints={
        "video": {
            "facingMode": "environment",
            "width": {"ideal": 1280},
            "height": {"ideal": 720},
        },
        "audio": False,
    },
    async_processing=True,
)

# ═══════════════════════════════════════════════════════════════════════
# PROPAGATE SETTINGS TO PROCESSOR
# ═══════════════════════════════════════════════════════════════════════
if ctx.video_processor:
    ctx.video_processor.mode = mode
    ctx.video_processor.hue_shift = float(hue_shift)
    ctx.video_processor.split_view = split_view
    ctx.video_processor.wheel_color_mode = wheel_color_mode
    ctx.video_processor.darkroom = darkroom
    ctx.video_processor.brightness = brightness
    ctx.video_processor.contrast = contrast
    ctx.video_processor.saturation = saturation
    ctx.video_processor.gamma = gamma

    if st.session_state.capture_trigger:
        ctx.video_processor.capture_requested = True
        st.session_state.capture_trigger = False
        time.sleep(0.15)
        if hasattr(ctx.video_processor, "_captured_data"):
            data = ctx.video_processor._captured_data
            card = create_capture_card(data["frame"], data["wheel"], {
                "timestamp": data["timestamp"],
                "mode": mode.upper(),
                "peak": data["peak"],
                "power": data["power"],
            })
            st.session_state.captures.insert(0, {
                "timestamp": data["timestamp"],
                "card_bgr": card,
                "frame_bgr": data["frame"],
                "wheel_bgr": data["wheel"],
                "metadata": {
                    "mode": mode,
                    "wheel_mode": wheel_color_mode,
                    "hue_shift": hue_shift,
                    "peak": data["peak"],
                    "power": data["power"],
                    "brightness": brightness,
                    "contrast": contrast,
                    "saturation": saturation,
                    "gamma": gamma,
                },
            })
            if len(st.session_state.captures) > MAX_GALLERY:
                st.session_state.captures = st.session_state.captures[:MAX_GALLERY]
            del ctx.video_processor._captured_data
            st.rerun()

if not ctx.state.playing:
    if not st.session_state.darkroom_mode:
        st.info("Click **START** above to grant camera access and begin the live feed.")

# ═══════════════════════════════════════════════════════════════════════
# KEYBOARD SHORTCUTS INFO
# ═══════════════════════════════════════════════════════════════════════
if not st.session_state.darkroom_mode:
    st.caption(
        "Keyboard shortcuts: SPACE = capture frame | "
        "F = toggle fullscreen | D = toggle darkroom mode | G = toggle gallery"
    )
    st.caption(
        "The inset wheel is a real 2D FFT of the live grayscale frame, "
        "remapped to polar coordinates. Hue encodes spatial-frequency "
        "orientation, brightness encodes power (log-scaled). It is a "
        "standard power spectral density (PSD) plot, computed live, "
        "not a view into any hidden field."
    )

# ═══════════════════════════════════════════════════════════════════════
# GALLERY SECTION
# ═══════════════════════════════════════════════════════════════════════
if st.session_state.show_gallery and len(st.session_state.captures) > 0:
    st.markdown("---")
    st.markdown("## CAPTURE GALLERY")

    gal_col1, gal_col2 = st.columns([1, 1])
    with gal_col1:
        if st.button("DOWNLOAD ALL AS ZIP", use_container_width=False):
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for i, cap in enumerate(st.session_state.captures):
                    img = frame_to_pil(cap["card_bgr"])
                    img_bytes = pil_to_bytes(img)
                    safe_ts = cap["timestamp"].replace(":", "-").replace(" ", "_")
                    zf.writestr(f"qcaus_capture_{i+1:03d}_{safe_ts}.png", img_bytes)
            zip_buf.seek(0)
            b64 = base64.b64encode(zip_buf.read()).decode()
            href = (
                f'<a href="data:application/zip;base64,{b64}" '
                f'download="qcaus_captures.zip" '
                f'style="font-family:Courier New;color:#38bdf8;">'
                f'Download ZIP</a>'
            )
            st.markdown(href, unsafe_allow_html=True)

    with gal_col2:
        if st.button("CLEAR GALLERY", use_container_width=False):
            st.session_state.captures = []
            st.rerun()

    cols = st.columns(3)
    for i, cap in enumerate(st.session_state.captures):
        with cols[i % 3]:
            card_pil = frame_to_pil(cap["card_bgr"])
            st.markdown("<div class='capture-card'>", unsafe_allow_html=True)
            st.image(card_pil, use_container_width=True)

            meta = cap["metadata"]
            st.markdown(
                f"<small style='color:#64748b;font-family:Courier New;'>"
                f"{cap['timestamp']}<br>"
                f"MODE:{meta['mode'].upper()} | PEAK:{meta['peak']:.0f}&deg;<br>"
                f"B:{meta['brightness']:+.1f} C:{meta['contrast']:.1f} "
                f"S:{meta['saturation']:.1f} &gamma;:{meta['gamma']:.1f}"
                f"</small>",
                unsafe_allow_html=True,
            )

            img_bytes = pil_to_bytes(card_pil)
            safe_ts = cap["timestamp"].replace(":", "-").replace(" ", "_")
            st.markdown(
                get_image_download_link(img_bytes, f"qcaus_{safe_ts}.png", "Download"),
                unsafe_allow_html=True,
            )

            with st.expander("View original"):
                frame_pil = frame_to_pil(cap["frame_bgr"])
                st.image(frame_pil, use_container_width=True)
                orig_bytes = pil_to_bytes(frame_pil)
                st.markdown(
                    get_image_download_link(
                        orig_bytes,
                        f"qcaus_original_{safe_ts}.png",
                        "Download original",
                    ),
                    unsafe_allow_html=True,
                )

            st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# KEYBOARD SHORTCUTS (JavaScript)
# ═══════════════════════════════════════════════════════════════════════
KEYBOARD_JS = """
<script>
document.addEventListener('keydown', function(e) {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    switch(e.key.toLowerCase()) {
        case ' ':
            e.preventDefault();
            var buttons = document.querySelectorAll('button');
            for (var i = 0; i < buttons.length; i++) {
                if (buttons[i].textContent.indexOf('CAPTURE FRAME') !== -1) {
                    buttons[i].click();
                    break;
                }
            }
            break;
        case 'f':
            e.preventDefault();
            if (!document.fullscreenElement) {
                document.documentElement.requestFullscreen();
            } else {
                document.exitFullscreen();
            }
            break;
        case 'd':
            e.preventDefault();
            var checkboxes = document.querySelectorAll('input[type="checkbox"]');
            for (var j = 0; j < checkboxes.length; j++) {
                var label = checkboxes[j].parentElement;
                if (label && label.textContent.indexOf('Darkroom') !== -1) {
                    checkboxes[j].click();
                    break;
                }
            }
            break;
        case 'g':
            e.preventDefault();
            var allButtons = document.querySelectorAll('button');
            for (var k = 0; k < allButtons.length; k++) {
                if (allButtons[k].textContent.indexOf('GALLERY') !== -1) {
                    allButtons[k].click();
                    break;
                }
            }
            break;
    }
});
</script>
"""
st.components.v1.html(KEYBOARD_JS, height=0)

# ═══════════════════════════════════════════════════════════════════════
# ABOUT / DEPLOYMENT
# ═══════════════════════════════════════════════════════════════════════
if not st.session_state.darkroom_mode:
    with st.expander("About / Deployment", expanded=False):
        about_lines = [
            "**Features:**",
            "- Real-time FFT wheel computed from live camera feed every ~80ms",
            "- Color filters: HSV-space inversion + hue rotation",
            "- Primordial mode: hue encodes power level via CLAHE dynamic-range remap",
            "- Display adjustments: brightness, contrast, saturation, gamma",
            "- Capture system: saves composited frame+wheel to session gallery",
            "- Darkroom mode: minimal overlay for immersive viewing or projection",
            "- Keyboard shortcuts: Space=Capture, F=Fullscreen, D=Darkroom, G=Gallery",
            "",
            "**Deployment:**",
            "```",
            "pip install streamlit streamlit-webrtc opencv-python-headless numpy av Pillow",
            "streamlit run spectrum_wheel_app.py",
            "```",
            "",
            "**Streamlit Cloud:** Add `packages.txt` with `libgl1` and these in `requirements.txt`:",
            "```",
            "streamlit>=1.28",
            "streamlit-webrtc>=0.45",
            "opencv-python-headless>=4.8",
            "numpy>=1.24",
            "av>=10",
            "Pillow>=10",
            "```",
        ]
        st.markdown("\n".join(about_lines))
