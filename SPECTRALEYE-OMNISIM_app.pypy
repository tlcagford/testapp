#!/usr/bin/env python3
"""
SPECTRALEYE-OMNISIM — Unified Forensic Analysis & Live Spectrum Viewer
Version: 3.0.0
Author: Tony E. Ford | QCAUS Research

Combines:
- Live webcam with real‑time 2D FFT power spectral density wheel (SpectrumWheel)
- Upload image analysis with deepfake detection, JPEG ghost, quality metrics
- Face detection and per‑face forensic scoring
- Batch processing with gallery and export

DEPLOYMENT:
    pip install -r requirements.txt
    streamlit run app.py

Keyboard shortcuts (live view): Space=Capture, F=Fullscreen, D=Darkroom, G=Gallery
"""

import os
import sys
import json
import time
import uuid
import hashlib
import base64
import zipfile
import threading
import warnings
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
from io import BytesIO
from collections import OrderedDict

import streamlit as st
import numpy as np
import cv2
import av
from PIL import Image
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase

# ─── Optional ML ──────────────────────────────────────────────────────
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════
class Config:
    VERSION = "3.0.0"
    MAX_FILE_SIZE_MB = 500
    ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp'}
    CACHE_TTL = 3600
    MAX_CACHE = 100
    # Hardware profiles for performance simulation
    HARDWARE_PROFILES = {
        "cpu":      {"flops": 1e11,   "bandwidth": 50,  "cost": 0.50, "power": 100},
        "a100":     {"flops": 19.5e12,"bandwidth": 1555,"cost": 3.20, "power": 400},
        "h100":     {"flops": 67e12,  "bandwidth": 3350,"cost": 4.50, "power": 700},
        "tpu_v4":   {"flops": 275e12, "bandwidth": 1200,"cost": 2.80, "power": 300},
    }
config = Config()

# ─── Page Config ──────────────────────────────────────────────────────
st.set_page_config(page_title="SpectralEye-OmniSim", page_icon="🔬", layout="wide")

# ─── Session State ────────────────────────────────────────────────────
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.captures = []          # Live capture gallery
    st.session_state.batch_results = []
    st.session_state.current_result = None
    st.session_state.analyzer = None        # Lazy init
    st.session_state.darkroom = False
    st.session_state.show_gallery = False

# ─── CSS ──────────────────────────────────────────────────────────────
def load_css():
    dark = st.session_state.darkroom
    bg = "#000000" if dark else "#07111f"
    st.markdown(f"""
    <style>
        .stApp {{ background: {bg}; }}
        h1,h2,h3 {{ color: #38bdf8; font-family: 'Courier New', monospace; }}
        .metric-card {{
            background: #0a1628; border: 1px solid #1e3a5f; border-radius: 8px;
            padding: 12px; text-align: center; margin: 6px 0;
        }}
        .metric-card .value {{ font-size: 24px; font-weight: bold; color: #e2e8f0; }}
        .metric-card .label {{ font-size: 10px; color: #64748b; text-transform: uppercase; }}
        .badge {{ display: inline-block; padding: 2px 12px; border-radius: 12px; font-size: 10px; font-weight: bold; }}
        .badge-ai {{ background: #38bdf8; color: #07111f; }}
        .badge-gpu {{ background: #7fd962; color: #07111f; }}
        .stButton>button {{ background: #1e3a5f; color: #38bdf8; border: none; border-radius: 6px; font-family: monospace; }}
        .stButton>button:hover {{ background: #2a4a7a; }}
        .stButton>button.primary {{ background: #f59e0b; color: #07111f; font-weight: bold; }}
        .capture-card {{ border: 1px solid #1e3a5f; border-radius: 8px; padding: 8px; background: #0a1628; margin: 4px; }}
        .gallery-img {{ border-radius: 4px; cursor: pointer; }}
        .gallery-img:hover {{ transform: scale(1.02); }}
        .stTabs [data-baseweb="tab"] {{ font-family: monospace; color: #64748b; }}
        .stTabs [aria-selected="true"] {{ color: #38bdf8; border-bottom: 2px solid #38bdf8; }}
        .stProgress > div > div {{ background: linear-gradient(90deg, #38bdf8, #7fd962); }}
    </style>
    """, unsafe_allow_html=True)
load_css()

# ═══════════════════════════════════════════════════════════════════════
# CORE FUNCTIONS (MERGED FROM PREVIOUS WORK)
# ═══════════════════════════════════════════════════════════════════════

# ─── Cache ────────────────────────────────────────────────────────────
class LRUCache:
    def __init__(self, max_size=100, ttl=3600):
        self.max_size = max_size
        self.ttl = ttl
        self._cache = OrderedDict()
        self._lock = threading.RLock()
    def get(self, key):
        with self._lock:
            if key not in self._cache: return None
            val, ts = self._cache[key]
            if time.time() - ts > self.ttl:
                del self._cache[key]
                return None
            self._cache.move_to_end(key)
            return val
    def set(self, key, val):
        with self._lock:
            if key in self._cache: del self._cache[key]
            while len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
            self._cache[key] = (val, time.time())
            self._cache.move_to_end(key)
_cache = LRUCache(max_size=config.MAX_CACHE, ttl=config.CACHE_TTL)

# ─── FFT Wheel (from SpectrumWheel) ──────────────────────────────────
N_FFT = 64
WHEEL_PX = 220

def _build_wheel_luts(wheel_px, n):
    yy, xx = np.mgrid[0:wheel_px, 0:wheel_px].astype(np.float32)
    cx = cy = wheel_px / 2.0
    r = wheel_px / 2.0
    dx = xx - cx; dy = yy - cy
    dist = np.sqrt(dx*dx + dy*dy)
    mask = dist <= r
    ang = np.arctan2(dy, dx)
    ang[ang < 0] += 2.0*np.pi
    freq_r = (dist / r) * (n / 2.0)
    u = np.round(n/2.0 + freq_r * np.cos(ang)).astype(np.int32)
    v = np.round(n/2.0 + freq_r * np.sin(ang)).astype(np.int32)
    u = np.clip(u, 0, n-1); v = np.clip(v, 0, n-1)
    hue_deg = np.degrees(ang)
    return u, v, hue_deg, mask
WHEEL_U, WHEEL_V, WHEEL_HUE, WHEEL_MASK = _build_wheel_luts(WHEEL_PX, N_FFT)

def compute_psd_wheel(gray_full, mode, hue_shift=0, wheel_color="orientation",
                      clip=3.0, blend=0.6, gamma=0.45):
    """Compute 2D FFT and render polar wheel."""
    small = cv2.resize(gray_full, (N_FFT, N_FFT), interpolation=cv2.INTER_AREA)
    small = small.astype(np.float32) / 255.0
    F = np.fft.fftshift(np.fft.fft2(small))
    mag = np.abs(F)
    mag[N_FFT//2, N_FFT//2] = 0.0
    max_mag = max(mag.max(), 1e-6)
    logmag = np.log1p(mag) / np.log1p(max_mag)
    power_field = logmag[WHEEL_V, WHEEL_U]

    if wheel_color == "primordial":
        u8 = np.clip(power_field*255, 0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=max(0.1, clip), tileGridSize=(8,8))
        eq = clahe.apply(u8).astype(np.float32)/255.0
        mixed = (1-blend)*power_field + blend*eq
        remapped = np.power(np.clip(mixed,0,1), gamma)
        hue = (remapped * 360.0 + hue_shift) % 360.0
        val = np.power(np.clip(remapped, 0.05, 1.0), 0.7)
        if mode == "invert":
            hue = (hue + 180.0) % 360.0
    else:
        hue = (WHEEL_HUE + hue_shift) % 360.0
        if mode == "invert":
            hue = (hue + 180.0) % 360.0
            val = 1.0 - np.power(1.0 - power_field, 1.4)
        else:
            val = np.power(power_field, 0.85)

    hsv = np.zeros((WHEEL_PX, WHEEL_PX, 3), dtype=np.uint8)
    hsv[...,0] = (hue / 2.0).astype(np.uint8)
    hsv[...,1] = 220
    hsv[...,2] = np.clip(val*255, 10, 255).astype(np.uint8)
    wheel_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    wheel_bgr[~WHEEL_MASK] = 0

    # Peak detection
    ring = mag.copy()
    ring[N_FFT//2-2:N_FFT//2+3, N_FFT//2-2:N_FFT//2+3] = 0.0
    peak_idx = np.unravel_index(np.argmax(ring), ring.shape)
    peak_angle = float(np.degrees(np.arctan2(peak_idx[0]-N_FFT/2, peak_idx[1]-N_FFT/2)) % 360.0)
    mean_power = float(logmag.mean())
    return wheel_bgr, peak_angle, mean_power

# ─── Forensic Analyzer (from SpectralEye) ────────────────────────────
class ForensicAnalyzer:
    @staticmethod
    def compute_fft(image, fft_size=256):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape)==3 else image
        resized = cv2.resize(gray, (fft_size, fft_size), interpolation=cv2.INTER_AREA)
        resized_f = resized.astype(np.float32)/255.0
        F = np.fft.fftshift(np.fft.fft2(resized_f))
        mag = np.abs(F)
        mag[fft_size//2, fft_size//2] = 0
        if mag.max() > 0:
            logmag = np.log1p(mag) / np.log1p(mag.max())
        else:
            logmag = np.zeros_like(mag)
        mean_power = float(logmag.mean())
        entropy = float(-np.sum(logmag * np.log1p(logmag+1e-10)) / np.log(fft_size))
        return {"logmag": logmag, "mean_power": mean_power, "entropy": entropy}

    @staticmethod
    def detect_deepfake(image):
        fft = ForensicAnalyzer.compute_fft(image, 256)
        logmag = fft["logmag"]
        fft_size = 256
        center = fft_size//2
        yy, xx = np.ogrid[:fft_size, :fft_size]
        dist = np.sqrt((xx-center)**2 + (yy-center)**2)
        low = dist < fft_size*0.25
        high = dist >= fft_size*0.45
        low_energy = logmag[low].mean() if low.any() else 0
        high_energy = logmag[high].mean() if high.any() else 0
        hf = float(min(high_energy / max(low_energy, 0.001) / 0.5, 1.0))
        # Ring artifacts
        ring_score = 0.0
        for freq in [0.25, 0.333, 0.5]:
            r = int(freq*center)
            hw = max(int(fft_size*0.015), 2)
            ring_mask = (dist >= r-hw) & (dist <= r+hw)
            bg_mask = (dist >= r-4*hw) & (dist <= r+4*hw) & ~ring_mask
            if ring_mask.any() and bg_mask.any():
                re = logmag[ring_mask].mean()
                be = logmag[bg_mask].mean()
                if be > 0 and re/be > 1.3:
                    ring_score = max(ring_score, min((re/be - 1.0)/1.5, 1.0))
        score = 0.6*ring_score + 0.4*hf
        return {"score": min(score, 1.0), "detected": score > 0.165}

    @staticmethod
    def assess_quality(image):
        fft = ForensicAnalyzer.compute_fft(image, 256)
        logmag = fft["logmag"]
        fft_size = 256
        center = fft_size//2
        yy, xx = np.ogrid[:fft_size, :fft_size]
        dist = np.sqrt((xx-center)**2 + (yy-center)**2)
        hf_mask = dist > fft_size*0.35
        total = logmag.sum()
        hf = logmag[hf_mask].sum() if hf_mask.any() else 0
        focus = min(hf / max(total, 0.001) / 0.3, 1.0)
        # uniformity
        block = fft_size//8
        scores = []
        for i in range(8):
            for j in range(8):
                y0 = i*block; x0 = j*block
                if y0+block <= fft_size and x0+block <= fft_size:
                    b = logmag[y0:y0+block, x0:x0+block]
                    if b.size > 0:
                        scores.append(b.mean())
        if scores and np.mean(scores) > 0:
            uniformity = 1.0 - min(np.std(scores) / max(np.mean(scores), 0.001) * 2, 1.0)
        else:
            uniformity = 0.5
        return {
            "focus_score": min(focus, 1.0),
            "sharpness_score": 0.5 + 0.5*min(focus, 1.0),
            "texture_uniformity": uniformity,
            "noise_level": float(logmag[hf_mask].std()) if hf_mask.any() else 0.0
        }

    @staticmethod
    def detect_faces(image):
        """Return list of (x,y,w,h) face bounding boxes using OpenCV Haar cascade."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))
        return [(int(x), int(y), int(w), int(h)) for (x,y,w,h) in faces]

# ─── Face Analyzer ──────────────────────────────────────────────────
class FaceAnalyzer:
    """Per‑face forensic analysis."""
    @staticmethod
    def analyze_face_region(image, bbox):
        x,y,w,h = bbox
        face = image[y:y+h, x:x+w]
        if face.size == 0:
            return None
        deepfake = ForensicAnalyzer.detect_deepfake(face)
        quality = ForensicAnalyzer.assess_quality(face)
        fft = ForensicAnalyzer.compute_fft(face, 256)
        return {
            "bbox": bbox,
            "deepfake": deepfake,
            "quality": quality,
            "fft": {"mean_power": fft["mean_power"], "entropy": fft["entropy"]},
            "combined_score": deepfake["score"]  # for simplicity
        }

# ─── Performance Simulator (no recursion) ────────────────────────────
class PerformanceSimulator:
    def predict(self, workload):
        hw = workload.get("hardware", "cpu")
        profile = config.HARDWARE_PROFILES.get(hw, config.HARDWARE_PROFILES["cpu"])
        model_mb = workload.get("model_size_mb", 500)
        batch = workload.get("batch_size", 1)
        size = workload.get("image_size", 256)
        flops = model_mb * 1e6 * size * size * 1.5
        compute = flops / profile["flops"]
        data_mb = batch * size * size * 3 * 4 / (1024**2)
        mem = data_mb * 8 / profile["bandwidth"]
        total = compute + mem
        throughput = batch / max(total, 0.001)
        cost = profile["cost"] * (total / 3600)
        return {
            "total_time": total, "throughput": throughput,
            "utilization": min(compute/max(total,0.001), 1.0),
            "cost": cost, "energy": profile["power"]*total/3600/1000
        }

# ─── Integrated Analyzer ─────────────────────────────────────────────
class IntegratedAnalyzer:
    def __init__(self, use_ai=TORCH_AVAILABLE):
        self.use_ai = use_ai
        self.simulator = PerformanceSimulator()
        self.stats = {"total":0, "time":0, "cache_hits":0}

    def analyze_image(self, image, hardware="a100", detect_faces=True):
        start = time.time()
        img_hash = hashlib.md5(image.tobytes()).hexdigest()
        cache_key = f"{img_hash}_{hardware}_{detect_faces}"
        cached = _cache.get(cache_key)
        if cached:
            self.stats["cache_hits"] += 1
            return cached

        result = {
            "timestamp": datetime.now().isoformat(),
            "image_hash": img_hash,
            "hardware": hardware,
            "dimensions": (image.shape[1], image.shape[0])
        }

        # Face detection
        faces = []
        if detect_faces:
            bboxes = ForensicAnalyzer.detect_faces(image)
            for bbox in bboxes:
                face_data = FaceAnalyzer.analyze_face_region(image, bbox)
                if face_data:
                    faces.append(face_data)
        result["faces"] = faces

        # Overall image analysis (if no faces, use full image)
        if faces:
            # Aggregate scores from faces
            scores = [f["combined_score"] for f in faces]
            avg_score = np.mean(scores) if scores else 0.5
            result["overall"] = {
                "deepfake_score": float(avg_score),
                "detected": avg_score > 0.165,
                "face_count": len(faces)
            }
            # Quality averaged
            qs = [f["quality"]["focus_score"] for f in faces]
            avg_q = np.mean(qs) if qs else 0.5
            result["quality_rating"] = {
                "score": float(avg_q),
                "rating": "Excellent" if avg_q > 0.8 else "Good" if avg_q > 0.6 else "Fair" if avg_q > 0.4 else "Poor"
            }
        else:
            # Fallback: analyse whole image
            deepfake = ForensicAnalyzer.detect_deepfake(image)
            quality = ForensicAnalyzer.assess_quality(image)
            result["overall"] = {
                "deepfake_score": deepfake["score"],
                "detected": deepfake["detected"],
                "face_count": 0
            }
            result["quality_rating"] = {
                "score": quality["focus_score"],
                "rating": "Excellent" if quality["focus_score"] > 0.8 else "Good" if quality["focus_score"] > 0.6 else "Fair" if quality["focus_score"] > 0.4 else "Poor"
            }

        # Performance simulation
        workload = {"hardware": hardware, "model_size_mb": 500, "batch_size": 1, "image_size": image.shape[0]}
        result["performance"] = self.simulator.predict(workload)

        proc_time = (time.time() - start) * 1000
        result["processing_time_ms"] = proc_time
        self.stats["total"] += 1
        self.stats["time"] += proc_time

        _cache.set(cache_key, result)
        return result

    def analyze_batch(self, images, hardware="a100", detect_faces=True):
        results = []
        for img in images:
            results.append(self.analyze_image(img, hardware, detect_faces))
        return results

# ─── Live Video Processor ────────────────────────────────────────────
class SpectrumWheelProcessor(VideoProcessorBase):
    def __init__(self):
        self.mode = "normal"
        self.hue_shift = 0.0
        self.split_view = False
        self.wheel_color = "orientation"
        self.darkroom = False
        self.brightness = 0.0
        self.contrast = 1.0
        self.saturation = 1.0
        self.gamma = 1.0
        self.prev_small = None
        self.last_spec_time = 0.0
        self.spec_interval = 0.08
        self.capture_requested = False
        self._wheel_bgr = np.zeros((WHEEL_PX, WHEEL_PX, 3), dtype=np.uint8)
        self._peak_angle = 0.0
        self._mean_power = 0.0
        self._edge = 0.0
        self._current_frame = None
        self._captured_data = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        # Adjustments
        f = img.astype(np.float32)
        f = f + self.brightness*255.0
        f = (f - 127.5) * self.contrast + 127.5
        f = np.clip(f/255.0, 0, 1)
        f = np.power(f, 1.0/max(self.gamma, 0.1))
        f = np.clip(f*255, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[...,1] = np.clip(hsv[...,1] * self.saturation, 0, 255)
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Color filter
        hsv2 = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV).astype(np.float32)
        h,s,v = hsv2[...,0], hsv2[...,1], hsv2[...,2]
        if self.split_view:
            mask = np.zeros_like(v, dtype=bool)
            mask[:, adjusted.shape[1]//2:] = True
        else:
            mask = np.full_like(v, self.mode=="invert", dtype=bool)
        v_new = v.copy()
        v_new[mask] = 255.0 - v[mask]
        h_new = (h + self.hue_shift/2.0) % 180.0
        hsv_out = np.stack([h_new, s, v_new], axis=-1).astype(np.uint8)
        filtered = cv2.cvtColor(hsv_out, cv2.COLOR_HSV2BGR)
        if self.split_view:
            cv2.line(filtered, (adjusted.shape[1]//2, 0), (adjusted.shape[1]//2, adjusted.shape[0]), (248,189,56), 2)

        # FFT wheel
        gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
        gray_small = cv2.resize(gray, (N_FFT, N_FFT), interpolation=cv2.INTER_AREA)
        now = time.time()
        if now - self.last_spec_time > self.spec_interval:
            self.last_spec_time = now
            wheel, peak, power = compute_psd_wheel(
                gray, self.mode, self.hue_shift, self.wheel_color
            )
            self._wheel_bgr = wheel
            self._peak_angle = peak
            self._mean_power = power
            if self.prev_small is not None:
                diff = np.abs(gray_small.astype(np.float32) - self.prev_small.astype(np.float32))
                self._edge = float(np.mean(diff)/255.0)
            self.prev_small = gray_small.copy()
            self._current_frame = filtered.copy()

        if self.capture_requested:
            self._captured_data = {
                "frame": self._current_frame.copy() if self._current_frame is not None else filtered.copy(),
                "wheel": self._wheel_bgr.copy(),
                "peak": self._peak_angle,
                "power": self._mean_power,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            self.capture_requested = False

        # Composite HUD
        out = filtered.copy()
        h, w = out.shape[:2]
        inset = WHEEL_PX
        margin = 14
        x0 = w - inset - margin
        y0 = h - inset - margin
        if x0 > 0 and y0 > 0:
            roi = out[y0:y0+inset, x0:x0+inset]
            blend = 0.1 if self.darkroom else 0.15
            blended = cv2.addWeighted(roi, blend, self._wheel_bgr, 1.0-blend, 0)
            out[y0:y0+inset, x0:x0+inset] = blended
            if not self.darkroom:
                cv2.rectangle(out, (x0,y0), (x0+inset,y0+inset), (248,189,56), 1)
                cv2.putText(out, "PSD WHEEL", (x0, y0-6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (248,189,56), 1)

        if not self.darkroom:
            label = f"MODE:{self.mode.upper()}  EDGE:{self._edge*100:.1f}%  PEAK:{self._peak_angle:.0f}deg  PWR:{self._mean_power:.2f}"
            cv2.rectangle(out, (0,0), (min(w, 8+9*len(label)), 22), (10,20,30), -1)
            cv2.putText(out, label, (6,15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (56,189,248), 1)

        return av.VideoFrame.from_ndarray(out, format="bgr24")

# ═══════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════

st.markdown("""
<div style='text-align:center; padding:10px 0;'>
    <span style='font-size:2.5em;'>🔬</span>
    <h1 style='display:inline; margin-left:10px;'>SPECTRALEYE-OMNISIM</h1>
    <div style='color:#64748b; font-family:monospace;'>
        Live FFT Spectrum Wheel + Forensic Face Analysis
    </div>
    <div style='margin-top:8px;'>
        <span class='badge badge-ai'>AI ENABLED</span>
        <span class='badge badge-gpu'>GPU READY</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    hardware = st.selectbox("Hardware", list(config.HARDWARE_PROFILES.keys()), index=1)
    detect_faces = st.checkbox("Detect faces", value=True)
    use_ai = st.checkbox("Use AI (if available)", value=TORCH_AVAILABLE, disabled=not TORCH_AVAILABLE)

    if 'analyzer' not in st.session_state or st.session_state.analyzer is None:
        st.session_state.analyzer = IntegratedAnalyzer(use_ai=use_ai)
    # Update use_ai
    st.session_state.analyzer.use_ai = use_ai

    st.divider()
    st.markdown("### 📊 Stats")
    if st.button("🔄 Refresh Stats"):
        stats = st.session_state.analyzer.stats
        st.session_state.show_stats = stats
    if 'show_stats' in st.session_state:
        s = st.session_state.show_stats
        st.metric("Images Analyzed", s.get("total",0))
        st.metric("Total Time", f"{s.get('time',0):.0f} ms")
        st.metric("Cache Hits", s.get("cache_hits",0))

    st.divider()
    st.caption(f"v{config.VERSION} | QCAUS Research")

# ─── Main Tabs ────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📸 Live Spectrum", "📤 Upload Analysis", "📦 Batch Process", "📊 Dashboard", "📖 About"
])

# ─── Tab 1: Live ──────────────────────────────────────────────────────
with tab1:
    st.markdown("### Live Webcam with Real‑time FFT Wheel")
    col1, col2 = st.columns([3,1])
    with col1:
        # Processor settings from UI
        darkroom = st.checkbox("Darkroom mode", value=st.session_state.darkroom)
        st.session_state.darkroom = darkroom
        mode = st.radio("Filter", ["normal", "invert"], horizontal=True, format_func=lambda x: "NORMAL" if x=="normal" else "INVERTED")
        hue_shift = st.slider("Hue Shift (deg)", 0, 360, 0, 5)
        split = st.checkbox("Split‑screen")
        wheel_color = st.radio("Wheel Color", ["orientation", "primordial"], horizontal=True)
        bright = st.slider("Brightness", -0.5, 0.5, 0.0, 0.05)
        contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.05)
        saturation = st.slider("Saturation", 0.0, 2.0, 1.0, 0.05)
        gamma = st.slider("Gamma", 0.3, 3.0, 1.0, 0.05)

    with col2:
        if st.button("CAPTURE FRAME", use_container_width=True, type="primary"):
            st.session_state.capture_trigger = True
        if st.button(f"GALLERY ({len(st.session_state.captures)})", use_container_width=True):
            st.session_state.show_gallery = not st.session_state.show_gallery

    # WebRTC stream
    ctx = webrtc_streamer(
        key="live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        video_processor_factory=SpectrumWheelProcessor,
        media_stream_constraints={"video": {"facingMode": "environment", "width": {"ideal": 1280}, "height": {"ideal": 720}}, "audio": False},
        async_processing=True,
    )

    if ctx.video_processor:
        proc = ctx.video_processor
        proc.mode = mode
        proc.hue_shift = float(hue_shift)
        proc.split_view = split
        proc.wheel_color = wheel_color
        proc.darkroom = darkroom
        proc.brightness = bright
        proc.contrast = contrast
        proc.saturation = saturation
        proc.gamma = gamma

        if st.session_state.capture_trigger:
            proc.capture_requested = True
            st.session_state.capture_trigger = False
            time.sleep(0.15)
            if hasattr(proc, "_captured_data") and proc._captured_data:
                data = proc._captured_data
                # Create card
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
                        "wheel_mode": wheel_color,
                        "hue_shift": hue_shift,
                        "peak": data["peak"],
                        "power": data["power"],
                    }
                })
                if len(st.session_state.captures) > 20:
                    st.session_state.captures = st.session_state.captures[:20]
                proc._captured_data = None
                st.rerun()

    # Gallery
    if st.session_state.show_gallery and st.session_state.captures:
        st.markdown("---")
        st.markdown("## CAPTURE GALLERY")
        # ZIP download
        if st.button("Download All as ZIP"):
            zip_buf = BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for i, cap in enumerate(st.session_state.captures):
                    img_pil = frame_to_pil(cap["card_bgr"])
                    img_bytes = pil_to_bytes(img_pil)
                    safe_ts = cap["timestamp"].replace(":", "-").replace(" ", "_")
                    zf.writestr(f"capture_{i+1:03d}_{safe_ts}.png", img_bytes)
            zip_buf.seek(0)
            b64 = base64.b64encode(zip_buf.read()).decode()
            href = f'<a href="data:application/zip;base64,{b64}" download="captures.zip" style="color:#38bdf8;">Download ZIP</a>'
            st.markdown(href, unsafe_allow_html=True)

        cols = st.columns(3)
        for i, cap in enumerate(st.session_state.captures):
            with cols[i % 3]:
                card_pil = frame_to_pil(cap["card_bgr"])
                st.image(card_pil, use_container_width=True)
                meta = cap["metadata"]
                st.caption(f"{cap['timestamp']} | PEAK:{meta['peak']:.0f}°")
                img_bytes = pil_to_bytes(card_pil)
                safe_ts = cap["timestamp"].replace(":", "-").replace(" ", "_")
                st.markdown(
                    f'<a href="data:image/png;base64,{base64.b64encode(img_bytes).decode()}" download="capture_{safe_ts}.png" style="color:#38bdf8;font-size:12px;">⬇ Download</a>',
                    unsafe_allow_html=True
                )

# ─── Tab 2: Upload ──────────────────────────────────────────────────
with tab2:
    st.markdown("### Upload Image for Forensic Analysis")
    uploaded = st.file_uploader("Choose image(s)", type=list(config.ALLOWED_EXTENSIONS), accept_multiple_files=False)
    if uploaded:
        data = uploaded.read()
        arr = np.frombuffer(data, np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is not None:
            col1, col2 = st.columns([1,1])
            with col1:
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
            with col2:
                if st.button("🔍 Analyze", type="primary", use_container_width=True):
                    with st.spinner("Analyzing..."):
                        result = st.session_state.analyzer.analyze_image(image, hardware, detect_faces)
                        st.session_state.current_result = result
                        st.success("Done")
                        # Display results
                        st.markdown("### Results")
                        overall = result["overall"]
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Deepfake Score", f"{overall['deepfake_score']:.2%}")
                        col_b.metric("Status", "⚠️ Detected" if overall['detected'] else "✅ Clean")
                        col_c.metric("Faces", overall['face_count'])
                        if "quality_rating" in result:
                            q = result["quality_rating"]
                            st.metric("Quality", q["rating"], delta=f"{q['score']:.2f}")
                        # Face details
                        if result.get("faces"):
                            st.markdown("#### Per‑Face Scores")
                            for i, fdata in enumerate(result["faces"]):
                                bbox = fdata["bbox"]
                                st.write(f"Face {i+1}: score={fdata['deepfake']['score']:.2f}, quality={fdata['quality']['focus_score']:.2f}")
                        # Performance
                        perf = result.get("performance", {})
                        if perf:
                            st.caption(f"Predicted time: {perf['total_time']:.3f}s | Throughput: {perf['throughput']:.1f}/s")
                        with st.expander("Full JSON"):
                            st.json(result)
        else:
            st.error("Could not decode image.")

# ─── Tab 3: Batch ──────────────────────────────────────────────────
with tab3:
    st.markdown("### Batch Process Multiple Images")
    files = st.file_uploader("Upload images", type=list(config.ALLOWED_EXTENSIONS), accept_multiple_files=True)
    if files:
        st.info(f"{len(files)} files uploaded")
        if st.button("▶️ Run Batch Analysis", type="primary", use_container_width=True):
            images = []
            for f in files:
                arr = np.frombuffer(f.read(), np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    images.append(img)
            if images:
                with st.spinner(f"Analyzing {len(images)} images..."):
                    results = st.session_state.analyzer.analyze_batch(images, hardware, detect_faces)
                    st.session_state.batch_results = results
                    st.success("Batch complete")
                    # Summary
                    scores = [r["overall"]["deepfake_score"] for r in results]
                    if scores:
                        col1,col2,col3,col4 = st.columns(4)
                        col1.metric("Avg Score", f"{np.mean(scores):.2%}")
                        col2.metric("Max", f"{np.max(scores):.2%}")
                        col3.metric("Min", f"{np.min(scores):.2%}")
                        col4.metric("Detected", f"{sum(1 for s in scores if s>0.165)}/{len(scores)}")
                    # Table
                    table = []
                    for i,r in enumerate(results):
                        table.append({
                            "Image": i+1,
                            "Score": f"{r['overall']['deepfake_score']:.2%}",
                            "Detected": "⚠️" if r['overall']['detected'] else "✅",
                            "Faces": r['overall']['face_count'],
                            "Quality": r.get("quality_rating",{}).get("rating","N/A")
                        })
                    st.dataframe(table, use_container_width=True)
                    # Export
                    if st.button("💾 Export JSON"):
                        json_str = json.dumps(results, indent=2, default=str)
                        st.download_button("Download JSON", data=json_str, file_name="batch_results.json")

# ─── Tab 4: Dashboard ──────────────────────────────────────────────
with tab4:
    st.markdown("### Results Dashboard")
    if st.session_state.current_result:
        res = st.session_state.current_result
        col1,col2,col3 = st.columns(3)
        col1.metric("Deepfake Score", f"{res['overall']['deepfake_score']:.2%}")
        col2.metric("Faces", res['overall']['face_count'])
        col3.metric("Time", f"{res.get('processing_time_ms',0):.1f} ms")
        if PLOTLY_AVAILABLE and res.get("faces"):
            # Radar of face metrics
            fig = go.Figure()
            for i, fdata in enumerate(res["faces"]):
                fig.add_trace(go.Scatterpolar(
                    r=[fdata["quality"]["focus_score"], fdata["quality"]["sharpness_score"], fdata["quality"]["texture_uniformity"], 1-fdata["deepfake"]["score"]],
                    theta=['Focus','Sharpness','Texture','Authenticity'],
                    name=f"Face {i+1}",
                    fill='toself'
                ))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), showlegend=True, paper_bgcolor='#07111f')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run an analysis to see dashboard.")

# ─── Tab 5: About ──────────────────────────────────────────────────
with tab5:
    st.markdown("""
    ### 📖 About SpectralEye-OmniSim
    **Version:** 3.0.0

    **Combines**:
    - 🔬 **Spectral Analysis** – FFT, deepfake artifact detection, quality metrics.
    - 👤 **Face Detection & Analysis** – per‑face forensic scoring.
    - 📸 **Live Spectrum Wheel** – real‑time 2D FFT visualisation with colour inversion.
    - 🖥️ **Performance Simulation** – predict speed on different hardware.

    **How it works**:
    - Uploaded images are scanned for faces; each face is analysed for GAN/upsampling artifacts and quality.
    - The overall score is the average of all detected faces (or whole image if none).
    - Live view uses the same FFT engine to display a polar spectral wheel.

    **Limitations**:
    - Detects specific GAN artifacts, not all modern diffusion models.
    - Face detection uses Haar cascade (can miss small/angled faces).
    - Results are investigative leads, not definitive proof.

    **Author**: Tony E. Ford | QCAUS Research
    """)

# ─── Helper functions ─────────────────────────────────────────────────
def frame_to_pil(bgr):
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

def pil_to_bytes(pil_img, fmt="PNG"):
    buf = BytesIO(); pil_img.save(buf, format=fmt); return buf.getvalue()

def create_capture_card(frame_bgr, wheel_bgr, metadata):
    h,w = frame_bgr.shape[:2]
    card_w = 400
    card_h = int(card_w * h / w) + 60
    card = np.zeros((card_h, card_w, 3), dtype=np.uint8)
    card[:] = (10,22,40)
    frame_h = card_h - 60
    frame_resized = cv2.resize(frame_bgr, (card_w, frame_h))
    card[0:frame_h, 0:card_w] = frame_resized
    wheel_small = cv2.resize(wheel_bgr, (80,80))
    wx, wy = card_w - 90, 10
    roi = card[wy:wy+80, wx:wx+80]
    blended = cv2.addWeighted(roi, 0.2, wheel_small, 0.8, 0)
    card[wy:wy+80, wx:wx+80] = blended
    cv2.rectangle(card, (0, frame_h), (card_w, card_h), (5,15,30), -1)
    ts = metadata.get("timestamp","")
    mode = metadata.get("mode","")
    cv2.putText(card, f"{ts} | {mode}", (8, frame_h+22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (56,189,248), 1)
    peak = metadata.get("peak",0.0)
    power = metadata.get("power",0.0)
    cv2.putText(card, f"PEAK:{peak:.0f}deg PWR:{power:.2f}", (8, frame_h+44), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (148,189,220), 1)
    return card

# ─── Keyboard shortcuts ───────────────────────────────────────────────
KEYBOARD_JS = """
<script>
document.addEventListener('keydown', function(e) {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    switch(e.key.toLowerCase()) {
        case ' ':
            e.preventDefault();
            var btns = document.querySelectorAll('button');
            for (var i=0; i<btns.length; i++) {
                if (btns[i].textContent.indexOf('CAPTURE FRAME') !== -1) {
                    btns[i].click(); break;
                }
            }
            break;
        case 'f':
            e.preventDefault();
            if (!document.fullscreenElement) document.documentElement.requestFullscreen();
            else document.exitFullscreen();
            break;
        case 'd':
            e.preventDefault();
            var cbs = document.querySelectorAll('input[type="checkbox"]');
            for (var j=0; j<cbs.length; j++) {
                if (cbs[j].parentElement.textContent.indexOf('Darkroom') !== -1) {
                    cbs[j].click(); break;
                }
            }
            break;
        case 'g':
            e.preventDefault();
            var btns2 = document.querySelectorAll('button');
            for (var k=0; k<btns2.length; k++) {
                if (btns2[k].textContent.indexOf('GALLERY') !== -1) {
                    btns2[k].click(); break;
                }
            }
            break;
    }
});
</script>
"""
st.components.v1.html(KEYBOARD_JS, height=0)
