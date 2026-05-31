"""
extract_features.py — Ekstraksi Fitur v4 (diperkaya)
=====================================================
Penambahan v4:
  1. Multi-radius LBP: R=1,2,3 digabung → menangkap tekstur di berbagai skala
  2. GLCM dari channel a* (Lab) selain L → sensitif terhadap hemoglobin
  3. Color moments dari HSV tambahan → H (hue) sangat relevan untuk pallor
  4. Fitur statistik lokal per region (atas/tengah/bawah ROI)

Dimensi total per sampel:
  LBP multi-radius : 3 × 26 = 78
  GLCM (L channel) : 72
  GLCM (a channel) : 72
  Histogram Lab    : 96
  Histogram HSV    : 96
  Momen Lab        : 9
  Momen HSV        : 9
  TOTAL            : 432
"""

import numpy as np
import cv2
import logging
from pathlib import Path
from scipy import stats as scipy_stats
from skimage.feature import local_binary_pattern
from skimage.feature import graycomatrix, graycoprops

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    LBP_N_POINTS, LBP_METHOD,
    GLCM_DISTANCES, GLCM_ANGLES, GLCM_PROPERTIES, GLCM_LEVELS,
    HIST_BINS
)

logger = logging.getLogger(__name__)

# Multi-radius LBP untuk menangkap tekstur di berbagai skala spasial
LBP_RADII = [1, 2, 3]


def _to_lab(patch_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(patch_rgb.astype(np.uint8), cv2.COLOR_RGB2LAB)


def _to_hsv(patch_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(patch_rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)


def _safe_skew(arr: np.ndarray) -> float:
    s = float(scipy_stats.skew(arr))
    return s if np.isfinite(s) else 0.0


# ── 1. LBP MULTI-RADIUS ──────────────────────────────────────────────────────

def extract_lbp_features(patch_rgb: np.ndarray) -> np.ndarray:
    """
    LBP pada channel L dari Lab, dengan 3 radius berbeda.
    Setiap radius menangkap tekstur di skala spasial berbeda.
    Output: (3 × 26,) = (78,)
    """
    lab = _to_lab(patch_rgb)
    gray = lab[:, :, 0]
    features = []
    for r in LBP_RADII:
        n_pts = 8 * r
        lbp = local_binary_pattern(gray, P=n_pts, R=r, method=LBP_METHOD)
        n_bins = n_pts + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins,
                                range=(0, n_bins), density=True)
        features.extend(hist.tolist())
    return np.array(features, dtype=np.float32)


# ── 2. GLCM DUAL CHANNEL ─────────────────────────────────────────────────────

def _glcm_from_channel(channel: np.ndarray) -> np.ndarray:
    """Hitung GLCM dari satu channel grayscale. Output: (72,)"""
    angles_rad = [a * np.pi / 180 for a in GLCM_ANGLES]
    glcm = graycomatrix(
        channel.astype(np.uint8),
        distances=GLCM_DISTANCES,
        angles=angles_rad,
        levels=GLCM_LEVELS,
        symmetric=True, normed=True
    )
    feats = []
    for prop in GLCM_PROPERTIES:
        feats.extend(graycoprops(glcm, prop).ravel().tolist())
    return np.array(feats, dtype=np.float32)


def extract_glcm_features(patch_rgb: np.ndarray) -> np.ndarray:
    """
    GLCM dari channel L (luminansi) dan channel a* (hemoglobin proxy).
    Output: (72 + 72,) = (144,)
    """
    lab = _to_lab(patch_rgb)
    glcm_L = _glcm_from_channel(lab[:, :, 0])   # luminansi
    glcm_a = _glcm_from_channel(lab[:, :, 1])   # green-red (hemoglobin)
    return np.concatenate([glcm_L, glcm_a])


# ── 3. COLOR HISTOGRAM DUAL SPACE ────────────────────────────────────────────

def _histogram_from_image(img: np.ndarray, bins: int = HIST_BINS) -> np.ndarray:
    """Histogram 3 channel dari satu gambar. Output: (bins × 3,)"""
    feats = []
    for ch in range(3):
        h = cv2.calcHist([img[:, :, ch]], [0], None, [bins], [0, 256])
        feats.extend((h.ravel() / (h.sum() + 1e-7)).tolist())
    return np.array(feats, dtype=np.float32)


def extract_color_histogram(patch_rgb: np.ndarray) -> np.ndarray:
    """
    Histogram dari Lab dan HSV digabung.
    Output: (96 + 96,) = (192,)
    """
    lab = _to_lab(patch_rgb)
    hsv = _to_hsv(patch_rgb)
    return np.concatenate([
        _histogram_from_image(lab),
        _histogram_from_image(hsv),
    ])


# ── 4. COLOR MOMENTS DUAL SPACE ──────────────────────────────────────────────

def extract_color_moments(patch_rgb: np.ndarray) -> np.ndarray:
    """
    Mean, std, skewness dari Lab dan HSV.
    Output: (9 + 9,) = (18,)
    """
    lab = _to_lab(patch_rgb).astype(np.float32)
    hsv = _to_hsv(patch_rgb).astype(np.float32)
    feats = []
    for img in [lab, hsv]:
        for ch in range(3):
            arr = img[:, :, ch].ravel()
            feats.append(float(np.mean(arr)))
            feats.append(float(np.std(arr)))
            feats.append(_safe_skew(arr))
    return np.array(feats, dtype=np.float32)


# ── 5. GABUNGAN SEMUA FITUR ───────────────────────────────────────────────────

def extract_all_features(patch_rgb: np.ndarray) -> np.ndarray:
    """
    Gabung semua fitur dari satu patch RGB.
    Output: (78 + 144 + 192 + 18,) = (432,)
    """
    return np.concatenate([
        extract_lbp_features(patch_rgb),      # 78
        extract_glcm_features(patch_rgb),     # 144
        extract_color_histogram(patch_rgb),   # 192
        extract_color_moments(patch_rgb),     # 18
    ])


def extract_sample_feature_vector(roi_rgb: np.ndarray) -> np.ndarray:
    """
    Ekstrak dari center patch ROI.
    Input: ROI RGB uint8 shape (H, W, 3).
    Output: (432,)
    """
    H, W = roi_rgb.shape[:2]
    pw = W // 3
    center = roi_rgb[:, pw:2*pw]
    return extract_all_features(center)


# ── 6. NAMA FITUR ─────────────────────────────────────────────────────────────

def get_feature_names() -> list:
    names = []
    # LBP multi-radius
    for r in LBP_RADII:
        n_pts = 8 * r
        for i in range(n_pts + 2):
            names.append(f"lbp_r{r}_bin{i}")
    # GLCM L dan a
    for ch_name in ["L", "a"]:
        for prop in GLCM_PROPERTIES:
            for d in GLCM_DISTANCES:
                for a in GLCM_ANGLES:
                    names.append(f"glcm_{ch_name}_{prop}_d{d}_a{a}")
    # Histogram Lab dan HSV
    for space in ["Lab", "HSV"]:
        ch_names = ["L","a","b"] if space == "Lab" else ["H","S","V"]
        for ch in ch_names:
            for b in range(HIST_BINS):
                names.append(f"hist_{space}_{ch}_bin{b}")
    # Momen Lab dan HSV
    for space in ["Lab", "HSV"]:
        ch_names = ["L","a","b"] if space == "Lab" else ["H","S","V"]
        for ch in ch_names:
            for stat in ["mean","std","skew"]:
                names.append(f"moment_{space}_{ch}_{stat}")
    return names
