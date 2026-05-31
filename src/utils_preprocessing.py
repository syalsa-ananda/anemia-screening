"""
utils_preprocessing.py — Preprocessing, ROI Extraction & Augmentasi
====================================================================
PERUBAHAN v2:
  - ROI dikembalikan dalam format RGB (bukan Lab) agar kompatibel
    dengan semua fungsi ekstraksi fitur dan visualisasi.
  - Konversi ke Lab/HSV dilakukan di dalam extract_features.py
    hanya saat dibutuhkan per fitur, bukan di sini.
  - Ini menghilangkan ambiguitas channel yang menyebabkan fitur
    LBP/GLCM tidak bermakna pada run sebelumnya.
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    IMG_RESIZE, ROI_OUTPUT_SIZE,
    CONJUNCTIVA_HSV_LOWER, CONJUNCTIVA_HSV_UPPER,
    BLUE_GLOVE_HSV_LOWER, BLUE_GLOVE_HSV_UPPER,
    PATCH_POSITIONS, ROTATION_RANGE, FLIP_HORIZONTAL,
    ROI_CROPS_DIR, MINORITY_CLASSES, CLASS_LABELS
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 1. UTILITAS DASAR
# ──────────────────────────────────────────────────────────────────────────────

def load_image(image_path) -> Optional[np.ndarray]:
    """Muat gambar dari disk, kembalikan dalam format RGB uint8."""
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        logger.warning(f"Gagal memuat gambar: {image_path}")
        return None
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def resize_image(img: np.ndarray, size: tuple = IMG_RESIZE) -> np.ndarray:
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


# ──────────────────────────────────────────────────────────────────────────────
# 2. SEGMENTASI ROI
# ──────────────────────────────────────────────────────────────────────────────

def create_conjunctiva_mask(img_rgb: np.ndarray) -> np.ndarray:
    """
    Buat binary mask area konjungtiva dari gambar RGB.
    Masking dilakukan di ruang HSV karena lebih intuitif untuk warna.
    """
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    lower_c = np.array(CONJUNCTIVA_HSV_LOWER, dtype=np.uint8)
    upper_c = np.array(CONJUNCTIVA_HSV_UPPER, dtype=np.uint8)
    mask_conjunctiva = cv2.inRange(img_hsv, lower_c, upper_c)

    lower_b = np.array(BLUE_GLOVE_HSV_LOWER, dtype=np.uint8)
    upper_b = np.array(BLUE_GLOVE_HSV_UPPER, dtype=np.uint8)
    mask_blue = cv2.inRange(img_hsv, lower_b, upper_b)

    mask_clean = cv2.bitwise_and(mask_conjunctiva, cv2.bitwise_not(mask_blue))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask_closed = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask_closed


def extract_roi_from_mask(img_rgb: np.ndarray,
                           mask: np.ndarray,
                           output_size: tuple = ROI_OUTPUT_SIZE) -> Optional[np.ndarray]:
    """
    Ekstrak bounding box ROI terbesar dari mask, resize ke output_size.
    Gambar masuk dan keluar dalam format RGB.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        pad_x = int(w * 0.05)
        pad_y = int(h * 0.05)
        H_img, W_img = img_rgb.shape[:2]
        x = max(0, x - pad_x)
        y = max(0, y - pad_y)
        w = min(W_img - x, w + 2 * pad_x)
        h = min(H_img - y, h + 2 * pad_y)
        roi = img_rgb[y:y+h, x:x+w]
    else:
        logger.warning("Mask kosong. Menggunakan crop tengah sebagai fallback ROI.")
        H_img, W_img = img_rgb.shape[:2]
        y1, y2 = H_img // 4, 3 * H_img // 4
        x1, x2 = W_img // 4, 3 * W_img // 4
        roi = img_rgb[y1:y2, x1:x2]

    return cv2.resize(roi, output_size, interpolation=cv2.INTER_AREA)


def preprocess_image(image_path,
                     save_roi: bool = False,
                     image_id: str = "") -> Optional[np.ndarray]:
    """
    Pipeline preprocessing lengkap. Output: ROI dalam format RGB uint8.
    Tidak ada konversi ruang warna di sini — ekstraksi fitur yang menangani itu.
    """
    img_rgb = load_image(image_path)
    if img_rgb is None:
        return None

    img_resized = resize_image(img_rgb)
    mask = create_conjunctiva_mask(img_resized)
    roi = extract_roi_from_mask(img_resized, mask)

    if roi is None:
        return None

    if save_roi and image_id:
        save_path = ROI_CROPS_DIR / f"{image_id}_roi.jpg"
        # Simpan sebagai BGR untuk cv2.imwrite
        cv2.imwrite(str(save_path), cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))

    return roi   # RGB uint8


# ──────────────────────────────────────────────────────────────────────────────
# 3. PATCH EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────

def extract_patches(roi: np.ndarray,
                    positions: list = PATCH_POSITIONS) -> dict:
    """Potong ROI menjadi sub-patch horizontal: left, center, right."""
    H, W = roi.shape[:2]
    patch_w = W // 3
    patches = {
        "left":   roi[:, 0:patch_w],
        "center": roi[:, patch_w:2*patch_w],
        "right":  roi[:, 2*patch_w:],
    }
    return {pos: patches[pos] for pos in positions if pos in patches}


# ──────────────────────────────────────────────────────────────────────────────
# 4. AUGMENTASI (hanya train set)
# ──────────────────────────────────────────────────────────────────────────────

def augment_patch(patch: np.ndarray, seed: int = 0) -> list:
    """Hasilkan variasi geometri aman dari satu patch RGB."""
    augmented = [patch.copy()]
    H, W = patch.shape[:2]
    center = (W / 2, H / 2)

    if FLIP_HORIZONTAL:
        augmented.append(cv2.flip(patch, 1))

    for angle in [ROTATION_RANGE[0], ROTATION_RANGE[1]]:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(patch, M, (W, H),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)
        augmented.append(rotated)

    return augmented


def augment_sample(roi: np.ndarray,
                   label: int,
                   image_id: str,
                   is_minority: bool = False) -> list:
    """
    Pipeline augmentasi per sampel.
    Minority: augmentasi pada patch center saja (lebih terkontrol).
    Majority: patch center tanpa augmentasi.
    """
    patches = extract_patches(roi)
    results = []

    # Selalu gunakan center patch sebagai representasi utama
    center_patch = patches.get("center", roi)
    patch_id = f"{image_id}_center"

    if is_minority:
        for idx, variant in enumerate(augment_patch(center_patch)):
            results.append((variant, label, f"{patch_id}_aug{idx}"))
    else:
        results.append((center_patch, label, patch_id))

    return results
