"""
utils_data.py — Utilitas Pemuatan & Pemisahan Data
===================================================
Modul ini bertanggung jawab untuk:
  1. Membaca file CSV metadata master.
  2. Mencari file gambar secara adaptif di folder Anemic/ atau Non-Anemic/.
  3. Melakukan stratified split menjadi train dan pure test set.
  4. TIDAK melakukan augmentasi atau transformasi apapun (data tetap murni).

PRINSIP ANTI-LEAKAGE:
  Pure Test Set diisolasi SEBELUM augmentasi dan SMOTE dilakukan.
  Fungsi-fungsi di modul ini HANYA menghasilkan path gambar & label,
  bukan fitur yang sudah ditransformasi.
"""

import logging
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    CSV_PATH, ANEMIC_DIR, NON_ANEMIC_DIR,
    COL_IMAGE_ID, COL_SEVERITY, CLASS_LABELS,
    VALID_IMG_EXTS, TEST_SIZE, RANDOM_STATE
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


# ──────────────────────────────────────────────────────────────────────────────
def find_image_path(image_id: str) -> Path | None:
    """
    Mencari file gambar secara adaptif di folder Anemic/ dan Non-Anemic/.

    Strategi pencarian:
      1. Coba folder Non-Anemic/ terlebih dahulu (kelas mayoritas).
      2. Jika tidak ada, coba folder Anemic/.
      3. Coba semua ekstensi gambar yang valid.

    Parameters
    ----------
    image_id : str
        ID gambar dari kolom IMAGE_ID di CSV, misal "Image_001".

    Returns
    -------
    Path | None
        Path lengkap file gambar jika ditemukan, atau None jika tidak ada.
    """
    search_dirs = [NON_ANEMIC_DIR, ANEMIC_DIR]

    for folder in search_dirs:
        for ext in VALID_IMG_EXTS:
            candidate = folder / f"{image_id}{ext}"
            if candidate.exists():
                logger.debug(f"Ditemukan: {candidate}")
                return candidate

    logger.warning(f"File gambar untuk '{image_id}' tidak ditemukan di kedua folder.")
    return None


# ──────────────────────────────────────────────────────────────────────────────
def load_metadata() -> pd.DataFrame:
    """
    Membaca CSV master dan memetakan label Severity ke integer.

    Langkah:
      1. Baca CSV.
      2. Cari path gambar untuk setiap IMAGE_ID (adaptif).
      3. Buang baris yang file gambarnya tidak ditemukan.
      4. Encode kolom Severity ke integer berdasarkan CLASS_LABELS.

    Returns
    -------
    pd.DataFrame
        DataFrame dengan kolom: ['IMAGE_ID', 'image_path', 'Severity', 'label']
        Hanya baris yang file gambarnya ditemukan yang dipertahankan.
    """
    logger.info(f"Membaca metadata dari: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    # Validasi kolom wajib
    required_cols = [COL_IMAGE_ID, COL_SEVERITY]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Kolom '{col}' tidak ditemukan di CSV. "
                             f"Kolom tersedia: {df.columns.tolist()}")

    # Normalisasi nilai Severity (strip whitespace, title case)
    df[COL_SEVERITY] = df[COL_SEVERITY].astype(str).str.strip().str.title()

    # Cari path gambar secara adaptif
    df["image_path"] = df[COL_IMAGE_ID].apply(find_image_path)

    # Buang baris tanpa gambar
    missing = df["image_path"].isna().sum()
    if missing > 0:
        logger.warning(f"{missing} gambar tidak ditemukan dan akan dibuang.")
    df = df.dropna(subset=["image_path"]).reset_index(drop=True)

    # Validasi label kelas
    unknown_labels = set(df[COL_SEVERITY].unique()) - set(CLASS_LABELS.keys())
    if unknown_labels:
        raise ValueError(f"Label tidak dikenal di CSV: {unknown_labels}. "
                         f"Label valid: {list(CLASS_LABELS.keys())}")

    # Encode label ke integer
    df["label"] = df[COL_SEVERITY].map(CLASS_LABELS)

    logger.info(f"Total sampel valid: {len(df)}")
    logger.info(f"Distribusi kelas:\n{df[COL_SEVERITY].value_counts().to_string()}")

    return df[[COL_IMAGE_ID, "image_path", COL_SEVERITY, "label"]]


# ──────────────────────────────────────────────────────────────────────────────
def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Melakukan stratified split menjadi Train Set (80%) dan Pure Test Set (20%).

    ANTI-LEAKAGE GUARANTEE:
      - Split dilakukan pada metadata (path + label), BUKAN pada fitur.
      - Augmentasi dan SMOTE HANYA diterapkan ke train_df, BUKAN test_df.
      - Fungsi ini adalah "gerbang isolasi" Pure Test Set.

    Parameters
    ----------
    df : pd.DataFrame
        Output dari load_metadata().

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df) — kedua DataFrame berisi path gambar & label asli.
    """
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"],   # stratified: proporsi kelas dijaga di kedua set
        shuffle=True,
    )

    train_df = train_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    logger.info("=" * 55)
    logger.info(f"SPLIT SELESAI → Train: {len(train_df)} | Test: {len(test_df)}")
    logger.info(f"Train distribusi:\n{train_df[COL_SEVERITY].value_counts().to_string()}")
    logger.info(f"Test distribusi:\n{test_df[COL_SEVERITY].value_counts().to_string()}")
    logger.info("Pure Test Set telah diisolasi. JANGAN disentuh sebelum evaluasi akhir.")
    logger.info("=" * 55)

    return train_df, test_df
