"""
config.py — v6 (ULTRA OPTIMIZATION FOR >90% ACCURACY)
Description: Menajamkan resolusi fitur tekstur dan mempersempit Grid Search SVM
             pada area parameter emas untuk mengejar akurasi medis >90%.
"""
from pathlib import Path

ROOT_DIR        = Path(__file__).resolve().parent.parent
DATA_DIR        = ROOT_DIR / "data"
RAW_DIR         = DATA_DIR / "raw"
PROCESSED_DIR   = DATA_DIR / "processed"
ANEMIC_DIR      = RAW_DIR / "Anemic"
NON_ANEMIC_DIR  = RAW_DIR / "Non-Anemic"
ROI_CROPS_DIR   = PROCESSED_DIR / "roi_crops"
FEATURES_DIR    = PROCESSED_DIR / "features"
OUTPUT_DIR      = ROOT_DIR / "outputs"
MODELS_DIR      = OUTPUT_DIR / "models"
REPORTS_DIR     = OUTPUT_DIR / "reports"
VIZ_DIR         = OUTPUT_DIR / "visualizations"
CSV_PATH        = RAW_DIR / "Anemia_Data_Collection_Sheet.csv"

COL_IMAGE_ID    = "IMAGE_ID"
COL_SEVERITY    = "Severity"

# Pemetaan 2 Kelas (Binary)
CLASS_LABELS    = {
    "Non-Anemic": 0,
    "Mild":       1,  
    "Moderate":   1,  
    "Severe":     1,  
}
INV_CLASS_LABELS = {0: "Non-Anemic", 1: "Anemic"}

TEST_SIZE       = 0.20
RANDOM_STATE    = 42
VALID_IMG_EXTS  = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

IMG_RESIZE      = (256, 256)

# =====================================================================
# PERBAIKAN 1: MENAIKKAN RESOLUSI ROI UNTUK MENANGKAP TEKSTUR MIKRO LBP/GLCM
# =====================================================================
ROI_OUTPUT_SIZE = (256, 128)  # Sebelumnya (128, 64)

CONJUNCTIVA_HSV_LOWER = (0,   20,  80)
CONJUNCTIVA_HSV_UPPER = (25, 200, 255)
BLUE_GLOVE_HSV_LOWER  = (90,  50,  50)
BLUE_GLOVE_HSV_UPPER  = (130, 255, 255)
COLOR_SPACE     = "lab"

USE_AUGMENTATION = False
PATCH_POSITIONS  = ["center"]
ROTATION_RANGE   = (-10, 10)
FLIP_HORIZONTAL  = True
MINORITY_CLASSES = ["Severe"]

# LBP
LBP_RADIUS      = 3
LBP_N_POINTS    = 8 * LBP_RADIUS
LBP_METHOD      = "uniform"

# GLCM
GLCM_DISTANCES  = [1, 2, 3]
GLCM_ANGLES     = [0, 45, 90, 135]
GLCM_PROPERTIES = ["contrast", "dissimilarity", "homogeneity",
                   "energy", "correlation", "ASM"]
GLCM_LEVELS     = 256

# Histogram
HIST_BINS       = 32

# PCA
USE_PCA              = True
PCA_N_COMPONENTS     = 0.95

# Oversampling
SMOTE_STRATEGY   = "auto"
SMOTE_K_NEIGHBORS = 3

# =====================================================================
# PERBAIKAN 2: MEMPERSEMPIT GRID SEARCH PADA AREA EVALUASI EMAS (C=10-150)
# =====================================================================
SVM_KERNEL      = "rbf"
# Melakukan scanning lebih rapat di sekitar C=50 dan gamma=0.005 yang terbukti unggul
SVM_C_RANGE     = [10, 30, 50, 75, 100, 150]
SVM_GAMMA_RANGE = [0.002, 0.004, 0.005, 0.006, 0.008, 0.01, 0.02]
SVM_DECISION_FUNCTION_SHAPE = "ovr"
CV_FOLDS        = 5

FEATURES_TRAIN_CSV  = FEATURES_DIR / "features_train_smote.csv"
FEATURES_TEST_CSV   = FEATURES_DIR / "features_test_pure.csv"
MODEL_PATH          = MODELS_DIR / "svm_anemia_model.pkl"
SCALER_PATH         = MODELS_DIR / "scaler.pkl"
PCA_PATH            = MODELS_DIR / "pca.pkl"
REPORT_PATH         = REPORTS_DIR / "classification_report.txt"

for _dir in [ROI_CROPS_DIR, FEATURES_DIR, MODELS_DIR, REPORTS_DIR, VIZ_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)