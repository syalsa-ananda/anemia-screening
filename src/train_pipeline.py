"""
train_pipeline.py — v4 (BorderlineSMOTE + optimasi grid)
=========================================================
Perubahan v4:
  - BorderlineSMOTE menggantikan SMOTE biasa: hanya mensintesis sampel
    di perbatasan decision boundary → lebih relevan untuk kelas yang
    susah dibedakan (Mild vs Moderate).
  - Grid search lebih fokus di area C=1-200 yang terbukti optimal.
  - Fitur 432 dimensi (dari extract_features v4).
"""

import logging, pickle, time
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                              balanced_accuracy_score, cohen_kappa_score)
from imblearn.over_sampling import BorderlineSMOTE
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    CLASS_LABELS, INV_CLASS_LABELS,
    SMOTE_STRATEGY, SMOTE_K_NEIGHBORS, RANDOM_STATE,
    SVM_KERNEL, SVM_C_RANGE, SVM_GAMMA_RANGE,
    SVM_DECISION_FUNCTION_SHAPE, CV_FOLDS,
    USE_PCA, PCA_N_COMPONENTS,
    FEATURES_TRAIN_CSV, FEATURES_TEST_CSV,
    MODEL_PATH, SCALER_PATH, PCA_PATH, REPORT_PATH, VIZ_DIR,
    FEATURES_DIR, COL_SEVERITY
)
from utils_data import load_metadata, split_dataset
from utils_preprocessing import preprocess_image
from extract_features import extract_sample_feature_vector, get_feature_names

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def build_feature_matrix(df, desc="Extracting"):
    X_list, y_list = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc, ncols=80):
        roi = preprocess_image(row["image_path"])
        if roi is None:
            logger.warning(f"Skipping {row['IMAGE_ID']}")
            continue
        feat = extract_sample_feature_vector(roi)
        X_list.append(feat)
        y_list.append(int(row["label"]))
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    logger.info(f"Matrix: {X.shape} | {dict(zip(*np.unique(y, return_counts=True)))}")
    return X, y


def apply_smote(X_train, y_train):
    logger.info("Menerapkan BorderlineSMOTE (fokus di decision boundary)...")
    logger.info(f"Sebelum: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    smote = BorderlineSMOTE(
        sampling_strategy=SMOTE_STRATEGY,
        k_neighbors=SMOTE_K_NEIGHBORS,
        random_state=RANDOM_STATE,
        kind="borderline-1"
    )
    X_res, y_res = smote.fit_resample(X_train, y_train)
    logger.info(f"Setelah:  {dict(zip(*np.unique(y_res, return_counts=True)))}")
    return X_res, y_res


def scale_and_reduce(X_train, X_test):
    logger.info("Fitting StandardScaler...")
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    pca = None
    if USE_PCA:
        logger.info(f"Fitting PCA (variance={PCA_N_COMPONENTS})...")
        pca = PCA(n_components=PCA_N_COMPONENTS, random_state=RANDOM_STATE)
        X_tr = pca.fit_transform(X_tr)
        X_te = pca.transform(X_te)
        logger.info(f"Dimensi setelah PCA: {X_tr.shape[1]}")
    return X_tr, X_te, scaler, pca


def train_svm(X_train, y_train):
    logger.info(f"Grid Search: C={SVM_C_RANGE}, gamma={SVM_GAMMA_RANGE}")
    param_grid = {"C": SVM_C_RANGE, "gamma": SVM_GAMMA_RANGE}
    svm = SVC(kernel=SVM_KERNEL,
              decision_function_shape=SVM_DECISION_FUNCTION_SHAPE,
              probability=True, class_weight="balanced",
              random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                         random_state=RANDOM_STATE)
    grid = GridSearchCV(svm, param_grid, cv=cv,
                        scoring="balanced_accuracy",
                        n_jobs=-1, verbose=2, refit=True)
    t0 = time.time()
    grid.fit(X_train, y_train)
    logger.info(f"Selesai dalam {time.time()-t0:.1f}s")
    logger.info(f"Best params: {grid.best_params_}")
    logger.info(f"Best CV balanced_accuracy: {grid.best_score_:.4f}")
    return grid.best_estimator_


def evaluate_and_report(model, X_test, y_test):
    logger.info("=" * 60)
    logger.info("EVALUASI AKHIR PADA PURE TEST SET")
    logger.info("=" * 60)
    y_pred  = model.predict(X_test)
    names   = [INV_CLASS_LABELS[i] for i in sorted(INV_CLASS_LABELS)]
    report  = classification_report(y_test, y_pred, target_names=names)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    kappa   = cohen_kappa_score(y_test, y_pred)
    cm      = confusion_matrix(y_test, y_pred)

    print(report)
    logger.info(f"Balanced Accuracy : {bal_acc:.4f}")
    logger.info(f"Cohen's Kappa     : {kappa:.4f}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write("CLASSIFICATION REPORT — PURE TEST SET\n\n")
        f.write(report)
        f.write(f"\nBalanced Accuracy : {bal_acc:.4f}\n")
        f.write(f"Cohen's Kappa     : {kappa:.4f}\n")

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=names, yticklabels=names, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — Pure Test Set", fontweight="bold")
    plt.tight_layout()
    fig.savefig(VIZ_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    return {"balanced_accuracy": bal_acc, "kappa": kappa}


def run_full_pipeline():
    logger.info("🚀 PIPELINE v4 — Multi-radius LBP + Dual GLCM + BorderlineSMOTE")
    logger.info("=" * 60)

    df = load_metadata()
    train_df, test_df = split_dataset(df)

    logger.info("📦 Ekstraksi fitur train...")
    X_train_raw, y_train = build_feature_matrix(train_df, "[TRAIN] Extract")

    logger.info("🔒 Ekstraksi fitur pure test...")
    X_test_raw, y_test = build_feature_matrix(test_df, "[TEST]  Extract")

    feat_names = get_feature_names()
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    for X, y, path, lbl in [
        (X_train_raw, y_train, FEATURES_TRAIN_CSV, "train_raw"),
        (X_test_raw,  y_test,  FEATURES_TEST_CSV,  "test_pure"),
    ]:
        df_f = pd.DataFrame(X, columns=feat_names[:X.shape[1]])
        df_f.insert(0, "label", y)
        df_f.to_csv(path, index=False)
        logger.info(f"Saved {lbl}: {X.shape}")

    X_train_bal, y_train_bal = apply_smote(X_train_raw, y_train)
    X_tr, X_te, scaler, pca = scale_and_reduce(X_train_bal, X_test_raw)
    model = train_svm(X_tr, y_train_bal)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f: pickle.dump(model, f)
    with open(SCALER_PATH, "wb") as f: pickle.dump(scaler, f)
    if pca: 
        with open(PCA_PATH, "wb") as f: pickle.dump(pca, f)
    logger.info("✅ Model, scaler, PCA tersimpan.")

    results = evaluate_and_report(model, X_te, y_test)

    logger.info("=" * 60)
    logger.info("🏁 SELESAI")
    logger.info(f"   CV  Balanced Accuracy: lihat 'Best CV' di atas")
    logger.info(f"   Test Balanced Accuracy: {results['balanced_accuracy']:.4f}")
    logger.info(f"   Cohen's Kappa         : {results['kappa']:.4f}")
    logger.info("=" * 60)
    return model, scaler, pca, results


if __name__ == "__main__":
    run_full_pipeline()
