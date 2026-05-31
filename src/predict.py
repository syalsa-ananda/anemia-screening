"""
predict.py — Inferensi v3 (dengan PCA)
"""
import pickle
import numpy as np
import logging
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import MODEL_PATH, SCALER_PATH, PCA_PATH, INV_CLASS_LABELS
from utils_preprocessing import preprocess_image
from extract_features import extract_sample_feature_vector

logger = logging.getLogger(__name__)

SEVERITY_COLORS = {
    "Non-Anemic": "#2ECC71",
    "Mild":       "#F1C40F",
    "Moderate":   "#E67E22",
    "Severe":     "#E74C3C",
}

def load_model_and_scaler():
    with open(MODEL_PATH, "rb") as f: model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f: scaler = pickle.load(f)
    pca = None
    if PCA_PATH.exists():
        with open(PCA_PATH, "rb") as f: pca = pickle.load(f)
    logger.info("Model, scaler, dan PCA berhasil dimuat.")
    return model, scaler, pca

def predict_image(image_path, model=None, scaler=None, pca=None) -> dict:
    if model is None:
        model, scaler, pca = load_model_and_scaler()

    roi = preprocess_image(image_path)
    if roi is None:
        return {"success": False, "error": "Preprocessing gagal."}

    feat = extract_sample_feature_vector(roi).reshape(1, -1)
    feat_scaled = scaler.transform(feat)
    if pca is not None:
        feat_scaled = pca.transform(feat_scaled)

    label_int  = int(model.predict(feat_scaled)[0])
    class_name = INV_CLASS_LABELS[label_int]
    proba      = model.predict_proba(feat_scaled)[0]
    names      = [INV_CLASS_LABELS[i] for i in range(len(proba))]
    proba_dict = {n: round(float(p), 4) for n, p in zip(names, proba)}

    return {
        "predicted_class": class_name,
        "label_int": label_int,
        "probabilities": proba_dict,
        "confidence": round(float(proba[label_int]), 4),
        "indicator_color": SEVERITY_COLORS.get(class_name, "#95A5A6"),
        "success": True,
        "error": None
    }
