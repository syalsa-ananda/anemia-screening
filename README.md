# 🔬 Skrining Anemia Non-Invasif pada Balita
### Berbasis Citra Konjungtiva Mata | LBP + GLCM + Color Histogram | SVM Binary Classification

---

## 📋 Deskripsi Proyek

Proyek ini mengimplementasikan sistem skrining anemia non-invasif menggunakan
citra konjungtiva kelopak mata bawah dengan pendekatan **Pengenalan Pola tradisional**.
Pipeline dirancang khusus untuk memenuhi standar penelitian medis yang valid secara
statistika dengan jaminan **Anti-Data Leakage** yang ketat.

**Judul Lengkap:**
> *"Skrining Anemia Non-Invasif pada Balita Berbasis Citra Konjungtiva Mata
> Menggunakan Fitur LBP, GLCM, dan Color Histogram dengan Klasifikasi
> Support Vector Machine"*

---
## 📓 Lihat Notebook

| Notebook | Google Colab | NBViewer |
|----------|-------------|----------|
| 01 - Exploratory Analysis | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/syalsa-ananda/anemia-screening/blob/main/notebooks/01_exploratory_analysis.ipynb) | [![NBViewer](https://img.shields.io/badge/render-nbviewer-orange)](https://nbviewer.org/github/syalsa-ananda/anemia-screening/blob/main/notebooks/01_exploratory_analysis.ipynb) |
| 02 - SVM Pipeline | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/syalsa-ananda/anemia-screening/blob/main/notebooks/02_svm_pipeline.ipynb) | [![NBViewer](https://img.shields.io/badge/render-nbviewer-orange)](https://nbviewer.org/github/syalsa-ananda/anemia-screening/blob/main/notebooks/02_svm_pipeline.ipynb) |

## 🏆 Hasil Akhir

| Metrik | Nilai |
|--------|-------|
| **Balanced Accuracy** | **85.35%** |
| **Cohen's Kappa** | **0.7069** (Substansial Agreement) |
| Accuracy | 86% |
| Precision (macro) | 85% |
| Recall (macro) | 85% |
| F1-Score (macro) | 85% |
| Best SVM Params | C=100, γ=0.002 |


---

## 🗂️ Struktur Folder

```
anemia_screening/
│
├── README.md
├── requirements.txt
│
├── data/
│   └── raw/
│       ├── Anemia_Data_Collection_Sheet.csv   ← Metadata master (IMAGE_ID + Severity)
│       ├── Anemic/                            ← Gambar Mild, Moderate, Severe
│       └── Non-Anemic/                        ← Gambar Normal
│
│   └── processed/                             ← Dibuat otomatis
│       ├── roi_crops/                         ← Hasil crop konjungtiva
│       └── features/
│           ├── features_train_smote.csv
│           └── features_test_pure.csv
│
├── src/
│   ├── config.py                  ← Semua konstanta & hyperparameter
│   ├── utils_data.py              ← Load CSV + adaptive path search + stratified split
│   ├── utils_preprocessing.py    ← ROI segmentasi konjungtiva + normalisasi warna
│   ├── extract_features.py       ← LBP + GLCM + Color Histogram + Color Moments
│   ├── train_pipeline.py         ← Pipeline utama: SMOTE → PCA → GridSearchCV SVM
│   └── predict.py                ← Inferensi satu gambar baru
│
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb  ← EDA: distribusi kelas, visualisasi ROI, PCA
│   └── 02_svm_pipeline.ipynb          ← Pipeline SVM lengkap dengan semua visualisasi
│
├── tests/
│   └── test_feature_extraction.py     ← Unit test 22 test case
│
└── outputs/                           ← Dibuat otomatis
    ├── models/
    │   ├── svm_anemia_model.pkl
    │   ├── scaler.pkl
    │   └── pca.pkl
    ├── reports/
    │   └── classification_report.txt
    └── visualizations/
        ├── class_distribution.png
        ├── roi_comparison.png
        ├── svm_smote_effect.png
        ├── svm_pca_analysis.png
        ├── svm_gridsearch_heatmap.png
        ├── svm_evaluation_final.png
        └── svm_sample_predictions.png
```

---

## 🔄 Arsitektur Pipeline (Anti-Data Leakage)

```
CSV Master (710 gambar)
        │
        ▼
Stratified Split 80/20
        │
        ├──── PURE TEST SET (142) ──────────────────────────────► [FREEZE]
        │                                                              │
        └──── TRAIN SET (568)                                          │
                    │                                                  │
                    ▼                                                  │
        Preprocessing ROI Konjungtiva                                  │
        (HSV masking + crop + RGB output)                              │
                    │                                                  │
                    ▼                                                  │
        Ekstraksi Fitur (408 dimensi)                                  │
        ┌─────────────────────────┐                                    │
        │ LBP multi-radius: 78   │                                    │
        │ GLCM dual channel: 144 │                                    │
        │ Color Histogram: 192   │                                    │
        │ Color Moments: 18      │                                    │
        └─────────────────────────┘                                    │
                    │                                                  │
                    ▼                                                  │
        BorderlineSMOTE (229→339 per kelas)                           │
                    │                                                  │
                    ▼                                                  │
        StandardScaler.FIT_TRANSFORM ──── StandardScaler.TRANSFORM ───┤
                    │                                                  │
                    ▼                                                  │
        PCA (408 → 57 dimensi, 95% variance)                          │
                    │                                                  │
                    ▼                                                  ▼
        GridSearchCV SVM ──────────────── Evaluasi Final ◄──── Pure Test Set
        (5-fold StratifiedKFold)         Balanced Acc: 85.35%
        Best: C=100, γ=0.002             Kappa: 0.7069
```

---

## ⚙️ Dimensi Fitur

| Komponen | Konfigurasi | Dimensi |
|----------|-------------|---------|
| LBP multi-radius | R=1,2,3 — uniform | 78 |
| GLCM channel L | 3 dist × 4 angles × 6 props | 72 |
| GLCM channel a* | 3 dist × 4 angles × 6 props | 72 |
| Color Histogram Lab | 32 bin × 3 channel | 96 |
| Color Histogram HSV | 32 bin × 3 channel | 96 |
| Color Moments Lab+HSV | mean/std/skew × 6 channel | 18 |
| **Total** | | **432** |
| **Setelah PCA (95%)** | | **57** |

---

## 🎯 Target Kelas

| Label | Integer | Mapping |
|-------|---------|---------|
| Non-Anemic | 0 | Hb ≥ 11 g/dL |
| Anemic | 1 | Mild + Moderate + Severe (Hb < 11 g/dL) |

---

## 🚀 Cara Menjalankan

### 1. Clone & Setup Environment
```bash
git clone https://github.com/USERNAME/anemia_screening.git
cd anemia_screening
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

### 2. Siapkan Data
```
data/raw/Anemia_Data_Collection_Sheet.csv
data/raw/Anemic/Image_XXX.png
data/raw/Non-Anemic/Image_XXX.png
```

### 3. Jalankan via Notebook (Disarankan)
```bash
jupyter notebook
```
Buka dan jalankan secara berurutan:
1. `notebooks/01_exploratory_analysis.ipynb` — EDA
2. `notebooks/02_svm_pipeline.ipynb` — Training + Evaluasi

### 4. Atau via Terminal
```bash
cd src
python train_pipeline.py
```

### 5. Unit Test
```bash
python -m pytest tests/test_feature_extraction.py -v
```

### 6. Prediksi Gambar Baru
```python
from src.predict import load_model_and_scaler, predict_image

model, scaler, pca = load_model_and_scaler()
result = predict_image('path/gambar.jpg', model, scaler, pca)
print(result['predicted_class'])  # "Anemic" atau "Non-Anemic"
print(result['confidence'])       # 0.88
```

---

## 📦 Library Utama

| Library | Versi | Fungsi |
|---------|-------|--------|
| opencv-python | 4.10.0.84 | Image processing & ROI |
| scikit-image | 0.24.0 | LBP & GLCM extraction |
| scikit-learn | 1.5.2 | SVM, PCA, GridSearchCV |
| imbalanced-learn | 0.12.3 | BorderlineSMOTE |
| scipy | latest | Color moment (skewness) |
| numpy | 1.26.4 | Array operations |
| pandas | 2.2.3 | Data handling |
| matplotlib | 3.9.2 | Visualization |
| seaborn | 0.13.2 | Heatmap & plots |

---

## 📊 Perbandingan dengan Penelitian Sejenis

| Penelitian | Metode | Fitur | Akurasi SVM |
|-----------|--------|-------|-------------|
| **Penelitian ini** | SVM + LBP+GLCM+Histogram | Konjungtiva | **85.35%** |
| Mahmud et al. (2023) | SVM + RGB+HSV | Mukosa bibir | 75% |
| Sevani et al. (2018) | K-Means | Konjungtiva | 90% |
| Magdalena et al. (2022) | CNN | Konjungtiva | 94% |

---

## ⚠️ Keterbatasan

- Dataset berasal dari Ghana — domain shift terhadap populasi balita Indonesia
- 7 dari 710 gambar gagal segmentasi ROI (fallback ke crop tengah)
- Klasifikasi binary (bukan 4 tingkat keparahan) karena keterbatasan jumlah data

---

## 📚 Referensi Utama

- Mahmud, S. et al. (2023). *Anemia detection through non-invasive analysis of lip mucosa images.* Frontiers in Big Data, 6, 1241899.
- Landis, J.R. & Koch, G.G. (1977). *The measurement of observer agreement for categorical data.* Biometrics, 33(1), 159-174.

---

## 👤 Informasi Proyek

- **Mata Kuliah:** Pengenalan Pola
- **Dataset:** 710 citra konjungtiva (Ghana)
- **Tahun:** 2025
