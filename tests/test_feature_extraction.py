"""
test_feature_extraction.py — v4
"""
import sys, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import GLCM_DISTANCES, GLCM_ANGLES, GLCM_PROPERTIES, HIST_BINS
from extract_features import (
    extract_lbp_features, extract_glcm_features,
    extract_color_histogram, extract_color_moments,
    extract_all_features, extract_sample_feature_vector, get_feature_names,
    LBP_RADII
)

N_LBP     = sum(8*r + 2 for r in LBP_RADII)          # 78
N_GLCM    = 2 * len(GLCM_DISTANCES) * len(GLCM_ANGLES) * len(GLCM_PROPERTIES)  # 144
N_HIST    = 2 * HIST_BINS * 3                          # 192
N_MOMENTS = 2 * 9                                      # 18
N_TOTAL   = N_LBP + N_GLCM + N_HIST + N_MOMENTS       # 432

@pytest.fixture
def patch():
    rng = np.random.default_rng(42)
    p = np.zeros((64, 64, 3), dtype=np.uint8)
    p[:,:,0] = rng.integers(180, 220, (64,64))
    p[:,:,1] = rng.integers(100, 150, (64,64))
    p[:,:,2] = rng.integers(100, 140, (64,64))
    return p

@pytest.fixture
def roi():
    rng = np.random.default_rng(7)
    r = np.zeros((64, 128, 3), dtype=np.uint8)
    r[:,:,0] = rng.integers(160, 220, (64,128))
    r[:,:,1] = rng.integers(90,  150, (64,128))
    r[:,:,2] = rng.integers(90,  140, (64,128))
    return r

class TestLBP:
    def test_shape(self, patch): assert extract_lbp_features(patch).shape == (N_LBP,)
    def test_no_nan(self, patch): assert not np.any(np.isnan(extract_lbp_features(patch)))
    def test_dtype(self, patch): assert extract_lbp_features(patch).dtype == np.float32

class TestGLCM:
    def test_shape(self, patch): assert extract_glcm_features(patch).shape == (N_GLCM,)
    def test_no_nan(self, patch): assert not np.any(np.isnan(extract_glcm_features(patch)))
    def test_dtype(self, patch): assert extract_glcm_features(patch).dtype == np.float32

class TestHistogram:
    def test_shape(self, patch): assert extract_color_histogram(patch).shape == (N_HIST,)
    def test_no_nan(self, patch): assert not np.any(np.isnan(extract_color_histogram(patch)))
    def test_non_neg(self, patch): assert np.all(extract_color_histogram(patch) >= 0)

class TestMoments:
    def test_shape(self, patch): assert extract_color_moments(patch).shape == (N_MOMENTS,)
    def test_no_nan(self, patch): assert not np.any(np.isnan(extract_color_moments(patch)))

class TestCombined:
    def test_shape(self, patch): assert extract_all_features(patch).shape == (N_TOTAL,)
    def test_no_nan(self, patch): assert not np.any(np.isnan(extract_all_features(patch)))

class TestROI:
    def test_shape(self, roi): assert extract_sample_feature_vector(roi).shape == (N_TOTAL,)
    def test_no_nan(self, roi): assert not np.any(np.isnan(extract_sample_feature_vector(roi)))
    def test_reproducible(self, roi):
        np.testing.assert_array_equal(
            extract_sample_feature_vector(roi),
            extract_sample_feature_vector(roi))

class TestNames:
    def test_length(self, roi):
        assert len(get_feature_names()) == extract_sample_feature_vector(roi).shape[0]
    def test_no_duplicates(self):
        n = get_feature_names()
        assert len(n) == len(set(n))
