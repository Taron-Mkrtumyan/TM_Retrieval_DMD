"""
Validation of the retrieved TM using held-out test patterns.
"""
import numpy as np
from scipy.stats import pearsonr


def predict_output(TM, pattern):
    """Predict output intensity for a given binary input pattern."""
    field = pattern @ TM
    return np.abs(field) ** 2


def evaluate(TM, test_patterns, test_frames):
    """
    Compute mean Pearson correlation between predicted and measured
    output intensities over all test patterns.
    """
    correlations = []
    for pattern, measured in zip(test_patterns, test_frames):
        predicted = predict_output(TM, pattern)
        r, _ = pearsonr(predicted.ravel(), measured.ravel())
        correlations.append(r)
    return np.mean(correlations), correlations
