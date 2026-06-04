"""
Bonus: PSF engineering using the retrieved TM.
Shape the output scattered light into a target spatial mode.
"""
import numpy as np


def compute_input_field(TM, target_field):
    """
    Compute the required input field to produce the target output field
    by pseudoinverse of the TM.
    """
    TM_pinv = np.linalg.pinv(TM)
    return TM_pinv @ target_field


def binarize(input_field, threshold=0.0):
    """
    Binarize a complex input field into a valid DMD pattern (0 or 1)
    based on the real part threshold.
    """
    return (input_field.real > threshold).astype(np.uint8)


def mode_overlap(measured, target):
    """Compute normalized mode overlap integral between two intensity maps."""
    num = np.abs(np.sum(np.sqrt(measured) * np.conj(np.sqrt(target)))) ** 2
    denom = np.sum(measured) * np.sum(target)
    return num / denom
