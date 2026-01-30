"""
AlphaOS Denoising Module (v4.0)

Signal denoising for price and feature series:
- KalmanFilter: Online-friendly 1D Kalman filter for real-time denoising
- WaveletDenoiser: Offline wavelet-based denoising for training data

Key Insight:
- Kalman filter provides `price_estimate` (denoised) + `residual` (noise level)
- Residual magnitude is a feature itself (high residual = noisy/uncertain regime)

Reference:
- "Advances in Financial Machine Learning" Chapter 18: Entropy Features
- PyWavelets documentation for wavelet denoising
"""

from alphaos.features.denoise.kalman import (
    KalmanFilter,
    AdaptiveKalmanFilter,
    KalmanState,
    compute_kalman_batch,
)
from alphaos.features.denoise.wavelet import (
    WaveletDenoiser,
    compute_wavelet_denoise_batch,
    compute_wavelet_features,
)

__all__ = [
    # Kalman
    "KalmanFilter",
    "AdaptiveKalmanFilter",
    "KalmanState",
    "compute_kalman_batch",
    # Wavelet
    "WaveletDenoiser",
    "compute_wavelet_denoise_batch",
    "compute_wavelet_features",
]
