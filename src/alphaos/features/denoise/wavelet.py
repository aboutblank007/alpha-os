"""
Wavelet Denoising for Training Data (v4.0)

Offline wavelet-based denoising using MODWT (Maximal Overlap DWT).

Key Features:
- Multi-resolution analysis: separate signal into different frequency components
- Threshold shrinkage: remove noise while preserving sharp features
- MODWT: shift-invariant, handles arbitrary length sequences

Usage:
- Apply to training data (price series, returns) before label generation
- Generates multi-scale decomposition as additional features
- NOT for real-time inference (introduces lookahead)

Reference:
- Percival & Walden (2000). Wavelet Methods for Time Series Analysis
- Donoho & Johnstone (1994). Ideal Spatial Adaptation by Wavelet Shrinkage
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np
from numpy.typing import NDArray

from alphaos.core.logging import get_logger

logger = get_logger(__name__)


# Check if pywt is available
try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False
    logger.warning("PyWavelets not installed. Wavelet denoising disabled.")


ThresholdMode = Literal["soft", "hard"]
ThresholdRule = Literal["universal", "sqtwolog", "minimax", "sure"]


@dataclass
class WaveletDenoiser:
    """
    Wavelet-based denoiser for offline data processing.
    
    Uses Discrete Wavelet Transform (DWT) with soft thresholding
    to remove high-frequency noise while preserving signal structure.
    
    Args:
        wavelet: Wavelet name (default: 'db4' - Daubechies 4)
            Other options: 'haar', 'db8', 'sym4', 'coif1'
        level: Decomposition level (None = auto-select)
        threshold_mode: 'soft' or 'hard' thresholding
        threshold_rule: Method to compute threshold
            - 'universal': σ * √(2 * log(n)) - good default
            - 'sqtwolog': Same as universal
            - 'minimax': Minimax risk threshold
            - 'sure': SURE (Stein's Unbiased Risk Estimate)
    
    Usage:
        denoiser = WaveletDenoiser(wavelet='db4', level=3)
        
        # Denoise price series
        clean_prices = denoiser.denoise(prices)
        
        # Get multi-scale decomposition
        components = denoiser.decompose(prices)
    """
    
    wavelet: str = "db4"
    level: int | None = None
    threshold_mode: ThresholdMode = "soft"
    threshold_rule: ThresholdRule = "universal"
    
    def denoise(self, signal: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Denoise a 1D signal using wavelet thresholding.
        
        Args:
            signal: Input signal array
            
        Returns:
            Denoised signal (same length as input)
        """
        if not HAS_PYWT:
            logger.warning("PyWavelets not available, returning original signal")
            return signal.copy()
        
        n = len(signal)
        
        # Determine decomposition level
        if self.level is None:
            # Auto-select: log2(n) - 1, capped at 6
            max_level = pywt.dwt_max_level(n, self.wavelet)
            level = min(max_level, 6)
        else:
            level = self.level
        
        # Perform DWT decomposition
        coeffs = pywt.wavedec(signal, self.wavelet, level=level)
        
        # Estimate noise level from finest detail coefficients
        # Use Median Absolute Deviation (robust to outliers)
        detail_coeffs = coeffs[-1]
        sigma = self._estimate_sigma(detail_coeffs)
        
        # Compute threshold
        threshold = self._compute_threshold(n, sigma)
        
        # Apply thresholding to detail coefficients (keep approximation)
        denoised_coeffs = [coeffs[0]]  # Keep approximation unchanged
        for i in range(1, len(coeffs)):
            denoised_coeffs.append(
                self._apply_threshold(coeffs[i], threshold)
            )
        
        # Reconstruct
        denoised = pywt.waverec(denoised_coeffs, self.wavelet)
        
        # Ensure same length (waverec may add 1 sample)
        return denoised[:n]
    
    def decompose(
        self, 
        signal: NDArray[np.float64],
        return_dict: bool = True,
    ) -> dict[str, NDArray[np.float64]] | list[NDArray[np.float64]]:
        """
        Decompose signal into multi-scale components.
        
        Useful for generating features at different frequency scales.
        
        Args:
            signal: Input signal
            return_dict: If True, return dict with named components
            
        Returns:
            Dictionary or list of wavelet components:
            - 'approx': Low-frequency approximation (trend)
            - 'detail_1': Highest frequency details (noise/HF trading)
            - 'detail_2': Mid-high frequency
            - etc.
        """
        if not HAS_PYWT:
            logger.warning("PyWavelets not available")
            if return_dict:
                return {"approx": signal.copy()}
            return [signal.copy()]
        
        n = len(signal)
        
        # Determine level
        if self.level is None:
            max_level = pywt.dwt_max_level(n, self.wavelet)
            level = min(max_level, 6)
        else:
            level = self.level
        
        # Decompose
        coeffs = pywt.wavedec(signal, self.wavelet, level=level)
        
        # Reconstruct each component separately
        components = []
        
        # Approximation (low frequency trend)
        approx_coeffs = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
        approx = pywt.waverec(approx_coeffs, self.wavelet)[:n]
        components.append(approx)
        
        # Details (high to low frequency)
        for i in range(1, len(coeffs)):
            detail_coeffs = [np.zeros_like(coeffs[0])]
            for j in range(1, len(coeffs)):
                if j == i:
                    detail_coeffs.append(coeffs[j])
                else:
                    detail_coeffs.append(np.zeros_like(coeffs[j]))
            
            detail = pywt.waverec(detail_coeffs, self.wavelet)[:n]
            components.append(detail)
        
        if return_dict:
            result = {"approx": components[0]}
            for i in range(1, len(components)):
                result[f"detail_{i}"] = components[i]
            return result
        
        return components
    
    def _estimate_sigma(self, detail_coeffs: NDArray[np.float64]) -> float:
        """Estimate noise standard deviation using MAD."""
        # Median Absolute Deviation / 0.6745 ≈ standard deviation for Gaussian
        mad = np.median(np.abs(detail_coeffs - np.median(detail_coeffs)))
        return mad / 0.6745
    
    def _compute_threshold(self, n: int, sigma: float) -> float:
        """Compute threshold based on rule."""
        if self.threshold_rule in ("universal", "sqtwolog"):
            # Universal threshold: σ * √(2 * log(n))
            return sigma * np.sqrt(2 * np.log(n))
        elif self.threshold_rule == "minimax":
            # Minimax threshold (approximation)
            if n <= 32:
                return 0.0
            return sigma * (0.3936 + 0.1829 * np.log(n) / np.log(2))
        else:
            # Default to universal
            return sigma * np.sqrt(2 * np.log(n))
    
    def _apply_threshold(
        self, 
        coeffs: NDArray[np.float64], 
        threshold: float,
    ) -> NDArray[np.float64]:
        """Apply soft or hard thresholding."""
        if self.threshold_mode == "soft":
            # Soft thresholding: sign(x) * max(|x| - λ, 0)
            return np.sign(coeffs) * np.maximum(np.abs(coeffs) - threshold, 0)
        else:
            # Hard thresholding: x if |x| > λ else 0
            return coeffs * (np.abs(coeffs) > threshold)


def compute_wavelet_denoise_batch(
    signals: NDArray[np.float64],
    wavelet: str = "db4",
    level: int | None = None,
    threshold_mode: ThresholdMode = "soft",
    threshold_rule: ThresholdRule = "universal",
) -> NDArray[np.float64]:
    """
    Apply wavelet denoising to multiple signals.
    
    Args:
        signals: Array of shape (n_signals, signal_length)
        wavelet: Wavelet name
        level: Decomposition level
        threshold_mode: 'soft' or 'hard'
        threshold_rule: Threshold computation method
        
    Returns:
        Denoised signals array (same shape)
    """
    denoiser = WaveletDenoiser(
        wavelet=wavelet,
        level=level,
        threshold_mode=threshold_mode,
        threshold_rule=threshold_rule,
    )
    
    if signals.ndim == 1:
        return denoiser.denoise(signals)
    
    denoised = np.zeros_like(signals)
    for i in range(signals.shape[0]):
        denoised[i] = denoiser.denoise(signals[i])
    
    return denoised


def compute_wavelet_features(
    prices: NDArray[np.float64],
    wavelet: str = "db4",
    level: int = 4,
) -> dict[str, NDArray[np.float64]]:
    """
    Compute wavelet-based features from price series.
    
    Generates multi-scale features for training:
    - denoised: Wavelet-denoised price series
    - trend: Low-frequency approximation
    - energy_1 to energy_N: Energy at each detail level
    - volatility_N: Rolling std of each detail component
    
    Args:
        prices: Price array
        wavelet: Wavelet name
        level: Decomposition level
        
    Returns:
        Dictionary of feature arrays
    """
    if not HAS_PYWT:
        # Fallback: return simple smoothing
        return {
            "denoised": prices.copy(),
            "trend": prices.copy(),
        }
    
    n = len(prices)
    denoiser = WaveletDenoiser(wavelet=wavelet, level=level)
    
    # Denoise
    denoised = denoiser.denoise(prices)
    
    # Decompose
    components = denoiser.decompose(prices, return_dict=True)
    
    features = {
        "denoised": denoised,
        "trend": components["approx"],
    }
    
    # Energy features (squared magnitude of details)
    for key, detail in components.items():
        if key.startswith("detail_"):
            level_num = key.split("_")[1]
            # Rolling energy (variance over window)
            energy = detail ** 2
            features[f"energy_{level_num}"] = energy
            
            # Rolling volatility of detail
            window = min(20, n // 10)
            if window > 1:
                vol = np.zeros(n)
                for i in range(window, n):
                    vol[i] = np.std(detail[i-window:i])
                vol[:window] = vol[window]  # Pad start
                features[f"volatility_{level_num}"] = vol
    
    return features
