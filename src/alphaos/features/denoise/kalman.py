"""
Online Kalman Filter for Price Denoising (v4.0)

A simple 1D Kalman filter optimized for online tick-by-tick processing.

Model:
    State: x_t = true price (hidden)
    Observation: z_t = observed mid price (noisy)
    
    State transition: x_t = x_{t-1} + process_noise (random walk)
    Observation: z_t = x_t + measurement_noise
    
Output:
    - price_estimate: Denoised price (posterior mean)
    - residual: z_t - price_estimate (innovation/surprise)
    - kalman_gain: How much we trust new observation
    - uncertainty: Posterior variance (estimation uncertainty)

Features for Trading:
    - |residual| / price → noise level indicator (high = regime change or noise)
    - kalman_gain → learning rate (high = adapting to new info)
    - Use price_estimate for trend/momentum calculations (smoother)

Reference:
- Kalman, R.E. (1960). A New Approach to Linear Filtering and Prediction Problems
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from alphaos.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class KalmanState:
    """
    Complete state output from KalmanFilter.
    
    Attributes:
        price_estimate: Denoised price (posterior mean)
        residual: Innovation = observation - prediction
        kalman_gain: How much new observation affects estimate (0-1)
        uncertainty: Posterior variance (estimation uncertainty)
        residual_bps: Residual in basis points (relative to price)
    """
    price_estimate: float
    residual: float
    kalman_gain: float
    uncertainty: float
    residual_bps: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "price_estimate": round(self.price_estimate, 4),
            "residual": round(self.residual, 4),
            "kalman_gain": round(self.kalman_gain, 6),
            "uncertainty": round(self.uncertainty, 6),
            "residual_bps": round(self.residual_bps, 4),
        }


@dataclass
class KalmanFilter:
    """
    Online 1D Kalman Filter for price denoising.
    
    Implements the classic Kalman filter for a random walk model:
    - State evolves as random walk: x_t = x_{t-1} + ε (ε ~ N(0, Q))
    - Observations are noisy: z_t = x_t + η (η ~ N(0, R))
    
    Args:
        process_variance: Q - variance of state transition noise
            Higher = price can change more between observations
            Typical: 0.01 to 1.0 (in price units squared)
        
        measurement_variance: R - variance of measurement noise
            Higher = observations are noisier, trust prior more
            Typical: 0.1 to 10.0 (in price units squared)
        
        initial_uncertainty: P_0 - initial estimation variance
            Higher = less confident in initial estimate
    
    Usage:
        kf = KalmanFilter(process_variance=0.1, measurement_variance=1.0)
        
        for tick in tick_stream:
            state = kf.update(tick.mid)
            denoised_price = state.price_estimate
            noise_level = abs(state.residual_bps)
    """
    
    process_variance: float = 0.1     # Q: state transition noise
    measurement_variance: float = 1.0  # R: observation noise
    initial_uncertainty: float = 100.0 # P_0: initial variance
    
    # Internal state
    _state_estimate: float = field(default=0.0, init=False)  # x_hat (posterior mean)
    _uncertainty: float = field(init=False)                   # P (posterior variance)
    _initialized: bool = field(default=False, init=False)
    _tick_count: int = field(default=0, init=False)
    
    def __post_init__(self) -> None:
        """Initialize uncertainty."""
        self._uncertainty = self.initial_uncertainty
    
    def update(self, observation: float) -> KalmanState:
        """
        Process a new observation and update state estimate.
        
        Kalman Filter Algorithm:
        1. Predict:  x_pred = x_hat, P_pred = P + Q
        2. Update:   K = P_pred / (P_pred + R)
                     x_hat = x_pred + K * (z - x_pred)
                     P = (1 - K) * P_pred
        
        Args:
            observation: Observed price (z_t)
            
        Returns:
            KalmanState with denoised estimate and diagnostics
        """
        self._tick_count += 1
        
        if not self._initialized:
            # Initialize with first observation
            self._state_estimate = observation
            self._uncertainty = self.initial_uncertainty
            self._initialized = True
            
            return KalmanState(
                price_estimate=observation,
                residual=0.0,
                kalman_gain=1.0,  # First obs fully trusted
                uncertainty=self._uncertainty,
                residual_bps=0.0,
            )
        
        # ===== PREDICT STEP =====
        # State prediction (random walk: x_pred = x_hat)
        x_pred = self._state_estimate
        
        # Uncertainty prediction: P_pred = P + Q
        P_pred = self._uncertainty + self.process_variance
        
        # ===== UPDATE STEP =====
        # Innovation (residual)
        residual = observation - x_pred
        
        # Kalman gain: K = P_pred / (P_pred + R)
        K = P_pred / (P_pred + self.measurement_variance)
        
        # State update: x_hat = x_pred + K * residual
        self._state_estimate = x_pred + K * residual
        
        # Uncertainty update: P = (1 - K) * P_pred
        self._uncertainty = (1 - K) * P_pred
        
        # Compute residual in basis points (relative measure)
        residual_bps = (residual / observation * 10000) if observation > 0 else 0.0
        
        return KalmanState(
            price_estimate=self._state_estimate,
            residual=residual,
            kalman_gain=K,
            uncertainty=self._uncertainty,
            residual_bps=residual_bps,
        )
    
    @property
    def state_estimate(self) -> float:
        """Get current state estimate (denoised price)."""
        return self._state_estimate
    
    @property
    def uncertainty(self) -> float:
        """Get current estimation uncertainty."""
        return self._uncertainty
    
    @property
    def is_initialized(self) -> bool:
        """Check if filter has been initialized."""
        return self._initialized
    
    def reset(self) -> None:
        """Reset filter state."""
        self._state_estimate = 0.0
        self._uncertainty = self.initial_uncertainty
        self._initialized = False
        self._tick_count = 0
    
    def initialize_from_prices(self, prices: Sequence[float]) -> None:
        """
        Initialize filter by processing historical prices.
        
        Args:
            prices: Historical price sequence (oldest first)
        """
        self.reset()
        
        for price in prices:
            self.update(price)
        
        logger.debug(
            "Kalman filter initialized",
            n_prices=len(prices),
            final_estimate=round(self._state_estimate, 2),
            final_uncertainty=round(self._uncertainty, 6),
        )


def compute_kalman_batch(
    prices: NDArray[np.float64],
    process_variance: float = 0.1,
    measurement_variance: float = 1.0,
    initial_uncertainty: float = 100.0,
) -> dict[str, NDArray[np.float64]]:
    """
    Apply Kalman filter to batch price data.
    
    Vectorized implementation for historical data processing.
    
    Args:
        prices: Array of observed prices
        process_variance: Q parameter
        measurement_variance: R parameter
        initial_uncertainty: P_0 parameter
        
    Returns:
        Dictionary with arrays:
        - 'price_estimate': Denoised prices
        - 'residual': Innovations
        - 'kalman_gain': Gain values
        - 'uncertainty': Posterior variances
        - 'residual_bps': Relative residuals
    """
    n = len(prices)
    
    # Output arrays
    estimates = np.zeros(n, dtype=np.float64)
    residuals = np.zeros(n, dtype=np.float64)
    gains = np.zeros(n, dtype=np.float64)
    uncertainties = np.zeros(n, dtype=np.float64)
    
    # Initialize
    x_hat = prices[0]
    P = initial_uncertainty
    Q = process_variance
    R = measurement_variance
    
    estimates[0] = prices[0]
    residuals[0] = 0.0
    gains[0] = 1.0
    uncertainties[0] = P
    
    # Iterate
    for i in range(1, n):
        z = prices[i]
        
        # Predict
        x_pred = x_hat
        P_pred = P + Q
        
        # Update
        residual = z - x_pred
        K = P_pred / (P_pred + R)
        x_hat = x_pred + K * residual
        P = (1 - K) * P_pred
        
        # Store
        estimates[i] = x_hat
        residuals[i] = residual
        gains[i] = K
        uncertainties[i] = P
    
    # Compute relative residuals
    residual_bps = np.where(
        prices > 0,
        residuals / prices * 10000,
        0.0
    )
    
    return {
        "price_estimate": estimates,
        "residual": residuals,
        "kalman_gain": gains,
        "uncertainty": uncertainties,
        "residual_bps": residual_bps,
    }


@dataclass
class AdaptiveKalmanFilter:
    """
    Adaptive Kalman Filter with dynamic noise estimation.
    
    Automatically adjusts process and measurement variance based on
    observed innovation sequence, making it more robust to regime changes.
    
    Uses Innovation-based Adaptive Estimation (IAE):
    - Track running variance of innovations
    - Adjust R (measurement noise) when innovations are consistently large/small
    - Adjust Q (process noise) when filter is too slow/fast to adapt
    
    Args:
        initial_Q: Initial process variance
        initial_R: Initial measurement variance
        adaptation_rate: How fast to adapt variances (0-1)
        innovation_window: Window for computing innovation variance
    """
    
    initial_Q: float = 0.1
    initial_R: float = 1.0
    adaptation_rate: float = 0.1
    innovation_window: int = 20
    
    # Dynamic parameters
    _Q: float = field(init=False)
    _R: float = field(init=False)
    
    # Base filter
    _base_filter: KalmanFilter = field(init=False)
    
    # Innovation history for adaptation
    _innovations: list[float] = field(default_factory=list, init=False)
    _innovation_variance: float = field(default=1.0, init=False)
    
    def __post_init__(self) -> None:
        """Initialize components."""
        self._Q = self.initial_Q
        self._R = self.initial_R
        self._base_filter = KalmanFilter(
            process_variance=self._Q,
            measurement_variance=self._R,
        )
    
    def update(self, observation: float) -> KalmanState:
        """
        Process observation with adaptive noise estimation.
        
        Args:
            observation: Observed price
            
        Returns:
            KalmanState with current estimates
        """
        # Get base filter update
        state = self._base_filter.update(observation)
        
        # Track innovations
        if self._base_filter.is_initialized:
            self._innovations.append(state.residual)
            if len(self._innovations) > self.innovation_window:
                self._innovations.pop(0)
            
            # Adapt noise parameters
            self._adapt_parameters(state)
        
        return state
    
    def _adapt_parameters(self, state: KalmanState) -> None:
        """Adapt Q and R based on innovation statistics."""
        if len(self._innovations) < self.innovation_window // 2:
            return
        
        # Compute innovation variance
        innov_arr = np.array(self._innovations)
        innov_var = np.var(innov_arr)
        
        # Expected innovation variance = P_pred + R ≈ Q + R (steady state)
        expected_var = self._Q + self._R
        
        # If innovations are larger than expected, increase R (more noise)
        # If innovations are smaller, decrease R (less noise)
        ratio = innov_var / (expected_var + 1e-10)
        
        # Adapt R (measurement noise)
        if ratio > 1.5:
            # Observations are noisier than model predicts
            self._R = self._R * (1 + self.adaptation_rate * 0.5)
        elif ratio < 0.5:
            # Observations are cleaner than model predicts
            self._R = max(self._R * (1 - self.adaptation_rate * 0.5), 0.01)
        
        # Adapt Q (process noise) based on autocorrelation
        # If consecutive innovations have same sign, Q is too low
        if len(self._innovations) >= 3:
            recent = self._innovations[-3:]
            same_sign = all(r > 0 for r in recent) or all(r < 0 for r in recent)
            if same_sign:
                # Filter is too slow, increase Q
                self._Q = self._Q * (1 + self.adaptation_rate)
            else:
                # Decay Q back toward initial
                self._Q = self._Q * (1 - self.adaptation_rate * 0.1) + self.initial_Q * self.adaptation_rate * 0.1
        
        # Update base filter parameters
        self._base_filter.process_variance = self._Q
        self._base_filter.measurement_variance = self._R
        
        self._innovation_variance = innov_var
    
    @property
    def current_Q(self) -> float:
        """Current process variance estimate."""
        return self._Q
    
    @property
    def current_R(self) -> float:
        """Current measurement variance estimate."""
        return self._R
    
    @property
    def innovation_variance(self) -> float:
        """Recent innovation variance."""
        return self._innovation_variance
    
    def reset(self) -> None:
        """Reset all state."""
        self._Q = self.initial_Q
        self._R = self.initial_R
        self._base_filter.reset()
        self._base_filter.process_variance = self._Q
        self._base_filter.measurement_variance = self._R
        self._innovations.clear()
        self._innovation_variance = 1.0
