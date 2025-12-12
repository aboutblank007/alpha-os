import numpy as np
from collections import deque

class StreamingARIMA_GARCH:
    """
    Lightweight Online ARIMA-GARCH Estimator
    Structure:
      - Mean Model: ARMA(1,1) -> r_t = c + phi*r_{t-1} + theta*e_{t-1} + e_t
      - Volatility Model: GARCH(1,1) Proxy via EWMA of squared residuals
    """
    def __init__(self, window_size=500):
        self.window_size = window_size
        self.returns = deque(maxlen=window_size)
        self.residuals = deque(maxlen=window_size)
        self.variances = deque(maxlen=window_size)
        
        # Initial Parameters (can be updated online)
        self.phi = 0.0   # AR(1)
        self.theta = 0.0 # MA(1)
        self.mu = 0.0    # Mean
        
        # GARCH(1,1) parameters (Fixed for standard EWMA-like behavior or adaptive)
        # sigma^2_t = omega + alpha*e^2_{t-1} + beta*sigma^2_{t-1}
        self.omega = 0.000001
        self.alpha = 0.05
        self.beta = 0.94
        
        self.last_residual = 0.0
        self.last_variance = 0.0001
        self.last_return = 0.0

    def update(self, price_close):
        """
        Update model with new closing price.
        Calculates log return and updates ARMA/GARCH states.
        """
        if self.last_return == 0.0 and len(self.returns) == 0:
            # First tick, just store price? No, we need returns.
            # Assume previous price was close to current or wait for second tick.
            pass 
            
        # We need price history to calc return.
        # If caller passes return directly, that's easier.
        # Let's assume caller passes log_return or we maintain state.
        # For flexibility, let's take log_return directly.
        pass

    def update_return(self, log_return):
        self.returns.append(log_return)
        
        # 1. Mean Model (ARMA 1,1) Prediction for *current* step was:
        # pred_r_t = mu + phi*r_{t-1} + theta*e_{t-1}
        pred_r = self.mu + self.phi * self.last_return + self.theta * self.last_residual
        
        # Actual Residual
        residual = log_return - pred_r
        self.residuals.append(residual)
        
        # 2. Volatility Model (GARCH 1,1)
        # sigma^2_t = omega + alpha*e^2_{t-1} + beta*sigma^2_{t-1}
        # Note: We use the residual from previous step for GARCH update of *current* variance?
        # Standard GARCH: sigma^2_t is conditional variance for t based on info up to t-1.
        # So we update sigma^2_next based on current residual.
        
        current_variance = self.omega + self.alpha * (self.last_residual**2) + self.beta * self.last_variance
        self.variances.append(current_variance)
        
        # 3. Online Parameter Update (Simple SGD for ARMA)
        # Minimize error^2 = (r_t - pred_r)^2
        # d(error^2)/d(phi) = -2*error * r_{t-1}
        lr = 0.01
        self.phi += lr * residual * self.last_return
        self.theta += lr * residual * self.last_residual
        self.mu += lr * residual
        
        # Constrain parameters for stationarity
        self.phi = np.clip(self.phi, -0.9, 0.9)
        self.theta = np.clip(self.theta, -0.9, 0.9)
        
        # Update State
        self.last_return = log_return
        self.last_residual = residual
        self.last_variance = current_variance
        
        return residual, current_variance

    def predict_next(self):
        """
        Predict next return and volatility
        """
        # E[r_{t+1}]
        pred_return = self.mu + self.phi * self.last_return + self.theta * self.last_residual
        
        # E[sigma^2_{t+1}]
        pred_variance = self.omega + self.alpha * (self.last_residual**2) + self.beta * self.last_variance
        
        return pred_return, np.sqrt(pred_variance)
        
    def get_volatility(self):
        return np.sqrt(self.last_variance)

if __name__ == "__main__":
    # Test
    model = StreamingARIMA_GARCH()
    
    # Simulate random walk with volatility
    returns = np.random.normal(0, 0.001, 1000)
    
    for r in returns:
        model.update_return(r)
        
    pred_r, pred_vol = model.predict_next()
    print(f"Next Return: {pred_r:.6f}, Volatility: {pred_vol:.6f}")

