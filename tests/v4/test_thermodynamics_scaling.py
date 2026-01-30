
import unittest
import numpy as np
from datetime import datetime, timedelta
from alphaos.data.event_bars.tick_imbalance import EventBar
from alphaos.v4.features.pipeline import FeaturePipelineV4, FeatureConfig
from alphaos.v4.schemas import FeatureSchema

class TestThermodynamicsScaling(unittest.TestCase):
    def setUp(self):
        self.config = FeatureConfig(thermo_window=50)
        self.schema = FeatureSchema.default()
        self.pipeline = FeaturePipelineV4(self.config, self.schema)
        
    def create_synthetic_bars(self, n=100, price_step=0.0001, dt_ms=1000):
        """
        Create synthetic bars with alternating price moves.
        Price step 1 pip (0.0001) on base 1.0 => log return approx 0.0001 (0.01%)
        Variance of returns in % should be approx (0.01)^2 = 1e-4
        Temp = Var / Mean_dt * 1e6
        Mean_dt = 1000
        Temp = 1e-4 / 1000 * 1e6 = 0.1
        """
        bars = []
        price = 1.0
        time = datetime.now()
        
        for i in range(n):
            direction = 1 if i % 2 == 0 else -1
            price += direction * price_step
            time += timedelta(milliseconds=dt_ms)
            
            bar = EventBar(
                time=time - timedelta(milliseconds=dt_ms),
                close_time=time,
                open=price - direction * price_step,
                high=max(price, price - direction * price_step),
                low=min(price, price - direction * price_step),
                close=price,
                tick_count=10,
                buy_count=5,
                sell_count=5,
                imbalance=0,
                spread_sum=0.1,
                duration_ms=dt_ms
            )
            bars.append(bar)
        return bars

    def test_market_temperature_scaling_batch(self):
        # Scenario: 1 pip move every 1 second
        # Expected Temp approx 0.1
        bars = self.create_synthetic_bars(n=100)
        
        result = self.pipeline.compute_batch(bars)
        
        temp_idx = self.schema.get_index("market_temperature")
        temps = result.features[:, temp_idx]
        
        # Check last few values (after warm-up)
        last_temp = temps[-1]
        print(f"Batch Market Temperature: {last_temp}")
        
        # Acceptable range: 0.05 to 0.2
        self.assertGreater(last_temp, 0.05, "Market temperature too low (likely not scaled)")
        self.assertLess(last_temp, 0.2, "Market temperature too high")
        
        ts_phase_idx = self.schema.get_index("ts_phase")
        phases = result.features[:, ts_phase_idx]
        last_phase = phases[-1]
        print(f"Batch TS Phase: {last_phase}")
        
        # Should not be FROZEN (0) if temp is around 0.1 (low threshold is exactly 0.1, might oscillate)
        # If we use strict > 0.1 logic, let's verify it's close.
        # Laminar is 1.
        
    def test_market_temperature_scaling_stream(self):
        self.pipeline.reset()
        bars = self.create_synthetic_bars(n=100)
        
        last_features = None
        for bar in bars:
            last_features = self.pipeline.update(bar)
            
        temp_idx = self.schema.get_index("market_temperature")
        last_temp = last_features[temp_idx]
        print(f"Stream Market Temperature: {last_temp}")
        
        self.assertGreater(last_temp, 0.05, "Market temperature too low (likely not scaled)")
        self.assertLess(last_temp, 0.2, "Market temperature too high")

if __name__ == '__main__':
    unittest.main()
