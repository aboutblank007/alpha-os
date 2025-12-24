I will implement the missing "Deep Analysis", "Quantum AI Telemetry", and "Journal Replay" features to complete the frontend system according to the design document.

### Phase 1: Core Visualization Components
1.  **Create `OrderBookHeatmap.tsx`**: Implement a Canvas-based Heatmap component to visualize Market Depth/Liquidity Walls (Section 4.1).
    *   *Ref:* `src/components/charts/OrderBookHeatmap.tsx`
2.  **Create `GradientNormChart.tsx`**: Implement a real-time chart for Quantum Gradient Norms to visualize "Barren Plateaus" (Section 5.1).
    *   *Ref:* `src/components/charts/GradientNormChart.tsx`

### Phase 2: Feature Integration
1.  **Enhance `dashboard/page.tsx`**:
    *   Integrate `OrderBookHeatmap` into a new "Deep Analysis" tab or toggleable view.
    *   Ensure `SystemVitals` (already existing) is prominently displayed.
2.  **Enhance `ai/page.tsx`**:
    *   Add the "Quantum Telemetry" section using `GradientNormChart`.
    *   Visualize the "Confidence Fan Chart" (Section 5.2).
3.  **Enhance `journal/page.tsx`**:
    *   Add a "Forensic Replay" (Time Travel) button to trade details.
    *   Implement a Modal that opens the Heatmap frozen at the trade's execution time (Section 6.2).

### Phase 3: Documentation & Verification
1.  **Update Documentation**:
    *   Update `docs/交易系统前端功能设计.MD` to mark sections as "Implemented" and add technical notes.
    *   Ensure all new code files contain the mandatory `[Ref: 交易系统前端功能设计.MD]` citation.
2.  **Verification**:
    *   Launch the dev server to verify the UI renders correctly.
    *   Check that "System Vitals" (Latency/Heartbeat) are reactive.
