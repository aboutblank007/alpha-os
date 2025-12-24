I will upgrade the frontend to "Live Mode" by connecting it to the real-time Python Gateway and providing a unified startup script.

### Phase 1: Frontend Data Binding (Switch to WebSocket)

1. **Update** **`dashboard/page.tsx`**:

   * Replace the legacy polling `fetch('/api/ai/latest')` with `useQuantumStore` hook.

   * Bind the AI Signal display to real-time telemetry data (`telemetry.ai_score`, `telemetry.regime`).

   * Bind the "System Vitals" to the `useBridgeStatus` and `useQuantumSocket` states.
2. **Update** **`SystemVitals.tsx`**:

   * Ensure it consumes `ticksPerSecond` and `latency` from the WebSocket store.

### Phase 2: Python Gateway Enhancement (Robustness & Mock)

1. **Enhance** **`api_gateway.py`**:

   * Add a `--mock` flag to generate synthetic High-Frequency Tick data and Quantum Telemetry if the ZMQ backend is offline.

   * This ensures the frontend "comes alive" immediately even without the full MT5/Python backend stack running.

### Phase 3: Infrastructure & Startup

1. **Create** **`dev.sh`**:

   * A shell script to launch both the Next.js frontend and the Python Gateway (in Mock mode by default for safety, or connected mode).

   * Installs Python dependencies if missing (`fastapi`, `uvicorn`, `zmq`, `loguru`).
2. **Update** **`next.config.ts`**:

   * Add API rewrites to proxy `/api/py/*` -> `http://127.0.0.1:8000/*` for direct access to the gateway status.

### Verification

1. Run `dev.sh`.
2. Open Dashboard.
3. Verify that the "System Status" indicator turns Green/Connected.
4. Verify that the Heatmap and Charts start updating with data (Mock or Real).

