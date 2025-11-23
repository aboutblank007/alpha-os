# Trading Bridge System Walkthrough

I have implemented the **Cross-Platform Trading Bridge System**. This system allows your Mac-based AlphaOS to control MetaTrader 5 running on an Ubuntu server.

## Architecture
- **Server (Ubuntu)**: Runs Docker with Wine, MT5, and a Python FastAPI bridge.
- **Client (Mac)**: AlphaOS uses a TypeScript client to send commands to the Python API.
- **Communication**:
    - **Mac -> Python**: REST API (HTTP POST/GET).
    - **Python -> MT5**: ZeroMQ (TCP/IPC).

## 1. Server Deployment (Ubuntu)
Since Docker is not available locally, you must deploy these files to your Ubuntu server.

### Steps:
1.  Copy the `trading-bridge` directory to your server.
2.  Navigate to `trading-bridge/docker`.
3.  Build and start the container:
    ```bash
    docker-compose up -d --build
    ```
4.  **Initial Setup**:
    - Connect via VNC to `your-server-ip:5900` (no password by default).
    - You will see the Wine desktop.
    - Install MT5 (you may need to download the installer manually or map it via volumes).
    - **Important**: Copy `BridgeEA.mq5` to the MT5 `MQL5/Experts` folder.
    - Compile and attach the EA to a chart.
    - Ensure "Allow DLL imports" is checked in EA settings (required for ZeroMQ).

## 2. Client Integration (AlphaOS)
I have added the following files to your AlphaOS project:
- `src/lib/bridge-client.ts`: The API client.
- `src/components/TradeControl.tsx`: A UI component to test trading.

### Usage:
Import and use the `TradeControl` component in your page:
```tsx
import TradeControl from '@/components/TradeControl';

export default function Page() {
  return <TradeControl />;
}
```

## 3. Verification
Once the server is running:
1.  Update `src/lib/bridge-client.ts` with your Ubuntu server's IP address (default is `localhost:8000`).
2.  Open the AlphaOS page with `TradeControl`.
3.  Click "BUY" or "SELL".
4.  Check the VNC session to see if the trade is executed in MT5.

> [!NOTE]
> Since I could not run Docker locally, I have not verified the build process. You may need to adjust the `Dockerfile` (e.g., installing specific Wine dependencies) depending on your exact server environment.
