# AlphaLink Browser Extension

This extension captures trade notifications from TradingView and sends them to your AlphaOS.

## Installation

1. Open Chrome and go to `chrome://extensions/`.
2. Enable **Developer mode** in the top right corner.
3. Click **Load unpacked**.
4. Select this directory (`alpha-link-extension`).

## Configuration

1. Click the AlphaLink icon in your browser toolbar.
2. Ensure the API URL is correct (default: `http://localhost:3000/api/trades`).
3. Click **Save**.

## Usage

1. Open TradingView and go to your chart.
2. Make a trade (or use Paper Trading).
3. When the "Order Filled" notification appears, AlphaLink will capture it.
4. Check your AlphaOS Dashboard to see the new trade.

## Troubleshooting

- Open the browser console (F12) on the TradingView tab to see logs from `content.js`.
- Open the extension's background page console (from `chrome://extensions/`) to see logs from `background.js`.
