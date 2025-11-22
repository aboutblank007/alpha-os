Design a professional, tech‑savvy financial trading dashboard in a modern web UI. The layout should be a modular grid with a dark theme (deep blues, dark grays, black) and teal/green/red accent colors for data highlights. Use clean sans-serif fonts (e.g. Roboto or Inter), subtle glassmorphism or gradient backgrounds on cards, and simple geometric icons. Include smooth animations and neon‑style hover effects for interactive elements. The dashboard must prioritize clarity of key metrics and intuitive navigation.

Welcome Banner: At the top, have a banner showing the workspace status indicator (e.g. “Default Workspace”, “Analysis Workspace”, “Strategy Workspace”), a personalized greeting (“Welcome back, Trader One”), and quick action buttons (“New Analysis”, “Share”, “Export”). Also display a brief portfolio overview here, such as the number of active signals and a small win-rate trend sparkline.

Core Metric Cards: Below the banner, place four prominent statistic cards:

Net P&L: Shows total profit/loss and the daily change.

Win Rate: Displays the percentage of winning trades over the last 20 trades.

Max Drawdown: Shows the maximum drawdown percentage from closed trades.

Execution Efficiency: A composite score combining profit factor and win rate.
Use distinct accent colors or icons for each card. The cards should have subtle shadows or glass effects and concise labels.

Equity Curve Chart: Include a large interactive line chart for Net Asset Value over time (the equity curve). Allow toggling time ranges (1W, 1M, 3M, YTD, ALL). Overlay technical indicators on the chart, such as an EMA line and Bollinger Bands. Enable annotating the chart with strategy markers or notes. Use a filled area under the curve and clean axis labeling.

Symbol Performance: Add a section with an interactive pie chart showing trade volume distribution by symbol. Next to it, list the top 5 traded symbols in detail, each showing that symbol’s profit/loss, total number of trades, and win rate. Use contrasting colors for each slice of the pie chart and matching labels in the list.

Ongoing Orders Panel: Provide a live table or panel listing all current open orders. Include columns such as Order Time, Symbol, Side (Buy/Sell), Entry Price, Quantity, and Status. This should update in real time, and support sorting/filtering by symbol or status. Use a clean table design with clear typography.

Recent Trades / Trade History: Below, show a table of the most recent trades (e.g. last 8 trades) with columns: Time, Symbol, Direction (Buy/Sell), Price, Quantity, and P/L. Support filtering and a “View All” option to see full history. Use alternating row shading or subtle separators, and make it easy to sort by any column.

Trading Insights: Include a section or widget that provides insights or highlights from the trade history. For example, summarize the best/worst trades, average holding time, or other performance notes. This could be shown as a text panel or small charts/metrics. Use concise wording and bullet point style highlights.

Sentiment Analysis: Add a component that visualizes market or trader sentiment. This could be a gauge, sentiment score, or a small bar/line chart. Use neutral/teal for positive sentiment and red for negative. Label it clearly (e.g. “Market Sentiment: Bullish / Bearish”) and update it in real time.

Risk Alerts: On a side or top panel, display automatic alert indicators:

Show a warning icon or highlight if Win Rate < 45%.

Show a warning if Profit Factor < 1.2.

Show a danger alert (with red color and an optional sound notification) if Total P/L < –$500.
Use clear icons and color changes (yellow for warning, red for danger) so alerts stand out immediately.

Extra Features: Mention in the prompt that the layout is drag‑and‑drop customizable, and data updates in real time (via something like Supabase). Include options for exporting data to CSV and sharing the dashboard via a link.

Emphasize the overall style: minimalist but sophisticated, with plenty of whitespace and a high-tech feel. Use subtle gradients or frosted-glass effects on cards, sleek line icons, and friendly button styles. Ensure the most important data (key stats and charts) are front and center. The design should feel cutting-edge and professional, making it easy for a trader to get quick insights into current orders, historical performance, and risk at a glance.