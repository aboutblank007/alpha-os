console.log('AlphaLink Content Script Loaded (v6 - DEBUG POLLING)');

// Helper to parse clean number strings like "4,048.99" or "-1.80"
function parseNumber(str) {
    if (!str) return 0;
    // Remove currency codes, commas, % and whitespace
    const clean = str.replace(/[A-Z]{3}|%|,|\s/g, '');
    return parseFloat(clean) || 0;
}

// Function to scan the positions table
function scanPositions() {
    const positions = [];
    
    // Strategy 1: Standard Table Rows
    let rows = Array.from(document.querySelectorAll('tbody tr'));
    
    // Strategy 2: Div-based rows (sometimes TradingView uses divs with role="row")
    if (rows.length === 0) {
        rows = Array.from(document.querySelectorAll('div[role="row"]'));
    }

    // Strategy 3: Find by cell and go up to row
    if (rows.length === 0) {
        const cell = document.querySelector('td[data-label="商品代码"], div[data-label="商品代码"]');
        if (cell) {
             // Find closest row-like parent
             const row = cell.closest('tr') || cell.closest('div[role="row"]');
             if (row && row.parentElement) {
                 rows = Array.from(row.parentElement.children);
             }
        }
    }

    // Debug: Log scanning stats
    // console.log(`AlphaLink Scan: Found ${rows.length} potential rows`);

    rows.forEach((row, index) => {
        // Check if this is a valid data row by looking for specific data-labels
        // Support both TD and DIV cells
        const symbolCell = row.querySelector('[data-label="商品代码"]');
        const sideCell = row.querySelector('[data-label="买/卖"]');
        const qtyCell = row.querySelector('[data-label="数量"]');
        const priceCell = row.querySelector('[data-label="成交均价"]');
        const pnlCell = row.querySelector('[data-label="未实现盈亏"]');

        // Aggressive Debug for first few rows
        if (index < 2) {
             // console.log(`Row ${index}:`, symbolCell ? symbolCell.innerText : 'No Symbol', sideCell ? sideCell.innerText : 'No Side');
        }

        if (symbolCell && sideCell && qtyCell) {
            // Extract text content
            const symbolRaw = symbolCell.innerText; 
            // Usually "BROKER:SYMBOL", e.g. "THINKMARKETS:XAUUSD". We want "XAUUSD".
            const symbol = symbolRaw.split(':').pop() || symbolRaw;

            const sideRaw = sideCell.innerText.trim(); // "做多" or "做空" or "Buy"
            let side = 'buy';
            if (sideRaw.includes('空') || sideRaw.toLowerCase().includes('sell') || sideRaw.toLowerCase().includes('short')) {
                side = 'sell';
            }

            const quantity = parseNumber(qtyCell.innerText);
            const entryPrice = parseNumber(priceCell ? priceCell.innerText : '0');
            const pnl = parseNumber(pnlCell ? pnlCell.innerText : '0');

            if (quantity > 0) {
                positions.push({
                    symbol,
                    side,
                    quantity,
                    entry_price: entryPrice,
                    pnl_net: pnl,
                    raw_id: `${symbol}-${side}-${entryPrice}` // Simple unique key for frontend dedup if needed
                });
            }
        }
    });

    return positions;
}

// Main polling loop
setInterval(() => {
    const positions = scanPositions();
    
    if (positions.length > 0) {
        chrome.runtime.sendMessage({ 
            type: 'SYNC_POSITIONS', 
            data: { 
                timestamp: Date.now(),
                positions: positions 
            } 
        });
    }

}, 2000); // Poll every 2 seconds
