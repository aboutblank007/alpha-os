// import { parse } from 'csv-parse/sync'; // Removed dependency

export interface CSVTradeRow {
    symbol: string;
    side: 'buy' | 'sell';
    type: string;
    quantity: number;
    filledQuantity: number;
    normalizedQuantity: number;
    price: number;
    status: string;
    time: Date;
    pnl: number | null;
    commission: number;
    swap: number;
    orderId: string;
    raw: any;
}

export interface ReconstructedTrade {
    symbol: string;
    side: 'buy' | 'sell';
    quantity: number;
    entryPrice: number;
    exitPrice: number;
    entryTime: Date;
    exitTime: Date;
    pnl: number;
    commission: number;
    swap: number;
    status: 'closed' | 'open';
    notes?: string;
    external_order_id: string;
    related_order_ids: string[];
}

export function normalizeVolume(symbol: string, volume: number): number {
    const sym = symbol.toUpperCase();
    if (sym.includes('XAU') || sym.includes('XAG')) {
        return volume / 100;
    }
    if (sym.includes('JPY') || sym.includes('EUR') || sym.includes('GBP') || sym.includes('AUD') || sym.includes('US30') || sym.includes('BTC')) {
        if (volume >= 1000) return volume / 100000;
    }
    return volume;
}

function estimateContractSize(symbol: string): number {
    const s = symbol.toUpperCase();
    // Metals
    if (s.includes('XAU') || s.includes('XAG')) return 100;
    // Forex (Standard Lot = 100,000 units)
    if (s.includes('JPY') || s.includes('EUR') || s.includes('GBP') || s.includes('AUD') || s.includes('NZD') || s.includes('CAD') || s.includes('CHF')) return 100000;
    // Crypto (Usually 1, sometimes 10 or 100, but ThinkMarkets 0.1 -> 0.1 seems to imply 1)
    if (s.includes('BTC') || s.includes('ETH')) return 1; 
    // Indices
    if (s.includes('US30') || s.includes('NAS100') || s.includes('SPX500') || s.includes('GER30') || s.includes('UK100')) return 1; 
    
    return 100000; // Default assumption
}

// Custom robust CSV parser
function parseCSV(text: string): any[] {
    const lines = text.split('\n');
    const result = [];
    const headers = parseCSVLine(lines[0]).map(h => h.trim());

    for (let i = 1; i < lines.length; i++) {
        const line = lines[i].trim();
        if (!line) continue;
        
        const values = parseCSVLine(line);
        if (values.length === 0) continue;

        const record: any = {};
        headers.forEach((header, index) => {
            record[header] = values[index];
        });
        result.push(record);
    }
    return result;
}

function parseCSVLine(line: string): string[] {
    const values = [];
    let current = '';
    let inQuotes = false;
    
    for (let i = 0; i < line.length; i++) {
        const char = line[i];
        if (char === '"') {
            inQuotes = !inQuotes;
        } else if (char === ',' && !inQuotes) {
            values.push(current.trim());
            current = '';
        } else {
            current += char;
        }
    }
    values.push(current.trim());
    return values.map(v => v.replace(/^"|"$/g, '').trim());
}

export function parseThinkMarketsCSV(csvContent: string): CSVTradeRow[] {
    const content = csvContent.replace(/^\uFEFF/, '');
    const records = parseCSV(content);

    return records.map((record: any) => {
        const symbol = record['商品代码'] || record['Symbol'];
        const sideStr = record['买/卖'] || record['Side'];
        const type = record['类型'] || record['Type'];
        const qty = parseFloat(record['数量'] || record['Quantity'] || '0');
        const filledQty = parseFloat(record['已成交数量'] || record['Filled Quantity'] || '0');
        const price = parseFloat(record['成交均价'] || record['Avg Price'] || '0');
        const status = record['状态'] || record['Status'];
        const timeStr = record['更新时间'] || record['Time'];
        const pnlStr = record['净盈亏'] || record['Net P/L'] || record['总盈亏'];
        const commStr = record['佣金'] || record['Commission'];
        const swapStr = record['隔夜利息'] || record['Swap'];
        const orderId = record['订单编号'] || record['Order ID'];

        let side: 'buy' | 'sell' = 'buy';
        if (sideStr?.includes('卖') || sideStr?.toLowerCase() === 'sell') side = 'sell';

        let pnl: number | null = null;
        if (pnlStr && pnlStr.trim() !== '') {
            pnl = parseFloat(pnlStr);
        }

        const rawQty = filledQty > 0 ? filledQty : qty;
        const normalizedQty = normalizeVolume(symbol, rawQty);

        // Validate essential fields
        if (!symbol || !timeStr) {
            return null;
        }

        return {
            symbol,
            side,
            type,
            quantity: rawQty,
            filledQuantity: filledQty,
            normalizedQuantity: normalizedQty,
            price,
            status,
            time: new Date(timeStr),
            pnl,
            commission: parseFloat(commStr || '0'),
            swap: parseFloat(swapStr || '0'),
            orderId,
            raw: record
        };
    }).filter((row: CSVTradeRow | null) => row !== null && (row.status === '已成交' || row.status === 'Filled'));
}

export function reconstructTrades(rows: CSVTradeRow[]): ReconstructedTrade[] {
    // Sort by time ascending
    rows.sort((a, b) => a.time.getTime() - b.time.getTime());

    const trades: ReconstructedTrade[] = [];
    const openPositions: Record<string, CSVTradeRow[]> = {};

    for (const row of rows) {
        // If row has PnL != 0, it is a CLOSING transaction
        if (row.pnl !== null && Math.abs(row.pnl) > 0.0001) {
            // Closing side is 'row.side'. Open side is opposite.
            const openSide = row.side === 'buy' ? 'sell' : 'buy';
            const queue = openPositions[row.symbol] || [];
            let remainingCloseQty = row.normalizedQuantity;
            
            let i = 0;
            while (i < queue.length && remainingCloseQty > 0.000001) {
                const openOrder = queue[i];
                
                if (openOrder.side === openSide) {
                    const matchQty = Math.min(remainingCloseQty, openOrder.normalizedQuantity);
                    
                    trades.push({
                        symbol: row.symbol,
                        side: openSide,
                        quantity: matchQty,
                        entryPrice: openOrder.price,
                        exitPrice: row.price,
                        entryTime: openOrder.time,
                        exitTime: row.time,
                        pnl: row.pnl * (matchQty / row.normalizedQuantity),
                        commission: row.commission * (matchQty / row.normalizedQuantity),
                        swap: row.swap * (matchQty / row.normalizedQuantity),
                        status: 'closed',
                        external_order_id: `${row.orderId}_${openOrder.orderId}`,
                        related_order_ids: [openOrder.orderId, row.orderId],
                        notes: `Matched ${openOrder.orderId} -> ${row.orderId}`
                    });
                    
                    remainingCloseQty -= matchQty;
                    openOrder.normalizedQuantity -= matchQty;
                    
                    if (openOrder.normalizedQuantity < 0.000001) {
                        queue.splice(i, 1);
                    } else {
                        // Partially consumed
                    }
                } else {
                    i++;
                }
            }
            
            if (remainingCloseQty > 0.000001) {
                // Unmatched / Partial History
                // Estimate Entry Price based on PnL
                const contractSize = estimateContractSize(row.symbol);
                // row.pnl is Net PnL. We need Gross PnL to estimate price diff.
                // Gross = Net - Commission - Swap
                // Note: Be careful with signs. Commission is usually negative.
                // If commission is -5, Net = Gross - 5 -> Gross = Net + 5.
                // But here row.commission is parsed as number. If it's negative in CSV, it's negative here.
                // PnL_Net = PnL_Gross + Comm + Swap.
                // PnL_Gross = PnL_Net - Comm - Swap.
                
                const proratedPnl = row.pnl * (remainingCloseQty / row.normalizedQuantity);
                const proratedComm = row.commission * (remainingCloseQty / row.normalizedQuantity);
                const proratedSwap = row.swap * (remainingCloseQty / row.normalizedQuantity);
                
                const grossPnl = proratedPnl - proratedComm - proratedSwap;
                
                // PnL = (Exit - Entry) * Vol * Size (Buy) -> Entry = Exit - PnL/(Vol*Size)
                // PnL = (Entry - Exit) * Vol * Size (Sell) -> Entry = Exit + PnL/(Vol*Size)
                
                // openSide is the side of the OPEN trade.
                // row.side is the side of the CLOSE trade.
                
                let estimatedEntryPrice = 0;
                const priceDelta = grossPnl / (remainingCloseQty * contractSize);
                
                if (openSide === 'buy') {
                    // Was Long, now Closing (Selling)
                    // Profit = (Exit - Entry) ...
                    // Entry = Exit - Delta
                    estimatedEntryPrice = row.price - priceDelta;
                } else {
                    // Was Short, now Closing (Buying)
                    // Profit = (Entry - Exit) ...
                    // Entry = Exit + Delta
                    estimatedEntryPrice = row.price + priceDelta;
                }
                
                // Sanity check: Price cannot be negative
                if (estimatedEntryPrice < 0) estimatedEntryPrice = 0;

                trades.push({
                    symbol: row.symbol,
                    side: openSide,
                    quantity: remainingCloseQty,
                    entryPrice: estimatedEntryPrice, // Use estimated price instead of 0
                    exitPrice: row.price,
                    entryTime: row.time, // Unknown entry time, use close time
                    exitTime: row.time,
                    pnl: proratedPnl,
                    commission: proratedComm,
                    swap: proratedSwap,
                    status: 'closed',
                    external_order_id: `${row.orderId}_partial`,
                    related_order_ids: [row.orderId],
                    notes: 'Partial/Unmatched History (Estimated Entry)'
                });
            }
            
        } else {
            if (!openPositions[row.symbol]) openPositions[row.symbol] = [];
            // Clone to avoid reference issues if needed, but here we just push
            openPositions[row.symbol].push({ ...row }); 
        }
    }
    
    return trades;
}
