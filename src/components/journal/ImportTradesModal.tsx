"use client";

import React, { useState, useRef } from 'react';
import { Modal } from '@/components/ui/Modal';
import { Upload, FileText, AlertCircle, CheckCircle2, X } from 'lucide-react';

interface ImportTradesModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSuccess?: () => void;
}

// Simple CSV parser that handles quoted strings containing commas
const parseCSVLine = (line: string) => {
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
    return values.map(v => v.replace(/^"|"$/g, '').trim()); // Remove surrounding quotes
};

export function ImportTradesModal({ open, onOpenChange, onSuccess }: ImportTradesModalProps) {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      if (selectedFile.type !== "text/csv" && !selectedFile.name.endsWith('.csv')) {
        setError("请上传有效的 CSV 文件。");
        return;
      }
      setFile(selectedFile);
      setError(null);
      parseCSV(selectedFile);
    }
  };

  const parseCSV = (file: File) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const text = e.target?.result;
        if (typeof text !== 'string') return;
        
        const lines = text.split('\n');
        if (lines.length < 1) return;
        
        // Handle empty file or just headers
        if (!lines[0].trim()) return;

        const headers = parseCSVLine(lines[0]).map(h => h.toLowerCase());
        
        const data = lines.slice(1)
          .filter(line => line && line.trim())
          .map(line => {
            const values = parseCSVLine(line);
            const entry: any = {};
            
              headers.forEach((header, index) => {
              let field = header;
              // 交易方向 -> side
              if (['type', 'direction', '方向', '类型', '交易方向', '买/卖'].includes(header)) field = 'side';
              // 产品代码 -> symbol
              if (['symbol', 'instrument', '产品代码', '品种', 'ticker', '商品代码'].includes(header)) field = 'symbol';
              // 数量 -> quantity
              if (['qty', 'size', 'amount', '数量', 'volume', '已成交数量'].includes(header)) field = 'quantity';
              // 持仓 -> entry_price
              if (['price', 'open price', 'entry', '开仓价', '价格', '持仓', '成交均价'].includes(header)) field = 'entry_price';
              // 平仓 -> exit_price
              if (['close price', 'exit', 'close', '平仓价', '平仓'].includes(header)) field = 'exit_price';
              // 净盈亏 -> pnl_net
              if (['pnl', 'profit', 'net profit', '盈亏', '净盈亏', '总盈亏'].includes(header)) field = 'pnl_net';
              // 佣金 -> commission
              if (['fee', 'comm', '手续费', '佣金'].includes(header)) field = 'commission';
              // 隔夜利息 -> swap
              if (['swap', '隔夜利息', 'overnight'].includes(header)) field = 'swap';
              // 开仓日期 -> date
              if (['time', 'date', 'created', '时间', '日期', '开仓日期', 'open time', '更新时间'].includes(header)) field = 'date';
              // 订单类型 -> order_type
              if (['order type', 'order_type', '类型'].includes(header) && !['方向', '交易方向'].includes(header)) field = 'order_type';
              // 状态 -> status
              if (['status', '状态'].includes(header)) field = 'status';
              // 订单编号 -> external_order_id
              if (['order id', 'order_id', 'external_id', 'trade_id', '订单编号', '交易编号', 'order number'].includes(header)) field = 'external_order_id';
              
              if (values[index] !== undefined && values[index] !== '') {
                  entry[field] = values[index];
              }
            });

            // Basic normalization
            if (entry.side) {
                entry.side = entry.side.toLowerCase();
                if (entry.side.includes('buy') || entry.side.includes('long') || entry.side.includes('买')) entry.side = 'buy';
                else if (entry.side.includes('sell') || entry.side.includes('short') || entry.side.includes('卖')) entry.side = 'sell';
            }
            
            // Ensure symbol exists
            if (!entry.symbol) entry.symbol = 'UNKNOWN';

            // Filter out non-market orders (止损、止盈等)
            if (entry.order_type && typeof entry.order_type === 'string') {
              const orderType = entry.order_type.toLowerCase();
              if (!orderType.includes('市价') && !orderType.includes('market')) {
                entry._skip = true;
              }
            }

            // Filter out cancelled orders
            if (entry.status && typeof entry.status === 'string') {
              const status = entry.status.toLowerCase();
              if (status.includes('已取消') || status.includes('cancelled') || status.includes('cancel')) {
                entry._skip = true;
              }
            }

            return entry;
          })
          .filter(entry => !entry._skip); // Filter out skipped entries

        setPreview(data.slice(0, 5));
      } catch (err) {
        console.error('CSV Parsing Error:', err);
        setError('解析 CSV 文件失败');
      }
    };
    reader.readAsText(file);
  };

  const handleImport = async () => {
    if (!file) return;
    
    setLoading(true);
    setError(null);

    try {
      const reader = new FileReader();
      reader.onload = async (e) => {
        try {
          const text = e.target?.result;
          if (typeof text !== 'string') {
            throw new Error('读取文件内容失败');
          }
          
          const lines = text.split('\n');
          if (lines.length < 1) throw new Error('文件为空');

          if (!lines[0].trim()) throw new Error('文件缺少表头');

          const headers = parseCSVLine(lines[0]).map(h => h.toLowerCase());
          
          const trades = lines.slice(1)
            .filter(line => line && line.trim())
            .map(line => {
              const values = parseCSVLine(line);
              const entry: any = {};
              
                  headers.forEach((header, index) => {
                     let field = header;
                     if (['type', 'direction', '方向', '类型', '交易方向', '买/卖'].includes(header)) field = 'side';
                     if (['symbol', 'instrument', '产品代码', '品种', 'ticker', '商品代码'].includes(header)) field = 'symbol';
                     if (['qty', 'size', 'amount', '数量', 'volume', '已成交数量'].includes(header)) field = 'quantity';
                     if (['price', 'open price', 'entry', '开仓价', '价格', '持仓', '成交均价'].includes(header)) field = 'entry_price';
                     if (['close price', 'exit', 'close', '平仓价', '平仓'].includes(header)) field = 'exit_price';
                     if (['pnl', 'profit', 'net profit', '盈亏', '净盈亏', '总盈亏'].includes(header)) field = 'pnl_net';
                     if (['fee', 'comm', '手续费', '佣金'].includes(header)) field = 'commission';
                     if (['swap', '隔夜利息', 'overnight'].includes(header)) field = 'swap';
                     if (['time', 'date', 'created', '时间', '日期', '开仓日期', 'open time', '更新时间'].includes(header)) field = 'date';
                     if (['order type', 'order_type', '类型'].includes(header) && !['方向', '交易方向'].includes(header)) field = 'order_type';
                     if (['status', '状态'].includes(header)) field = 'status';
                     if (['order id', 'order_id', 'external_id', 'trade_id', '订单编号', '交易编号', 'order number'].includes(header)) field = 'external_order_id';
                     
                     if (values[index] !== undefined && values[index] !== '') {
                        entry[field] = values[index];
                     }
                  });
                  
                  // 规范化交易方向
                  if (entry.side) {
                    entry.side = entry.side.toLowerCase();
                    if (entry.side.includes('buy') || entry.side.includes('long') || entry.side.includes('买')) entry.side = 'buy';
                    else if (entry.side.includes('sell') || entry.side.includes('short') || entry.side.includes('卖')) entry.side = 'sell';
                  }
                  
                  if (!entry.symbol) entry.symbol = 'UNKNOWN';
                  
                  // 规范化数量 - ThinkMarkets格式转换
                  // ThinkMarkets导出: USDJPY用基础单位(10000=0.1手), XAUUSD用手数
                  if (entry.quantity && entry.symbol) {
                    const qty = parseFloat(entry.quantity);
                    const symbol = entry.symbol.toUpperCase();
                    
                    // 如果是JPY货币对且数量>100，说明是ThinkMarkets格式(基础单位)
                    if (symbol.includes('JPY') && qty > 100) {
                      entry.quantity = (qty / 100000).toString(); // 转换为手数
                    }
                    // 如果是其他非JPY货币对且数量>100，也可能是基础单位
                    else if (!symbol.includes('XAU') && !symbol.includes('XAG') && !symbol.includes('BTC') && qty > 100) {
                      entry.quantity = (qty / 100000).toString(); // 转换为手数
                    }
                  }

              // Filter out non-market orders (止损、止盈等)
              if (entry.order_type && typeof entry.order_type === 'string') {
                const orderType = entry.order_type.toLowerCase();
                if (!orderType.includes('市价') && !orderType.includes('market')) {
                  entry._skip = true;
                }
              }

              // Filter out cancelled orders
              if (entry.status && typeof entry.status === 'string') {
                const status = entry.status.toLowerCase();
                if (status.includes('已取消') || status.includes('cancelled') || status.includes('cancel')) {
                  entry._skip = true;
                }
              }

              return entry;
            })
            .filter(entry => !entry._skip);

          if (trades.length === 0) {
             throw new Error('CSV 中未找到有效交易记录');
          }

          const response = await fetch('/api/trades/import', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ trades })
          });

          const result = await response.json();

          if (!response.ok) {
              throw new Error(result.error || '导入失败');
          }

          // 显示成功消息和跳过的订单信息
          let successMessage = `成功导入 ${result.count} 条交易记录`;
          if (result.skipped > 0) {
            successMessage += `\n跳过 ${result.skipped} 条重复记录`;
          }
          
          // 可以在这里使用 toast 或其他方式显示消息
          alert(successMessage);

          if (onSuccess) onSuccess();
          onOpenChange(false);
          setFile(null);
          setPreview([]);
        } catch (innerErr: any) {
           console.error('Import Processing Error:', innerErr);
           setError(innerErr.message || '处理 CSV 数据失败');
           setLoading(false);
        }
      };
      reader.readAsText(file);
    } catch (err: any) {
        setError(err.message);
        setLoading(false);
    }
  };

  return (
    <Modal
      open={open}
      onOpenChange={onOpenChange}
      title="导入交易记录"
      className="max-w-2xl"
    >
      <div className="space-y-6">
        {!file ? (
            <div 
                onClick={() => fileInputRef.current?.click()}
                className="border-2 border-dashed border-surface-border rounded-xl p-10 text-center cursor-pointer hover:border-accent-primary hover:bg-accent-primary/5 transition-all group"
            >
                <input 
                    type="file" 
                    ref={fileInputRef} 
                    onChange={handleFileChange} 
                    accept=".csv" 
                    className="hidden" 
                />
                <div className="w-12 h-12 rounded-full bg-surface-border flex items-center justify-center mx-auto mb-4 group-hover:bg-accent-primary/20 group-hover:text-accent-primary transition-colors">
                    <Upload size={24} />
                </div>
                <h3 className="text-lg font-medium text-white mb-2">点击上传 CSV</h3>
                <p className="text-slate-400 text-sm">
                    支持多种交易记录格式 · 自动过滤非市价订单 · 智能去重
                </p>
            </div>
        ) : (
            <div className="bg-surface-glass border border-surface-border rounded-xl p-4 flex items-center justify-between">
                <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-lg bg-accent-primary/20 flex items-center justify-center text-accent-primary">
                        <FileText size={20} />
                    </div>
                    <div>
                        <div className="text-white font-medium">{file.name}</div>
                        <div className="text-xs text-slate-400">{(file.size / 1024).toFixed(1)} KB</div>
                    </div>
                </div>
                <button 
                    onClick={() => { setFile(null); setPreview([]); }}
                    className="p-2 hover:bg-white/10 rounded-lg text-slate-400 hover:text-white transition-colors"
                >
                    <X size={18} />
                </button>
            </div>
        )}

        {error && (
            <div className="bg-accent-danger/10 border border-accent-danger/20 text-accent-danger text-sm p-3 rounded-lg flex items-center gap-2">
                <AlertCircle size={16} />
                {error}
            </div>
        )}

        {preview.length > 0 && (
            <div className="space-y-3">
                <h4 className="text-sm font-medium text-slate-300">预览（前 5 行）</h4>
                <div className="overflow-x-auto rounded-lg border border-surface-border">
                    <table className="w-full text-sm text-left">
                        <thead className="bg-white/5 text-slate-400">
                            <tr>
                                <th className="px-4 py-2 font-medium">品种</th>
                                <th className="px-4 py-2 font-medium">方向</th>
                                <th className="px-4 py-2 font-medium">数量</th>
                                <th className="px-4 py-2 font-medium">开仓价</th>
                                <th className="px-4 py-2 font-medium">盈亏</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-surface-border/50">
                            {preview.map((row, i) => (
                                <tr key={i} className="text-slate-300">
                                    <td className="px-4 py-2">{row.symbol}</td>
                                    <td className="px-4 py-2">
                                        <span className={row.side === 'buy' ? 'text-accent-success' : 'text-accent-danger'}>
                                            {row.side?.toUpperCase()}
                                        </span>
                                    </td>
                                    <td className="px-4 py-2">{row.quantity}</td>
                                    <td className="px-4 py-2">{row.entry_price}</td>
                                    <td className={`px-4 py-2 ${Number(row.pnl_net) >= 0 ? 'text-accent-success' : 'text-accent-danger'}`}>
                                        {row.pnl_net}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        )}

        <div className="flex justify-end gap-3 pt-4">
            <button 
                onClick={() => onOpenChange(false)}
                className="px-4 py-2 text-slate-400 hover:text-white transition-colors"
            >
                取消
            </button>
            <button
                onClick={handleImport}
                disabled={!file || loading}
                className="flex items-center gap-2 px-6 py-2 bg-accent-primary hover:bg-accent-primary/90 text-white rounded-xl transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
                {loading ? (
                    <>处理中...</>
                ) : (
                    <>
                        <CheckCircle2 size={18} />
                        导入交易
                    </>
                )}
            </button>
        </div>
      </div>
    </Modal>
  );
}
