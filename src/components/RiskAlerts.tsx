import { AlertTriangle, RotateCcw } from 'lucide-react';
import { Button } from '@/components/ui/Button';

interface DashboardStats {
    totalPnL: number;
    winRate: number;
    profitFactor: number;
}

interface RiskAlertsProps {
    stats: DashboardStats;
    onResetLayout: () => void;
}

export function RiskAlerts({ stats, onResetLayout }: RiskAlertsProps) {
    const alerts: Array<{ level: 'warning' | 'danger'; text: string }> = [];
    
    if (stats.winRate < 45) alerts.push({ level: 'warning', text: '胜率低于 45%' });
    if (stats.profitFactor < 1.2 && stats.profitFactor > 0) alerts.push({ level: 'warning', text: '盈亏比低于 1.2' });
    if (stats.totalPnL < -500) alerts.push({ level: 'danger', text: '回撤超过阈值 (-$500)' });

    return (
        <div className="glass-panel rounded-xl p-6 h-full">
            <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                    <AlertTriangle className="text-accent-warning" size={20} />
                    <h3 className="text-lg font-semibold text-white">风险预警</h3>
                </div>
                <Button variant="ghost" size="sm" onClick={onResetLayout} className="text-slate-400 hover:text-white">
                    <RotateCcw size={14} className="mr-1"/> 重置
                </Button>
            </div>
            
            {alerts.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-32 text-slate-500">
                    <div className="p-3 rounded-full bg-accent-success/10 text-accent-success mb-2">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M20 6 9 17l-5-5"/></svg>
                    </div>
                    <p className="text-sm">系统正常</p>
                </div>
            ) : (
                <ul className="space-y-3">
                    {alerts.map((a, i) => (
                        <li key={i} className={`flex items-center justify-between rounded-lg px-4 py-3 border ${a.level === 'danger' ? 'bg-accent-danger/5 border-accent-danger/20 text-accent-danger' : 'bg-accent-warning/5 border-accent-warning/20 text-accent-warning'}`}>
                            <span className="text-sm font-medium">{a.text}</span>
                            <div className={`h-2 w-2 rounded-full ${a.level === 'danger' ? 'bg-accent-danger animate-pulse' : 'bg-accent-warning'}`}></div>
                        </li>
                    ))}
                </ul>
            )}
        </div>
    );
}

