import { cn } from "@/lib/utils";
import { ArrowUpRight, ArrowDownRight, Minus } from 'lucide-react';

interface CardProps {
    children: React.ReactNode;
    className?: string;
}

export function Card({ children, className }: CardProps) {
    return (
        <div className={cn(
            "glass-panel p-6 rounded-xl relative overflow-hidden",
            className
        )}>
            {children}
        </div>
    );
}

interface StatCardProps {
    label: string;
    value: string | number;
    trend?: number;
    icon?: React.ReactNode;
    subValue?: string;
}

export function StatCard({ label, value, trend, icon, subValue }: StatCardProps) {
    const isPositive = trend && trend > 0;
    const isNegative = trend && trend < 0;
    const isNeutral = trend === 0 || trend === undefined;

    return (
        <Card className="group hover:bg-white/[0.02] transition-colors duration-300">
            <div className="flex items-start justify-between">
                <div>
                    <p className="text-sm font-medium text-slate-500">{label}</p>
                    <div className="mt-2 flex items-baseline gap-2">
                        <h3 className="text-2xl font-bold text-white tracking-tight">{value}</h3>
                        {subValue && <span className="text-xs text-slate-500">{subValue}</span>}
                    </div>

                    <div className="mt-3 flex items-center gap-2">
                        <div className={cn(
                            "flex items-center gap-0.5 text-xs font-medium",
                            isPositive ? "text-accent-success" :
                                isNegative ? "text-accent-danger" :
                                    "text-slate-400"
                        )}>
                            {isPositive && <ArrowUpRight size={14} />}
                            {isNegative && <ArrowDownRight size={14} />}
                            {isNeutral && <Minus size={14} />}
                            {trend ? Math.abs(trend).toFixed(1) + '%' : '0.0%'}
                        </div>
                        <span className="text-xs text-slate-600">较上月</span>
                    </div>
                </div>

                {icon && (
                    <div className="p-2.5 rounded-lg bg-white/5 text-slate-400 group-hover:text-white transition-colors">
                        {icon}
                    </div>
                )}
            </div>
        </Card>
    );
}
