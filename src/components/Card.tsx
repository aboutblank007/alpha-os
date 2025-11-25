import React from 'react';

export type CardProps = React.HTMLAttributes<HTMLDivElement>;

export function Card({ className, ...props }: CardProps) {
    return (
        <div
            className={`glass-panel rounded-2xl p-6 ${className}`}
            {...props}
        />
    );
}

export interface StatCardProps {
    label: string;
    value: string;
    trend?: number;
    icon?: React.ReactNode;
    subValue?: string;
    className?: string;
}

export function StatCard({ label, value, trend, icon, subValue, className }: StatCardProps) {
    const isPositive = trend ? trend > 0 : true;

    return (
        <div className={`glass-panel p-6 rounded-2xl relative overflow-hidden group hover:border-white/10 transition-colors ${className}`}>
            {/* Background Gradient Glow */}
            <div className="absolute top-0 right-0 w-32 h-32 bg-accent-primary/5 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2 group-hover:bg-accent-primary/10 transition-colors duration-500"></div>

            <div className="relative z-10 flex justify-between items-start">
                <div>
                    <p className="text-sm font-medium text-slate-400 mb-1">{label}</p>
                    <h3 className="text-3xl font-bold text-white tracking-tight">{value}</h3>

                    {(trend !== undefined || subValue) && (
                        <div className="flex items-center gap-2 mt-2">
                            {trend !== undefined && (
                                <span className={`text-xs font-bold px-1.5 py-0.5 rounded ${isPositive ? 'text-accent-success bg-accent-success/10' : 'text-accent-danger bg-accent-danger/10'}`}>
                                    {isPositive ? '+' : ''}{trend}%
                                </span>
                            )}
                            {subValue && (
                                <span className="text-xs text-slate-500">{subValue}</span>
                            )}
                        </div>
                    )}
                </div>

                <div className={`p-3 rounded-xl ${isPositive ? 'bg-accent-primary/10 text-accent-primary' : 'bg-accent-danger/10 text-accent-danger'} ring-1 ring-white/5 group-hover:scale-110 transition-transform duration-300`}>
                    {icon}
                </div>
            </div>
        </div>
    );
}
