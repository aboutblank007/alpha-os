import React from 'react';

export default function ReviewPage() {
    return (
        <div className="flex flex-col gap-6 p-6">
            <div className="flex items-center justify-between">
                <h1 className="text-2xl font-bold text-white">AI Trading Review</h1>
                <span className="px-3 py-1 text-xs font-medium text-amber-400 bg-amber-400/10 rounded-full border border-amber-400/20">
                    Coming Soon
                </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Placeholder: Daily Summary */}
                <div className="p-6 rounded-xl bg-white/5 border border-white/10 backdrop-blur-sm">
                    <h2 className="text-lg font-semibold text-slate-200 mb-4">Daily Summary</h2>
                    <div className="space-y-4">
                        <div className="h-4 bg-white/5 rounded w-3/4 animate-pulse"></div>
                        <div className="h-4 bg-white/5 rounded w-1/2 animate-pulse"></div>
                        <div className="h-4 bg-white/5 rounded w-5/6 animate-pulse"></div>
                    </div>
                </div>

                {/* Placeholder: Psychological Analysis */}
                <div className="p-6 rounded-xl bg-white/5 border border-white/10 backdrop-blur-sm">
                    <h2 className="text-lg font-semibold text-slate-200 mb-4">Psychological Analysis</h2>
                    <div className="flex items-center justify-center h-32 text-slate-500 text-sm">
                        AI analysis module not connected
                    </div>
                </div>
            </div>
        </div>
    );
}
