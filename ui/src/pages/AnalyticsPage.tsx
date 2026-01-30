
import React, { useEffect, useState } from 'react';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';
import { Activity, Clock } from 'lucide-react';

interface RuntimeData {
    timestamp: number;
    temperature: number;
    entropy: number;
    market_phase: string;
}

const AnalyticsPage: React.FC = () => {
    const [data, setData] = useState<RuntimeData[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const controller = new AbortController();

        const toEpochMs = (ts: any) => {
            if (typeof ts === 'number') {
                return ts > 1e12 ? ts : ts * 1000;
            }
            const parsed = Date.parse(ts);
            return Number.isNaN(parsed) ? Date.now() : parsed;
        };

        const fetchData = async () => {
            try {
                const response = await fetch('/api/history/runtime?limit=200', { signal: controller.signal });
                if (!response.ok) throw new Error('Failed to fetch data');
                const jsonData = await response.json();

                // Process data for charts
                const processed = jsonData.reverse().map((d: any) => ({
                    ...d,
                    time: new Date(toEpochMs(d.timestamp)).toLocaleTimeString(),
                }));

                setData(processed);
                setError(null);
            } catch (err: any) {
                if (err?.name === 'AbortError') return;
                console.error(err);
                setError('Failed to load historical data. Is the backend API running on port 8000?');
            } finally {
                setLoading(false);
            }
        };

        fetchData();
        const interval = setInterval(fetchData, 5000); // Refresh every 5s
        return () => {
            clearInterval(interval);
            controller.abort();
        };
    }, []);

    if (loading) return <div className="p-8 text-slate-500">Loading analytics...</div>;
    if (error) return <div className="p-8 text-red-400 border border-red-500/20 bg-red-500/5 rounded m-4">{error}</div>;

    return (
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
            <header className="flex justify-between items-center mb-6">
                <div>
                    <h1 className="text-2xl font-bold text-white tracking-tight flex items-center gap-2">
                        <Activity className="text-primary" />
                        Market Thermodynamics
                    </h1>
                    <p className="text-slate-400 text-sm mt-1">Historical analysis of Temperature & Entropy</p>
                </div>
                <div className="flex items-center gap-2 text-xs text-slate-500 px-3 py-1 bg-slate-900 rounded-full border border-slate-800">
                    <Clock size={12} />
                    Live Sync (5s)
                </div>
            </header>

            {/* Temperature Chart */}
            <div className="glass-panel p-6 rounded-xl">
                <h3 className="text-sm font-medium text-slate-400 mb-4">Temperature Trend</h3>
                <div className="h-[300px] min-h-[240px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={data}>
                            <defs>
                                <linearGradient id="colorTemp" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#38bdf8" stopOpacity={0.3} />
                                    <stop offset="95%" stopColor="#38bdf8" stopOpacity={0} />
                                </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                            <XAxis dataKey="time" stroke="#64748b" fontSize={10} tickMargin={10} />
                            <YAxis stroke="#64748b" fontSize={10} domain={[0, 1]} />
                            <Tooltip
                                contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', color: '#f8fafc' }}
                                itemStyle={{ color: '#38bdf8' }}
                            />
                            <Area
                                type="monotone"
                                dataKey="temperature"
                                stroke="#38bdf8"
                                strokeWidth={2}
                                fillOpacity={1}
                                fill="url(#colorTemp)"
                            />
                        </AreaChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {/* Entropy Chart */}
            <div className="glass-panel p-6 rounded-xl">
                <h3 className="text-sm font-medium text-slate-400 mb-4">Entropy Trend</h3>
                <div className="h-[300px] min-h-[240px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={data}>
                            <defs>
                                <linearGradient id="colorEntropy" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#818cf8" stopOpacity={0.3} />
                                    <stop offset="95%" stopColor="#818cf8" stopOpacity={0} />
                                </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                            <XAxis dataKey="time" stroke="#64748b" fontSize={10} tickMargin={10} />
                            <YAxis stroke="#64748b" fontSize={10} domain={[0, 1]} />
                            <Tooltip
                                contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', color: '#f8fafc' }}
                                itemStyle={{ color: '#818cf8' }}
                            />
                            <Area
                                type="monotone"
                                dataKey="entropy"
                                stroke="#818cf8"
                                strokeWidth={2}
                                fillOpacity={1}
                                fill="url(#colorEntropy)"
                            />
                        </AreaChart>
                    </ResponsiveContainer>
                </div>
            </div>
        </div>
    );
};

export default AnalyticsPage;
