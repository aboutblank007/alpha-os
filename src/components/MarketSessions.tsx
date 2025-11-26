"use client";

import React, { useEffect, useState } from 'react';
import { Clock, Globe } from 'lucide-react';

interface Session {
    name: string;
    city: string;
    timezone: string;
    start: number; // Hour in UTC
    end: number;   // Hour in UTC
}

const sessions: Session[] = [
    { name: 'Sydney', city: '悉尼', timezone: 'Australia/Sydney', start: 22, end: 7 },
    { name: 'Tokyo', city: '东京', timezone: 'Asia/Tokyo', start: 0, end: 9 },
    { name: 'London', city: '伦敦', timezone: 'Europe/London', start: 8, end: 17 },
    { name: 'New York', city: '纽约', timezone: 'America/New_York', start: 13, end: 22 },
];

export function MarketSessions() {
    const [now, setNow] = useState(new Date());

    useEffect(() => {
        const timer = setInterval(() => setNow(new Date()), 60000); // Update every minute
        return () => clearInterval(timer);
    }, []);

    const currentHour = now.getUTCHours() + now.getUTCMinutes() / 60;

    const getStatus = (start: number, end: number) => {
        // Handle crossing midnight
        const isOvernight = end < start;
        const isOpen = isOvernight
            ? (currentHour >= start || currentHour < end)
            : (currentHour >= start && currentHour < end);
        
        return isOpen ? 'open' : 'closed';
    };

    const getProgress = (start: number, end: number) => {
        const isOvernight = end < start;
        const duration = isOvernight ? (24 - start + end) : (end - start);
        let elapsed = 0;

        if (isOvernight) {
            if (currentHour >= start) elapsed = currentHour - start;
            else elapsed = (24 - start) + currentHour;
        } else {
            elapsed = currentHour - start;
        }

        if (elapsed < 0) return 0;
        if (elapsed > duration) return 100;
        return (elapsed / duration) * 100;
    };

    return (
        <div className="glass-panel p-6 rounded-xl h-full flex flex-col relative overflow-hidden">
            {/* Header */}
            <div className="flex items-center justify-between mb-6 relative z-10">
                <div className="flex items-center gap-2">
                    <div className="p-2 rounded-lg bg-indigo-500/10 text-indigo-400">
                        <Globe size={18} />
                    </div>
                    <h3 className="font-medium text-slate-200">市场时段</h3>
                </div>
                <div className="flex items-center gap-1.5 text-xs font-mono text-slate-500 bg-white/5 px-2 py-1 rounded-md">
                    <Clock size={12} />
                    <span>UTC {now.toISOString().slice(11, 16)}</span>
                </div>
            </div>

            {/* Sessions List */}
            <div className="flex-1 flex flex-col justify-between gap-2 relative z-10">
                {sessions.map((session) => {
                    const status = getStatus(session.start, session.end);
                    const progress = getProgress(session.start, session.end);
                    const isOpen = status === 'open';

                    return (
                        <div key={session.name} className="space-y-1.5">
                            <div className="flex items-center justify-between text-sm">
                                <span className={`font-medium ${isOpen ? 'text-white' : 'text-slate-500'}`}>
                                    {session.city}
                                </span>
                                <span className={`text-xs px-1.5 py-0.5 rounded flex items-center gap-1.5 ${
                                    isOpen 
                                        ? 'bg-accent-success/10 text-accent-success border border-accent-success/20' 
                                        : 'text-slate-600'
                                }`}>
                                    {isOpen && <span className="relative flex h-1.5 w-1.5">
                                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-accent-success opacity-75"></span>
                                        <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-accent-success"></span>
                                    </span>}
                                    {isOpen ? 'Open' : 'Closed'}
                                </span>
                            </div>
                            
                            {/* Progress Bar */}
                            <div className="h-1.5 w-full bg-white/5 rounded-full overflow-hidden">
                                <div 
                                    className={`h-full rounded-full transition-all duration-1000 ${
                                        isOpen ? 'bg-indigo-500 shadow-[0_0_10px_rgba(99,102,241,0.5)]' : 'bg-slate-700'
                                    }`}
                                    style={{ width: isOpen ? `${progress}%` : '0%' }}
                                />
                            </div>
                        </div>
                    );
                })}
            </div>

            {/* Decorative Background */}
            <div className="absolute -bottom-10 -right-10 w-32 h-32 bg-indigo-500/10 rounded-full blur-3xl pointer-events-none"></div>
        </div>
    );
}

