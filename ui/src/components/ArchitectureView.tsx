import React, { useState } from 'react';
import { FLOWCHART_DIAGRAM, SEQUENCE_DIAGRAM, METRICS_DEFINITIONS } from '../architecture/diagrams';
import MermaidDiagram from './MermaidDiagram';
import { useAlphaOS } from '../context/AlphaOSContext';
import { Activity, Shield, Database, Lock, PlayCircle, BarChart2 } from 'lucide-react';

const ArchitectureView = () => {
    const [activeTab, setActiveTab] = useState('flowchart');
    const { status, positions, decision, runtimeSnapshot } = useAlphaOS();

    // Runtime State from SSOT (DB/WS)
    const rs = runtimeSnapshot;

    const runtimeState = {
        warmup: ((rs.warmup_progress || 0) * 100).toFixed(1) + '%',
        ticks: rs.ticks_total || 0,
        positions: rs.open_positions || 0,
        guardian: rs.guardian_halt ? 'HALTED' : 'ACTIVE',
        exitV21: rs.exit_v21_enabled ? 'ENABLED' : 'DISABLED',
        dbWrite: rs.db_snapshot_count || 0,
    };

    return (
        <div className="flex-1 flex overflow-hidden bg-dark">
            {/* 左侧：图表区域 */}
            <div className="flex-1 flex flex-col relative border-r border-panel">
                {/* Tabs */}
                <div className="flex items-center gap-1 p-2 border-b border-panel bg-panel/30">
                    <TabButton 
                        active={activeTab === 'flowchart'} 
                        onClick={() => setActiveTab('flowchart')}
                        icon={Activity}
                        label="System Flowchart (SSOT)"
                    />
                    <TabButton 
                        active={activeTab === 'sequence'} 
                        onClick={() => setActiveTab('sequence')}
                        icon={PlayCircle}
                        label="Sequence Diagram"
                    />
                </div>

                {/* Diagram Content */}
                <div className="flex-1 overflow-hidden bg-[#0d1117]"> {/* GitHub Dark Dimmed 背景色适配 Mermaid Dark */}
                    {activeTab === 'flowchart' && (
                        <MermaidDiagram chart={FLOWCHART_DIAGRAM} id="flowchart-view" />
                    )}
                    {activeTab === 'sequence' && (
                        <MermaidDiagram chart={SEQUENCE_DIAGRAM} id="sequence-view" />
                    )}
                </div>
            </div>

            {/* 右侧：运行时状态面板 */}
            <div className="w-80 flex flex-col bg-dark border-l border-panel">
                <div className="p-4 border-b border-panel">
                    <h3 className="text-sm font-bold text-dim uppercase tracking-wider flex items-center gap-2">
                        <Database size={14} />
                        Runtime State (DB SSOT)
                    </h3>
                </div>

                <div className="p-4 space-y-4 overflow-y-auto flex-1">
                    {/* 关键状态卡片 */}
                    <StatusCard 
                        label="Warmup Progress" 
                        value={runtimeState.warmup} 
                        icon={BarChart2}
                        status={status.warmup_complete ? 'success' : 'warning'}
                    />
                    <StatusCard 
                        label="Total Ticks" 
                        value={runtimeState.ticks} 
                        icon={Activity}
                        status="neutral"
                    />
                    <StatusCard 
                        label="Open Positions" 
                        value={runtimeState.positions} 
                        icon={Lock}
                        status={runtimeState.positions > 0 ? 'warning' : 'neutral'}
                    />
                    <StatusCard 
                        label="Model Guardian" 
                        value={runtimeState.guardian} 
                        icon={Shield}
                        status={status.model_halted ? 'danger' : 'success'}
                    />
                    <StatusCard 
                        label="Exit Policy v2.1" 
                        value={runtimeState.exitV21} 
                        icon={Shield}
                        status="success"
                    />
                    
                    {/* Metrics List (Reference) */}
                    <div className="mt-8">
                        <h4 className="text-xs font-bold text-dim uppercase mb-3">Observability Metrics</h4>
                        <div className="space-y-2">
                            {METRICS_DEFINITIONS.map((m, i) => (
                                <div key={i} className="text-xs p-2 bg-panel rounded border border-panel/50">
                                    <div className="font-mono text-secondary">{m.name}</div>
                                    <div className="flex justify-between mt-1 text-dim">
                                        <span>{m.type}</span>
                                        <span className="opacity-70">{m.desc}</span>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

const TabButton = ({ active, onClick, icon: Icon, label }) => (
    <button
        onClick={onClick}
        className={`flex items-center gap-2 px-3 py-1.5 rounded text-sm transition-colors ${
            active 
                ? 'bg-primary/20 text-primary border border-primary/30' 
                : 'text-dim hover:bg-panel hover:text-main'
        }`}
    >
        <Icon size={14} />
        {label}
    </button>
);

const StatusCard = ({ label, value, icon: Icon, status }) => {
    const statusColors = {
        success: 'text-success border-success/30 bg-success/5',
        warning: 'text-warning border-warning/30 bg-warning/5',
        danger: 'text-danger border-danger/30 bg-danger/5',
        neutral: 'text-main border-panel bg-panel/50',
    };
    
    return (
        <div className={`p-3 rounded border ${statusColors[status] || statusColors.neutral}`}>
            <div className="flex items-center justify-between mb-1">
                <span className="text-xs opacity-70 uppercase tracking-wider">{label}</span>
                <Icon size={14} className="opacity-80" />
            </div>
            <div className="text-xl font-mono font-bold">
                {value}
            </div>
        </div>
    );
};

export default ArchitectureView;
