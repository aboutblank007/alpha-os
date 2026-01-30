
import React from 'react';
import { Outlet, NavLink, useLocation } from 'react-router-dom';
import { LayoutDashboard, BarChart2, Layers, Cpu } from 'lucide-react';
import { AlphaOSProvider } from '../../context/AlphaOSContext';

const SidebarItem: React.FC<{ to: string; icon: React.ElementType; label: string }> = ({ to, icon: Icon, label }) => (
    <NavLink
        to={to}
        className={({ isActive }) =>
            `flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200 group ${isActive
                ? 'bg-primary/10 text-primary border border-primary/20 shadow-[0_0_15px_rgba(56,189,248,0.15)]'
                : 'text-slate-400 hover:text-slate-100 hover:bg-slate-800/50'
            }`
        }
    >
        <Icon size={18} />
        <span className="font-medium text-sm">{label}</span>
    </NavLink>
);

const DashboardLayout: React.FC = () => {
    const location = useLocation();

    return (
        <AlphaOSProvider>
            <div className="flex h-screen w-screen bg-bg-dark text-text-main overflow-hidden font-sans">
                {/* Sidebar */}
                <div className="w-64 border-r border-border-highlight bg-bg-panel/50 backdrop-blur-xl flex flex-col">
                    {/* Logo Area */}
                    <div className="h-16 flex items-center px-6 border-b border-border-highlight/50">
                        <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-primary to-secondary flex items-center justify-center mr-3 shadow-[0_0_15px_rgba(56,189,248,0.4)]">
                            <Cpu size={18} className="text-white" />
                        </div>
                        <div>
                            <h1 className="font-bold text-lg tracking-wider text-white">AlphaOS</h1>
                            <span className="text-[10px] text-primary tracking-[0.2em] font-mono block -mt-1 opacity-80">INTELLIGENCE</span>
                        </div>
                    </div>

                    {/* Navigation */}
                    <nav className="flex-1 p-4 space-y-2">
                        <div className="text-xs font-bold text-slate-500 uppercase tracking-wider px-4 mb-2 mt-2">Platform</div>
                        <SidebarItem to="/live" icon={LayoutDashboard} label="Live Operations" />
                        <SidebarItem to="/analytics" icon={BarChart2} label="Thermodynamics" />

                        <div className="text-xs font-bold text-slate-500 uppercase tracking-wider px-4 mb-2 mt-6">System</div>
                        <SidebarItem to="/architecture" icon={Layers} label="Architecture" />
                    </nav>

                    {/* Footer / User */}
                    <div className="p-4 border-t border-border-highlight/50 bg-slate-900/30">
                        <div className="flex items-center gap-3">
                            <div className="w-8 h-8 rounded bg-slate-800 border border-slate-700 flex items-center justify-center text-slate-400">
                                <span className="font-mono text-xs">V4</span>
                            </div>
                            <div className="flex-1 min-w-0">
                                <div className="text-xs font-medium text-slate-300 truncate">v4.0.0-PROD</div>
                                <div className="text-[10px] text-emerald-400 flex items-center gap-1">
                                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
                                    System Online
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Main Content */}
                <div className="flex-1 flex flex-col min-w-0 bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-indigo-900/20 via-slate-900/0 to-slate-900/0">
                    <Outlet />
                </div>
            </div>
        </AlphaOSProvider>
    );
};

export default DashboardLayout;
