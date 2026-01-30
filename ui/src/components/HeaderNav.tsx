import React from 'react';
import { NavLink } from 'react-router-dom';
import { Activity, Layers } from 'lucide-react';

const NavButton: React.FC<{ to: string; icon: React.ElementType; label: string }> = ({ to, icon: Icon, label }) => (
    <NavLink
        to={to}
        className={({ isActive }) => `
            flex items-center gap-1.5 px-3 py-1 rounded text-xs font-bold transition-all
            ${isActive 
                ? 'bg-primary/20 text-primary shadow-sm' 
                : 'text-dim hover:text-main'
            }
        `}
    >
        <Icon size={12} />
        {label}
    </NavLink>
);

export const HeaderNav: React.FC = () => {
    return (
        <div className="flex items-center bg-panel rounded p-0.5 border border-panel">
            <NavButton to="/live" icon={Activity} label="LIVE" />
            <NavButton to="/architecture" icon={Layers} label="ARCH" />
        </div>
    );
};
