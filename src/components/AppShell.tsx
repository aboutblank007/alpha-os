"use client";

import React, { useState, useEffect } from 'react';
import {
  LayoutDashboard,
  BookOpen,
  BarChart2,
  Settings,
  Menu,
  ChevronLeft,
  Bell,
  Search,
  Command
} from 'lucide-react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';

export function AppShell({ children }: { children: React.ReactNode }) {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [scrolled, setScrolled] = useState(false);
  const [netAsset, setNetAsset] = useState<number | null>(null);
  const [totalPnl, setTotalPnl] = useState<number>(0);
  const pathname = usePathname();

  // Handle scroll effect for header
  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // 加载账户余额
  useEffect(() => {
    loadBalance();

    // 每30秒刷新一次
    const interval = setInterval(loadBalance, 30000);
    return () => clearInterval(interval);
  }, []);

  const loadBalance = async () => {
    try {
      const response = await fetch('/api/account/balance');
      if (response.ok) {
        const data = await response.json();
        setNetAsset(data.net_asset);
        setTotalPnl(data.total_pnl);
      }
    } catch (error) {
      console.error('加载账户余额失败:', error);
    }
  };

  // Handle screen resize
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    const checkMobile = () => {
      const mobile = window.innerWidth < 768;
      setIsMobile(mobile);
      if (mobile) setIsSidebarOpen(false);
      else setIsSidebarOpen(true);
    };

    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  const navItems = [
    { icon: LayoutDashboard, label: '仪表板', href: '/dashboard' },
    { icon: BookOpen, label: '交易日志', href: '/journal' },
    { icon: BarChart2, label: '数据分析', href: '/analytics' },
    { icon: Settings, label: '设置', href: '/settings' },
  ];

  const sidebarWidth = isMobile ? '0px' : (isSidebarOpen ? '280px' : '80px');

  return (
    <div className="min-h-screen bg-background text-foreground selection:bg-accent-primary/30">
      {/* Mobile Overlay */}
      <div
        className={`fixed inset-0 z-40 bg-black/60 backdrop-blur-sm transition-opacity duration-500 ${
          isMobile && isSidebarOpen ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'
        }`}
        onClick={() => setIsSidebarOpen(false)}
      />

      {/* Sidebar */}
      <aside
        className={`fixed left-0 top-0 z-50 h-screen border-r border-white/5 bg-surface-glass-strong backdrop-blur-xl transition-all duration-500 cubic-bezier(0.2, 0, 0, 1) ${isMobile
            ? (isSidebarOpen ? 'translate-x-0 w-[280px]' : '-translate-x-full w-[280px]')
            : ''
          }`}
        style={!isMobile ? { width: isSidebarOpen ? '280px' : '80px' } : undefined}
      >
        <div className="flex h-full flex-col">
          {/* Logo Section */}
          <div className="flex h-20 items-center justify-between px-6 border-b border-white/5">
            <div className={`flex items-center gap-3 overflow-hidden transition-all duration-500 ${isSidebarOpen ? 'opacity-100' : 'opacity-0 w-0'}`}>
              <div className="relative flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-accent-primary to-accent-secondary shadow-[0_0_20px_rgba(59,130,246,0.5)]">
                <span className="text-lg font-bold text-white">A</span>
              </div>
              <span className="text-lg font-bold tracking-tight text-white">
                Alpha<span className="text-accent-primary">OS</span>
              </span>
            </div>
            {!isMobile && (
              <button
                onClick={() => setIsSidebarOpen(!isSidebarOpen)}
                className="group rounded-lg p-2 text-slate-400 hover:bg-white/5 hover:text-white transition-all"
              >
                {isSidebarOpen ? <ChevronLeft size={18} /> : <Menu size={18} />}
              </button>
            )}
            {isMobile && (
              <button
                onClick={() => setIsSidebarOpen(false)}
                className="group rounded-lg p-2 text-slate-400 hover:bg-white/5 hover:text-white transition-all"
              >
                <ChevronLeft size={18} />
              </button>
            )}
          </div>

          {/* Navigation */}
          <nav className="flex-1 space-y-2 px-3 py-6">
            {navItems.map((item) => {
              const isActive = pathname === item.href;
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  onClick={() => isMobile && setIsSidebarOpen(false)}
                  className={`
                    group relative flex items-center gap-3 rounded-xl px-3.5 py-3 text-sm font-medium transition-all duration-300
                    ${isActive
                      ? 'text-white bg-gradient-to-r from-accent-primary/10 to-transparent border border-white/5 shadow-[inset_0_1px_0_0_rgba(255,255,255,0.05)]'
                      : 'text-slate-400 hover:bg-white/5 hover:text-slate-200'
                    }
                  `}
                >
                  {isActive && (
                    <div className="absolute left-0 top-1/2 h-6 w-1 -translate-y-1/2 rounded-r-full bg-accent-primary shadow-[0_0_12px_#3b82f6]"></div>
                  )}
                  <item.icon
                    size={20}
                    className={`transition-colors duration-300 ${isActive ? 'text-accent-primary drop-shadow-[0_0_8px_rgba(59,130,246,0.5)]' : 'group-hover:text-slate-300'}`}
                  />
                  <span className={`whitespace-nowrap transition-all duration-500 ${isSidebarOpen ? 'opacity-100 translate-x-0' : 'opacity-0 -translate-x-4 hidden'}`}>
                    {item.label}
                  </span>
                </Link>
              );
            })}
          </nav>

          {/* User Profile */}
          <div className="border-t border-white/5 p-4 bg-black/20">
            <button className={`group flex w-full items-center gap-3 rounded-xl p-2 transition-all hover:bg-white/5 ${!isSidebarOpen && 'justify-center'}`}>
              <div className="relative h-9 w-9 overflow-hidden rounded-full ring-1 ring-white/10 transition-all group-hover:ring-white/20 shadow-lg">
                <div className="flex h-full w-full items-center justify-center bg-gradient-to-br from-slate-800 to-slate-950 text-xs font-medium text-white">
                  TR
                </div>
                <div className="absolute bottom-0 right-0 h-2.5 w-2.5 rounded-full bg-accent-success ring-2 ring-[#030712] shadow-[0_0_8px_rgba(16,185,129,0.5)]"></div>
              </div>
              <div className={`flex flex-col items-start overflow-hidden transition-all duration-500 ${isSidebarOpen ? 'opacity-100 w-auto' : 'opacity-0 w-0 hidden'}`}>
                <span className="text-sm font-medium text-slate-200 group-hover:text-white transition-colors">交易员</span>
                <span className="text-[10px] uppercase tracking-wider text-accent-primary font-semibold">专业账户</span>
              </div>
            </button>
          </div>
        </div>
      </aside>

      {/* Main Content Wrapper */}
      <div
        className="min-h-screen transition-all duration-500 cubic-bezier(0.2, 0, 0, 1)"
        style={{ paddingLeft: sidebarWidth }}
      >
        {/* Header */}
        <header
          className={`
            sticky top-0 z-40 flex h-16 items-center justify-between px-4 md:px-8 transition-all duration-300
            ${scrolled ? 'bg-[#030712]/80 backdrop-blur-xl border-b border-white/5' : 'bg-transparent'}
          `}
        >
          {/* Left: Mobile Menu Trigger & Search */}
          <div className="flex items-center gap-4">
            {isMobile && (
              <button
                onClick={() => setIsSidebarOpen(true)}
                className="p-2 -ml-2 text-slate-400 hover:text-white"
              >
                <Menu size={24} />
              </button>
            )}

            {/* Search Bar */}
            <div className="hidden md:flex items-center gap-3 rounded-full bg-white/5 px-4 py-2 border border-white/5 transition-all focus-within:bg-white/10 focus-within:border-white/10 hover:bg-white/10 w-64">
              <Search size={14} className="text-slate-500" />
              <input
                type="text"
                placeholder="搜索市场..."
                className="bg-transparent border-none outline-none text-sm text-slate-200 placeholder:text-slate-600 w-full"
              />
              <div className="flex items-center gap-1 px-1.5 py-0.5 rounded border border-white/10 bg-white/5">
                <Command size={10} className="text-slate-500" />
                <span className="text-[10px] text-slate-500 font-medium">K</span>
              </div>
            </div>
          </div>

          {/* Right Actions */}
          <div className="flex items-center gap-4 md:gap-6">
            <div className="hidden md:flex flex-col items-end">
              <span className="text-[10px] uppercase tracking-wider text-slate-500 font-medium">净资产</span>
              <div className="flex items-baseline gap-2">
                <span className="text-sm font-bold text-white font-mono tracking-tight">
                  ${netAsset !== null ? netAsset.toFixed(2) : '---'}
                </span>
                {totalPnl !== 0 && (
                  <span className={`text-[10px] font-medium ${totalPnl >= 0 ? 'text-accent-success' : 'text-accent-danger'}`}>
                    {totalPnl >= 0 ? '+' : ''}{totalPnl.toFixed(2)}
                  </span>
                )}
              </div>
            </div>
            <div className="hidden md:block h-8 w-[1px] bg-white/10"></div>
            <button className="relative rounded-full p-2 text-slate-400 hover:bg-white/5 hover:text-white transition-all">
              <Bell size={18} />
              <span className="absolute right-2 top-2 h-1.5 w-1.5 rounded-full bg-accent-danger ring-2 ring-[#030712]"></span>
            </button>
          </div>
        </header>

        {/* Page Content */}
        <main className="p-4 md:p-8 animate-fade-in-up">
          {children}
        </main>
      </div>
    </div>
  );
}
