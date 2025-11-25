"use client";

import { Card } from "@/components/Card";
import { Calendar, Filter, Download, Upload, Edit, BookOpen } from "lucide-react";
import { ImportTradesModal } from "@/components/journal/ImportTradesModal";
import { JournalNoteModal } from "@/components/journal/JournalNoteModal";
import { useState, useEffect } from "react";

interface DayData {
  date: Date;
  day: number;
  hasTrade: boolean;
  hasNote: boolean;
  isWin?: boolean;
  pnl?: number;
  tradeCount?: number;
}

interface DailyStats {
  date: string;
  total_pnl: number;
  total_trades: number;
  winning_trades: number;
}

export default function JournalPage() {
  const [isImportModalOpen, setIsImportModalOpen] = useState(false);
  const [isNoteModalOpen, setIsNoteModalOpen] = useState(false);
  const [selectedDate, setSelectedDate] = useState<Date>(new Date());
  const [currentMonth, setCurrentMonth] = useState(new Date());
  const [notes, setNotes] = useState<Record<string, { id?: string; content: string; mood?: string; tags?: string[] }>>({});
  const [currentNote, setCurrentNote] = useState<{ id?: string; content: string; mood?: string; tags?: string[] } | null>(null);
  const [dailyStats, setDailyStats] = useState<Record<string, DailyStats>>({});
  const [todayStats, setTodayStats] = useState<DailyStats | null>(null);

  // 生成当前月份的日历数据
  const generateCalendarDays = (): DayData[] => {
    const year = currentMonth.getFullYear();
    const month = currentMonth.getMonth();
    const firstDay = new Date(year, month, 1);
    const lastDay = new Date(year, month + 1, 0);
    const daysInMonth = lastDay.getDate();
    const startDayOfWeek = firstDay.getDay();

    const days: DayData[] = [];

    // 填充月初空白
    for (let i = 0; i < startDayOfWeek; i++) {
      const date = new Date(year, month, -startDayOfWeek + i + 1);
      days.push({
        date,
        day: date.getDate(),
        hasTrade: false,
        hasNote: false,
      });
    }

    // 填充当月日期
    for (let day = 1; day <= daysInMonth; day++) {
      const date = new Date(year, month, day);
      const dateStr = date.toISOString().split('T')[0];
      const stats = dailyStats[dateStr];

      days.push({
        date,
        day,
        hasTrade: !!stats && stats.total_trades > 0,
        hasNote: !!notes[dateStr],
        isWin: stats ? stats.total_pnl >= 0 : undefined,
        pnl: stats?.total_pnl || 0,
        tradeCount: stats?.total_trades || 0,
      });
    }

    return days;
  };

  const days = generateCalendarDays();

  // 加载数据
  useEffect(() => {
    // 加载当月笔记
    const loadNotes = async () => {
      const firstDay = new Date(currentMonth.getFullYear(), currentMonth.getMonth(), 1);
      const lastDay = new Date(currentMonth.getFullYear(), currentMonth.getMonth() + 1, 0);

      try {
        const response = await fetch(
          `/api/journal/notes?startDate=${firstDay.toISOString().split('T')[0]}&endDate=${lastDay.toISOString().split('T')[0]}`
        );
        if (response.ok) {
          const data = await response.json();
          const notesMap: Record<string, { id?: string; content: string; mood?: string; tags?: string[] }> = {};
          data.notes?.forEach((note: { date: string; id?: string; content: string; mood?: string; tags?: string[] }) => {
            notesMap[note.date] = note;
          });
          setNotes(notesMap);
        }
      } catch (error) {
        console.error('加载笔记失败:', error);
      }
    };

    // 加载每日交易统计
    const loadDailyStats = async () => {
      const firstDay = new Date(currentMonth.getFullYear(), currentMonth.getMonth(), 1);
      const lastDay = new Date(currentMonth.getFullYear(), currentMonth.getMonth() + 1, 0);

      try {
        const response = await fetch(
          `/api/trades/daily-stats?startDate=${firstDay.toISOString().split('T')[0]}&endDate=${lastDay.toISOString().split('T')[0]}`
        );
        if (response.ok) {
          const data = await response.json();
          const statsMap: Record<string, DailyStats> = {};
          data.stats?.forEach((stat: DailyStats) => {
            statsMap[stat.date] = stat;
          });
          setDailyStats(statsMap);
        }
      } catch (error) {
        console.error('加载交易统计失败:', error);
      }
    };

    // 加载今日统计
    const loadTodayStats = async () => {
      const today = new Date().toISOString().split('T')[0];
      try {
        const response = await fetch(`/api/trades/daily-stats?date=${today}`);
        if (response.ok) {
          const data = await response.json();
          setTodayStats(data.stats);
        }
      } catch (error) {
        console.error('加载今日统计失败:', error);
      }
    };

    loadNotes();
    loadDailyStats();
    loadTodayStats();
  }, [currentMonth]);

  // 加载选定日期的笔记
  const loadNoteForDate = async (date: Date) => {
    const dateStr = date.toISOString().split('T')[0];
    try {
      const response = await fetch(`/api/journal/notes?date=${dateStr}`);
      if (response.ok) {
        const data = await response.json();
        setCurrentNote(data.note);
      }
    } catch (error) {
      console.error('加载笔记失败:', error);
    }
  };

  // 处理日期点击
  const handleDateClick = (date: Date) => {
    setSelectedDate(date);
    loadNoteForDate(date);
    setIsNoteModalOpen(true);
  };

  // 保存笔记
  const handleSaveNote = async (note: { content: string; mood?: string; tags?: string[] }) => {
    const dateStr = selectedDate.toISOString().split('T')[0];
    try {
      const response = await fetch('/api/journal/notes', {
        method: currentNote ? 'PUT' : 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          date: dateStr,
          ...note,
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error);
      }

      // 重新加载笔记 (Need to trigger reload, simplest is to refresh page or refetch)
      // For now, we can just reload the window or refetch manually if we extract fetch logic.
      // But since we moved fetch logic inside useEffect, we can't call it easily.
      // Let's just reload the page for now as a simple fix, or we could move fetch logic back to useCallback.
      window.location.reload();
    } catch (error: unknown) {
      console.error('保存笔记失败:', error);
      throw error;
    }
  };

  // 切换月份
  const changeMonth = (delta: number) => {
    setCurrentMonth(new Date(currentMonth.getFullYear(), currentMonth.getMonth() + delta, 1));
  };

  const getMonthName = () => {
    return `${currentMonth.getFullYear()}年 ${currentMonth.getMonth() + 1}月`;
  };

  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white tracking-tight">交易日志</h1>
          <p className="text-slate-400 mt-2">回顾与分析您的交易表现</p>
        </div>
        <div className="flex gap-3">
          <button className="flex items-center gap-2 px-4 py-2 rounded-xl bg-surface-glass border border-surface-border text-slate-300 hover:text-white hover:bg-white/5 transition-all">
            <Filter size={18} />
            <span>筛选</span>
          </button>
          <button
            onClick={() => setIsImportModalOpen(true)}
            className="flex items-center gap-2 px-4 py-2 rounded-xl bg-surface-glass border border-surface-border text-slate-300 hover:text-white hover:bg-white/5 transition-all"
          >
            <Upload size={18} />
            <span>导入 CSV</span>
          </button>
          <button className="flex items-center gap-2 px-4 py-2 rounded-xl bg-accent-primary text-white shadow-lg shadow-accent-primary/20 hover:bg-accent-primary/90 transition-all">
            <Download size={18} />
            <span>导出</span>
          </button>
        </div>
      </div>

      <ImportTradesModal
        open={isImportModalOpen}
        onOpenChange={setIsImportModalOpen}
        onSuccess={() => {
          console.log("导入成功");
        }}
      />

      <JournalNoteModal
        open={isNoteModalOpen}
        onOpenChange={setIsNoteModalOpen}
        date={selectedDate}
        existingNote={currentNote as { id?: string; content: string; mood?: string; tags?: string[] } | undefined}
        onSave={handleSaveNote}
      />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Calendar View */}
        <Card className="lg:col-span-2">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-semibold text-white flex items-center gap-2">
              <Calendar className="text-accent-primary" size={20} />
              {getMonthName()}
            </h2>
            <div className="flex gap-2">
              <button
                onClick={() => changeMonth(-1)}
                className="p-2 rounded-lg hover:bg-white/5 text-slate-400 hover:text-white transition-colors"
              >
                &larr;
              </button>
              <button
                onClick={() => setCurrentMonth(new Date())}
                className="px-3 py-2 rounded-lg text-sm hover:bg-white/5 text-slate-400 hover:text-white transition-colors"
              >
                今天
              </button>
              <button
                onClick={() => changeMonth(1)}
                className="p-2 rounded-lg hover:bg-white/5 text-slate-400 hover:text-white transition-colors"
              >
                &rarr;
              </button>
            </div>
          </div>

          <div className="grid grid-cols-7 gap-4 mb-4">
            {['日', '一', '二', '三', '四', '五', '六'].map(day => (
              <div key={day} className="text-center text-sm font-medium text-slate-500 uppercase tracking-wider">
                {day}
              </div>
            ))}
          </div>

          <div className="grid grid-cols-7 gap-4">
            {days.map((d, i) => {
              const isCurrentMonth = d.date.getMonth() === currentMonth.getMonth();
              const isToday = d.date.toDateString() === new Date().toDateString();

              return (
                <div
                  key={i}
                  onClick={() => isCurrentMonth && handleDateClick(d.date)}
                  className={`
                                        aspect-square rounded-xl border p-3 relative group transition-all duration-300
                                        ${isCurrentMonth ? 'cursor-pointer hover:scale-105 hover:shadow-lg' : 'opacity-30'}
                                        ${isToday ? 'ring-2 ring-accent-primary' : 'border-surface-border'}
                  ${d.hasTrade && d.isWin ? 'bg-accent-success/10 border-accent-success/20' : ''}
                  ${d.hasTrade && !d.isWin ? 'bg-accent-danger/10 border-accent-danger/20' : ''}
                                        ${!d.hasTrade && isCurrentMonth ? 'bg-white/[0.02]' : ''}
                `}
                >
                  <div className="flex flex-col h-full">
                    <span className={`text-sm font-medium ${isCurrentMonth ? 'text-white' : 'text-slate-600'}`}>
                      {d.day}
                    </span>

                    {/* 笔记指示器 */}
                    {d.hasNote && isCurrentMonth && (
                      <div className="absolute top-2 right-2">
                        <BookOpen size={12} className="text-accent-primary" />
                      </div>
                    )}

                    {/* 盈亏显示 */}
                    {d.hasTrade && isCurrentMonth && d.pnl !== undefined && (
                      <div className="absolute bottom-2 right-2 text-xs font-bold">
                        <span className={d.pnl >= 0 ? 'text-accent-success' : 'text-accent-danger'}>
                          {d.pnl >= 0 ? '+' : ''}{d.pnl.toFixed(0)}
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </Card>

        {/* Daily Summary & Quick Note */}
        <div className="space-y-6">
          <Card>
            <h3 className="text-lg font-semibold text-white mb-4">今日摘要</h3>
            <div className="space-y-4">
              <div className="p-4 rounded-xl bg-surface-glass border border-surface-border">
                <div className="text-sm text-slate-400 mb-1">净盈亏</div>
                <div className={`text-2xl font-bold ${(todayStats?.total_pnl || 0) >= 0 ? 'text-accent-success' : 'text-accent-danger'
                  }`}>
                  {todayStats?.total_pnl !== undefined
                    ? `${(todayStats.total_pnl || 0) >= 0 ? '+' : ''}${(todayStats.total_pnl || 0).toFixed(2)}`
                    : '$0.00'
                  }
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="p-4 rounded-xl bg-surface-glass border border-surface-border">
                  <div className="text-sm text-slate-400 mb-1">交易笔数</div>
                  <div className="text-xl font-bold text-white">
                    {todayStats?.total_trades || 0}
                  </div>
                </div>
                <div className="p-4 rounded-xl bg-surface-glass border border-surface-border">
                  <div className="text-sm text-slate-400 mb-1">胜率</div>
                  <div className="text-xl font-bold text-accent-primary">
                    {(todayStats?.total_trades || 0) > 0
                      ? `${Math.round(((todayStats?.winning_trades || 0) / (todayStats?.total_trades || 1)) * 100)}%`
                      : '0%'
                    }
                  </div>
                </div>
              </div>
            </div>
          </Card>

          <Card>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-white">快速笔记</h3>
              <button
                onClick={() => {
                  setSelectedDate(new Date());
                  loadNoteForDate(new Date());
                  setIsNoteModalOpen(true);
                }}
                className="p-2 rounded-lg hover:bg-white/5 text-slate-400 hover:text-accent-primary transition-colors"
              >
                <Edit size={18} />
              </button>
            </div>
            <div className="text-sm text-slate-400 mb-3">
              点击日历上的日期来添加或编辑笔记
            </div>
            <div className="p-4 rounded-xl bg-black/20 border border-surface-border">
              {currentNote ? (
                <div className="space-y-2">
                  <div className="text-slate-300 text-sm line-clamp-4">
                    {currentNote.content}
                  </div>
                  {currentNote.tags && currentNote.tags.length > 0 && (
                    <div className="flex flex-wrap gap-2 mt-2">
                      {currentNote.tags.map((tag: string) => (
                        <span
                          key={tag}
                          className="px-2 py-1 bg-accent-primary/20 text-accent-primary rounded text-xs"
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-slate-500 text-sm text-center py-4">
                  暂无笔记
                </div>
              )}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
