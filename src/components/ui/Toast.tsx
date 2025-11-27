"use client";
import React, { useEffect } from "react";
import { X } from "lucide-react";
import { cn } from "@/lib/utils";

type ToastItem = { id: number; title?: React.ReactNode; description?: React.ReactNode; action?: React.ReactNode; duration?: number };

const ToastContext = React.createContext<{
  toasts: ToastItem[];
  show: (t: Omit<ToastItem, "id">) => void;
  remove: (id: number) => void;
} | null>(null);

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = React.useState<ToastItem[]>([]);
  const seq = React.useRef(0);
  const show = (t: Omit<ToastItem, "id">) => {
    seq.current += 1;
    const id = seq.current;
    setToasts(prev => [...prev, { id, ...t }]);
    if (t.duration !== Infinity) {
        setTimeout(() => remove(id), t.duration || 3000);
    }
  };
  const remove = (id: number) => setToasts(prev => prev.filter(t => t.id !== id));
  
  return (
    <ToastContext.Provider value={{ toasts, show, remove }}>
      {children}
      <div aria-live="polite" aria-atomic="true" className="fixed bottom-4 right-4 z-50 flex flex-col gap-2 pointer-events-none">
        {toasts.map(t => (
          <div key={t.id} role="status" className="glass-panel rounded-xl px-4 py-3 text-sm text-white pointer-events-auto animate-in slide-in-from-right-full duration-300">
            <div className="flex justify-between items-start gap-4">
                <div>
                    {t.title && <div className="font-semibold mb-1">{t.title}</div>}
                    {t.description && <div className="text-slate-300 text-xs">{t.description}</div>}
                </div>
                {t.action && <div>{t.action}</div>}
            </div>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}

export function useToast() {
  const ctx = React.useContext(ToastContext);
  if (!ctx) throw new Error("useToast must be used within ToastProvider");
  return ctx;
}

interface ToastProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: React.ReactNode;
  description?: React.ReactNode;
  action?: React.ReactNode;
  duration?: number;
  className?: string;
}

export function Toast({ open, onOpenChange, title, description, action, duration = 5000, className }: ToastProps) {
  useEffect(() => {
    if (open && duration && duration !== Infinity) {
      const timer = setTimeout(() => {
        onOpenChange(false);
      }, duration);
      return () => clearTimeout(timer);
    }
  }, [open, duration, onOpenChange]);

  if (!open) return null;

  return (
    <div className={cn("fixed bottom-4 right-4 z-[60] flex flex-col gap-2 animate-in slide-in-from-right-full duration-300", className)}>
      <div role="status" className="glass-panel rounded-xl px-4 py-4 text-sm text-white shadow-2xl border border-white/10 bg-[#030712]/90 backdrop-blur-md min-w-[300px]">
        <div className="flex justify-between items-start gap-4">
          <div className="flex-1">
            {title && <div className="font-semibold mb-1.5 text-base">{title}</div>}
            {description && <div className="text-slate-300 text-xs leading-relaxed">{description}</div>}
          </div>
          <button onClick={() => onOpenChange(false)} className="text-slate-500 hover:text-white transition-colors">
            <X size={16} />
          </button>
        </div>
        {action && <div className="mt-3 flex justify-end">{action}</div>}
      </div>
    </div>
  );
}
