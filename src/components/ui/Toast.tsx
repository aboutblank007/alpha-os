"use client";
import React from "react";

type ToastItem = { id: number; title?: string; description?: string };

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
    setTimeout(() => remove(id), 3000);
  };
  const remove = (id: number) => setToasts(prev => prev.filter(t => t.id !== id));
  return (
    <ToastContext.Provider value={{ toasts, show, remove }}>
      {children}
      <div aria-live="polite" aria-atomic="true" className="fixed bottom-4 right-4 z-50 flex flex-col gap-2">
        {toasts.map(t => (
          <div key={t.id} role="status" className="glass-panel rounded-xl px-4 py-3 text-sm text-white">
            {t.title && <div className="font-semibold">{t.title}</div>}
            {t.description && <div className="text-slate-300">{t.description}</div>}
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
