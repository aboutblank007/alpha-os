"use client";
import { cn } from "@/lib/utils";
import React from "react";

interface ModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title?: string;
  children: React.ReactNode;
  footer?: React.ReactNode;
  className?: string;
  hideCloseButton?: boolean;
}

export function Modal({ open, onOpenChange, title, children, footer, className, hideCloseButton }: ModalProps) {
  const overlayRef = React.useRef<HTMLDivElement | null>(null);
  const dialogRef = React.useRef<HTMLDivElement | null>(null);

  React.useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onOpenChange(false);
    };
    if (open) document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, onOpenChange]);

  React.useEffect(() => {
    if (!open) return;
    const el = dialogRef.current;
    if (!el) return;
    const focusables = el.querySelectorAll<HTMLElement>("a, button, input, select, textarea, [tabindex]:not([tabindex='-1'])");
    const first = focusables[0];
    first?.focus();
  }, [open]);

  return (
    <div
      aria-hidden={!open}
      className={cn("fixed inset-0 z-50 grid place-items-center p-4 transition-all duration-200", open ? "visible opacity-100" : "invisible opacity-0")}
    >
      <div
        ref={overlayRef}
        onClick={() => onOpenChange(false)}
        className="absolute inset-0 bg-black/50 backdrop-blur-sm"
      />
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-label={title}
        className={cn("glass-panel w-full max-w-lg rounded-2xl p-6 relative z-10", className)}
      >
        {title && (
          <div className="mb-4 text-lg font-semibold text-white">{title}</div>
        )}
        <div>{children}</div>
        {footer && (
          <div className="mt-6 flex justify-end gap-2">{footer}</div>
        )}
        {!hideCloseButton && (
          <button
            className="absolute right-4 top-4 h-8 w-8 rounded-lg text-slate-400 hover:text-white hover:bg-white/5"
            onClick={() => onOpenChange(false)}
            aria-label="Close"
          >
            ×
          </button>
        )}
      </div>
    </div>
  );
}

