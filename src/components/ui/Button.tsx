"use client";
import { cn } from "@/lib/utils";
import React from "react";

type Variant = "primary" | "secondary" | "outline" | "ghost" | "danger" | "success";
type Size = "sm" | "md" | "lg";

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  loading?: boolean;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
}

const base = "inline-flex items-center justify-center rounded-xl font-medium transition-all focus:outline-none focus-visible:ring-2 focus-visible:ring-white/30 disabled:opacity-50 disabled:cursor-not-allowed";

const variants: Record<Variant, string> = {
  primary: "bg-accent-primary text-white hover:bg-accent-primary/90 shadow-lg shadow-accent-primary/20",
  secondary: "bg-white/5 text-slate-200 hover:bg-white/10 border border-white/10",
  outline: "border border-white/20 text-slate-200 hover:bg-white/5",
  ghost: "text-slate-300 hover:bg-white/5",
  danger: "bg-accent-danger text-white hover:bg-accent-danger/90",
  success: "bg-accent-success text-white hover:bg-accent-success/90",
};

const sizes: Record<Size, string> = {
  sm: "h-9 px-3 text-sm gap-2",
  md: "h-11 px-4 text-sm gap-2",
  lg: "h-12 px-5 text-base gap-3",
};

export const Button = React.memo(function Button({ 
  variant = "primary", 
  size = "md", 
  loading, 
  leftIcon, 
  rightIcon, 
  children, 
  className, 
  ...props 
}: ButtonProps) {
  return (
    <button
      className={cn(base, variants[variant], sizes[size], className)}
      type={props.type ?? "button"}
      aria-busy={loading ? true : undefined}
      {...props}
    >
      {leftIcon && <span className="mr-1.5">{leftIcon}</span>}
      {loading && (
        <span className="mr-2 inline-flex">
          <span className="h-4 w-4 rounded-full border-2 border-white/40 border-t-white animate-spin"></span>
        </span>
      )}
      {children}
      {rightIcon && <span className="ml-1.5">{rightIcon}</span>}
    </button>
  );
});

