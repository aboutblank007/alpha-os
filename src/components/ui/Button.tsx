"use client";

import React from "react";
import { cn } from "@/lib/utils";

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
    variant?: "primary" | "secondary" | "ghost" | "danger" | "outline";
    size?: "sm" | "md" | "lg" | "icon";
    isLoading?: boolean;
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
    ({ className, variant = "primary", size = "md", isLoading, children, ...props }, ref) => {

        const baseStyles = "inline-flex items-center justify-center rounded-lg font-medium transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-primary/20 disabled:opacity-50 disabled:pointer-events-none active:scale-[0.98]";

        const variants = {
            primary: "bg-primary text-white hover:bg-primary/90 shadow-[0_0_15px_rgba(37,99,235,0.3)] hover:shadow-[0_0_25px_rgba(37,99,235,0.5)] border border-transparent",
            secondary: "bg-bg-subtle text-text-secondary border border-border-subtle hover:text-white hover:border-border-active hover:bg-white/5",
            ghost: "text-text-secondary hover:text-white hover:bg-white/5",
            danger: "bg-danger/10 text-danger border border-danger/20 hover:bg-danger/20",
            outline: "border border-border-subtle text-text-secondary hover:text-white hover:border-white/20 bg-transparent",
        };

        const sizes = {
            sm: "h-8 px-3 text-xs",
            md: "h-10 px-4 text-sm",
            lg: "h-12 px-6 text-base",
            icon: "h-9 w-9 p-0",
        };

        return (
            <button
                ref={ref}
                className={cn(baseStyles, variants[variant], sizes[size], className)}
                disabled={isLoading || props.disabled}
                {...props}
            >
                {isLoading ? (
                    <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin mr-2" />
                ) : null}
                {children}
            </button>
        );
    }
);
Button.displayName = "Button";
