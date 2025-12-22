import React from "react";
import { cn } from "@/lib/utils";

export function GlassCard({ className, children, ...props }: React.HTMLAttributes<HTMLDivElement>) {
    return (
        <div
            className={cn(
                "quantum-glass rounded-xl overflow-hidden transition-all duration-200",
                className
            )}
            {...props}
        >
            {children}
        </div>
    );
}

export function CardHeader({ className, children }: React.HTMLAttributes<HTMLDivElement>) {
    return <div className={cn("px-5 py-4 border-b border-white/5", className)}>{children}</div>;
}

export function CardTitle({ className, children }: React.HTMLAttributes<HTMLHeadingElement>) {
    return <h3 className={cn("text-sm font-medium text-text-primary tracking-wide", className)}>{children}</h3>;
}

export function CardContent({ className, children }: React.HTMLAttributes<HTMLDivElement>) {
    return <div className={cn("p-5", className)}>{children}</div>;
}
