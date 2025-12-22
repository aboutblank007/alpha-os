import React from "react";
import { cn } from "@/lib/utils";

export interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
    leftIcon?: React.ReactNode;
}

export const Input = React.forwardRef<HTMLInputElement, InputProps>(
    ({ className, leftIcon, ...props }, ref) => {
        return (
            <div className="relative flex items-center w-full group">
                {leftIcon && (
                    <div className="absolute left-3 text-text-muted group-focus-within:text-primary transition-colors">
                        {leftIcon}
                    </div>
                )}
                <input
                    className={cn(
                        "flex h-9 w-full rounded-lg border border-border-subtle bg-bg-subtle px-3 py-1 text-sm shadow-sm transition-colors",
                        "file:border-0 file:bg-transparent file:text-sm file:font-medium",
                        "placeholder:text-text-muted text-text-primary",
                        "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary focus-visible:border-primary/50",
                        "disabled:cursor-not-allowed disabled:opacity-50",
                        leftIcon && "pl-9",
                        className
                    )}
                    ref={ref}
                    {...props}
                />
            </div>
        );
    }
);
Input.displayName = "Input";
