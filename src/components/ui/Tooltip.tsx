"use client";
import { cn } from "@/lib/utils";
import React from "react";

type HandlerProps = Partial<{
  onMouseEnter: (e: React.MouseEvent<Element>) => void;
  onMouseLeave: (e: React.MouseEvent<Element>) => void;
  onFocus: (e: React.FocusEvent<Element>) => void;
  onBlur: (e: React.FocusEvent<Element>) => void;
  'aria-describedby': string;
}>;

interface TooltipProps {
  content: React.ReactNode;
  children: React.ReactElement<HandlerProps>;
  className?: string;
}

export function Tooltip({ content, children, className }: TooltipProps) {
  const [open, setOpen] = React.useState(false);
  const id = React.useId();
  const childEl = children as React.ReactElement<HandlerProps>;
  const child = React.cloneElement<HandlerProps>(childEl, {
    onMouseEnter: (e: React.MouseEvent) => {
      childEl.props.onMouseEnter?.(e);
      setOpen(true);
    },
    onMouseLeave: (e: React.MouseEvent) => {
      childEl.props.onMouseLeave?.(e);
      setOpen(false);
    },
    onFocus: (e: React.FocusEvent) => {
      childEl.props.onFocus?.(e);
      setOpen(true);
    },
    onBlur: (e: React.FocusEvent) => {
      childEl.props.onBlur?.(e);
      setOpen(false);
    },
    'aria-describedby': open ? id : undefined,
  });
  return (
    <span className={cn("relative inline-block", className)}>
      {child}
      <span
        id={id}
        role="tooltip"
        className={cn("pointer-events-none absolute -top-10 left-1/2 -translate-x-1/2 whitespace-nowrap rounded-lg bg-black/80 px-3 py-1.5 text-xs text-white shadow", open ? "opacity-100" : "opacity-0")}
      >
        {content}
      </span>
    </span>
  );
}
