"use client";

import React, { useState } from "react";
import { ChevronDown, ChevronUp, ArrowUpDown } from "lucide-react";
import { cn } from "@/lib/utils";

export interface Column<T> {
  key: keyof T | string;
  header: string;
  render?: (item: T) => React.ReactNode;
  sortable?: boolean;
  className?: string;
  align?: "left" | "center" | "right";
}

interface DataGridProps<T> {
  data: T[];
  columns: Column<T>[];
  keyExtractor: (item: T) => string;
  onRowClick?: (item: T) => void;
  loading?: boolean;
  emptyMessage?: string;
  className?: string;
}

type SortDirection = "asc" | "desc" | null;

export function DataGrid<T>({
  data,
  columns,
  keyExtractor,
  onRowClick,
  loading = false,
  emptyMessage = "无数据",
  className,
}: DataGridProps<T>) {
  const [sortConfig, setSortConfig] = useState<{ key: string; direction: SortDirection } | null>(null);

  const handleSort = (key: string) => {
    setSortConfig((current) => {
      if (current?.key === key) {
        if (current.direction === "asc") return { key, direction: "desc" };
        if (current.direction === "desc") return null;
      }
      return { key, direction: "asc" };
    });
  };

  const sortedData = React.useMemo(() => {
    if (!sortConfig || !sortConfig.direction) return data;

    return [...data].sort((a, b) => {
      const valA = (a as Record<string, unknown>)[sortConfig.key as string];
      const valB = (b as Record<string, unknown>)[sortConfig.key as string];

      // Handle null/undefined values for comparison
      if (valA === null || valA === undefined) return sortConfig.direction === "asc" ? -1 : 1;
      if (valB === null || valB === undefined) return sortConfig.direction === "asc" ? 1 : -1;

      if (valA < valB) return sortConfig.direction === "asc" ? -1 : 1;
      if (valA > valB) return sortConfig.direction === "asc" ? 1 : -1;
      return 0;
    });
  }, [data, sortConfig]);

  return (
    <div className={cn("w-full overflow-hidden rounded-xl border border-surface-border bg-surface-glass", className)}>
      <div className="overflow-x-auto">
        <table className="w-full text-sm text-left">
          <thead className="text-xs text-slate-400 uppercase bg-surface-glass-strong border-b border-surface-border">
            <tr>
              {columns.map((col) => (
                <th
                  key={col.key as string}
                  className={cn(
                    "px-4 py-3 font-medium select-none whitespace-nowrap",
                    col.sortable && "cursor-pointer hover:text-white transition-colors group",
                    col.align === "right" && "text-right",
                    col.align === "center" && "text-center",
                    col.className
                  )}
                  onClick={() => col.sortable && handleSort(col.key as string)}
                >
                  <div className={cn(
                    "flex items-center gap-1",
                    col.align === "right" && "justify-end",
                    col.align === "center" && "justify-center"
                  )}>
                    {col.header}
                    {col.sortable && (
                      <span className="text-slate-600 group-hover:text-slate-400">
                        {sortConfig?.key === col.key ? (
                          sortConfig.direction === "asc" ? <ChevronUp size={12} /> : <ChevronDown size={12} />
                        ) : (
                          <ArrowUpDown size={12} />
                        )}
                      </span>
                    )}
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-surface-border/50">
            {loading ? (
              <tr>
                <td colSpan={columns.length} className="px-4 py-12 text-center text-slate-500">
                  <div className="flex flex-col items-center gap-2">
                    <div className="h-5 w-5 rounded-full border-2 border-slate-600 border-t-transparent animate-spin"></div>
                    <span>加载中...</span>
                  </div>
                </td>
              </tr>
            ) : sortedData.length === 0 ? (
              <tr>
                <td colSpan={columns.length} className="px-4 py-12 text-center text-slate-500">
                  {emptyMessage}
                </td>
              </tr>
            ) : (
              sortedData.map((item) => (
                <tr
                  key={keyExtractor(item)}
                  onClick={() => onRowClick?.(item)}
                  className={cn(
                    "hover:bg-white/5 transition-colors",
                    onRowClick && "cursor-pointer active:bg-white/10"
                  )}
                >
                  {columns.map((col) => (
                    <td
                      key={col.key as string}
                      className={cn(
                        "px-4 py-3 whitespace-nowrap text-slate-300",
                        col.align === "right" && "text-right",
                        col.align === "center" && "text-center",
                        col.className
                      )}
                    >
                      {col.render ? col.render(item) : String((item as Record<string, unknown>)[col.key as string] ?? '')}
                    </td>
                  ))}
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

