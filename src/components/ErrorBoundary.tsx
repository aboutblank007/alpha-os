"use client";

import React, { Component, ErrorInfo, ReactNode } from "react";
import { AlertTriangle, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/Button";

interface Props {
  children?: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("Uncaught error:", error, errorInfo);
  }

  public render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div className="flex flex-col items-center justify-center h-full min-h-[200px] p-6 rounded-xl bg-accent-danger/5 border border-accent-danger/20 text-center">
          <div className="w-12 h-12 rounded-full bg-accent-danger/10 flex items-center justify-center mb-4 text-accent-danger">
            <AlertTriangle size={24} />
          </div>
          <h3 className="text-lg font-medium text-white mb-2">组件加载失败</h3>
          <p className="text-sm text-slate-400 mb-6 max-w-xs">
            {this.state.error?.message || "发生未知错误，请稍后重试。"}
          </p>
          <Button
            variant="secondary"
            onClick={() => this.setState({ hasError: false, error: null })}
            leftIcon={<RefreshCw size={16} />}
          >
            重试
          </Button>
        </div>
      );
    }

    return this.props.children;
  }
}

