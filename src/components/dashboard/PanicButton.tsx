/**
 * 紧急平仓按钮组件
 * 
 * 一键发送 CLOSE_ALL 命令到 Command Bus
 */

'use client';

import { useState, useCallback } from 'react';
import { Button } from '@/components/ui/Button';
import { cn } from '@/lib/utils';
import { AlertTriangle, X, Check } from 'lucide-react';

interface PanicButtonProps {
    /** 发送命令的回调 */
    onCloseAll: () => boolean | Promise<boolean>;
    /** 是否禁用 */
    disabled?: boolean;
    className?: string;
}

export function PanicButton({ onCloseAll, disabled, className }: PanicButtonProps) {
    const [state, setState] = useState<'idle' | 'confirming' | 'executing' | 'success' | 'error'>('idle');
    const [confirmTimeout, setConfirmTimeout] = useState<NodeJS.Timeout | null>(null);

    // 第一次点击: 进入确认状态
    const handleFirstClick = useCallback(() => {
        setState('confirming');

        // 5 秒后自动取消确认状态
        const timeout = setTimeout(() => {
            setState('idle');
        }, 5000);

        setConfirmTimeout(timeout);
    }, []);

    // 第二次点击: 执行平仓
    const handleConfirmClick = useCallback(async () => {
        if (confirmTimeout) {
            clearTimeout(confirmTimeout);
            setConfirmTimeout(null);
        }

        setState('executing');

        try {
            const result = await onCloseAll();
            setState(result ? 'success' : 'error');
        } catch (error) {
            console.error('[PanicButton] 平仓失败:', error);
            setState('error');
        }

        // 3 秒后恢复
        setTimeout(() => setState('idle'), 3000);
    }, [onCloseAll, confirmTimeout]);

    // 取消确认
    const handleCancel = useCallback(() => {
        if (confirmTimeout) {
            clearTimeout(confirmTimeout);
            setConfirmTimeout(null);
        }
        setState('idle');
    }, [confirmTimeout]);

    return (
        <div className={cn('relative', className)}>
            {state === 'idle' && (
                <Button
                    variant="danger"
                    onClick={handleFirstClick}
                    disabled={disabled}
                    className="w-full h-12 text-lg font-bold shadow-lg hover:shadow-red-500/25 transition-shadow"
                >
                    <AlertTriangle className="mr-2" size={20} />
                    紧急平仓
                </Button>
            )}

            {state === 'confirming' && (
                <div className="flex gap-2 animate-pulse">
                    <Button
                        variant="danger"
                        onClick={handleConfirmClick}
                        className="flex-1 h-12 text-lg font-bold bg-red-600 hover:bg-red-700"
                    >
                        <AlertTriangle className="mr-2" size={20} />
                        确认平仓?
                    </Button>
                    <Button
                        variant="secondary"
                        onClick={handleCancel}
                        className="h-12 w-12"
                    >
                        <X size={20} />
                    </Button>
                </div>
            )}

            {state === 'executing' && (
                <Button
                    variant="danger"
                    disabled
                    className="w-full h-12 text-lg font-bold"
                >
                    <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin mr-2" />
                    执行中...
                </Button>
            )}

            {state === 'success' && (
                <Button
                    variant="primary"
                    disabled
                    className="w-full h-12 text-lg font-bold bg-green-600"
                >
                    <Check className="mr-2" size={20} />
                    已平仓
                </Button>
            )}

            {state === 'error' && (
                <Button
                    variant="danger"
                    disabled
                    className="w-full h-12 text-lg font-bold"
                >
                    <X className="mr-2" size={20} />
                    平仓失败
                </Button>
            )}

            {/* 确认状态倒计时指示器 */}
            {state === 'confirming' && (
                <div className="absolute -bottom-1 left-0 right-0 h-1 bg-red-900 rounded-full overflow-hidden">
                    <div className="h-full bg-red-500 animate-[shrink_5s_linear]" />
                </div>
            )}
        </div>
    );
}
