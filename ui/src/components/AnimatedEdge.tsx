/**
 * AnimatedEdge - 带粒子流动效果的自定义边
 * 
 * 功能：
 * - 沿路径流动的光点粒子
 * - 不同颜色表示不同数据类型
 * - 根据数据流状态控制动画
 */

import React, { memo, useMemo } from 'react';
import { BaseEdge, getSmoothStepPath, getBezierPath, EdgeProps } from 'reactflow';

// 粒子颜色配置
const PARTICLE_COLORS = {
    tick: '#00f3ff',      // 青色 - Tick 数据
    order: '#ff0055',     // 红色 - 订单数据
    websocket: '#00ff9d', // 绿色 - WebSocket 推送
    feature: '#ffbe00',   // 黄色 - 特征/持久化
    model: '#7000ff',     // 紫色 - 模型推理
    default: '#00f3ff',
};

// 粒子数量配置
const PARTICLE_COUNTS = {
    high: 4,    // 高频数据流
    medium: 3,  // 中等频率
    low: 2,     // 低频数据流
};

/**
 * 流动粒子边组件
 */
export const FlowingEdge = memo(({
    id,
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
    style = {},
    markerEnd,
    data = {},
}: EdgeProps) => {
    const {
        particleColor = 'default',
        particleCount = 'medium',
        flowSpeed = 'normal',
        isActive = true,
        edgeType = 'smoothstep',
    } = data;

    // 计算路径
    const [edgePath] = useMemo(() => {
        if (edgeType === 'bezier') {
            return getBezierPath({
                sourceX, sourceY, sourcePosition,
                targetX, targetY, targetPosition,
            });
        }
        return getSmoothStepPath({
            sourceX, sourceY, sourcePosition,
            targetX, targetY, targetPosition,
            borderRadius: 8,
        });
    }, [sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, edgeType]);

    // 获取粒子配置
    const color = PARTICLE_COLORS[particleColor] || PARTICLE_COLORS.default;
    const count = PARTICLE_COUNTS[particleCount] || PARTICLE_COUNTS.medium;
    
    // 动画速度
    const duration = {
        fast: 1.5,
        normal: 2.5,
        slow: 4,
    }[flowSpeed] || 2.5;

    // 生成粒子
    const particles = useMemo(() => {
        if (!isActive) return [];
        
        return Array.from({ length: count }, (_, i) => ({
            id: `${id}-particle-${i}`,
            delay: (i / count) * duration,
            size: 3 + Math.random() * 2,
            opacity: 0.6 + Math.random() * 0.4,
        }));
    }, [id, count, duration, isActive]);

    return (
        <g className="animated-edge-group">
            {/* 基础边线（发光效果） */}
            <path
                d={edgePath}
                fill="none"
                stroke={color}
                strokeWidth={style.strokeWidth || 1.5}
                strokeOpacity={style.opacity || 0.3}
                filter="url(#glow)"
            />
            
            {/* 主边线 */}
            <BaseEdge
                id={id}
                path={edgePath}
                style={{
                    ...style,
                    stroke: style.stroke || color,
                    strokeWidth: style.strokeWidth || 1.5,
                }}
                markerEnd={markerEnd}
            />

            {/* 流动粒子 */}
            {particles.map((particle) => (
                <circle
                    key={particle.id}
                    r={particle.size}
                    fill={color}
                    opacity={particle.opacity}
                    filter="url(#particleGlow)"
                >
                    <animateMotion
                        dur={`${duration}s`}
                        repeatCount="indefinite"
                        begin={`${particle.delay}s`}
                        path={edgePath}
                    />
                </circle>
            ))}
        </g>
    );
});

/**
 * 脉冲边 - 数据传输时的脉冲效果
 */
export const PulseEdge = memo(({
    id,
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
    style = {},
    markerEnd,
    data = {},
}: EdgeProps) => {
    const { pulseColor = '#00f3ff', isPulsing = false } = data;

    const [edgePath] = getSmoothStepPath({
        sourceX, sourceY, sourcePosition,
        targetX, targetY, targetPosition,
        borderRadius: 8,
    });

    return (
        <g className={`pulse-edge-group ${isPulsing ? 'pulsing' : ''}`}>
            {/* 脉冲光晕 */}
            {isPulsing && (
                <path
                    d={edgePath}
                    fill="none"
                    stroke={pulseColor}
                    strokeWidth={6}
                    strokeOpacity={0}
                    className="pulse-glow"
                />
            )}
            
            {/* 主边线 */}
            <BaseEdge
                id={id}
                path={edgePath}
                style={{
                    ...style,
                    stroke: style.stroke || pulseColor,
                }}
                markerEnd={markerEnd}
            />
        </g>
    );
});

/**
 * SVG 滤镜定义（需要在 FlowChart 中引用）
 */
export const EdgeFilters = () => (
    <svg style={{ position: 'absolute', width: 0, height: 0 }}>
        <defs>
            {/* 粒子发光效果 */}
            <filter id="particleGlow" x="-50%" y="-50%" width="200%" height="200%">
                <feGaussianBlur stdDeviation="2" result="blur" />
                <feMerge>
                    <feMergeNode in="blur" />
                    <feMergeNode in="SourceGraphic" />
                </feMerge>
            </filter>
            
            {/* 边线发光效果 */}
            <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
                <feGaussianBlur stdDeviation="3" result="blur" />
                <feMerge>
                    <feMergeNode in="blur" />
                    <feMergeNode in="SourceGraphic" />
                </feMerge>
            </filter>
            
            {/* 脉冲动画滤镜 */}
            <filter id="pulseFilter" x="-50%" y="-50%" width="200%" height="200%">
                <feGaussianBlur stdDeviation="4" result="blur">
                    <animate
                        attributeName="stdDeviation"
                        values="2;6;2"
                        dur="1s"
                        repeatCount="indefinite"
                    />
                </feGaussianBlur>
            </filter>
        </defs>
    </svg>
);

// 导出边类型映射
export const edgeTypes = {
    flowing: FlowingEdge,
    pulse: PulseEdge,
};

export default FlowingEdge;
