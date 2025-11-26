import {
    CustomSeriesPricePlotValues,
    ICustomSeriesPaneView,
    PaneRendererCustomData,
    Time,
    WhitespaceData,
    CustomSeriesOptions,
} from 'lightweight-charts';

export interface CloudData {
    time: Time;
    ema1: number;
    ema2: number;
}

interface BitmapCoordinatesRenderingScope {
    context: CanvasRenderingContext2D;
    horizontalPixelRatio: number;
    verticalPixelRatio: number;
}

interface CanvasRenderingTarget2D {
    useBitmapCoordinateSpace(
        scope: (scope: BitmapCoordinatesRenderingScope) => void
    ): void;
}

export class CloudSeriesRenderer {
    _data: PaneRendererCustomData<Time, CloudData> | null = null;
    _options: CustomSeriesOptions | null = null;

    draw(target: CanvasRenderingTarget2D, priceConverter: (price: number) => number | null) {
        target.useBitmapCoordinateSpace((scope: BitmapCoordinatesRenderingScope) => {
            if (this._data === null || this._data.bars.length === 0 || this._data.visibleRange === null) {
                return;
            }

            const ctx = scope.context;
            const pixelRatio = scope.horizontalPixelRatio;

            ctx.save();

            const bars = this._data.bars;

            // First pass: Draw the cloud fill
            for (let i = 0; i < bars.length - 1; i++) {
                const bar = bars[i];
                const nextBar = bars[i + 1];

                // Skip if data is missing
                if (!bar.originalData || !nextBar.originalData) continue;

                const curr = bar.originalData as CloudData;
                const next = nextBar.originalData as CloudData;

                // Coordinates
                const x1 = bar.x * pixelRatio;
                const x2 = nextBar.x * pixelRatio;

                const y1_ema1 = priceConverter(curr.ema1);
                const y1_ema2 = priceConverter(curr.ema2);

                const y2_ema1 = priceConverter(next.ema1);
                const y2_ema2 = priceConverter(next.ema2);

                if (y1_ema1 === null || y1_ema2 === null || y2_ema1 === null || y2_ema2 === null) {
                    continue;
                }

                const y1_ema1_scaled = y1_ema1 * pixelRatio;
                const y1_ema2_scaled = y1_ema2 * pixelRatio;
                const y2_ema1_scaled = y2_ema1 * pixelRatio;
                const y2_ema2_scaled = y2_ema2 * pixelRatio;

                // Determine color based on trend of the *current* bar
                const trendUp = curr.ema1 > curr.ema2;

                // Cloud fill with transparency - matching the chart's green/red theme
                ctx.fillStyle = trendUp ? 'rgba(16, 185, 129, 0.12)' : 'rgba(239, 68, 68, 0.12)';

                // Draw Quad
                ctx.beginPath();
                ctx.moveTo(x1, y1_ema1_scaled);
                ctx.lineTo(x2, y2_ema1_scaled);
                ctx.lineTo(x2, y2_ema2_scaled);
                ctx.lineTo(x1, y1_ema2_scaled);
                ctx.closePath();
                ctx.fill();
            }

            // Second pass: Draw EMA1 line (faster - green/teal)
            ctx.strokeStyle = 'rgba(16, 185, 129, 0.9)';
            ctx.lineWidth = 2 * pixelRatio;
            ctx.lineJoin = 'round';
            ctx.lineCap = 'round';
            ctx.beginPath();

            let firstPoint = true;
            for (let i = 0; i < bars.length; i++) {
                const bar = bars[i];
                if (!bar.originalData) continue;

                const data = bar.originalData as CloudData;
                const x = bar.x * pixelRatio;
                const y = priceConverter(data.ema1);

                if (y === null) {
                     firstPoint = true;
                     continue;
                }

                const yScaled = y * pixelRatio;

                if (firstPoint) {
                    ctx.moveTo(x, yScaled);
                    firstPoint = false;
                } else {
                    ctx.lineTo(x, yScaled);
                }
            }
            ctx.stroke();

            // Third pass: Draw EMA2 line (slower - red)
            ctx.strokeStyle = 'rgba(239, 68, 68, 0.9)';
            ctx.lineWidth = 2 * pixelRatio;
            ctx.lineJoin = 'round';
            ctx.lineCap = 'round';
            ctx.beginPath();

            firstPoint = true;
            for (let i = 0; i < bars.length; i++) {
                const bar = bars[i];
                if (!bar.originalData) continue;

                const data = bar.originalData as CloudData;
                const x = bar.x * pixelRatio;
                const y = priceConverter(data.ema2);

                if (y === null) {
                     firstPoint = true; // Reset path if point is invalid to avoid zig-zag
                     continue;
                }

                const yScaled = y * pixelRatio;

                if (firstPoint) {
                    ctx.moveTo(x, yScaled);
                    firstPoint = false;
                } else {
                    ctx.lineTo(x, yScaled);
                }
            }
            ctx.stroke();

            ctx.restore();
        });
    }

    update(data: PaneRendererCustomData<Time, CloudData>, options: CustomSeriesOptions) {
        this._data = data;
        this._options = options;
    }
}

export class CloudSeries implements ICustomSeriesPaneView<Time, CloudData, CustomSeriesOptions> {
    _renderer: CloudSeriesRenderer;

    constructor() {
        this._renderer = new CloudSeriesRenderer();
    }

    priceValueBuilder(plotRow: CloudData): CustomSeriesPricePlotValues {
        // Use the average or max/min for auto-scaling
        return [plotRow.ema1, plotRow.ema2, Math.max(plotRow.ema1, plotRow.ema2)];
    }

    isWhitespace(data: CloudData | WhitespaceData): data is WhitespaceData {
        return (data as Partial<CloudData>).ema1 === undefined;
    }

    renderer(): CloudSeriesRenderer {
        return this._renderer;
    }

    update(
        data: PaneRendererCustomData<Time, CloudData>,
        options: CustomSeriesOptions
    ): void {
        this._renderer.update(data, options);
    }

    defaultOptions(): CustomSeriesOptions {
        return {
            color: 'rgba(0, 0, 0, 0)',
            priceLineVisible: false,
            lastValueVisible: false,
        } as CustomSeriesOptions;
    }
}