export interface TradeRequest {
    action: 'BUY' | 'SELL';
    symbol: string;
    volume: number;
    sl?: number;
    tp?: number;
}

export interface BridgeResponse {
    status: string;
    command?: any;
    detail?: string;
}

export class BridgeClient {
    private baseUrl: string;

    constructor(baseUrl: string = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }

    async executeTrade(trade: TradeRequest): Promise<BridgeResponse> {
        try {
            const response = await fetch(`${this.baseUrl}/trade/execute`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(trade),
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Trade execution failed');
            }

            return await response.json();
        } catch (error) {
            console.error('Bridge API Error:', error);
            throw error;
        }
    }

    async healthCheck(): Promise<boolean> {
        try {
            const response = await fetch(`${this.baseUrl}/health`);
            return response.ok;
        } catch (error) {
            return false;
        }
    }
}

export const bridgeClient = new BridgeClient();
