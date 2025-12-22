
import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export type UserRole = 'novice' | 'trader' | 'admin';
export type Density = 'compact' | 'balanced' | 'dense';
export type DefaultTerminalMode = 'trading' | 'monitoring';

interface SettingsState {
    displayName: string;
    email: string;
    timezone: string;
    defaultCurrency: string;
    riskLevel: 'low' | 'medium' | 'high';
    userRole: UserRole;
    density: Density;
    defaultTerminalMode: DefaultTerminalMode;
    showLivePrice: boolean;
    autoSync: boolean;
    emailNotifications: boolean;
    tradeAlerts: boolean;
    riskAlerts: boolean;
    dailySummary: boolean;
    theme: 'light' | 'dark' | 'auto';
    accentColor: string;
    updateSettings: (settings: Partial<SettingsState>) => void;
}

export const useSettingsStore = create<SettingsState>()(
    persist(
        (set) => ({
            displayName: "交易员",
            email: "trader@alphaos.com",
            timezone: "Asia/Shanghai",
            defaultCurrency: "USD",
            riskLevel: "medium",
            userRole: "trader",
            density: "balanced",
            defaultTerminalMode: "trading",
            showLivePrice: true,
            autoSync: true,
            emailNotifications: true,
            tradeAlerts: true,
            riskAlerts: true,
            dailySummary: false,
            theme: "dark",
            accentColor: "blue",
            updateSettings: (newSettings) => set((state) => ({ ...state, ...newSettings })),
        }),
        {
            name: 'alphaos-settings-storage',
        }
    )
);
