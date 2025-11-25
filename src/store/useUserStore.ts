import { create } from 'zustand';

export interface UserPreferences {
  display_name: string;
  theme: 'light' | 'dark' | 'auto';
  risk_level: 'low' | 'medium' | 'high' | 'aggressive';
  default_currency: string;
  show_live_price: boolean;
  auto_sync: boolean;
  email_notifications: boolean;
  trade_alerts: boolean;
  risk_alerts: boolean;
}

interface UserStore {
  preferences: UserPreferences;
  isLoading: boolean;
  
  // Actions
  setPreferences: (prefs: Partial<UserPreferences>) => void;
  setLoading: (loading: boolean) => void;
}

const defaultPreferences: UserPreferences = {
  display_name: 'Trader',
  theme: 'dark',
  risk_level: 'medium',
  default_currency: 'USD',
  show_live_price: true,
  auto_sync: true,
  email_notifications: true,
  trade_alerts: true,
  risk_alerts: true,
};

export const useUserStore = create<UserStore>((set) => ({
  preferences: defaultPreferences,
  isLoading: false,

  setPreferences: (prefs) => set((state) => ({
    preferences: { ...state.preferences, ...prefs }
  })),
  
  setLoading: (loading) => set({ isLoading: loading }),
}));

