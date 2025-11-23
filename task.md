# 📋 AlphaOS Optimization Tasks

## 🚀 Phase 1: User Experience & Visuals (Current Focus)
- [x] **Fix Chart Indicator Fidelity** <!-- id: 0 -->
    - [x] Analyze current `TradingViewChart` implementation
    - [x] Identify discrepancies between Lightweight Charts and actual TV
    - [x] Implement "Pivot Trend Signals" style rendering
- [x] **MT5 Symbol Sync** <!-- id: 1 -->
    - [x] Update MT5 EA to report active symbols
    - [x] Update Bridge API to expose active symbols endpoint
    - [x] Update Frontend to fetch and display dynamic symbol list

## 🛠 Phase 2: Stability & Core Architecture
- [x] **Environment Variable Validation** <!-- id: 2 -->
    - [x] Add `zod` validation script
    - [x] Handle missing/optional keys gracefully
- [x] **Bridge Health Monitoring** <!-- id: 3 -->
    - [x] Add Heartbeat check
    - [x] Add Dashboard status indicator

## 🎨 Phase 4: UI/UX Polish (Current)
- [x] **Global Styling**
    - [x] Update color palette and gradients
    - [x] Add glassmorphism utilities
## 🎨 Phase 4: UI/UX Polish (Current)
- [x] **Global Styling**
    - [x] Update color palette and gradients
    - [x] Add glassmorphism utilities
- [x] **Dashboard Refinement**
    - [x] Polish Stat Cards (Gradients, Icons)
    - [x] Refactor Chart Header (Controls, Status)
    - [x] Improve Sidebar and Layout
- [x] **Market Watch Implementation**
    - [x] Create MarketWatch component
    - [x] Integrate into Dashboard layout
    - [x] Clean up Chart controls
- [x] **Mobile Optimization**
    - [x] Responsive AppShell (Sidebar/Drawer)
    - [x] Responsive Dashboard Grid
    - [x] Component Tuning (MarketWatch, Chart)

## 🤖 Phase 3: Automation (Future)
- [x] Auto-Journaling (MT5 -> Supabase)
- [x] AI Review Interface (Design only)
