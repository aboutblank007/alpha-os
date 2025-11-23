# Implementation Plan - Mobile Optimization & Responsive Layout

The goal is to ensure the AlphaOS dashboard and all its components look and function perfectly on all device sizes, from mobile phones to large desktop monitors.

## User Review Required
> [!NOTE]
> I will be making significant changes to the `AppShell` to introduce a mobile-friendly navigation (e.g., a bottom nav or a hamburger menu drawer) and adjusting the dashboard grid to stack vertically on mobile.

## Proposed Changes

### 1. App Shell & Navigation
#### [MODIFY] [AppShell.tsx](file:///Users/hanjianglin/Documents/GitHub/alpha-os/src/components/AppShell.tsx)
- **Mobile**: Hide the permanent sidebar. Add a top header with a hamburger menu that opens a drawer, OR a bottom navigation bar for quick access.
- **Tablet**: Collapsed sidebar (icon only).
- **Desktop**: Expanded sidebar (as is).

### 2. Dashboard Layout
#### [MODIFY] [page.tsx](file:///Users/hanjianglin/Documents/GitHub/alpha-os/src/app/dashboard/page.tsx)
- **Grid**: Ensure `grid-cols-1` on mobile, `grid-cols-2` on tablet, `grid-cols-4` on desktop.
- **Spacing**: Reduce padding/gap on mobile (`p-4`, `gap-4`) vs desktop (`p-8`, `gap-6`).
- **Welcome Banner**: Stack text and buttons vertically on mobile. Hide or resize the background blobs/mesh for performance and visual clarity on small screens.

### 3. Component Responsiveness
#### [MODIFY] [MarketWatch.tsx](file:///Users/hanjianglin/Documents/GitHub/alpha-os/src/components/MarketWatch.tsx)
- **List Items**: Adjust font sizes for mobile. Ensure Buy/Sell buttons are touch-friendly (larger tap targets).
- **Layout**: On mobile, this might need to be a full-width card or a swipeable panel if it takes up too much vertical space.

#### [MODIFY] [TradingViewChart.tsx](file:///Users/hanjianglin/Documents/GitHub/alpha-os/src/components/charts/TradingViewChart.tsx)
- **Height**: Adjust height for mobile (e.g., `h-[350px]` instead of `500px`).
- **Controls**: The floating controls bar needs to wrap gracefully or hide less critical elements (like specific timeframes) into a dropdown on mobile.

#### [MODIFY] [StatCard.tsx](file:///Users/hanjianglin/Documents/GitHub/alpha-os/src/components/Card.tsx)
- **Font Sizes**: Scale down value text on very small screens.

## Verification Plan
### Manual Verification
- **Mobile View (375px)**: Check layout stacking, navigation access, and touch targets.
- **Tablet View (768px)**: Check grid reflow (2 columns) and sidebar behavior.
- **Desktop View (1440px+)**: Ensure no regression in the premium wide-screen experience.
