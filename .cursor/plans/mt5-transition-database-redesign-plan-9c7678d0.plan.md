<!-- 9c7678d0-ca73-4a72-b33c-9d7565eec321 ff21eb23-5ccc-4470-b576-3a9fe70abdb9 -->
# MT5 Transition & Database Redesign Plan

This plan outlines the steps to remove legacy dependencies (OANDA, Browser Extension), implement a robust MT5-native database schema, and plan for missing features.

## 1. Cleanup Legacy Code (Immediate)

- **Delete Directories**:
- `alpha-link-extension/` (No longer needed as we rely on MT5 Bridge).
- **Delete Files**:
- `src/lib/oanda.ts`
- **Refactor**:
- `src/env.ts`: Remove `OANDA_*` environment variables.
- `src/app/api/prices/route.ts`: Remove OANDA imports/helpers. Update to strictly use MT5 Bridge or Mock fallback.
- `src/components/OngoingOrders.tsx`: Remove "OANDA API not configured" warnings.
- `src/app/api/test-env/route.ts`: Remove OANDA checks.

## 2. Database Redesign (New Schema)

We will replace the generic `trades` table with MT5-specific tables to accurately reflect "Positions" (Live) vs "Deals" (History).

### New Tables

1.  **`mt5_accounts`**

-   Stores account info synced from MT5.
-   Columns: `login` (PK), `balance`, `equity`, `margin`, `free_margin`, `leverage`, `currency`, `server`, `updated_at`.

2.  **`mt5_deals` (History)**

-   Immutable record of executed trades (Entry & Exit are separate deals in MT5, or Netting).
-   Columns: `ticket` (PK), `order`, `time`, `type` (BUY/SELL), `entry` (IN/OUT), `symbol`, `volume`, `price`, `commission`, `swap`, `profit`, `magic`, `comment`.

3.  **`mt5_positions` (Live State)**

-   Current open positions.
-   Columns: `ticket` (PK), `symbol`, `type`, `volume`, `price_open`, `price_current`, `sl`, `tp`, `swap`, `profit`, `magic`, `updated_at`.

4.  **`app_settings`**

-   Store user preferences (missing feature).
-   Columns: `key` (PK), `value` (JSON), `updated_at`.

### SQL Migration Script (`DB_MT5_MIGRATION.sql`)

(See implementation below)

## 3. Trading Bridge Update (`trading-bridge/src/main.py`)

- **Sync Logic**:
- **On Startup**: Clear `mt5_positions` table and re-populate from MT5 status.
- **On Trade**: Insert into `mt5_deals` when a trade is reported.
- **On Status Update**: Upsert `mt5_accounts` and sync `mt5_positions` (Update existing, insert new, delete missing).

## 4. Unfinished Features Implementation

- **Settings Persistence**:
- Implement `src/app/api/settings` to read/write to `app_settings` table.
- Use for "Risk % per trade", "Theme", "Default Symbols".
- **Journaling Integration**:
- Update `journal_notes` table to add `deal_ticket` (BigInt) FK to `mt5_deals`.
- Allow users to "Attach Note" to a specific past trade in the UI.
- **Advanced Analytics**:
- Create Views: `view_daily_pnl`, `view_symbol_performance` based on `mt5_deals`.
- Implement `Win Rate`, `Profit Factor`, `Avg Win/Loss` calculations using SQL views.

## 5. Implementation Roadmap

1.  **Execute Cleanup**: Delete files and fix build errors.
2.  **Run Migration**: Execute `DB_MT5_MIGRATION.sql` in Supabase.
3.  **Update Bridge**: Modify Python code to write to new tables.
4.  **Update Frontend**: Point generic components (`RecentTrades`) to new API endpoints that return `mt5_deals` data.

### To-dos

- [ ] Delete legacy files (alpha-link-extension, oanda.ts) and remove OANDA references from code
- [ ] Create and execute DB_MT5_MIGRATION.sql for new MT5 schema
- [ ] Update trading-bridge/src/main.py to sync data to new tables
- [ ] Update API routes and Frontend to use new MT5 tables