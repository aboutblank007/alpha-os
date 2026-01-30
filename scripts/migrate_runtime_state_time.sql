-- Migrate runtime_state from legacy timestamp FLOAT to time TIMESTAMPTZ.
-- Safe to run multiple times.

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'runtime_state'
          AND column_name = 'timestamp'
    ) THEN
        ALTER TABLE runtime_state
            ALTER COLUMN timestamp TYPE TIMESTAMPTZ USING to_timestamp(timestamp);
        ALTER TABLE runtime_state
            RENAME COLUMN timestamp TO time;
    END IF;
END $$;

ALTER TABLE runtime_state
    ALTER COLUMN time SET NOT NULL;
