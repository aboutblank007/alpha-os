DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'runtime_state'
          AND column_name = 'timestamp'
    ) THEN
        ALTER TABLE runtime_state RENAME COLUMN timestamp TO time;
    END IF;

    IF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'runtime_state'
          AND column_name = 'time'
          AND data_type IN ('double precision', 'real')
    ) THEN
        ALTER TABLE runtime_state
            ALTER COLUMN time TYPE TIMESTAMPTZ
            USING to_timestamp(time);
    END IF;

    IF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'runtime_state'
          AND column_name = 'snapshot_count'
    ) THEN
        ALTER TABLE runtime_state RENAME COLUMN snapshot_count TO db_snapshot_count;
    END IF;
END $$;
