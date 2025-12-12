#!/bin/bash
# Script to apply schema migrations to remote Supabase

set -e

REMOTE_HOST="macOS"
REMOTE_DIR="~/alpha-os"

echo "📋 Applying Schema Migrations to Remote Supabase..."

# 1. Copy schema file to remote
echo "   📤 Uploading schema file..."
scp src/db/cloud_schema.sql $REMOTE_HOST:$REMOTE_DIR/

# 2. Apply schema via docker exec
echo "   🔧 Applying schema to Supabase DB container..."
ssh $REMOTE_HOST << 'ENDSSH'
    cd ~/alpha-os-data/supabase/docker
    
    # Find the actual container name (might be supabase-db or similar)
    DB_CONTAINER=$(docker ps --filter "name=supabase-db" --format "{{.Names}}" | head -n 1)
    
    if [ -z "$DB_CONTAINER" ]; then
        echo "❌ Supabase DB container not found!"
        exit 1
    fi
    
    echo "   📦 Using container: $DB_CONTAINER"
    
    # Apply schema
    cat ~/alpha-os/cloud_schema.sql | docker exec -i $DB_CONTAINER psql -U postgres -d postgres
    
    echo "   ✅ Schema applied successfully!"
ENDSSH

echo "✅ Schema Migration Complete!"
