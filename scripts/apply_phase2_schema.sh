#!/bin/bash

# Apply Phase 2 Session Tracking System Schema
# This script applies the session tracking tables to the MegaMind database

set -e  # Exit on error

# Configuration
DB_HOST=${MEGAMIND_DB_HOST:-10.255.250.22}
DB_PORT=${MEGAMIND_DB_PORT:-3309}
DB_NAME=${MEGAMIND_DB_NAME:-megamind_database}
DB_USER=${MEGAMIND_DB_USER:-megamind_user}

# Check if password is set
if [ -z "$MEGAMIND_DB_PASSWORD" ]; then
    echo "Error: MEGAMIND_DB_PASSWORD environment variable not set"
    exit 1
fi

SCHEMA_FILE="database/schema_updates/phase2_session_tracking_system.sql"

# Check if schema file exists
if [ ! -f "$SCHEMA_FILE" ]; then
    echo "Error: Schema file not found: $SCHEMA_FILE"
    exit 1
fi

echo "Applying Phase 2 Session Tracking System schema..."
echo "Database: $DB_NAME @ $DB_HOST:$DB_PORT"

# Apply the schema
mysql -h "$DB_HOST" -P "$DB_PORT" -u "$DB_USER" -p"$MEGAMIND_DB_PASSWORD" "$DB_NAME" < "$SCHEMA_FILE"

if [ $? -eq 0 ]; then
    echo "✅ Phase 2 schema applied successfully!"
    echo ""
    echo "New tables created:"
    echo "  - megamind_embedding_sessions"
    echo "  - megamind_session_chunks"
    echo "  - megamind_session_state"
    echo "  - megamind_session_documents"
    echo "  - megamind_session_metrics"
    echo ""
    echo "Tables modified:"
    echo "  - megamind_document_structures (added session tracking)"
    echo "  - megamind_chunk_metadata (added session tracking)"
    echo "  - megamind_entry_embeddings (added session tracking)"
    echo "  - megamind_quality_assessments (added session tracking)"
    echo ""
    echo "Next steps:"
    echo "1. Rebuild the HTTP container: docker compose build megamind-mcp-server-http"
    echo "2. Restart the container: docker compose up -d megamind-mcp-server-http"
    echo "3. Test the new MCP functions"
else
    echo "❌ Error applying Phase 2 schema"
    exit 1
fi