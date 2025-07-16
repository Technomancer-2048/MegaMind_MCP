#!/bin/bash

# Phase 4 Schema Application Script
# AI Enhancement System - Quality Improvement, Adaptive Learning, and Performance Optimization

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}Phase 4: AI Enhancement Schema${NC}"
echo -e "${GREEN}======================================${NC}"

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Check if .env file exists
if [ ! -f "$PROJECT_ROOT/.env" ]; then
    echo -e "${RED}Error: .env file not found in $PROJECT_ROOT${NC}"
    echo "Please create a .env file with database credentials"
    exit 1
fi

# Load environment variables
source "$PROJECT_ROOT/.env"

# Database connection parameters
DB_HOST="${MEGAMIND_DB_HOST:-10.255.250.22}"
DB_PORT="${MEGAMIND_DB_PORT:-3309}"
DB_NAME="${MEGAMIND_DB_NAME:-megamind_database}"
DB_USER="${MEGAMIND_DB_USER:-megamind_user}"
DB_PASS="${MEGAMIND_DB_PASSWORD}"

# Check if password is set
if [ -z "$DB_PASS" ]; then
    echo -e "${RED}Error: Database password not set. Please set MEGAMIND_DB_PASSWORD in .env${NC}"
    exit 1
fi

# Schema file
SCHEMA_FILE="$PROJECT_ROOT/database/schema_updates/phase4_ai_enhancement.sql"

# Check if schema file exists
if [ ! -f "$SCHEMA_FILE" ]; then
    echo -e "${RED}Error: Schema file not found: $SCHEMA_FILE${NC}"
    exit 1
fi

echo -e "${YELLOW}Applying Phase 4 schema to database...${NC}"
echo "Host: $DB_HOST:$DB_PORT"
echo "Database: $DB_NAME"
echo "Schema file: $SCHEMA_FILE"

# Apply the schema
mysql -h "$DB_HOST" -P "$DB_PORT" -u "$DB_USER" -p"$DB_PASS" "$DB_NAME" < "$SCHEMA_FILE"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Phase 4 schema applied successfully!${NC}"
    echo ""
    echo "New tables created:"
    echo "  - megamind_quality_improvements"
    echo "  - megamind_user_feedback"
    echo "  - megamind_learning_patterns"
    echo "  - megamind_adaptive_strategies"
    echo "  - megamind_curation_decisions"
    echo "  - megamind_curation_workflows"
    echo "  - megamind_performance_metrics"
    echo "  - megamind_optimization_history"
    echo "  - megamind_embedding_cache"
    echo "  - megamind_ai_reports"
    echo ""
    echo "New views created:"
    echo "  - megamind_quality_effectiveness_view"
    echo "  - megamind_learning_effectiveness_view"
    echo "  - megamind_curation_performance_view"
    echo ""
    echo "New procedures created:"
    echo "  - apply_quality_improvement"
    echo "  - record_feedback_and_learn"
    echo ""
    echo -e "${GREEN}Phase 4 database schema is ready!${NC}"
else
    echo -e "${RED}✗ Failed to apply Phase 4 schema${NC}"
    echo "Please check the error messages above"
    exit 1
fi