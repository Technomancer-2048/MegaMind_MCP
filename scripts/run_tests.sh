#!/bin/bash
# Run MegaMind Database Tests
# Usage: ./scripts/run_tests.sh

set -e

echo "Running MegaMind Context Database Tests..."

# Check if database is running
if ! docker ps | grep -q megamind-mysql; then
    echo "❌ Test database is not running. Starting database..."
    ./scripts/start_database.sh
    sleep 10
fi

# Set test environment variables
export TEST_DB_HOST=localhost
export TEST_DB_PORT=3309
export TEST_DB_NAME=megamind_database
export TEST_DB_USER=megamind_user
export TEST_DB_PASSWORD=${MYSQL_PASSWORD:-megamind_secure_pass}

# Create test virtual environment if it doesn't exist
if [ ! -d "test_venv" ]; then
    echo "Creating test virtual environment..."
    python3 -m venv test_venv
fi

# Activate virtual environment
source test_venv/bin/activate

# Install test dependencies
echo "Installing test dependencies..."
pip install -r tools/requirements.txt
pip install -r mcp_server/requirements.txt

# Run the tests
echo "Running Phase 1 validation tests..."
cd tests
python test_phase1_validation.py

echo "✅ All tests completed!"

# Deactivate virtual environment
deactivate