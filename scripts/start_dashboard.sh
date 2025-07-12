#!/bin/bash
# Start MegaMind Analytics Dashboard
# Usage: ./scripts/start_dashboard.sh

set -e

echo "Starting MegaMind Analytics Dashboard..."

# Check if database is running
if ! docker ps | grep -q megamind-mysql; then
    echo "❌ Database is not running. Please start it first:"
    echo "   ./scripts/start_database.sh"
    exit 1
fi

# Set environment variables
export MEGAMIND_DB_HOST=localhost
export MEGAMIND_DB_PORT=3309
export MEGAMIND_DB_NAME=megamind_database
export MEGAMIND_DB_USER=megamind_user
export MEGAMIND_DB_PASSWORD=${MYSQL_PASSWORD:-megamind_secure_pass}
export DASHBOARD_PORT=5000
export FLASK_DEBUG=false

# Create dashboard virtual environment if it doesn't exist
if [ ! -d "dashboard_venv" ]; then
    echo "Creating dashboard virtual environment..."
    python3 -m venv dashboard_venv
fi

# Activate virtual environment
source dashboard_venv/bin/activate

# Install dashboard dependencies
echo "Installing dashboard dependencies..."
pip install -r dashboard/requirements.txt

# Start the dashboard
echo "Starting analytics dashboard on port ${DASHBOARD_PORT}..."
echo "Dashboard will be available at: http://localhost:${DASHBOARD_PORT}"

cd dashboard
python context_analytics.py

# Deactivate virtual environment on exit
deactivate