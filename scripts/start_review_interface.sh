#!/bin/bash
# Start MegaMind Change Review Interface
# Usage: ./scripts/start_review_interface.sh

set -e

echo "Starting MegaMind Change Review Interface..."

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
export REVIEW_PORT=5001
export FLASK_DEBUG=false

# Create review virtual environment if it doesn't exist
if [ ! -d "review_venv" ]; then
    echo "Creating review interface virtual environment..."
    python3 -m venv review_venv
fi

# Activate virtual environment
source review_venv/bin/activate

# Install review interface dependencies
echo "Installing review interface dependencies..."
pip install -r review/requirements.txt

# Start the review interface
echo "Starting change review interface on port ${REVIEW_PORT}..."
echo "Review interface will be available at: http://localhost:${REVIEW_PORT}"

cd review
python change_reviewer.py

# Deactivate virtual environment on exit
deactivate