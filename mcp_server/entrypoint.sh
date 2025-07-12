#!/bin/bash
set -e

echo "MegaMind MCP Server starting..."

# Initialize database schema
echo "Initializing database schema..."
python init_database.py

if [ $? -ne 0 ]; then
    echo "Database initialization failed"
    exit 1
fi

echo "Database initialization successful"

# Start the MCP server
echo "Starting MCP server..."
exec python megamind_database_server.py