#!/bin/bash
# Start MegaMind MCP Server
# Usage: ./scripts/start_mcp_server.sh

set -e

echo "Starting MegaMind MCP Server..."

# Check if database is running
if ! docker ps | grep -q megamind-mysql; then
    echo "❌ MySQL database is not running. Please start it first:"
    echo "   ./scripts/start_database.sh"
    exit 1
fi

# Start the MCP server
echo "Building and starting MCP server..."
docker-compose up -d megamind-mcp-server

# Wait for server to be ready
echo "Waiting for MCP server to be ready..."
timeout=30
counter=0
while ! docker logs megamind-mcp-server 2>/dev/null | grep -q "MegaMind MCP Server starting"; do
    sleep 2
    counter=$((counter + 2))
    if [ $counter -ge $timeout ]; then
        echo "❌ MCP server failed to start within $timeout seconds"
        echo "Check logs with: docker logs megamind-mcp-server"
        exit 1
    fi
done

echo "✅ MegaMind MCP Server is running!"
echo ""
echo "MCP Server: localhost:8002"
echo ""
echo "Check status: docker-compose ps"
echo "View logs: docker logs megamind-mcp-server"
echo "Stop server: docker-compose stop megamind-mcp-server"