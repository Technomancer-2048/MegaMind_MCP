#!/bin/bash
# Start MegaMind Database System
# Usage: ./scripts/start_database.sh

set -e

echo "Starting MegaMind Context Database System..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please update the passwords in .env file before production use!"
fi

# Start the services
echo "Starting MySQL and Redis services..."
docker-compose up -d megamind-mysql megamind-redis

# Wait for MySQL to be ready
echo "Waiting for MySQL to be ready..."
timeout=60
counter=0
while ! docker exec megamind-mysql mysqladmin ping -h localhost --silent; do
    sleep 2
    counter=$((counter + 2))
    if [ $counter -ge $timeout ]; then
        echo "❌ MySQL failed to start within $timeout seconds"
        exit 1
    fi
done

echo "✅ MySQL is ready!"

# Verify database setup
echo "Verifying database setup..."
docker exec megamind-mysql mysql -u megamind_user -p${MYSQL_PASSWORD:-megamind_secure_pass} -e "USE megamind_database; SHOW TABLES;"

echo "✅ MegaMind Database System is running!"
echo ""
echo "MySQL: 10.255.250.22:3309"
echo "Redis: 10.255.250.22:6379"
echo ""
echo "To start the MCP server: ./scripts/start_mcp_server.sh"
echo "To stop all services: docker-compose down"