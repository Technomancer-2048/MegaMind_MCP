#!/bin/bash
# Run MegaMind Semantic Analysis
# Usage: ./scripts/run_semantic_analysis.sh

set -e

echo "Running MegaMind Semantic Analysis..."

# Check if database is running
if ! docker ps | grep -q megamind-mysql; then
    echo "❌ Database is not running. Please start it first:"
    echo "   ./scripts/start_database.sh"
    exit 1
fi

# Set environment variables
export MEGAMIND_DB_HOST=10.255.250.22
export MEGAMIND_DB_PORT=3309
export MEGAMIND_DB_NAME=megamind_database
export MEGAMIND_DB_USER=megamind_user
export MEGAMIND_DB_PASSWORD=${MYSQL_PASSWORD:-megamind_secure_pass}

# Create analysis virtual environment if it doesn't exist
if [ ! -d "analysis_venv" ]; then
    echo "Creating analysis virtual environment..."
    python3 -m venv analysis_venv
fi

# Activate virtual environment
source analysis_venv/bin/activate

# Install analysis dependencies
echo "Installing analysis dependencies..."
pip install -r analysis/requirements.txt

# Run semantic analysis
echo "Starting semantic analysis with embedding generation..."
echo "This may take several minutes for large document sets..."

cd analysis
python semantic_analyzer.py --password ${MEGAMIND_DB_PASSWORD} --batch-size 50

if [ $? -eq 0 ]; then
    echo "✅ Semantic analysis completed successfully!"
    echo ""
    echo "Generated:"
    echo "- Embeddings for all chunks"
    echo "- Semantic relationships between chunks"
    echo "- Automated tags for chunk classification"
    echo ""
    echo "You can now use advanced MCP functions:"
    echo "- mcp__megamind_db__search_by_embedding()"
    echo "- mcp__megamind_db__search_by_tags()"
    echo "- mcp__megamind_db__get_related_chunks()"
else
    echo "❌ Semantic analysis failed!"
    exit 1
fi

# Deactivate virtual environment
deactivate