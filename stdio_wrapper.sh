#!/bin/bash
# stdio_wrapper.sh - Wrapper to properly activate venv before running MCP server

set -e  # Exit on any error

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Function to find and activate virtual environment
activate_venv() {
    local venv_paths=(
        "$SCRIPT_DIR/venv"
        "$SCRIPT_DIR/.venv"
        "$(dirname "$SCRIPT_DIR")/venv"
        "$(dirname "$SCRIPT_DIR")/.venv"
    )
    
    for venv_path in "${venv_paths[@]}"; do
        if [[ -f "$venv_path/bin/activate" ]]; then
            echo "Found virtual environment: $venv_path" >&2
            source "$venv_path/bin/activate"
            echo "Activated virtual environment" >&2
            return 0
        fi
    done
    
    echo "Warning: No virtual environment found, using system Python" >&2
    return 1
}

# Function to test Python dependencies
test_dependencies() {
    echo "Testing Python dependencies..." >&2
    python3 -c "
import sys
try:
    import mysql.connector
    print('✓ mysql.connector available', file=sys.stderr)
except ImportError as e:
    print(f'✗ mysql.connector missing: {e}', file=sys.stderr)
    sys.exit(1)

try:
    # Test local imports - add mcp_server to path
    sys.path.insert(0, '$SCRIPT_DIR/mcp_server')
    import megamind_database_server
    print('✓ megamind_database_server available', file=sys.stderr)
except ImportError as e:
    print(f'✗ megamind_database_server missing: {e}', file=sys.stderr)
    sys.exit(1)

print('✓ All dependencies available', file=sys.stderr)
"
}

# Main execution
main() {
    echo "=== MegaMind MCP STDIO Wrapper ===" >&2
    echo "Script directory: $SCRIPT_DIR" >&2
    
    # Try to activate virtual environment
    activate_venv || true
    
    # Show Python info
    echo "Python executable: $(which python3)" >&2
    echo "Python version: $(python3 --version)" >&2
    
    # Test dependencies
    if ! test_dependencies; then
        echo "Dependency test failed - cannot proceed" >&2
        exit 1
    fi
    
    # Load environment variables if .env exists
    if [[ -f "$SCRIPT_DIR/.env" ]]; then
        echo "Loading .env file" >&2
        set -a  # Automatically export variables
        source "$SCRIPT_DIR/.env"
        set +a
    fi
    
    # Execute the actual Python MCP server
    echo "Starting MegaMind MCP server..." >&2
    exec python3 "$SCRIPT_DIR/mcp_server/stdio_transport.py" "$@"
}

main "$@"