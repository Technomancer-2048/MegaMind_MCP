#!/usr/bin/env python3
"""
STDIO Transport for MegaMind MCP Server
Direct connection for Claude Code via standard input/output
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add the server directory to Python path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Configure logging to stderr so it doesn't interfere with STDIO protocol
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr  # Important: log to stderr, not stdout
)

logger = logging.getLogger(__name__)

def load_env_file():
    """Load environment variables from .env file"""
    env_file = current_dir.parent / '.env'
    if env_file.exists():
        logger.info(f"Loading environment from: {env_file}")
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Only set if not already in environment
                    if key not in os.environ:
                        os.environ[key] = value
    else:
        logger.warning(f"No .env file found at: {env_file}")

def setup_environment():
    """Setup environment variables for MegaMind realm context and database connection"""
    # First load from .env file
    load_env_file()
    
    # Set default realm configuration (can be overridden by .mcp.json env settings)
    default_realm_env = {
        'MEGAMIND_PROJECT_REALM': 'MegaMind_MCP',
        'MEGAMIND_PROJECT_NAME': 'MegaMind Context Database',
        'MEGAMIND_DEFAULT_TARGET': 'PROJECT',
        'MEGAMIND_ROOT': str(current_dir.parent),
    }
    
    for key, value in default_realm_env.items():
        if key not in os.environ:
            os.environ[key] = value
    
    # Map Docker container addresses to external access
    if 'MEGAMIND_DB_HOST' not in os.environ:
        # Use the same IP that Docker containers are bound to
        bind_ip = os.environ.get('MYSQL_BIND_IP', '10.255.250.22')
        os.environ['MEGAMIND_DB_HOST'] = bind_ip
    
    if 'MEGAMIND_DB_PORT' not in os.environ:
        # Use external port mapping from Docker
        db_port = os.environ.get('MYSQL_PORT', '3309')
        os.environ['MEGAMIND_DB_PORT'] = db_port
    
    logger.info(f"Environment configured - Realm: {os.environ.get('MEGAMIND_PROJECT_REALM')}")
    logger.info(f"Database: {os.environ.get('MEGAMIND_DB_HOST')}:{os.environ.get('MEGAMIND_DB_PORT')}/{os.environ.get('MEGAMIND_DB_NAME')}")
    logger.info(f"Database User: {os.environ.get('MEGAMIND_DB_USER')}")

async def main():
    """Main entry point for STDIO transport"""
    try:
        # Setup environment first
        setup_environment()
        
        logger.info("=== MegaMind MCP Server - STDIO Transport ===")
        logger.info(f"Project Realm: {os.environ.get('MEGAMIND_PROJECT_REALM')}")
        logger.info(f"Default Target: {os.environ.get('MEGAMIND_DEFAULT_TARGET')}")
        
        # Import and initialize the main server
        from megamind_database_server import main as server_main
        
        # Run the main server logic
        await server_main()
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"STDIO transport error: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())