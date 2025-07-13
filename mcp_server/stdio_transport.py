#!/usr/bin/env python3
"""
STDIO Transport for MegaMind MCP Server
Direct connection for Claude Code via standard input/output
FIXED: Properly handles virtual environment dependencies
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# CRITICAL: Set up paths and environment BEFORE any other imports
current_dir = Path(__file__).parent
project_root = current_dir.parent

def setup_python_path():
    """Setup Python path to find both local modules and venv packages"""
    # Add current directory first for local imports
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    # Add project root for broader imports
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Look for virtual environment and add to path
    venv_paths = [
        current_dir / 'venv' / 'lib' / 'python3.11' / 'site-packages',
        current_dir / 'venv' / 'lib' / 'python3.10' / 'site-packages',
        current_dir / 'venv' / 'lib' / 'python3.9' / 'site-packages',
        project_root / 'venv' / 'lib' / 'python3.11' / 'site-packages',
        project_root / 'venv' / 'lib' / 'python3.10' / 'site-packages',
        # Check for common venv names
        current_dir / '.venv' / 'lib' / 'python3.11' / 'site-packages',
        project_root / '.venv' / 'lib' / 'python3.11' / 'site-packages',
    ]
    
    venv_found = False
    for venv_path in venv_paths:
        if venv_path.exists():
            venv_str = str(venv_path)
            if venv_str not in sys.path:
                sys.path.insert(0, venv_str)
                logging.info(f"Added venv to path: {venv_str}")
                venv_found = True
                break
    
    if not venv_found:
        logging.warning("No virtual environment found - using system packages")
    
    # Also check for pip user packages
    import site
    user_site = site.getusersitepackages()
    if user_site and os.path.exists(user_site) and user_site not in sys.path:
        sys.path.append(user_site)

def load_env_file():
    """Load environment variables from .env file"""
    env_files = [
        current_dir / '.env',
        project_root / '.env',
        current_dir / '.env.local',
        project_root / '.env.local'
    ]
    
    for env_file in env_files:
        if env_file.exists():
            logging.info(f"Loading environment from: {env_file}")
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        # Strip quotes if present
                        value = value.strip('"\'')
                        # Only set if not already in environment
                        if key not in os.environ:
                            os.environ[key] = value
            break
    else:
        logging.warning("No .env file found in expected locations")

def setup_environment():
    """Setup environment variables for MegaMind realm context and database connection"""
    # First load from .env file
    load_env_file()
    
    # Set default realm configuration (can be overridden by .mcp.json env settings)
    default_realm_env = {
        'MEGAMIND_PROJECT_REALM': 'MegaMind_MCP',
        'MEGAMIND_PROJECT_NAME': 'MegaMind Context Database',
        'MEGAMIND_DEFAULT_TARGET': 'PROJECT',
        'MEGAMIND_ROOT': str(project_root),
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

def test_imports():
    """Test that critical imports work before proceeding"""
    try:
        # Test database imports
        import mysql.connector
        logging.info("✓ mysql.connector available")
    except ImportError as e:
        logging.error(f"✗ mysql.connector missing: {e}")
        return False
    
    try:
        # Test other critical dependencies
        import asyncio
        import json
        logging.info("✓ Standard library imports OK")
    except ImportError as e:
        logging.error(f"✗ Standard library import failed: {e}")
        return False
    
    return True

# CRITICAL: Setup paths and environment FIRST
setup_python_path()

# Configure logging to stderr so it doesn't interfere with STDIO protocol
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr  # Important: log to stderr, not stdout
)

logger = logging.getLogger(__name__)

async def main():
    """Main entry point for STDIO transport"""
    try:
        # Setup environment
        setup_environment()
        
        logger.info("=== MegaMind MCP Server - STDIO Transport ===")
        logger.info(f"Project Root: {project_root}")
        logger.info(f"Current Directory: {current_dir}")
        logger.info(f"Python Path: {sys.path[:3]}...")  # Show first 3 entries
        logger.info(f"Project Realm: {os.environ.get('MEGAMIND_PROJECT_REALM')}")
        logger.info(f"Default Target: {os.environ.get('MEGAMIND_DEFAULT_TARGET')}")
        
        # Test critical imports
        if not test_imports():
            logger.error("Critical dependencies missing - cannot proceed")
            sys.exit(1)
        
        # NOW import the main server (after path/env setup)
        try:
            # Try relative import first, then absolute path
            try:
                from megamind_database_server import main as server_main
            except ImportError:
                # Add mcp_server directory to path for container environment
                mcp_server_path = current_dir / 'mcp_server'
                if not mcp_server_path.exists():
                    mcp_server_path = project_root / 'mcp_server'
                if mcp_server_path.exists() and str(mcp_server_path) not in sys.path:
                    sys.path.insert(0, str(mcp_server_path))
                from megamind_database_server import main as server_main
            logger.info("✓ Successfully imported megamind_database_server")
        except ImportError as e:
            logger.error(f"✗ Failed to import megamind_database_server: {e}")
            logger.error("Available modules in current directory:")
            for item in current_dir.iterdir():
                if item.suffix == '.py':
                    logger.error(f"  - {item.name}")
            raise
        
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