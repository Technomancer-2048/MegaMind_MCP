#!/usr/bin/env python3
"""
HTTP MCP Server Entry Point
Simple entry point for running the MCP server in HTTP mode
"""

import os
import sys
import asyncio
import logging
from typing import Dict, Any

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from transport_manager import main_transport_server
    from megamind_database_server import setup_environment_paths, load_config
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_http_config() -> Dict[str, Any]:
    """Load configuration for HTTP server"""
    # Setup environment paths
    path_config = setup_environment_paths()
    
    # Load database configuration
    db_config = load_config()
    
    # HTTP server specific configuration
    # Store HTTP server settings before merging database config
    http_host = os.getenv('MCP_HOST', '0.0.0.0')
    http_port = int(os.getenv('MCP_PORT', 8080))
    
    http_config = {
        # Database configuration (with db_ prefix to avoid conflicts)
        'db_host': db_config['host'],
        'db_port': db_config['port'], 
        'db_database': db_config['database'],
        'db_user': db_config['user'],
        'db_password': db_config['password'],
        'db_pool_size': db_config['pool_size'],
        
        # Path configuration
        **path_config,
        
        # Transport settings (MUST be after db_config to avoid override)
        'transport': 'http',
        'host': http_host,
        'port': http_port,
        
        # Realm management
        'realm_factory_type': os.getenv('MCP_REALM_FACTORY', 'dynamic'),
        'default_realm': os.getenv('MCP_DEFAULT_REALM', 'PROJECT'),
        
        # Enhanced features
        'enhanced_monitoring': os.getenv('MCP_ENHANCED_MONITORING', 'true').lower() == 'true',
    }
    
    return http_config

async def main():
    """Main entry point for HTTP MCP server"""
    try:
        logger.info("Starting MegaMind MCP HTTP Server...")
        
        # Load configuration
        config = load_http_config()
        
        # Validate required configuration
        if not config.get('db_password'):
            logger.error("Database password not configured. Set MEGAMIND_DB_PASSWORD environment variable.")
            return 1
        
        logger.info(f"HTTP Server Configuration:")
        logger.info(f"  HTTP Host: {config['host']}")
        logger.info(f"  HTTP Port: {config['port']}")
        logger.info(f"  Realm Factory: {config['realm_factory_type']}")
        logger.info(f"  Default Realm: {config['default_realm']}")
        logger.info(f"  Enhanced Monitoring: {config['enhanced_monitoring']}")
        
        # Log database configuration separately
        db_config = load_config()
        logger.info(f"Database Configuration:")
        logger.info(f"  DB Host: {db_config['host']}")
        logger.info(f"  DB Port: {db_config['port']}")
        logger.info(f"  DB Name: {db_config['database']}")
        logger.info(f"  DB User: {db_config['user']}")
        logger.info(f"  DB Pool Size: {db_config['pool_size']}")
        
        # Start the transport server
        await main_transport_server(config)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("HTTP Server stopped by user")
        return 0
    except Exception as e:
        logger.error(f"HTTP Server failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))