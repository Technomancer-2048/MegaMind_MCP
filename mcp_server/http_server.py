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
    http_config = {
        # Transport settings
        'transport': 'http',
        'host': os.getenv('MCP_HOST', '0.0.0.0'),
        'port': int(os.getenv('MCP_PORT', 8080)),
        
        # Realm management
        'realm_factory_type': os.getenv('MCP_REALM_FACTORY', 'dynamic'),
        'default_realm': os.getenv('MCP_DEFAULT_REALM', 'PROJECT'),
        
        # Enhanced features
        'enhanced_monitoring': os.getenv('MCP_ENHANCED_MONITORING', 'true').lower() == 'true',
        
        # Database configuration (merge from load_config)
        **db_config,
        
        # Path configuration
        **path_config
    }
    
    return http_config

async def main():
    """Main entry point for HTTP MCP server"""
    try:
        logger.info("Starting MegaMind MCP HTTP Server...")
        
        # Load configuration
        config = load_http_config()
        
        # Validate required configuration
        if not config.get('password'):
            logger.error("Database password not configured. Set MEGAMIND_DB_PASSWORD environment variable.")
            return 1
        
        logger.info(f"HTTP Server Configuration:")
        logger.info(f"  Host: {config['host']}")
        logger.info(f"  Port: {config['port']}")
        logger.info(f"  Realm Factory: {config['realm_factory_type']}")
        logger.info(f"  Default Realm: {config['default_realm']}")
        logger.info(f"  Enhanced Monitoring: {config['enhanced_monitoring']}")
        logger.info(f"  Database Host: {config['host']}")
        
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