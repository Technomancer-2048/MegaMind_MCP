#!/usr/bin/env python3
"""
Transport Manager for MCP Server
Manages both stdio and HTTP transports with dynamic switching capabilities
"""

import os
import sys
import json
import logging
import asyncio
from typing import Dict, Any, Optional, Union
from enum import Enum

try:
    from .http_transport import HTTPMCPTransport, RealmAwareHTTPMCPServer
    from .realm_manager_factory import RealmManagerFactory, DynamicRealmManagerFactory
    from .realm_aware_database import RealmAwareMegaMindDatabase
    from .megamind_database_server import MCPServer
except ImportError:
    from http_transport import HTTPMCPTransport, RealmAwareHTTPMCPServer
    from realm_manager_factory import RealmManagerFactory, DynamicRealmManagerFactory
    from realm_aware_database import RealmAwareMegaMindDatabase
    from megamind_database_server import MCPServer

logger = logging.getLogger(__name__)

class TransportType(Enum):
    """Supported transport types"""
    STDIO = "stdio"
    HTTP = "http"
    AUTO = "auto"

class TransportManager:
    """Manages both stdio and HTTP transports for MCP server"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.transport_type = self._determine_transport_type()
        self.realm_factory = None
        self.http_server = None
        self.stdio_server = None
        
        logger.info(f"TransportManager initialized with transport: {self.transport_type.value}")
    
    def _determine_transport_type(self) -> TransportType:
        """Determine transport type from configuration"""
        transport_config = self.config.get('transport', 'auto').lower()
        
        if transport_config == 'http':
            return TransportType.HTTP
        elif transport_config == 'stdio':
            return TransportType.STDIO
        elif transport_config == 'auto':
            # Auto-detect based on environment
            if os.getenv('MCP_TRANSPORT') == 'http':
                return TransportType.HTTP
            elif sys.stdin.isatty():
                # Interactive terminal, prefer HTTP for development
                return TransportType.HTTP
            else:
                # Non-interactive (pipe/subprocess), use stdio
                return TransportType.STDIO
        else:
            logger.warning(f"Unknown transport type '{transport_config}', defaulting to stdio")
            return TransportType.STDIO
    
    async def initialize_realm_factory(self):
        """Initialize the realm factory based on configuration"""
        try:
            # Determine factory type
            factory_type = self.config.get('realm_factory_type', 'standard')
            
            if factory_type == 'dynamic':
                logger.info("Using DynamicRealmManagerFactory")
                self.realm_factory = DynamicRealmManagerFactory(self.config)
            else:
                logger.info("Using standard RealmManagerFactory")
                self.realm_factory = RealmManagerFactory(self.config)
            
            # Initialize shared services
            await self.realm_factory.initialize_shared_services()
            
            logger.info("Realm factory initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize realm factory: {e}")
            raise
    
    async def start_server(self):
        """Start server with configured transport"""
        try:
            # Initialize realm factory first
            await self.initialize_realm_factory()
            
            if self.transport_type == TransportType.HTTP:
                return await self.start_http_server()
            elif self.transport_type == TransportType.STDIO:
                return await self.start_stdio_server()
            else:
                raise ValueError(f"Unsupported transport type: {self.transport_type}")
                
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            raise
    
    async def start_http_server(self):
        """Start HTTP server"""
        try:
            logger.info("Starting HTTP transport...")
            
            self.http_server = RealmAwareHTTPMCPServer(
                realm_factory=self.realm_factory,
                config=self.config
            )
            
            # Start server in background task
            server_task = asyncio.create_task(self.http_server.run())
            
            logger.info(f"HTTP MCP Server started on {self.config.get('host', 'localhost')}:{self.config.get('port', 8080)}")
            
            return server_task
            
        except Exception as e:
            logger.error(f"Failed to start HTTP server: {e}")
            raise
    
    async def start_stdio_server(self):
        """Start stdio server (traditional MCP)"""
        try:
            logger.info("Starting stdio transport...")
            
            # Create default realm manager for stdio mode
            default_realm_manager = await self.realm_factory.get_default_realm_manager()
            
            # Create MCP server
            self.stdio_server = MCPServer(default_realm_manager)
            
            # Start stdio server
            await self.stdio_server.run()
            
            logger.info("Stdio MCP Server completed")
            
        except Exception as e:
            logger.error(f"Failed to start stdio server: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown all transports and cleanup resources"""
        try:
            logger.info("Shutting down TransportManager...")
            
            if self.http_server:
                await self.http_server.http_transport.shutdown()
            
            if self.realm_factory:
                await self.realm_factory.shutdown()
            
            logger.info("TransportManager shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during TransportManager shutdown: {e}")
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get information about the current server configuration"""
        return {
            "transport_type": self.transport_type.value,
            "config": {
                "host": self.config.get('host', 'localhost'),
                "port": self.config.get('port', 8080),
                "realm_factory_type": self.config.get('realm_factory_type', 'standard'),
                "default_realm": self.config.get('default_realm', 'PROJECT')
            },
            "realm_factory_initialized": self.realm_factory is not None and self.realm_factory.initialized if self.realm_factory else False,
            "active_realms": len(self.realm_factory.realm_managers) if self.realm_factory else 0
        }

class EnhancedTransportManager(TransportManager):
    """Enhanced transport manager with additional features"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.metrics = {
            'start_time': None,
            'requests_processed': 0,
            'errors_encountered': 0,
            'active_connections': 0
        }
    
    async def start_server_with_monitoring(self):
        """Start server with enhanced monitoring and recovery"""
        from datetime import datetime
        
        self.metrics['start_time'] = datetime.now()
        
        try:
            # Start server with monitoring
            server_task = await self.start_server()
            
            # If HTTP server, add monitoring
            if self.transport_type == TransportType.HTTP and self.http_server:
                # Monitor server health periodically
                monitor_task = asyncio.create_task(self._monitor_server_health())
                
                # Wait for either server completion or monitoring failure
                done, pending = await asyncio.wait(
                    [server_task, monitor_task],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Cleanup pending tasks
                for task in pending:
                    task.cancel()
                    
                return server_task
            else:
                return server_task
                
        except Exception as e:
            self.metrics['errors_encountered'] += 1
            logger.error(f"Enhanced server start failed: {e}")
            raise
    
    async def _monitor_server_health(self):
        """Monitor server health and log metrics periodically"""
        try:
            while True:
                await asyncio.sleep(60)  # Check every minute
                
                if self.realm_factory:
                    realm_count = len(self.realm_factory.realm_managers)
                    logger.info(f"Server health check: {realm_count} active realms")
                    
                    # Check individual realm health
                    for realm_id in list(self.realm_factory.realm_managers.keys()):
                        try:
                            health = await self.realm_factory.check_realm_health(realm_id)
                            if not health.get('healthy', False):
                                logger.warning(f"Realm {realm_id} health issue: {health}")
                        except Exception as e:
                            logger.error(f"Health check failed for realm {realm_id}: {e}")
                
        except asyncio.CancelledError:
            logger.info("Server health monitoring stopped")
        except Exception as e:
            logger.error(f"Server health monitoring error: {e}")
    
    def get_enhanced_server_info(self) -> Dict[str, Any]:
        """Get enhanced server information including metrics"""
        base_info = self.get_server_info()
        base_info['metrics'] = self.metrics.copy()
        
        if self.metrics['start_time']:
            from datetime import datetime
            uptime = (datetime.now() - self.metrics['start_time']).total_seconds()
            base_info['metrics']['uptime_seconds'] = round(uptime, 2)
        
        return base_info

async def create_transport_manager(config: Dict[str, Any]) -> TransportManager:
    """Factory function to create transport manager"""
    enhanced = config.get('enhanced_monitoring', False)
    
    if enhanced:
        return EnhancedTransportManager(config)
    else:
        return TransportManager(config)

async def main_transport_server(config: Dict[str, Any]):
    """Main entry point for transport server"""
    transport_manager = None
    
    try:
        # Create transport manager
        transport_manager = await create_transport_manager(config)
        
        # Start server
        if hasattr(transport_manager, 'start_server_with_monitoring'):
            await transport_manager.start_server_with_monitoring()
        else:
            await transport_manager.start_server()
            
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Transport server failed: {e}")
        raise
    finally:
        if transport_manager:
            await transport_manager.shutdown()

if __name__ == "__main__":
    # Example configuration
    config = {
        'transport': os.getenv('MCP_TRANSPORT', 'auto'),
        'host': os.getenv('MCP_HOST', 'localhost'),
        'port': int(os.getenv('MCP_PORT', 8080)),
        'realm_factory_type': os.getenv('MCP_REALM_FACTORY', 'standard'),
        'default_realm': os.getenv('MCP_DEFAULT_REALM', 'PROJECT'),
        'enhanced_monitoring': os.getenv('MCP_ENHANCED_MONITORING', 'false').lower() == 'true',
        
        # Database configuration
        'host': os.getenv('MEGAMIND_DB_HOST', '10.255.250.21'),
        'port': os.getenv('MEGAMIND_DB_PORT', '3309'),
        'database': os.getenv('MEGAMIND_DB_NAME', 'megamind_database'),
        'user': os.getenv('MEGAMIND_DB_USER', 'megamind_user'),
        'password': os.getenv('MEGAMIND_DB_PASSWORD', ''),
        'pool_size': os.getenv('CONNECTION_POOL_SIZE', '10')
    }
    
    # Run the transport server
    asyncio.run(main_transport_server(config))