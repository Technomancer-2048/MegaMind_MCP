#!/usr/bin/env python3
"""
HTTP Transport for MCP with Dynamic Realm Support
Implements persistent HTTP-based MCP server with realm parameter passing through JSON-RPC
"""

import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List
from aiohttp import web, ClientSession, WSMsgType
from aiohttp.web_request import Request
from aiohttp.web_response import Response
import aiohttp_cors

try:
    from .realm_manager_factory import RealmManagerFactory, DynamicRealmManagerFactory, RealmContext
    from .realm_aware_database import RealmAwareMegaMindDatabase
    from .megamind_database_server import MCPServer
except ImportError:
    from realm_manager_factory import RealmManagerFactory, DynamicRealmManagerFactory, RealmContext
    from realm_aware_database import RealmAwareMegaMindDatabase
    from megamind_database_server import MCPServer

logger = logging.getLogger(__name__)

class HTTPMCPTransport:
    """HTTP transport for MCP with dynamic realm support and comprehensive API"""
    
    def __init__(self, realm_factory: RealmManagerFactory, host: str = "localhost", port: int = 8080):
        self.realm_factory = realm_factory
        self.host = host
        self.port = port
        self.app = web.Application()
        self.server_start_time = datetime.now()
        self.request_count = 0
        self.setup_routes()
        # Skip CORS setup for now - can be added later if needed
        
        logger.info(f"HTTPMCPTransport initialized on {host}:{port}")
    
    def setup_cors(self):
        """Setup CORS for cross-origin requests"""
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*"
            )
        })
        
        # Add CORS to all routes except OPTIONS handlers
        for route in list(self.app.router.routes()):
            if route.method != 'OPTIONS':
                cors.add(route)
    
    def setup_routes(self):
        """Setup HTTP routes for MCP protocol and management"""
        # Core MCP endpoints
        self.app.router.add_post('/mcp/jsonrpc', self.handle_jsonrpc)
        self.app.router.add_options('/mcp/jsonrpc', self.handle_options)
        
        # Health and status endpoints
        self.app.router.add_get('/mcp/health', self.health_check)
        self.app.router.add_get('/mcp/status', self.server_status)
        
        # Realm management endpoints
        self.app.router.add_get('/mcp/realms', self.list_realms)
        self.app.router.add_get('/mcp/realms/{realm_id}/health', self.realm_health)
        self.app.router.add_post('/mcp/realms/{realm_id}', self.create_realm)
        self.app.router.add_delete('/mcp/realms/{realm_id}', self.delete_realm)
        
        # API documentation endpoint
        self.app.router.add_get('/mcp/api', self.api_documentation)
        
        # Root endpoint
        self.app.router.add_get('/', self.root_endpoint)
        
        logger.info("HTTP routes configured successfully")
    
    async def handle_options(self, request: Request) -> Response:
        """Handle CORS preflight requests"""
        return web.Response(
            status=200,
            headers={
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type, Authorization',
                'Access-Control-Max-Age': '3600'
            }
        )
    
    def extract_realm_context(self, data: Dict[str, Any], request: Request) -> RealmContext:
        """Extract realm context with dynamic configuration support"""
        try:
            # 1. Try to get full realm configuration from header (highest priority)
            realm_config_header = request.headers.get('X-MCP-Realm-Config')
            if realm_config_header:
                try:
                    realm_config = json.loads(realm_config_header)
                    logger.debug(f"Using dynamic realm configuration: {realm_config}")
                    
                    # Validate required fields
                    required_fields = ['project_realm', 'project_name', 'default_target']
                    if all(field in realm_config for field in required_fields):
                        context = RealmContext.from_dynamic_config(realm_config)
                        logger.info(f"✅ Created dynamic realm context for {context.realm_id}")
                        return context
                    else:
                        missing = [f for f in required_fields if f not in realm_config]
                        logger.warning(f"Incomplete realm configuration in header, missing: {missing}")
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON in X-MCP-Realm-Config header: {e}")
            
            # 2. Fallback to existing realm ID extraction logic
            realm_id = self._extract_realm_id(data, request)
            
            # 3. Create minimal context for backward compatibility
            context = RealmContext(
                realm_id=realm_id,
                project_name=f"HTTP Project {realm_id}",
                default_target="PROJECT"
            )
            
            logger.debug(f"Using fallback realm context: {context.to_dict()}")
            return context
            
        except Exception as e:
            logger.error(f"Failed to extract realm context: {e}")
            return self._get_default_realm_context()

    def _extract_realm_id(self, data: Dict[str, Any], request: Request) -> str:
        """Extract realm ID using existing priority logic"""
        realm_id = None
        
        # 1. Check tool arguments first (highest priority)
        if 'params' in data and 'arguments' in data['params']:
            realm_id = data['params']['arguments'].get('realm_id')
        
        # 2. Check JSON-RPC params
        if not realm_id and 'params' in data:
            realm_id = data['params'].get('realm_id')
        
        # 3. Check HTTP headers
        if not realm_id:
            realm_id = request.headers.get('X-MCP-Realm-ID')
        
        # 4. Check query parameters
        if not realm_id:
            realm_id = request.query.get('realm_id')
        
        # Use default if no realm specified
        return realm_id or "PROJECT"
    
    def _get_default_realm_context(self) -> RealmContext:
        """Get safe default realm context"""
        return RealmContext(
            realm_id="PROJECT",
            project_name="Default HTTP Project", 
            default_target="PROJECT"
        )
    
    async def handle_jsonrpc(self, request: Request) -> Response:
        """Handle JSON-RPC requests over HTTP with realm context"""
        self.request_count += 1
        start_time = datetime.now()
        
        try:
            # Parse JSON request
            data = await request.json()
            logger.debug(f"Received JSON-RPC request: {data.get('method', 'unknown')}")
            
            # Extract realm context from request
            realm_context = self.extract_realm_context(data, request)
            
            # Create realm manager with dynamic configuration if available
            has_dynamic_config = request.headers.get('X-MCP-Realm-Config') is not None
            if has_dynamic_config:
                # Use dynamic realm manager creation
                realm_manager = await self.realm_factory.create_dynamic_realm_manager(realm_context)
                logger.debug(f"Using dynamic realm manager for {realm_context.realm_id}")
            else:
                # Fallback to static realm manager
                realm_manager = await self.realm_factory.get_realm_manager(realm_context.realm_id)
                logger.debug(f"Using static realm manager for {realm_context.realm_id}")
            
            # Create MCP server instance for this request
            mcp_server = MCPServer(realm_manager)
            
            # Process MCP request
            response = await mcp_server.handle_request(data)
            
            # Add performance metrics to response
            processing_time = (datetime.now() - start_time).total_seconds()
            if isinstance(response, dict):
                response['_meta'] = {
                    'realm_id': realm_context.realm_id,
                    'processing_time_ms': round(processing_time * 1000, 2),
                    'server_request_count': self.request_count
                }
            
            logger.debug(f"JSON-RPC request processed in {processing_time:.3f}s for realm {realm_context.realm_id}")
            
            # Clean any Decimal objects from the entire response before JSON serialization
            from decimal import Decimal
            def clean_response_decimals(obj):
                if isinstance(obj, dict):
                    return {k: clean_response_decimals(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [clean_response_decimals(item) for item in obj]
                elif isinstance(obj, Decimal):
                    return float(obj)
                elif hasattr(obj, 'isoformat'):
                    return obj.isoformat()
                return obj
            
            clean_response = clean_response_decimals(response)
            
            return web.json_response(clean_response, headers={
                'X-MCP-Realm-ID': realm_context.realm_id,
                'X-Processing-Time-Ms': str(round(processing_time * 1000, 2))
            })
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in request: {e}")
            error_response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32700,
                    "message": "Parse error: Invalid JSON"
                }
            }
            return web.json_response(error_response, status=400)
            
        except Exception as e:
            logger.error(f"HTTP JSON-RPC request failed: {e}")
            import traceback
            traceback.print_exc()
            
            error_response = {
                "jsonrpc": "2.0",
                "id": data.get('id') if 'data' in locals() else None,
                "error": {
                    "code": -32603,
                    "message": f"Internal server error: {str(e)}"
                }
            }
            return web.json_response(error_response, status=500)
    
    async def health_check(self, request: Request) -> Response:
        """Basic health check endpoint"""
        try:
            # Test realm factory health
            await self.realm_factory.initialize_shared_services()
            
            # Get basic health info
            uptime = (datetime.now() - self.server_start_time).total_seconds()
            
            health_data = {
                "status": "healthy",
                "uptime_seconds": round(uptime, 2),
                "request_count": self.request_count,
                "realm_factory_initialized": self.realm_factory.initialized,
                "timestamp": datetime.now().isoformat()
            }
            
            return web.json_response(health_data)
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            error_data = {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            return web.json_response(error_data, status=503)
    
    async def server_status(self, request: Request) -> Response:
        """Detailed server status endpoint"""
        try:
            uptime = (datetime.now() - self.server_start_time).total_seconds()
            
            status_data = {
                "server": {
                    "name": "MegaMind MCP HTTP Server",
                    "version": "2.0.0-http",
                    "host": self.host,
                    "port": self.port,
                    "uptime_seconds": round(uptime, 2),
                    "start_time": self.server_start_time.isoformat()
                },
                "metrics": {
                    "total_requests": self.request_count,
                    "requests_per_minute": round(self.request_count / max(uptime / 60, 1), 2) if uptime > 0 else 0
                },
                "realm_factory": {
                    "initialized": self.realm_factory.initialized,
                    "active_realms": len(self.realm_factory.realm_managers),
                    "realm_list": list(self.realm_factory.realm_managers.keys())
                },
                "timestamp": datetime.now().isoformat()
            }
            
            return web.json_response(status_data)
            
        except Exception as e:
            logger.error(f"Status check failed: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def list_realms(self, request: Request) -> Response:
        """List all available realms and their status"""
        try:
            realms_info = self.realm_factory.list_realms()
            
            response_data = {
                "realms": realms_info,
                "total_count": len(realms_info),
                "timestamp": datetime.now().isoformat()
            }
            
            return web.json_response(response_data)
            
        except Exception as e:
            logger.error(f"List realms failed: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def realm_health(self, request: Request) -> Response:
        """Check health of specific realm"""
        try:
            realm_id = request.match_info['realm_id']
            health_status = await self.realm_factory.check_realm_health(realm_id)
            
            status_code = 200 if health_status.get('healthy', False) else 503
            return web.json_response(health_status, status=status_code)
            
        except Exception as e:
            logger.error(f"Realm health check failed: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def create_realm(self, request: Request) -> Response:
        """Create a new realm (if using DynamicRealmManagerFactory)"""
        try:
            realm_id = request.match_info['realm_id']
            
            if not isinstance(self.realm_factory, DynamicRealmManagerFactory):
                return web.json_response({
                    "error": "Dynamic realm creation not supported with current factory"
                }, status=501)
            
            # Parse realm configuration from request body
            try:
                realm_config = await request.json()
            except json.JSONDecodeError:
                realm_config = {}
            
            # Create the realm
            success = await self.realm_factory.create_realm(realm_id, realm_config)
            
            if success:
                return web.json_response({
                    "message": f"Realm {realm_id} created successfully",
                    "realm_id": realm_id,
                    "config": realm_config
                }, status=201)
            else:
                return web.json_response({
                    "error": f"Failed to create realm {realm_id}"
                }, status=500)
                
        except Exception as e:
            logger.error(f"Create realm failed: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def delete_realm(self, request: Request) -> Response:
        """Delete/cleanup a realm"""
        try:
            realm_id = request.match_info['realm_id']
            
            # Don't allow deletion of default realms
            if realm_id in ['PROJECT', 'GLOBAL']:
                return web.json_response({
                    "error": f"Cannot delete system realm: {realm_id}"
                }, status=403)
            
            success = await self.realm_factory.cleanup_realm(realm_id)
            
            if success:
                return web.json_response({
                    "message": f"Realm {realm_id} deleted successfully"
                })
            else:
                return web.json_response({
                    "error": f"Realm {realm_id} not found or failed to delete"
                }, status=404)
                
        except Exception as e:
            logger.error(f"Delete realm failed: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def api_documentation(self, request: Request) -> Response:
        """API documentation endpoint"""
        documentation = {
            "name": "MegaMind MCP HTTP Server API",
            "version": "2.0.0",
            "description": "HTTP transport for MCP with dynamic realm support",
            "endpoints": {
                "/mcp/jsonrpc": {
                    "method": "POST",
                    "description": "Main JSON-RPC endpoint for MCP protocol",
                    "headers": {
                        "Content-Type": "application/json",
                        "X-MCP-Realm-ID": "Optional realm identifier"
                    },
                    "query_params": {
                        "realm_id": "Optional realm identifier"
                    }
                },
                "/mcp/health": {
                    "method": "GET", 
                    "description": "Basic health check"
                },
                "/mcp/status": {
                    "method": "GET",
                    "description": "Detailed server status and metrics"
                },
                "/mcp/realms": {
                    "method": "GET",
                    "description": "List all available realms"
                },
                "/mcp/realms/{realm_id}/health": {
                    "method": "GET",
                    "description": "Check health of specific realm"
                },
                "/mcp/realms/{realm_id}": {
                    "method": "POST",
                    "description": "Create new realm (dynamic factory only)"
                },
                "/mcp/api": {
                    "method": "GET",
                    "description": "This API documentation"
                }
            },
            "realm_parameter_sources": [
                "tool_arguments.realm_id (highest priority)",
                "params.realm_id",
                "headers['X-MCP-Realm-ID']",
                "query_params.realm_id",
                "default: PROJECT"
            ]
        }
        
        return web.json_response(documentation)
    
    async def root_endpoint(self, request: Request) -> Response:
        """Root endpoint with basic server info"""
        return web.json_response({
            "name": "MegaMind MCP HTTP Server",
            "version": "2.0.0-http", 
            "status": "running",
            "endpoints": {
                "jsonrpc": "/mcp/jsonrpc",
                "health": "/mcp/health",
                "status": "/mcp/status",
                "realms": "/mcp/realms",
                "api": "/mcp/api"
            },
            "documentation": "/mcp/api"
        })
    
    async def start_server(self):
        """Start the HTTP server"""
        try:
            logger.info(f"=== Starting HTTP MCP Server ===")
            logger.info(f"Target binding: {self.host}:{self.port}")
            
            # Initialize shared services (this is where database connections are made)
            logger.info("Initializing shared services (database connections)...")
            await self.realm_factory.initialize_shared_services()
            logger.info("✓ Shared services initialized successfully")
            
            # Create and start server
            logger.info("Creating web application runner...")
            runner = web.AppRunner(self.app)
            await runner.setup()
            logger.info("✓ Web application runner created")
            
            logger.info(f"Creating TCP site for binding to {self.host}:{self.port}...")
            site = web.TCPSite(runner, host=self.host, port=self.port)
            await site.start()
            logger.info("✓ TCP site started successfully")
            
            logger.info(f"🚀 HTTP MCP Server started successfully on http://{self.host}:{self.port}")
            logger.info(f"📚 API documentation available at: http://{self.host}:{self.port}/mcp/api")
            logger.info(f"❤️  Health check available at: http://{self.host}:{self.port}/mcp/health")
            
            return runner
            
        except Exception as e:
            logger.error(f"✗ Failed to start HTTP server: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise
    
    async def shutdown(self):
        """Shutdown the HTTP server and cleanup resources"""
        try:
            logger.info("Shutting down HTTP MCP Server...")
            
            # Cleanup realm factory
            await self.realm_factory.shutdown()
            
            logger.info("HTTP MCP Server shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during HTTP server shutdown: {e}")

class RealmAwareHTTPMCPServer:
    """Enhanced HTTP MCP Server with realm-aware request routing"""
    
    def __init__(self, realm_factory: RealmManagerFactory, config: Dict[str, Any]):
        self.realm_factory = realm_factory
        self.config = config
        self.http_transport = HTTPMCPTransport(
            realm_factory=realm_factory,
            host=config.get('host', 'localhost'),
            port=config.get('port', 8080)
        )
        
    async def run(self):
        """Run the HTTP server indefinitely"""
        try:
            runner = await self.http_transport.start_server()
            
            # Keep server running
            while True:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        except Exception as e:
            logger.error(f"Server error: {e}")
        finally:
            await self.http_transport.shutdown()
            if 'runner' in locals():
                await runner.cleanup()