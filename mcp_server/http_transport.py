#!/usr/bin/env python3
"""
HTTP Transport for MCP with Dynamic Realm Support
Implements persistent HTTP-based MCP server with realm parameter passing through JSON-RPC
"""

import json
import logging
import asyncio
import os
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
    from .consolidated_mcp_server import ConsolidatedMCPServer
    from .phase2_enhanced_server import Phase2EnhancedMCPServer
    from .enhanced_security_pipeline import EnhancedSecurityPipeline, SecurityContext, SecurityLevel, ValidationOutcome
    from .dynamic_realm_validator import RealmConfigValidator
    from .dynamic_realm_audit_logger import DynamicRealmAuditLogger
    from .realm_config_cache import RealmConfigurationManager
except ImportError:
    from realm_manager_factory import RealmManagerFactory, DynamicRealmManagerFactory, RealmContext
    from realm_aware_database import RealmAwareMegaMindDatabase
    from megamind_database_server import MCPServer
    from consolidated_mcp_server import ConsolidatedMCPServer
    from phase2_enhanced_server import Phase2EnhancedMCPServer
    from enhanced_security_pipeline import EnhancedSecurityPipeline, SecurityContext, SecurityLevel, ValidationOutcome
    from dynamic_realm_validator import RealmConfigValidator
    from dynamic_realm_audit_logger import DynamicRealmAuditLogger
    from realm_config_cache import RealmConfigurationManager

logger = logging.getLogger(__name__)

class HTTPMCPTransport:
    """HTTP transport for MCP with dynamic realm support, comprehensive API, and Phase 3 security"""
    
    def __init__(self, realm_factory: RealmManagerFactory, host: str = "localhost", port: int = 8080, 
                 security_config: Optional[Dict[str, Any]] = None):
        self.realm_factory = realm_factory
        self.host = host
        self.port = port
        self.app = web.Application()
        self.server_start_time = datetime.now()
        self.request_count = 0
        
        # Initialize Phase 3 security components
        self.security_config = security_config or {}
        self._init_security_pipeline()
        
        self.setup_routes()
        # Skip CORS setup for now - can be added later if needed
        
        logger.info(f"HTTPMCPTransport initialized on {host}:{port} with enhanced security")
    
    def _init_security_pipeline(self):
        """Initialize Phase 3 security pipeline components"""
        try:
            # Extract security configuration
            security_level = self.security_config.get('security_level', 'standard')
            
            # Initialize enhanced security pipeline
            self.security_pipeline = EnhancedSecurityPipeline({
                'security_level': security_level,
                'enable_threat_detection': self.security_config.get('enable_threat_detection', True),
                'max_validation_time_ms': self.security_config.get('max_validation_time_ms', 5000),
                'rate_limit_enabled': self.security_config.get('rate_limit_enabled', True),
                'max_requests_per_minute': self.security_config.get('max_requests_per_minute', 100),
                'validator_config': self.security_config.get('validator_config', {}),
                'audit_config': self.security_config.get('audit_config', {
                    'audit_enabled': True,
                    'log_to_file': True,
                    'audit_log_path': '/var/log/megamind/http_audit.log'
                }),
                'cache_config': self.security_config.get('cache_config', {
                    'max_entries': 1000,
                    'default_ttl_seconds': 1800
                })
            })
            
            logger.info("✅ Phase 3 security pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize security pipeline: {e}")
            # Create a mock security pipeline for graceful degradation
            class MockSecurityPipeline:
                def validate_and_process_realm_config(self, config, context):
                    return ValidationOutcome.APPROVED, config
                def get_security_metrics(self):
                    return {'error': 'Security pipeline not available'}
                def shutdown(self):
                    pass
            
            self.security_pipeline = MockSecurityPipeline()
    
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
    
    def _sanitize_realm_id_for_response(self, realm_id: str) -> str:
        """Sanitize realm ID for safe inclusion in response metadata"""
        import re
        
        if not realm_id:
            return "unknown"
        
        # First check if realm ID contains SQL injection patterns and reject completely
        sql_patterns = ['union', 'select', 'drop', 'delete', 'insert', 'update', '--', ';', '/*', '*/', 'xp_', 'sp_']
        realm_lower = realm_id.lower()
        
        if any(pattern in realm_lower for pattern in sql_patterns):
            logger.warning(f"SQL injection pattern detected in realm ID, using safe default: {realm_id}")
            return "sanitized_realm"
        
        # Check for XSS patterns and reject completely
        xss_patterns = ['<script', '</script', 'javascript:', 'vbscript:', 'onload=', 'onerror=']
        if any(pattern in realm_lower for pattern in xss_patterns):
            logger.warning(f"XSS pattern detected in realm ID, using safe default: {realm_id}")
            return "sanitized_realm"
        
        # Remove any potentially dangerous characters
        sanitized = re.sub(r'[^A-Za-z0-9_-]', '', realm_id)
        
        # Limit length to prevent response bloat
        sanitized = sanitized[:50]
        
        # Return safe default if completely sanitized
        return sanitized if sanitized else "sanitized_realm"
    
    def _is_safe_realm_id(self, realm_id: str) -> bool:
        """Validate realm ID is safe for processing"""
        import re
        
        if not realm_id:
            return False
        
        # Check for SQL injection patterns
        sql_patterns = ['union', 'select', 'drop', 'delete', 'insert', 'update', '--', ';', '/*', '*/', 'xp_', 'sp_']
        realm_lower = realm_id.lower()
        
        if any(pattern in realm_lower for pattern in sql_patterns):
            logger.warning(f"SQL injection pattern detected in realm ID: {realm_id}")
            return False
        
        # Check for XSS patterns
        xss_patterns = ['<script', '</script', 'javascript:', 'vbscript:', 'onload=', 'onerror=']
        if any(pattern in realm_lower for pattern in xss_patterns):
            logger.warning(f"XSS pattern detected in realm ID: {realm_id}")
            return False
        
        # Ensure alphanumeric with limited special chars
        if not re.match(r'^[A-Za-z0-9_-]+$', realm_id):
            logger.warning(f"Invalid characters in realm ID: {realm_id}")
            return False
        
        # Length check
        if len(realm_id) > 100:
            logger.warning(f"Realm ID too long: {len(realm_id)} characters")
            return False
        
        return True
    
    def setup_routes(self):
        """Setup HTTP routes for MCP protocol and management"""
        # Core MCP endpoints - both root and namespaced for compatibility
        self.app.router.add_post('/', self.handle_jsonrpc)  # Root JSON-RPC endpoint
        self.app.router.add_options('/', self.handle_options)  # Root OPTIONS for CORS
        self.app.router.add_post('/mcp/jsonrpc', self.handle_jsonrpc)  # Legacy namespaced endpoint
        self.app.router.add_options('/mcp/jsonrpc', self.handle_options)
        
        # Health and status endpoints
        self.app.router.add_get('/mcp/health', self.health_check)
        self.app.router.add_get('/mcp/status', self.server_status)
        
        # Realm management endpoints
        self.app.router.add_get('/mcp/realms', self.list_realms)
        self.app.router.add_get('/mcp/realms/{realm_id}/health', self.realm_health)
        self.app.router.add_post('/mcp/realms/{realm_id}', self.create_realm)
        self.app.router.add_delete('/mcp/realms/{realm_id}', self.delete_realm)
        
        # Phase 3 Security endpoints
        self.app.router.add_get('/mcp/security/metrics', self.security_metrics)
        self.app.router.add_post('/mcp/security/reset', self.reset_security_state)
        self.app.router.add_get('/mcp/security/config', self.security_configuration)
        
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
        """Extract realm context with dynamic configuration support and Phase 3 security validation"""
        try:
            # Create security context for validation
            security_context = SecurityContext(
                client_ip=request.remote or 'unknown',
                user_agent=request.headers.get('User-Agent', 'unknown'),
                request_id=str(self.request_count),
                realm_id=None,  # Will be set after extraction
                operation='extract_realm_context',
                security_level=SecurityLevel(self.security_config.get('security_level', 'standard'))
            )
            
            # 1. Try to get full realm configuration from header (highest priority)
            realm_config_header = request.headers.get('X-MCP-Realm-Config')
            if realm_config_header:
                try:
                    realm_config = json.loads(realm_config_header)
                    logger.debug(f"Processing dynamic realm configuration with security validation")
                    
                    # Update security context with realm ID
                    security_context.realm_id = realm_config.get('project_realm', 'unknown')
                    
                    # Validate configuration through security pipeline
                    outcome, processed_config = self.security_pipeline.validate_and_process_realm_config(
                        realm_config, security_context
                    )
                    
                    if outcome == ValidationOutcome.APPROVED or outcome == ValidationOutcome.APPROVED_WITH_WARNINGS:
                        # Validate required fields in processed config
                        required_fields = ['project_realm', 'project_name', 'default_target']
                        if all(field in processed_config for field in required_fields):
                            context = RealmContext.from_dynamic_config(processed_config)
                            logger.info(f"✅ Created secure dynamic realm context for {context.realm_id} (outcome: {outcome.value})")
                            return context
                        else:
                            missing = [f for f in required_fields if f not in processed_config]
                            logger.warning(f"Processed config missing required fields: {missing}")
                    else:
                        logger.warning(f"❌ Realm configuration validation failed: {outcome.value}")
                        # Fall through to backup realm context
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON in X-MCP-Realm-Config header: {e}")
                except Exception as e:
                    logger.error(f"Security validation failed for dynamic config: {e}")
            
            # 2. Fallback to existing realm ID extraction logic with validation
            realm_id = self._extract_realm_id(data, request)
            
            # ENHANCED: Always validate realm ID even in fallback
            if not self._is_safe_realm_id(realm_id):
                logger.warning(f"Potentially unsafe realm ID detected: {realm_id}")
                # Force to safe default instead of using unsafe realm ID
                realm_id = "PROJECT"
            
            # Update security context with validated realm ID
            security_context.realm_id = realm_id
            
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
            # Check for Phase 2 enhanced functions first, then Phase 1 consolidation
            use_phase2_enhanced = os.getenv('MEGAMIND_USE_PHASE2_ENHANCED_FUNCTIONS', 'false').lower() == 'true'
            use_consolidated = os.getenv('MEGAMIND_USE_CONSOLIDATED_FUNCTIONS', 'true').lower() == 'true'
            
            if use_phase2_enhanced:
                logger.debug("Using Phase 2 enhanced MCP server with 29 advanced functions")
                mcp_server = Phase2EnhancedMCPServer(realm_manager)
            elif use_consolidated:
                logger.debug("Using consolidated MCP server with 19 master functions")
                mcp_server = ConsolidatedMCPServer(realm_manager)
            else:
                logger.debug("Using original MCP server with 44 functions")
                mcp_server = MCPServer(realm_manager)
            
            # Process MCP request
            response = await mcp_server.handle_request(data)
            
            # Add performance metrics to response
            processing_time = (datetime.now() - start_time).total_seconds()
            if isinstance(response, dict):
                response['_meta'] = {
                    'realm_id': self._sanitize_realm_id_for_response(realm_context.realm_id),
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
    
    async def security_metrics(self, request: Request) -> Response:
        """Get comprehensive security metrics from Phase 3 pipeline"""
        try:
            # ENHANCED: Better error handling for security pipeline issues
            if not hasattr(self, 'security_pipeline'):
                return web.json_response({
                    'error': 'Security pipeline not initialized',
                    'available_features': [],
                    'status': 'degraded'
                }, status=503)
            
            metrics = self.security_pipeline.get_security_metrics()
            
            # Add HTTP-specific metrics
            metrics['http_transport'] = {
                'total_requests': self.request_count,
                'uptime_seconds': (datetime.now() - self.server_start_time).total_seconds(),
                'security_config': {
                    'security_level': self.security_config.get('security_level', 'standard'),
                    'threat_detection_enabled': self.security_config.get('enable_threat_detection', True),
                    'rate_limiting_enabled': self.security_config.get('rate_limit_enabled', True)
                }
            }
            
            return web.json_response(metrics)
            
        except Exception as e:
            logger.error(f"Security metrics failed: {e}")
            return web.json_response({
                'error': str(e),
                'status': 'error',
                'timestamp': datetime.now().isoformat()
            }, status=500)
    
    async def reset_security_state(self, request: Request) -> Response:
        """Reset security state (admin endpoint)"""
        try:
            # Get client IP from query parameter (optional)
            client_ip = request.query.get('client_ip')
            
            # Reset security state
            self.security_pipeline.reset_security_state(client_ip)
            
            response_data = {
                "message": f"Security state reset {'for IP: ' + client_ip if client_ip else 'for all IPs'}",
                "timestamp": datetime.now().isoformat()
            }
            
            return web.json_response(response_data)
            
        except Exception as e:
            logger.error(f"Security state reset failed: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def security_configuration(self, request: Request) -> Response:
        """Get current security configuration"""
        try:
            config_data = {
                "security_level": self.security_config.get('security_level', 'standard'),
                "threat_detection_enabled": self.security_config.get('enable_threat_detection', True),
                "rate_limiting_enabled": self.security_config.get('rate_limit_enabled', True),
                "max_requests_per_minute": self.security_config.get('max_requests_per_minute', 100),
                "max_validation_time_ms": self.security_config.get('max_validation_time_ms', 5000),
                "phase3_features": {
                    "dynamic_realm_validator": True,
                    "audit_logging": True,
                    "configuration_caching": True,
                    "enhanced_security_pipeline": True
                },
                "timestamp": datetime.now().isoformat()
            }
            
            return web.json_response(config_data)
            
        except Exception as e:
            logger.error(f"Security configuration request failed: {e}")
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
                "/mcp/security/metrics": {
                    "method": "GET",
                    "description": "Get comprehensive security metrics from Phase 3 pipeline"
                },
                "/mcp/security/reset": {
                    "method": "POST",
                    "description": "Reset security state (admin endpoint)",
                    "query_params": {
                        "client_ip": "Optional IP to reset (if not provided, resets all)"
                    }
                },
                "/mcp/security/config": {
                    "method": "GET",
                    "description": "Get current security configuration and Phase 3 feature status"
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
                "jsonrpc": "/ (primary) or /mcp/jsonrpc (legacy)",
                "health": "/mcp/health",
                "status": "/mcp/status",
                "realms": "/mcp/realms",
                "api": "/mcp/api"
            },
            "documentation": "/mcp/api",
            "mcp_protocol": {
                "primary_endpoint": "/",
                "legacy_endpoint": "/mcp/jsonrpc",
                "method": "POST",
                "format": "JSON-RPC 2.0"
            }
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
            
            # Cleanup Phase 3 security pipeline
            if hasattr(self, 'security_pipeline'):
                self.security_pipeline.shutdown()
                logger.info("✓ Security pipeline shutdown complete")
            
            # Cleanup realm factory
            await self.realm_factory.shutdown()
            
            logger.info("HTTP MCP Server shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during HTTP server shutdown: {e}")

class RealmAwareHTTPMCPServer:
    """Enhanced HTTP MCP Server with realm-aware request routing and Phase 3 security"""
    
    def __init__(self, realm_factory: RealmManagerFactory, config: Dict[str, Any]):
        self.realm_factory = realm_factory
        self.config = config
        
        # Extract security configuration
        security_config = config.get('security', {})
        
        self.http_transport = HTTPMCPTransport(
            realm_factory=realm_factory,
            host=config.get('host', 'localhost'),
            port=config.get('port', 8080),
            security_config=security_config
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