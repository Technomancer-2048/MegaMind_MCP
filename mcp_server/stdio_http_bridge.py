#!/usr/bin/env python3
"""
Optimized STDIO-to-HTTP Bridge for MegaMind MCP Server
Non-blocking initialization with proper environment variable handling
"""

import json
import sys
import os
import asyncio
import logging
import urllib.request
import urllib.parse
import urllib.error
from typing import Dict, Any, Optional

# Configure logging level from environment
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)

logger = logging.getLogger(__name__)

class OptimizedSTDIOHttpBridge:
    """STDIO-to-HTTP bridge with non-blocking initialization and environment config"""
    
    def __init__(self):
        # Load configuration from environment variables
        self.http_endpoint = os.getenv('MEGAMIND_HTTP_ENDPOINT', 'http://10.255.250.22:8080/mcp/jsonrpc')
        self.project_realm = os.getenv('MEGAMIND_PROJECT_REALM', 'MegaMind_MCP')
        self.project_name = os.getenv('MEGAMIND_PROJECT_NAME', 'MegaMind Context Database')
        self.default_target = os.getenv('MEGAMIND_DEFAULT_TARGET', 'PROJECT')
        self.mcp_timeout = int(os.getenv('MCP_TIMEOUT', '30'))
        
        # Create comprehensive realm configuration for dynamic handling
        realm_config = {
            'project_realm': self.project_realm,
            'project_name': self.project_name,
            'default_target': self.default_target,
            'global_realm': 'GLOBAL',
            'cross_realm_search_enabled': True,
            'project_priority_weight': 1.2,
            'global_priority_weight': 1.0
        }
        
        self.request_headers = {
            'Content-Type': 'application/json',
            'X-MCP-Realm-ID': self.project_realm,
            'X-MCP-Project-Name': self.project_name,
            'X-MCP-Realm-Config': json.dumps(realm_config)
        }
        self.tools_cache = None  # Will be populated lazily
        self.initialized = False
        
        # Validate configuration on initialization
        if not self.validate_realm_config():
            logger.error("Bridge initialization failed due to invalid realm configuration")
            sys.exit(1)
            
        logger.info(f"Bridge configured: realm={self.project_realm}, timeout={self.mcp_timeout}s")
        logger.debug(f"Realm config header: {self.request_headers['X-MCP-Realm-Config']}")
    
    def validate_realm_config(self) -> bool:
        """Validate that all required realm configuration is present"""
        required_vars = [
            'MEGAMIND_PROJECT_REALM',
            'MEGAMIND_PROJECT_NAME', 
            'MEGAMIND_DEFAULT_TARGET'
        ]
        
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            logger.error(f"Missing required environment variables: {missing}")
            logger.error("Required for dynamic realm configuration:")
            for var in required_vars:
                logger.error(f"  {var}={os.getenv(var, 'NOT SET')}")
            return False
        
        # Validate default_target value
        if self.default_target not in ['PROJECT', 'GLOBAL']:
            logger.error(f"Invalid MEGAMIND_DEFAULT_TARGET: {self.default_target}. Must be 'PROJECT' or 'GLOBAL'")
            return False
        
        logger.debug("Realm configuration validation passed")
        return True
    
    async def fetch_backend_capabilities(self) -> Optional[Dict[str, Any]]:
        """Fetch actual tool capabilities from HTTP backend (non-blocking)"""
        try:
            logger.debug("Fetching capabilities from backend...")
            
            # Create a tools/list request to get available tools
            tools_request = {
                "jsonrpc": "2.0",
                "id": "capability_discovery",
                "method": "tools/list"
            }
            
            json_data = json.dumps(tools_request).encode('utf-8')
            req = urllib.request.Request(
                self.http_endpoint,
                data=json_data,
                headers=self.request_headers,
                method='POST'
            )
            
            # Use asyncio.to_thread for non-blocking HTTP request
            def _make_request():
                with urllib.request.urlopen(req, timeout=5) as response:
                    return response.read().decode('utf-8')
            
            response_body = await asyncio.to_thread(_make_request)
            response_data = json.loads(response_body)
            
            # Clean up response - remove non-standard fields
            if isinstance(response_data, dict):
                response_data.pop('_meta', None)
                if 'result' in response_data and isinstance(response_data['result'], dict):
                    response_data['result'].pop('_meta', None)
            
            if 'result' in response_data and 'tools' in response_data['result']:
                tools = response_data['result']['tools']
                logger.info(f"✅ Discovered {len(tools)} tools from backend")
                
                # Keep tools in proper MCP format (list, not dict)
                capabilities = {
                    "tools": {},  # MCP initialize expects object format
                    "_tools_list": tools  # Store original list for tools/list calls
                }
                return capabilities
            else:
                logger.warning("Backend didn't return tools list")
                return None
                
        except Exception as e:
            logger.warning(f"Could not fetch backend capabilities: {e}")
            return None
    
    def handle_mcp_initialize(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP initialize request locally - always succeeds immediately"""
        request_id = request_data.get('id')
        
        logger.info("🤝 Handling MCP initialize request locally")
        
        # Always provide basic capabilities - tools will be discovered on first use
        capabilities = {
            "tools": {},  # Will be populated when backend is available
            "resources": {}
        }
        
        response = {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": capabilities,
                "serverInfo": {
                    "name": "megamind-stdio-bridge",
                    "version": "2.0.0"
                }
            }
        }
        
        self.initialized = True
        logger.info("✅ MCP initialized - tools will be discovered on demand")
        
        # Trigger background capability fetch (non-blocking)
        asyncio.create_task(self._fetch_capabilities_background())
        
        return response
    
    def handle_mcp_initialized_notification(self, request_data: Dict[str, Any]) -> None:
        """Handle initialized notification (no response needed)"""
        logger.info("🎉 Client initialization complete - ready for operations")
        # No response for notifications
        return None
    
    async def _fetch_capabilities_background(self):
        """Background task to fetch capabilities without blocking initialization"""
        try:
            capabilities = await self.fetch_backend_capabilities()
            if capabilities:
                self.tools_cache = capabilities
                tools_count = len(capabilities.get('_tools_list', []))
                logger.info(f"🔄 Background fetch complete: {tools_count} tools cached")
        except Exception as e:
            logger.warning(f"Background capability fetch failed: {e}")
    
    async def handle_tools_list(self, request_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle tools/list request - fetch from backend if needed"""
        request_id = request_data.get('id')
        
        # If we have cached tools, return them
        if self.tools_cache and '_tools_list' in self.tools_cache:
            tools_list = self.tools_cache['_tools_list']
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "tools": tools_list
                }
            }
            logger.info(f"📋 Returned {len(tools_list)} tools from cache")
            return response
        
        # Try to fetch capabilities if not cached yet
        try:
            capabilities = await self.fetch_backend_capabilities()
            if capabilities and '_tools_list' in capabilities:
                self.tools_cache = capabilities
                tools_list = capabilities['_tools_list']
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "tools": tools_list
                    }
                }
                logger.info(f"📋 Fetched and returned {len(tools_list)} tools")
                return response
        except Exception as e:
            logger.warning(f"Failed to fetch tools for tools/list: {e}")
        
        # Forward to backend as fallback
        logger.info("📋 Forwarding tools/list to backend")
        return None
    
    def handle_resources_list(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resources/list request - return empty resources list"""
        request_id = request_data.get('id')
        
        # Return empty resources list - this MCP server doesn't provide resources
        response = {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "resources": []
            }
        }
        logger.info("📋 Returned empty resources list")
        return response
    
    def sanitize_request_params(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize request to block GLOBAL realm access and enforce project realm"""
        sanitized_data = request_data.copy()
        
        if 'params' in sanitized_data and 'arguments' in sanitized_data['params']:
            args = sanitized_data['params']['arguments']
            if 'realm_id' in args:
                realm_id = args['realm_id']
                if realm_id == 'GLOBAL':
                    logger.warning(f"🔒 Blocking GLOBAL realm access, forcing PROJECT")
                    args['realm_id'] = self.default_target
                elif realm_id not in ['PROJECT', self.project_realm]:
                    logger.warning(f"🔒 Blocking unauthorized realm '{realm_id}', forcing {self.default_target}")
                    args['realm_id'] = self.default_target
            
            # Also check target_realm parameter for promotion functions
            if 'target_realm' in args and args['target_realm'] == 'GLOBAL':
                logger.warning(f"🔒 Blocking GLOBAL target_realm, forcing PROJECT")
                args['target_realm'] = self.default_target
        
        return sanitized_data

    async def send_http_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send HTTP request to backend with proper error handling and timeout"""
        import time
        start_time = time.time()
        
        method = request_data.get('method', 'unknown')
        request_id = request_data.get('id', 'unknown')
        
        try:
            logger.debug(f"🔄 Forwarding to backend: {request_id} ({method})")
            
            # Sanitize request
            sanitized_request = self.sanitize_request_params(request_data)
            
            # Check payload size before sending to prevent large payload issues
            json_data = json.dumps(sanitized_request).encode('utf-8')
            payload_size = len(json_data)
            
            # Define payload size limits (adjust as needed)
            MAX_PAYLOAD_SIZE = 1024 * 1024  # 1MB limit
            WARN_PAYLOAD_SIZE = 100 * 1024   # 100KB warning threshold
            
            if payload_size > MAX_PAYLOAD_SIZE:
                logger.warning(f"🚫 Payload too large: {payload_size} bytes > {MAX_PAYLOAD_SIZE} bytes")
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32602,  # Invalid params - payload too large
                        "message": f"Request payload too large: {payload_size} bytes exceeds maximum of {MAX_PAYLOAD_SIZE} bytes",
                        "data": {
                            "payload_size": payload_size,
                            "max_size": MAX_PAYLOAD_SIZE,
                            "error_type": "payload_size_exceeded"
                        }
                    }
                }
            elif payload_size > WARN_PAYLOAD_SIZE:
                logger.warning(f"⚠️ Large payload detected: {payload_size} bytes (threshold: {WARN_PAYLOAD_SIZE})")
            
            req = urllib.request.Request(
                self.http_endpoint,
                data=json_data,
                headers=self.request_headers,
                method='POST'
            )
            
            # Send request with configurable timeout
            def _make_request():
                try:
                    with urllib.request.urlopen(req, timeout=self.mcp_timeout) as response:
                        return response.read().decode('utf-8'), response.status
                except urllib.error.HTTPError as e:
                    # Handle HTTP error responses
                    error_body = e.read().decode('utf-8') if e.fp else ''
                    return error_body, e.code
            
            response_body, status_code = await asyncio.to_thread(_make_request)
            
            # Handle HTTP error status codes by converting to JSON-RPC errors
            if status_code != 200:
                logger.warning(f"⚠️ HTTP {status_code} from backend for {request_id}")
                
                # Try to parse error response, fallback to generic error
                try:
                    error_data = json.loads(response_body) if response_body else {}
                    
                    # If backend already returned JSON-RPC error, use it
                    if isinstance(error_data, dict) and 'error' in error_data:
                        return error_data
                        
                except json.JSONDecodeError:
                    pass
                
                # Convert HTTP status codes to JSON-RPC error codes
                if status_code == 400:
                    json_rpc_code = -32602  # Invalid params
                    error_msg = "Invalid request parameters"
                elif status_code == 404:
                    json_rpc_code = -32601  # Method not found
                    error_msg = "Method not found"
                elif status_code == 413:  # Payload too large
                    json_rpc_code = -32602  # Invalid params
                    error_msg = "Request payload too large"
                elif status_code == 429:  # Too many requests
                    json_rpc_code = -32603  # Internal error (rate limiting)
                    error_msg = "Rate limit exceeded"
                elif status_code >= 500:
                    json_rpc_code = -32603  # Internal error
                    error_msg = "Server internal error"
                else:
                    json_rpc_code = -32603  # Internal error
                    error_msg = f"HTTP {status_code} error"
                
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": json_rpc_code,
                        "message": error_msg,
                        "data": {
                            "http_status": status_code,
                            "original_response": response_body[:500] if response_body else None  # Truncate for safety
                        }
                    }
                }
            
            response_data = json.loads(response_body)
            
            # Clean up response - remove non-standard fields that break Claude Code validation
            if isinstance(response_data, dict):
                # Remove _meta field which causes JSON-RPC validation errors
                response_data.pop('_meta', None)
                
                # Clean result object if present
                if 'result' in response_data and isinstance(response_data['result'], dict):
                    response_data['result'].pop('_meta', None)
            
            elapsed = time.time() - start_time
            logger.debug(f"✅ Backend response for {request_id}: {elapsed:.3f}s")
            
            return response_data
                
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"❌ Backend request failed for {request_id} after {elapsed:.3f}s: {e}")
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": f"Backend error: {str(e)}"
                }
            }

    async def initialize_bridge(self):
        """Initialize bridge - non-blocking startup"""
        logger.info("🚀 Initializing STDIO-HTTP bridge (non-blocking)...")
        
        # Don't block on backend capabilities - fetch in background
        logger.info("✅ Bridge ready for MCP protocol - capabilities will be fetched on demand")

    async def run_stdio_loop(self):
        """Main STDIO loop with proper MCP protocol handling"""
        logger.info("🔄 Starting STDIO-HTTP bridge loop...")
        
        try:
            while True:
                # Read JSON-RPC request from stdin
                line = await asyncio.to_thread(sys.stdin.readline)
                if not line:
                    logger.info("📜 EOF received, shutting down")
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # Parse JSON-RPC request
                    request_data = json.loads(line)
                    request_id = request_data.get('id', 'unknown')
                    method = request_data.get('method', 'unknown')
                    
                    logger.debug(f"📥 Request: {request_id} ({method})")
                    
                    # Handle MCP protocol requests locally
                    local_response = None
                    
                    if method == 'initialize':
                        local_response = self.handle_mcp_initialize(request_data)
                    elif method == 'notifications/initialized':
                        self.handle_mcp_initialized_notification(request_data)
                        # No response for notifications
                        continue
                    elif method == 'tools/list':
                        local_response = await self.handle_tools_list(request_data)
                    elif method == 'resources/list':
                        local_response = self.handle_resources_list(request_data)
                    
                    if local_response is not None:
                        # Send local response
                        response_json = json.dumps(local_response)
                        print(response_json, flush=True)
                        logger.debug(f"📤 Local response sent for {request_id}")
                    else:
                        # Forward to backend
                        response_data = await self.send_http_request(request_data)
                        
                        response_json = json.dumps(response_data)
                        print(response_json, flush=True)
                        logger.debug(f"📤 Backend response sent for {request_id}")
                    
                except json.JSONDecodeError as e:
                    logger.error(f"❌ Invalid JSON: {e}")
                    error_response = {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {
                            "code": -32700,
                            "message": "Parse error: Invalid JSON"
                        }
                    }
                    print(json.dumps(error_response), flush=True)
                    
                except Exception as e:
                    logger.error(f"❌ Request processing error: {e}")
                    # Safe error response generation
                    error_id = None
                    try:
                        if 'request_data' in locals() and isinstance(request_data, dict):
                            error_id = request_data.get('id')
                    except:
                        pass
                    
                    error_response = {
                        "jsonrpc": "2.0",
                        "id": error_id,
                        "error": {
                            "code": -32603,
                            "message": f"Internal error: {str(e)}"
                        }
                    }
                    print(json.dumps(error_response), flush=True)
                    
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        except Exception as e:
            logger.error(f"Fatal error in STDIO loop: {e}")
            raise

async def main():
    """Main entry point"""
    logger.info("🎆 === Optimized MegaMind MCP STDIO-HTTP Bridge v2.0 ===")
    logger.info(f"🔧 Configuration: realm={os.getenv('MEGAMIND_PROJECT_REALM', 'MegaMind_MCP')}, timeout={os.getenv('MCP_TIMEOUT', '30')}s")
    
    # Initialize bridge
    bridge = OptimizedSTDIOHttpBridge()
    
    try:
        # Non-blocking initialization
        await bridge.initialize_bridge()
        
        # Run the STDIO loop
        await bridge.run_stdio_loop()
        
    except Exception as e:
        logger.error(f"Bridge failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())