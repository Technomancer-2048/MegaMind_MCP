#!/usr/bin/env python3
"""
STDIO-to-HTTP Bridge for MegaMind MCP Server
Lightweight proxy using only standard library - no external dependencies
Enhanced with robust path and environment setup like STDIO transport
"""

import json
import sys
import os
import asyncio
import logging
import urllib.request
import urllib.parse
import urllib.error
from typing import Dict, Any
from pathlib import Path

# CRITICAL: Set up paths and environment BEFORE any other imports
current_dir = Path(__file__).parent
project_root = current_dir.parent

def setup_python_path():
    """Setup Python path to find both local modules and system packages"""
    # Add current directory first for local imports
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    # Add project root for broader imports
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
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
        logging.info("No .env file found - using environment variables only")

def setup_environment():
    """Setup environment variables for MegaMind realm context"""
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

# CRITICAL: Setup paths and environment FIRST
setup_python_path()
setup_environment()

# Configure comprehensive DEBUG logging to stderr
logging.basicConfig(
    level=logging.INFO,  # Default to INFO, can be overridden by LOG_LEVEL env var
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)

# Override log level from environment
log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
if hasattr(logging, log_level):
    logging.getLogger().setLevel(getattr(logging, log_level))

logger = logging.getLogger(__name__)

class STDIOHttpBridge:
    """STDIO-to-HTTP bridge for MCP protocol using standard library"""
    
    def __init__(self, http_endpoint: str = "http://10.255.250.22:8080/mcp/jsonrpc"):
        self.http_endpoint = http_endpoint
        
        # Use environment variables for realm configuration
        realm_id = os.environ.get('MEGAMIND_PROJECT_REALM', 'MegaMind_MCP')
        project_name = os.environ.get('MEGAMIND_PROJECT_NAME', 'MegaMind_MCP')
        
        self.request_headers = {
            'Content-Type': 'application/json',
            'X-MCP-Realm-ID': realm_id,
            'X-MCP-Project-Name': project_name
        }
    
    def sanitize_request_params(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize request to block GLOBAL realm access and enforce PROJECT-only operations"""
        sanitized_data = request_data.copy()
        
        # Check for realm_id in function arguments and block GLOBAL access
        if 'params' in sanitized_data and 'arguments' in sanitized_data['params']:
            args = sanitized_data['params']['arguments']
            if 'realm_id' in args:
                realm_id = args['realm_id']
                if realm_id == 'GLOBAL':
                    logger.warning(f"Blocking GLOBAL realm access attempt in request {request_data.get('id')}")
                    # Force to PROJECT realm instead
                    args['realm_id'] = 'PROJECT'
                elif realm_id not in ['PROJECT', 'MegaMind_MCP']:
                    logger.warning(f"Blocking unauthorized realm '{realm_id}' in request {request_data.get('id')}")
                    # Force to PROJECT realm
                    args['realm_id'] = 'PROJECT'
        
        return sanitized_data

    def send_http_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send HTTP request to backend using urllib with realm access controls"""
        import time
        start_time = time.time()
        
        try:
            method = request_data.get('method', 'unknown')
            request_id = request_data.get('id', 'unknown')
            
            logger.info(f"🔄 Processing request {request_id}: {method}")
            logger.debug(f"📥 Raw request data: {json.dumps(request_data, indent=2)[:500]}...")
            
            # Sanitize request to enforce PROJECT-only realm access
            sanitized_request = self.sanitize_request_params(request_data)
            
            if sanitized_request != request_data:
                logger.warning(f"🔒 Request sanitized for {request_id}")
                logger.debug(f"📝 Sanitized request: {json.dumps(sanitized_request, indent=2)[:500]}...")
            
            # Prepare HTTP request
            json_data = json.dumps(sanitized_request).encode('utf-8')
            req = urllib.request.Request(
                self.http_endpoint,
                data=json_data,
                headers=self.request_headers,
                method='POST'
            )
            
            logger.debug(f"🌐 HTTP Request to {self.http_endpoint}")
            logger.debug(f"📤 Headers: {self.request_headers}")
            logger.debug(f"📤 Body size: {len(json_data)} bytes")
            logger.debug(f"📤 Body preview: {json_data[:200].decode('utf-8')}...")
            
            # Send request and get response
            with urllib.request.urlopen(req, timeout=30) as response:
                response_body = response.read().decode('utf-8')
                response_data = json.loads(response_body)
                
                elapsed = time.time() - start_time
                logger.info(f"✅ Response for {request_id}: HTTP {response.status} ({elapsed:.3f}s)")
                logger.debug(f"📨 Response headers: {dict(response.headers)}")
                logger.debug(f"📨 Response body: {response_body[:500]}...")
                
                return response_data
                
        except urllib.error.HTTPError as e:
            logger.error(f"HTTP error for request {request_id}: {e.code} {e.reason}")
            try:
                error_body = e.read().decode('utf-8')
                logger.error(f"Error body: {error_body}")
            except:
                pass
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": f"HTTP {e.code}: {e.reason}"
                }
            }
        except urllib.error.URLError as e:
            logger.error(f"URL error for request {request_id}: {e}")
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": f"Connection error: {str(e)}"
                }
            }
        except Exception as e:
            logger.error(f"Unexpected error for request {request_id}: {e}")
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": f"Bridge error: {str(e)}"
                }
            }
    
    def handle_local_mcp_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP initialization requests - forward initialize to get actual capabilities"""
        method = request_data.get('method', '')
        request_id = request_data.get('id')
        
        if method == 'initialize':
            logger.info("🤝 Forwarding MCP initialize request to HTTP backend for capabilities")
            # Forward to HTTP backend to get actual tool capabilities
            return None  # Signal to forward to HTTP backend
        elif method == 'notifications/initialized':
            logger.info("🎉 Client initialization complete - ready for normal operations")
            return None  # Notifications don't return responses
        else:
            return None  # Not a local request, forward to HTTP backend

    async def run_stdio_loop(self):
        """Main STDIO loop - read from stdin, send to HTTP, write to stdout"""
        logger.info("🚀 Starting STDIO-HTTP bridge loop...")
        logger.info(f"🌐 HTTP Endpoint: {self.http_endpoint}")
        logger.info(f"📤 Request Headers: {self.request_headers}")
        
        try:
            while True:
                # Read JSON-RPC request from stdin
                line = await asyncio.to_thread(sys.stdin.readline)
                if not line:
                    logger.info("📜 EOF received on stdin, shutting down")
                    break
                
                line = line.strip()
                if not line:
                    logger.debug("🔄 Empty line received, skipping")
                    continue
                
                try:
                    logger.debug(f"📥 STDIN received: {line[:200]}...")
                    
                    # Parse JSON-RPC request
                    request_data = json.loads(line)
                    request_id = request_data.get('id', 'unknown')
                    method = request_data.get('method', 'unknown')
                    
                    logger.info(f"🔍 Parsed JSON-RPC: ID={request_id}, Method={method}")
                    
                    # Check if this is a local MCP protocol request
                    local_response = self.handle_local_mcp_request(request_data)
                    
                    if local_response is not None:
                        # Handle locally (MCP protocol requests)
                        response_json = json.dumps(local_response)
                        logger.debug(f"📤 STDOUT sending (local): {response_json[:200]}...")
                        print(response_json, flush=True)
                        logger.info(f"✅ Completed local request {request_id}")
                    elif method == 'notifications/initialized':
                        # Notification - no response needed
                        logger.info(f"✅ Processed notification {method}")
                    else:
                        # Forward to HTTP backend for actual MCP tool calls
                        response_data = await asyncio.to_thread(self.send_http_request, request_data)
                        
                        # Write JSON-RPC response to stdout
                        response_json = json.dumps(response_data)
                        logger.debug(f"📤 STDOUT sending (HTTP): {response_json[:200]}...")
                        print(response_json, flush=True)
                        logger.info(f"✅ Completed HTTP request {request_id}")
                    
                except json.JSONDecodeError as e:
                    logger.error(f"❌ Invalid JSON received: {e}")
                    logger.error(f"❌ Raw line: {line[:100]}...")
                    error_response = {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {
                            "code": -32700,
                            "message": "Parse error: Invalid JSON"
                        }
                    }
                    error_json = json.dumps(error_response)
                    logger.debug(f"📤 STDOUT error: {error_json}")
                    print(error_json, flush=True)
                    
                except Exception as e:
                    logger.error(f"Error processing request: {e}")
                    error_response = {
                        "jsonrpc": "2.0",
                        "id": request_data.get('id') if 'request_data' in locals() else None,
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

def test_backend_connectivity():
    """Test if HTTP backend is accessible"""
    logger.info("🔍 Testing HTTP backend connectivity...")
    try:
        health_url = "http://10.255.250.22:8080/mcp/health"
        logger.debug(f"🌐 Testing: {health_url}")
        
        req = urllib.request.Request(health_url)
        with urllib.request.urlopen(req, timeout=5) as response:
            response_body = response.read().decode('utf-8')
            logger.debug(f"📨 Health response: {response_body}")
            
            if response.status == 200:
                logger.info("✅ HTTP backend is accessible and healthy")
                return True
            else:
                logger.warning(f"⚠️ HTTP backend returned status {response.status}")
                return False
    except Exception as e:
        logger.error(f"❌ Cannot reach HTTP backend: {e}")
        logger.debug(f"❌ Full error details: {type(e).__name__}: {str(e)}")
        return False

async def main():
    """Main entry point for STDIO-HTTP bridge"""
    logger.info("🎆 === MegaMind MCP STDIO-HTTP Bridge ===")
    logger.info("🔗 Connecting Claude Code (STDIO) → HTTP MCP Server")
    logger.info(f"💻 Python version: {sys.version}")
    logger.info(f"📁 Working directory: {os.getcwd()}")
    logger.info(f"📁 Project Root: {project_root}")
    logger.info(f"📁 Current Directory: {current_dir}")
    logger.info(f"🎯 Project Realm: {os.environ.get('MEGAMIND_PROJECT_REALM')}")
    logger.info(f"🎯 Default Target: {os.environ.get('MEGAMIND_DEFAULT_TARGET')}")
    logger.info(f"📊 Log Level: {os.environ.get('LOG_LEVEL', 'INFO')}")
    
    # Test backend connectivity
    if not test_backend_connectivity():
        logger.error("❌ Backend connectivity test failed")
        sys.exit(1)
    
    # Initialize and run bridge
    logger.info("🔄 Initializing STDIO-HTTP bridge...")
    bridge = STDIOHttpBridge()
    logger.info("✅ Bridge initialized successfully")
    
    try:
        await bridge.run_stdio_loop()
    except Exception as e:
        logger.error(f"Bridge failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())