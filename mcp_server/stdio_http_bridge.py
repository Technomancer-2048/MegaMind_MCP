#!/usr/bin/env python3
"""
STDIO-to-HTTP Bridge for MegaMind MCP Server
Lightweight proxy using only standard library - no external dependencies
"""

import json
import sys
import asyncio
import logging
import urllib.request
import urllib.parse
import urllib.error
from typing import Dict, Any

# Configure logging to stderr so it doesn't interfere with STDIO protocol
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)

logger = logging.getLogger(__name__)

class STDIOHttpBridge:
    """STDIO-to-HTTP bridge for MCP protocol using standard library"""
    
    def __init__(self, http_endpoint: str = "http://10.255.250.22:8080/mcp/jsonrpc"):
        self.http_endpoint = http_endpoint
        self.request_headers = {
            'Content-Type': 'application/json',
            'X-MCP-Realm-ID': 'MegaMind_MCP',
            'X-MCP-Project-Name': 'MegaMind_MCP'
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
        try:
            method = request_data.get('method', 'unknown')
            request_id = request_data.get('id', 'unknown')
            
            logger.debug(f"Sending request {request_id}: {method}")
            
            # Sanitize request to enforce PROJECT-only realm access
            sanitized_request = self.sanitize_request_params(request_data)
            
            # Prepare HTTP request
            json_data = json.dumps(sanitized_request).encode('utf-8')
            req = urllib.request.Request(
                self.http_endpoint,
                data=json_data,
                headers=self.request_headers,
                method='POST'
            )
            
            # Send request and get response
            with urllib.request.urlopen(req, timeout=30) as response:
                response_data = json.loads(response.read().decode('utf-8'))
                logger.debug(f"Received response for {request_id}: {response.status}")
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
    
    async def run_stdio_loop(self):
        """Main STDIO loop - read from stdin, send to HTTP, write to stdout"""
        logger.info("Starting STDIO-HTTP bridge loop...")
        
        try:
            while True:
                # Read JSON-RPC request from stdin
                line = await asyncio.to_thread(sys.stdin.readline)
                if not line:
                    logger.info("EOF received on stdin, shutting down")
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # Parse JSON-RPC request
                    request_data = json.loads(line)
                    
                    # Send request to HTTP backend (run in thread to avoid blocking)
                    response_data = await asyncio.to_thread(self.send_http_request, request_data)
                    
                    # Write JSON-RPC response to stdout
                    print(json.dumps(response_data), flush=True)
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON received: {e}")
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
    try:
        req = urllib.request.Request("http://10.255.250.22:8080/mcp/health")
        with urllib.request.urlopen(req, timeout=5) as response:
            if response.status == 200:
                logger.info("✓ HTTP backend is accessible and healthy")
                return True
            else:
                logger.warning(f"⚠ HTTP backend returned status {response.status}")
                return False
    except Exception as e:
        logger.error(f"✗ Cannot reach HTTP backend: {e}")
        return False

async def main():
    """Main entry point for STDIO-HTTP bridge"""
    logger.info("=== MegaMind MCP STDIO-HTTP Bridge ===")
    logger.info("Connecting Claude Code (STDIO) → HTTP MCP Server")
    
    # Test backend connectivity
    if not test_backend_connectivity():
        logger.error("Backend connectivity test failed")
        sys.exit(1)
    
    # Initialize and run bridge
    bridge = STDIOHttpBridge()
    
    try:
        await bridge.run_stdio_loop()
    except Exception as e:
        logger.error(f"Bridge failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())