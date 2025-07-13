#!/usr/bin/env python3
"""
STDIO-to-HTTP Proxy Bridge for MegaMind MCP Server
Provides Claude Code with proper MCP protocol over STDIO while proxying to HTTP backend
"""

import json
import sys
import asyncio
import logging
import aiohttp
from typing import Dict, Any, Optional

# Configure logging to stderr so it doesn't interfere with STDIO protocol
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)

logger = logging.getLogger(__name__)

class STDIOHttpProxy:
    """STDIO-to-HTTP proxy for MCP protocol"""
    
    def __init__(self, http_endpoint: str = "http://10.255.250.22:8080/mcp/jsonrpc"):
        self.http_endpoint = http_endpoint
        self.session: Optional[aiohttp.ClientSession] = None
        self.request_headers = {
            'Content-Type': 'application/json',
            'X-MCP-Realm-ID': 'MegaMind_MCP',
            'X-MCP-Project-Name': 'MegaMind_MCP'
        }
    
    async def initialize(self):
        """Initialize HTTP client session"""
        self.session = aiohttp.ClientSession()
        logger.info(f"STDIO-HTTP Proxy initialized, backend: {self.http_endpoint}")
    
    async def cleanup(self):
        """Cleanup HTTP client session"""
        if self.session:
            await self.session.close()
            logger.info("STDIO-HTTP Proxy session closed")
    
    async def proxy_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Proxy MCP request from STDIO to HTTP backend"""
        try:
            if not self.session:
                raise RuntimeError("Proxy not initialized")
            
            method = request_data.get('method', 'unknown')
            request_id = request_data.get('id', 'unknown')
            
            logger.debug(f"Proxying request {request_id}: {method}")
            
            # Send request to HTTP backend
            async with self.session.post(
                self.http_endpoint,
                json=request_data,
                headers=self.request_headers,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                
                if response.content_type == 'application/json':
                    response_data = await response.json()
                else:
                    # Handle non-JSON responses
                    text = await response.text()
                    response_data = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {
                            "code": -32603,
                            "message": f"HTTP {response.status}: {text}"
                        }
                    }
                
                logger.debug(f"Received response for {request_id}: {response.status}")
                return response_data
                
        except asyncio.TimeoutError:
            logger.error(f"Request {request_id} timed out")
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": "Request timeout"
                }
            }
        except aiohttp.ClientError as e:
            logger.error(f"HTTP client error for request {request_id}: {e}")
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": f"HTTP client error: {str(e)}"
                }
            }
        except Exception as e:
            logger.error(f"Unexpected error proxying request {request_id}: {e}")
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": f"Proxy error: {str(e)}"
                }
            }
    
    async def run_stdio_loop(self):
        """Main STDIO loop - read from stdin, proxy to HTTP, write to stdout"""
        logger.info("Starting STDIO-HTTP proxy loop...")
        
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
                    
                    # Proxy request to HTTP backend
                    response_data = await self.proxy_request(request_data)
                    
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

async def main():
    """Main entry point for STDIO-HTTP proxy"""
    proxy = STDIOHttpProxy()
    
    try:
        logger.info("=== MegaMind MCP STDIO-HTTP Proxy ===")
        logger.info("Connecting Claude Code (STDIO) → HTTP MCP Server")
        
        # Test backend connectivity
        async with aiohttp.ClientSession() as test_session:
            try:
                async with test_session.get("http://10.255.250.22:8080/mcp/health", timeout=5) as response:
                    if response.status == 200:
                        logger.info("✓ HTTP backend is accessible and healthy")
                    else:
                        logger.warning(f"⚠ HTTP backend returned status {response.status}")
            except Exception as e:
                logger.error(f"✗ Cannot reach HTTP backend: {e}")
                sys.exit(1)
        
        # Initialize and run proxy
        await proxy.initialize()
        await proxy.run_stdio_loop()
        
    except Exception as e:
        logger.error(f"Proxy failed: {e}")
        sys.exit(1)
    finally:
        await proxy.cleanup()

if __name__ == "__main__":
    asyncio.run(main())