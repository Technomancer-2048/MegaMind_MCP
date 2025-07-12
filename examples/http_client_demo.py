#!/usr/bin/env python3
"""
HTTP MCP Client Demo
Demonstrates how to use the new HTTP transport with realm parameters
"""

import json
import asyncio
import aiohttp
from typing import Dict, Any, List

class MCPHTTPClient:
    """Simple HTTP client for MCP JSON-RPC protocol"""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.session = None
        self.request_id = 0
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def get_next_id(self) -> str:
        """Get next request ID"""
        self.request_id += 1
        return str(self.request_id)
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any], realm_id: str = None) -> Dict[str, Any]:
        """Call an MCP tool via HTTP"""
        
        # Prepare JSON-RPC request
        request_data = {
            "jsonrpc": "2.0",
            "id": self.get_next_id(),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        
        # Add realm_id to arguments if specified
        if realm_id:
            request_data["params"]["arguments"]["realm_id"] = realm_id
        
        # Set headers
        headers = {"Content-Type": "application/json"}
        if realm_id:
            headers["X-MCP-Realm-ID"] = realm_id
        
        try:
            async with self.session.post(
                f"{self.base_url}/mcp/jsonrpc",
                json=request_data,
                headers=headers
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result
                else:
                    error_text = await response.text()
                    return {
                        "error": {
                            "code": response.status,
                            "message": f"HTTP {response.status}: {error_text}"
                        }
                    }
        except Exception as e:
            return {
                "error": {
                    "code": -1,
                    "message": f"Request failed: {str(e)}"
                }
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check server health"""
        try:
            async with self.session.get(f"{self.base_url}/mcp/health") as response:
                return await response.json()
        except Exception as e:
            return {"error": str(e)}
    
    async def server_status(self) -> Dict[str, Any]:
        """Get detailed server status"""
        try:
            async with self.session.get(f"{self.base_url}/mcp/status") as response:
                return await response.json()
        except Exception as e:
            return {"error": str(e)}
    
    async def list_realms(self) -> Dict[str, Any]:
        """List available realms"""
        try:
            async with self.session.get(f"{self.base_url}/mcp/realms") as response:
                return await response.json()
        except Exception as e:
            return {"error": str(e)}

async def demo_http_transport():
    """Demonstrate HTTP transport features"""
    
    print("🚀 MCP HTTP Transport Demo")
    print("=" * 50)
    
    async with MCPHTTPClient() as client:
        
        # 1. Health Check
        print("\n1. Server Health Check")
        print("-" * 30)
        health = await client.health_check()
        if "error" not in health:
            print(f"✅ Server Status: {health.get('status', 'unknown')}")
            print(f"📊 Uptime: {health.get('uptime_seconds', 0):.2f} seconds")
            print(f"📈 Requests: {health.get('request_count', 0)}")
        else:
            print(f"❌ Health check failed: {health['error']}")
            return
        
        # 2. Server Status
        print("\n2. Detailed Server Status")
        print("-" * 30)
        status = await client.server_status()
        if "error" not in status:
            server_info = status.get('server', {})
            print(f"🏷️  Name: {server_info.get('name', 'Unknown')}")
            print(f"🔢 Version: {server_info.get('version', 'Unknown')}")
            print(f"🌐 Host: {server_info.get('host', 'Unknown')}:{server_info.get('port', 'Unknown')}")
            
            realm_info = status.get('realm_factory', {})
            print(f"🏰 Active Realms: {realm_info.get('active_realms', 0)}")
        else:
            print(f"❌ Status check failed: {status['error']}")
        
        # 3. List Realms
        print("\n3. Available Realms")
        print("-" * 30)
        realms = await client.list_realms()
        if "error" not in realms:
            realm_list = realms.get('realms', {})
            print(f"📋 Total Realms: {realms.get('total_count', 0)}")
            for realm_id, realm_info in realm_list.items():
                print(f"   🏰 {realm_id}: {'✅ Active' if realm_info.get('active', False) else '❌ Inactive'}")
        else:
            print(f"❌ Realm listing failed: {realms['error']}")
        
        # 4. Test MCP Tool Calls
        print("\n4. MCP Tool Calls")
        print("-" * 30)
        
        # Test without realm_id (backward compatibility)
        print("📞 Search chunks (default realm):")
        search_result = await client.call_tool(
            "mcp__context_db__search_chunks",
            {"query": "test", "limit": 2}
        )
        
        if "error" not in search_result:
            print("✅ Default realm search successful")
            if "result" in search_result and "content" in search_result["result"]:
                content = search_result["result"]["content"][0]["text"]
                try:
                    chunks = json.loads(content)
                    print(f"   📊 Found {len(chunks)} chunks")
                except:
                    print("   📄 Response received")
        else:
            print(f"❌ Search failed: {search_result.get('error', {}).get('message', 'Unknown error')}")
        
        # Test with explicit realm_id
        print("\n📞 Search chunks (explicit realm):")
        realm_search_result = await client.call_tool(
            "mcp__context_db__search_chunks",
            {"query": "test", "limit": 2},
            realm_id="PROJECT"
        )
        
        if "error" not in realm_search_result:
            print("✅ Explicit realm search successful")
            # Check for realm metadata
            meta = realm_search_result.get("_meta", {})
            if meta:
                print(f"   🏰 Realm: {meta.get('realm_id', 'unknown')}")
                print(f"   ⏱️  Processing Time: {meta.get('processing_time_ms', 0)}ms")
        else:
            print(f"❌ Realm search failed: {realm_search_result.get('error', {}).get('message', 'Unknown error')}")
        
        # 5. Demonstrate Realm Parameter Sources
        print("\n5. Realm Parameter Sources")
        print("-" * 30)
        
        test_cases = [
            {
                "name": "Tool Arguments",
                "method": "arguments",
                "realm": "TEST_REALM_1"
            },
            {
                "name": "HTTP Header",
                "method": "header",
                "realm": "TEST_REALM_2"
            }
        ]
        
        for test_case in test_cases:
            print(f"🧪 Testing {test_case['name']}:")
            
            if test_case["method"] == "arguments":
                # Pass realm in tool arguments
                result = await client.call_tool(
                    "mcp__context_db__search_chunks",
                    {"query": "test", "limit": 1, "realm_id": test_case["realm"]}
                )
            else:
                # Pass realm in HTTP header
                result = await client.call_tool(
                    "mcp__context_db__search_chunks",
                    {"query": "test", "limit": 1},
                    realm_id=test_case["realm"]
                )
            
            if "error" not in result:
                meta = result.get("_meta", {})
                extracted_realm = meta.get('realm_id', 'unknown')
                if extracted_realm == test_case["realm"]:
                    print(f"   ✅ Realm extracted correctly: {extracted_realm}")
                else:
                    print(f"   ⚠️  Realm mismatch: expected {test_case['realm']}, got {extracted_realm}")
            else:
                print(f"   ❌ Test failed: {result.get('error', {}).get('message', 'Unknown error')}")

async def main():
    """Main demo function"""
    try:
        await demo_http_transport()
        print("\n🎉 Demo completed successfully!")
        
    except Exception as e:
        print(f"\n💥 Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("MCP HTTP Transport Demo")
    print("Make sure the HTTP server is running on localhost:8080")
    print("Start server with: python http_server.py")
    print()
    
    asyncio.run(main())