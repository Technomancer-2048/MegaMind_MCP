#!/usr/bin/env python3
"""
Container Health Check Script - Phase 3 Containerization
Comprehensive health monitoring for MegaMind MCP HTTP Server
"""

import os
import sys
import json
import time
import requests
import subprocess
from typing import Dict, Any, List
from datetime import datetime

class ContainerHealthChecker:
    """Comprehensive health checker for containerized MCP server"""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.timeout = 10
        self.start_time = time.time()
        
    def check_http_server_health(self) -> Dict[str, Any]:
        """Check HTTP server health endpoint"""
        try:
            response = requests.get(
                f"{self.base_url}/mcp/health",
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                health_data = response.json()
                return {
                    "status": "healthy",
                    "response_time_ms": response.elapsed.total_seconds() * 1000,
                    "server_data": health_data
                }
            else:
                return {
                    "status": "unhealthy",
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
                
        except requests.exceptions.ConnectionError:
            return {
                "status": "unreachable",
                "error": "Connection refused - server may not be running"
            }
        except requests.exceptions.Timeout:
            return {
                "status": "timeout",
                "error": f"Request timeout after {self.timeout}s"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def check_server_status(self) -> Dict[str, Any]:
        """Check detailed server status"""
        try:
            response = requests.get(
                f"{self.base_url}/mcp/status",
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return {
                    "status": "healthy",
                    "status_data": response.json()
                }
            else:
                return {
                    "status": "unhealthy",
                    "error": f"Status endpoint failed: HTTP {response.status_code}"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def check_realm_functionality(self) -> Dict[str, Any]:
        """Check realm management functionality"""
        try:
            response = requests.get(
                f"{self.base_url}/mcp/realms",
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                realms_data = response.json()
                realm_count = realms_data.get('total_count', 0)
                
                return {
                    "status": "healthy",
                    "realm_count": realm_count,
                    "realms": list(realms_data.get('realms', {}).keys())
                }
            else:
                return {
                    "status": "unhealthy",
                    "error": f"Realms endpoint failed: HTTP {response.status_code}"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def check_mcp_functionality(self) -> Dict[str, Any]:
        """Check core MCP functionality with a test call"""
        try:
            # Test MCP JSON-RPC call
            test_request = {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "mcp__context_db__search_chunks",
                    "arguments": {
                        "query": "health_check_test",
                        "limit": 1
                    }
                },
                "id": "health_check_test"
            }
            
            response = requests.post(
                f"{self.base_url}/mcp/jsonrpc",
                json=test_request,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if "error" not in result:
                    return {
                        "status": "healthy",
                        "mcp_response_time_ms": response.elapsed.total_seconds() * 1000,
                        "realm_id": result.get("_meta", {}).get("realm_id", "unknown")
                    }
                else:
                    return {
                        "status": "functional_error",
                        "error": result.get("error", {}).get("message", "Unknown MCP error")
                    }
            else:
                return {
                    "status": "unhealthy",
                    "error": f"MCP endpoint failed: HTTP {response.status_code}"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def check_database_connectivity(self) -> Dict[str, Any]:
        """Check database connectivity through MCP server"""
        try:
            # Try a simple database operation
            test_request = {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "mcp__context_db__get_hot_contexts",
                    "arguments": {
                        "limit": 1
                    }
                },
                "id": "db_health_check"
            }
            
            response = requests.post(
                f"{self.base_url}/mcp/jsonrpc",
                json=test_request,
                timeout=self.timeout * 2,  # Database operations may take longer
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if "error" not in result:
                    return {
                        "status": "healthy",
                        "db_response_time_ms": response.elapsed.total_seconds() * 1000
                    }
                else:
                    return {
                        "status": "database_error",
                        "error": result.get("error", {}).get("message", "Database operation failed")
                    }
            else:
                return {
                    "status": "unhealthy",
                    "error": f"Database check failed: HTTP {response.status_code}"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def check_container_resources(self) -> Dict[str, Any]:
        """Check container resource usage (if running in container)"""
        try:
            # Check if we're in a container
            if os.path.exists('/.dockerenv'):
                # Get memory usage
                try:
                    with open('/proc/meminfo', 'r') as f:
                        meminfo = f.read()
                    
                    memory_data = {}
                    for line in meminfo.split('\n'):
                        if line.startswith(('MemTotal:', 'MemAvailable:', 'MemFree:')):
                            key, value = line.split(':')
                            # Extract numeric value in kB
                            memory_data[key.strip()] = int(value.strip().split()[0])
                    
                    memory_usage_percent = (
                        (memory_data['MemTotal'] - memory_data['MemAvailable']) / 
                        memory_data['MemTotal'] * 100
                    )
                    
                    return {
                        "status": "healthy",
                        "in_container": True,
                        "memory_usage_percent": round(memory_usage_percent, 2),
                        "memory_total_mb": round(memory_data['MemTotal'] / 1024, 2),
                        "memory_available_mb": round(memory_data['MemAvailable'] / 1024, 2)
                    }
                    
                except Exception as e:
                    return {
                        "status": "error",
                        "in_container": True,
                        "error": f"Failed to read memory info: {e}"
                    }
            else:
                return {
                    "status": "not_applicable",
                    "in_container": False,
                    "message": "Not running in container"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def run_comprehensive_health_check(self) -> Dict[str, Any]:
        """Run all health checks and return comprehensive report"""
        
        health_report = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "unknown",
            "checks": {},
            "summary": {}
        }
        
        # Run all health checks
        checks = {
            "http_server": self.check_http_server_health,
            "server_status": self.check_server_status,
            "realm_functionality": self.check_realm_functionality,
            "mcp_functionality": self.check_mcp_functionality,
            "database_connectivity": self.check_database_connectivity,
            "container_resources": self.check_container_resources
        }
        
        healthy_checks = 0
        total_checks = len(checks)
        
        for check_name, check_func in checks.items():
            try:
                result = check_func()
                health_report["checks"][check_name] = result
                
                if result["status"] in ["healthy", "not_applicable"]:
                    healthy_checks += 1
                    
            except Exception as e:
                health_report["checks"][check_name] = {
                    "status": "error",
                    "error": f"Health check failed: {e}"
                }
        
        # Determine overall status
        if healthy_checks == total_checks:
            health_report["overall_status"] = "healthy"
        elif healthy_checks >= total_checks * 0.7:  # 70% threshold
            health_report["overall_status"] = "degraded"
        else:
            health_report["overall_status"] = "unhealthy"
        
        # Create summary
        health_report["summary"] = {
            "healthy_checks": healthy_checks,
            "total_checks": total_checks,
            "health_percentage": round((healthy_checks / total_checks) * 100, 2),
            "check_duration_ms": round((time.time() - self.start_time) * 1000, 2)
        }
        
        return health_report

def main():
    """Main health check entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='MegaMind MCP Container Health Check')
    parser.add_argument('--url', default='http://localhost:8080', help='Base URL for health checks')
    parser.add_argument('--format', choices=['json', 'text'], default='json', help='Output format')
    parser.add_argument('--check', choices=['basic', 'full'], default='full', help='Check level')
    parser.add_argument('--exit-code', action='store_true', help='Exit with non-zero code if unhealthy')
    
    args = parser.parse_args()
    
    checker = ContainerHealthChecker(args.url)
    
    if args.check == 'basic':
        # Basic health check - just HTTP server
        result = checker.check_http_server_health()
        
        if args.format == 'json':
            print(json.dumps(result, indent=2))
        else:
            print(f"Health Status: {result['status']}")
            if 'error' in result:
                print(f"Error: {result['error']}")
    else:
        # Full comprehensive health check
        result = checker.run_comprehensive_health_check()
        
        if args.format == 'json':
            print(json.dumps(result, indent=2))
        else:
            print(f"Overall Status: {result['overall_status']}")
            print(f"Health Percentage: {result['summary']['health_percentage']}%")
            print(f"Checks: {result['summary']['healthy_checks']}/{result['summary']['total_checks']}")
            print(f"Duration: {result['summary']['check_duration_ms']}ms")
            
            # Show failed checks
            for check_name, check_result in result['checks'].items():
                if check_result['status'] not in ['healthy', 'not_applicable']:
                    print(f"❌ {check_name}: {check_result['status']} - {check_result.get('error', 'No details')}")
    
    # Exit with appropriate code
    if args.exit_code:
        if args.check == 'basic':
            sys.exit(0 if result['status'] == 'healthy' else 1)
        else:
            sys.exit(0 if result['overall_status'] == 'healthy' else 1)

if __name__ == "__main__":
    main()