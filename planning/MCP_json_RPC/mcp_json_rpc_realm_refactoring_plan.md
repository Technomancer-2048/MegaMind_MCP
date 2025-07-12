# MCP JSON-RPC Realm Refactoring Plan

**Document Version:** 1.0  
**Date:** July 12, 2025  
**Status:** Planning Phase  

## Executive Summary

This document outlines the comprehensive refactoring plan to transform the MegaMind MCP server from a process-spawning stdio transport to a persistent HTTP-based service with dynamic realm parameter passing through JSON-RPC. This architectural change addresses critical issues with Claude Code environment variable passing and eliminates the ~30-90 second startup overhead per session.

## Current Architecture Analysis

### Existing Implementation Issues

1. **Environment Variable Dependencies**: Server relies on complex environment variable passing through `env` command wrapper
2. **Process Spawning Overhead**: Each Claude Code session spawns new server process (~30-90 seconds initialization)
3. **Resource Duplication**: Multiple embedding service instances across sessions
4. **Configuration Complexity**: Complex `.mcp.json` configuration with environment variable passing issues
5. **Startup Failure Points**: Heavy initialization during startup (database + embedding service) causes connection failures

### Current Transport Mechanism

```python
# Current: stdio transport with environment-based realm configuration
async def run(self):
    """Run the MCP server on stdin/stdout"""
    while True:
        line = await asyncio.to_thread(sys.stdin.readline)
        request = json.loads(line.strip())
        response = await self.handle_request(request)
        print(json.dumps(response), flush=True)
```

### Current Realm Configuration

```python
# Environment-based realm setup
self.realm_config = get_realm_config()  # From environment variables
self.realm_access = get_realm_access_controller()
```

## Proposed Architecture

### 1. Persistent HTTP Server with Dynamic Realm Handling

**Core Principle**: Single persistent server instance handling multiple clients with realm information passed per request.

#### HTTP Transport Implementation

```python
class HTTPMCPServer:
    """Persistent HTTP MCP Server with dynamic realm support"""
    
    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config
        self.realm_managers = {}  # Cache of realm-specific managers
        self.embedding_service = None  # Shared across all realms
        
    async def handle_http_request(self, request: web.Request) -> web.Response:
        """Handle HTTP JSON-RPC requests with realm context"""
        data = await request.json()
        
        # Extract realm from request parameters or headers
        realm_context = self.extract_realm_context(data)
        
        # Get or create realm-specific manager
        realm_manager = await self.get_realm_manager(realm_context)
        
        # Process request with realm context
        response = await self.process_mcp_request(data, realm_manager)
        
        return web.json_response(response)
```

#### Dynamic Realm Manager Factory

```python
class DynamicRealmManagerFactory:
    """Factory for creating and caching realm-specific database managers"""
    
    def __init__(self, base_db_config: Dict[str, Any], shared_embedding_service):
        self.base_config = base_db_config
        self.embedding_service = shared_embedding_service
        self.realm_managers = {}
        
    async def get_realm_manager(self, realm_id: str) -> RealmAwareMegaMindDatabase:
        """Get or create realm-specific manager with shared resources"""
        if realm_id not in self.realm_managers:
            realm_config = RealmConfiguration(
                project_realm=realm_id,
                global_realm="GLOBAL",
                project_name=f"Dynamic Realm {realm_id}"
            )
            
            manager = RealmAwareMegaMindDatabase(
                config=self.base_config,
                realm_config=realm_config,
                embedding_service=self.embedding_service  # Shared service
            )
            
            self.realm_managers[realm_id] = manager
            
        return self.realm_managers[realm_id]
```

### 2. JSON-RPC Realm Parameter Design

#### Tool Schema Enhancement

**Current Schema (Environment-based)**:
```json
{
  "name": "mcp__context_db__search_chunks",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {"type": "string"},
      "limit": {"type": "integer", "default": 10}
    },
    "required": ["query"]
  }
}
```

**Enhanced Schema (Realm-aware)**:
```json
{
  "name": "mcp__context_db__search_chunks",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {"type": "string"},
      "limit": {"type": "integer", "default": 10},
      "realm_id": {"type": "string", "description": "Target realm identifier (optional, defaults to server realm)"},
      "search_type": {"type": "string", "default": "hybrid"}
    },
    "required": ["query"]
  }
}
```

#### Request Processing with Realm Context

```python
async def handle_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle tool calls with dynamic realm context"""
    
    # Extract realm from arguments
    realm_id = arguments.pop('realm_id', self.default_realm)
    
    # Get realm-specific manager
    realm_manager = await self.realm_factory.get_realm_manager(realm_id)
    
    # Route to appropriate handler with realm context
    if tool_name == "mcp__context_db__search_chunks":
        return await self.search_chunks_with_realm(arguments, realm_manager)
    elif tool_name == "mcp__context_db__get_chunk":
        return await self.get_chunk_with_realm(arguments, realm_manager)
    # ... other tools
```

### 3. Backward Compatibility Strategy

#### Default Realm Support

```python
class BackwardCompatibleMCPServer:
    """MCP Server supporting both legacy and dynamic realm modes"""
    
    def __init__(self, config):
        self.default_realm = config.get('default_realm', 'PROJECT')
        self.realm_factory = DynamicRealmManagerFactory(config)
        self.legacy_mode = config.get('legacy_mode', False)
        
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process requests with backward compatibility"""
        
        if self.legacy_mode:
            # Use single realm from environment (existing behavior)
            return await self.legacy_handler.handle_request(request)
        else:
            # Dynamic realm handling
            return await self.dynamic_handler.handle_request(request)
```

#### Migration Path

1. **Phase 1**: Add realm parameters as optional to all tools
2. **Phase 2**: Implement HTTP transport alongside stdio
3. **Phase 3**: Deprecate stdio transport after validation
4. **Phase 4**: Remove legacy environment-based configuration

## Implementation Roadmap

### Phase 1: Foundation (Week 1)

#### 1.1 Enhanced Tool Schema Design
- Add optional `realm_id` parameter to all existing tools
- Maintain backward compatibility with existing clients
- Update JSON schema definitions in server

#### 1.2 Realm Manager Factory Implementation
```python
# File: mcp_server/realm_manager_factory.py
class RealmManagerFactory:
    """Factory for managing multiple realm contexts with shared resources"""
    
    def __init__(self, base_config: Dict[str, Any]):
        self.base_config = base_config
        self.shared_embedding_service = None
        self.realm_managers = {}
        
    async def initialize_shared_services(self):
        """Initialize services shared across all realms"""
        self.shared_embedding_service = get_embedding_service()
        await self.shared_embedding_service.initialize()
        
    async def get_realm_manager(self, realm_id: str) -> RealmAwareMegaMindDatabase:
        """Get or create realm-specific manager"""
        if realm_id not in self.realm_managers:
            # Create realm-specific configuration
            realm_config = self.create_realm_config(realm_id)
            
            # Create manager with shared embedding service
            manager = RealmAwareMegaMindDatabase(
                config=self.base_config,
                realm_config=realm_config,
                shared_embedding_service=self.shared_embedding_service
            )
            
            self.realm_managers[realm_id] = manager
            
        return self.realm_managers[realm_id]
```

#### 1.3 Database Manager Refactoring
- Modify `RealmAwareMegaMindDatabase` to accept injected realm configuration
- Enable shared embedding service across realm instances
- Implement connection pooling optimization for multi-realm access

### Phase 2: HTTP Transport Implementation (Week 2)

#### 2.1 HTTP Server Framework
```python
# File: mcp_server/http_transport.py
from aiohttp import web, ClientSession
import json
from typing import Dict, Any

class HTTPMCPTransport:
    """HTTP transport for MCP with SSE support"""
    
    def __init__(self, realm_factory: RealmManagerFactory, port: int = 8080):
        self.realm_factory = realm_factory
        self.port = port
        self.app = web.Application()
        self.setup_routes()
        
    def setup_routes(self):
        """Setup HTTP routes for MCP protocol"""
        self.app.router.add_post('/mcp/jsonrpc', self.handle_jsonrpc)
        self.app.router.add_get('/mcp/health', self.health_check)
        self.app.router.add_get('/mcp/realms', self.list_realms)
        
    async def handle_jsonrpc(self, request: web.Request) -> web.Response:
        """Handle JSON-RPC requests over HTTP"""
        try:
            data = await request.json()
            
            # Extract realm context from request
            realm_context = self.extract_realm_context(data, request)
            
            # Get realm manager
            realm_manager = await self.realm_factory.get_realm_manager(
                realm_context.get('realm_id', 'PROJECT')
            )
            
            # Process MCP request
            response = await self.process_mcp_request(data, realm_manager)
            
            return web.json_response(response)
            
        except Exception as e:
            error_response = {
                "jsonrpc": "2.0",
                "id": data.get('id') if 'data' in locals() else None,
                "error": {
                    "code": -32603,
                    "message": str(e)
                }
            }
            return web.json_response(error_response, status=500)
```

#### 2.2 Dual Transport Support
```python
# File: mcp_server/transport_manager.py
class TransportManager:
    """Manages both stdio and HTTP transports"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.realm_factory = RealmManagerFactory(config)
        
    async def start_server(self):
        """Start server with configured transport"""
        transport_type = self.config.get('transport', 'stdio')
        
        if transport_type == 'http':
            return await self.start_http_server()
        elif transport_type == 'stdio':
            return await self.start_stdio_server()
        else:
            raise ValueError(f"Unsupported transport: {transport_type}")
            
    async def start_http_server(self):
        """Start HTTP server"""
        http_transport = HTTPMCPTransport(
            realm_factory=self.realm_factory,
            port=self.config.get('port', 8080)
        )
        
        runner = web.AppRunner(http_transport.app)
        await runner.setup()
        
        site = web.TCPSite(runner, 
                          host=self.config.get('host', 'localhost'),
                          port=self.config.get('port', 8080))
        await site.start()
        
        logger.info(f"HTTP MCP Server started on {self.config.get('host', 'localhost')}:{self.config.get('port', 8080)}")
```

### Phase 3: Containerization Strategy (Week 3)

#### 3.1 Docker Container Design
```dockerfile
# File: Dockerfile.mcp-server
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create application directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY mcp_server/ ./mcp_server/
COPY models/ ./models/

# Create non-root user
RUN useradd -m -u 1000 mcpuser && chown -R mcpuser:mcpuser /app
USER mcpuser

# Expose HTTP port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/mcp/health || exit 1

# Start server
CMD ["python", "-m", "mcp_server.http_server"]
```

#### 3.2 Docker Compose Configuration
```yaml
# File: docker-compose.yml
version: '3.8'

services:
  megamind-mcp-server:
    build:
      context: .
      dockerfile: Dockerfile.mcp-server
    container_name: megamind-mcp-server
    ports:
      - "8080:8080"
    environment:
      # Database configuration
      MEGAMIND_DB_HOST: ${MEGAMIND_DB_HOST:-10.255.250.21}
      MEGAMIND_DB_PORT: ${MEGAMIND_DB_PORT:-3309}
      MEGAMIND_DB_NAME: ${MEGAMIND_DB_NAME:-megamind_database}
      MEGAMIND_DB_USER: ${MEGAMIND_DB_USER:-megamind_user}
      MEGAMIND_DB_PASSWORD: ${MEGAMIND_DB_PASSWORD}
      
      # Server configuration
      MCP_TRANSPORT: http
      MCP_HOST: 0.0.0.0
      MCP_PORT: 8080
      MCP_DEFAULT_REALM: PROJECT
      
      # Performance tuning
      CONNECTION_POOL_SIZE: 20
      EMBEDDING_CACHE_SIZE: 1000
      
      # Logging
      MEGAMIND_LOG_LEVEL: INFO
      
    volumes:
      - ./models:/app/models:ro
      - ./logs:/app/logs
      
    restart: unless-stopped
    
    depends_on:
      - mysql
      
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/mcp/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  mysql:
    image: mysql:8.0
    container_name: megamind-mysql
    environment:
      MYSQL_ROOT_PASSWORD: ${MYSQL_ROOT_PASSWORD}
      MYSQL_DATABASE: ${MEGAMIND_DB_NAME:-megamind_database}
      MYSQL_USER: ${MEGAMIND_DB_USER:-megamind_user}
      MYSQL_PASSWORD: ${MEGAMIND_DB_PASSWORD}
    ports:
      - "3309:3306"
    volumes:
      - mysql_data:/var/lib/mysql
      - ./init_schema.sql:/docker-entrypoint-initdb.d/init_schema.sql
    restart: unless-stopped

volumes:
  mysql_data:
```

### Phase 4: Client Configuration Updates (Week 4)

#### 4.1 New MCP Client Configuration
```json
{
  "mcpServers": {
    "megamind-database": {
      "command": "curl",
      "args": [
        "-X", "POST",
        "-H", "Content-Type: application/json",
        "-d", "@-",
        "http://localhost:8080/mcp/jsonrpc"
      ],
      "env": {
        "MEGAMIND_DEFAULT_REALM": "MegaMind_MCP"
      }
    }
  }
}
```

#### 4.2 Enhanced Client Configuration (Direct HTTP)
```json
{
  "mcpServers": {
    "megamind-database": {
      "transport": "http",
      "endpoint": "http://localhost:8080/mcp/jsonrpc",
      "timeout": 30000,
      "defaultRealm": "MegaMind_MCP"
    }
  }
}
```

### Phase 5: Advanced Features (Week 5)

#### 5.1 Realm Management API
```python
# Additional HTTP endpoints for realm management
class RealmManagementAPI:
    """HTTP API for dynamic realm management"""
    
    async def create_realm(self, request: web.Request) -> web.Response:
        """Create new realm configuration"""
        data = await request.json()
        realm_id = data['realm_id']
        realm_config = data['config']
        
        # Validate and create realm
        success = await self.realm_factory.create_realm(realm_id, realm_config)
        
        return web.json_response({"success": success})
        
    async def list_realms(self, request: web.Request) -> web.Response:
        """List available realms"""
        realms = await self.realm_factory.list_realms()
        return web.json_response({"realms": realms})
        
    async def realm_health(self, request: web.Request) -> web.Response:
        """Check health of specific realm"""
        realm_id = request.match_info['realm_id']
        health = await self.realm_factory.check_realm_health(realm_id)
        return web.json_response(health)
```

#### 5.2 Connection Pooling Optimization
```python
class OptimizedConnectionPool:
    """Optimized connection pooling for multi-realm access"""
    
    def __init__(self, base_config: Dict[str, Any]):
        self.base_config = base_config
        self.pools = {}  # Per-realm connection pools
        self.shared_pool = None  # Shared pool for read operations
        
    async def get_connection(self, realm_id: str, operation_type: str = 'read'):
        """Get optimized connection based on operation type"""
        if operation_type == 'read' and self.shared_pool:
            return await self.shared_pool.get_connection()
        
        if realm_id not in self.pools:
            self.pools[realm_id] = await self.create_realm_pool(realm_id)
            
        return await self.pools[realm_id].get_connection()
```

## Configuration Migration

### Current Environment Variables
```bash
# Current complex environment setup
MEGAMIND_ROOT=/Data/MCP_Servers/MegaMind_MCP
MEGAMIND_PROJECT_REALM=MegaMind_MCP
MEGAMIND_PROJECT_NAME="MegaMind MCP Platform"
MEGAMIND_DEFAULT_TARGET=PROJECT
MEGAMIND_DB_HOST=10.255.250.21
MEGAMIND_DB_PORT=3309
MEGAMIND_DB_NAME=megamind_database
MEGAMIND_DB_USER=megamind_user
MEGAMIND_DB_PASSWORD=6Q93XLI6D1b7CM9QA1sm
CONNECTION_POOL_SIZE=10
MEGAMIND_DEBUG=false
MEGAMIND_LOG_LEVEL=INFO
```

### Simplified Container Configuration
```bash
# Simplified container environment
MEGAMIND_DB_HOST=10.255.250.21
MEGAMIND_DB_PORT=3309
MEGAMIND_DB_NAME=megamind_database
MEGAMIND_DB_USER=megamind_user
MEGAMIND_DB_PASSWORD=6Q93XLI6D1b7CM9QA1sm
MCP_TRANSPORT=http
MCP_PORT=8080
MCP_DEFAULT_REALM=PROJECT
```

## Performance Benefits

### Startup Time Comparison
- **Current**: ~30-90 seconds per Claude Code session
- **Proposed**: ~0.1-2 seconds per request (after initial server startup)

### Resource Utilization
- **Current**: N embedding service instances for N clients
- **Proposed**: 1 shared embedding service for all clients

### Scalability Improvements
- **Current**: Process spawning limits concurrent sessions
- **Proposed**: HTTP server handles multiple concurrent clients

## Security Considerations

### 1. Realm Isolation
```python
class RealmSecurityManager:
    """Security manager for multi-realm access"""
    
    def validate_realm_access(self, client_id: str, realm_id: str) -> bool:
        """Validate client access to specific realm"""
        # Implement realm-based access control
        return self.access_controller.check_permission(client_id, realm_id)
        
    def sanitize_realm_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize realm parameters to prevent injection"""
        sanitized = {}
        for key, value in params.items():
            if key == 'realm_id':
                sanitized[key] = self.validate_realm_id(value)
            else:
                sanitized[key] = value
        return sanitized
```

### 2. HTTP Security Headers
```python
# Security middleware for HTTP transport
async def security_middleware(request, handler):
    """Add security headers to HTTP responses"""
    response = await handler(request)
    
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    
    return response
```

## Testing Strategy

### 1. Unit Tests
```python
# Test realm parameter handling
async def test_realm_parameter_extraction():
    """Test extraction of realm parameters from JSON-RPC requests"""
    request = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "mcp__context_db__search_chunks",
            "arguments": {
                "query": "test query",
                "realm_id": "test_realm"
            }
        },
        "id": "1"
    }
    
    realm_id = extract_realm_from_request(request)
    assert realm_id == "test_realm"
```

### 2. Integration Tests
```python
# Test HTTP transport with multiple realms
async def test_multi_realm_http_transport():
    """Test HTTP transport handling multiple realm requests"""
    async with aiohttp.ClientSession() as session:
        # Test realm A
        response_a = await session.post('http://localhost:8080/mcp/jsonrpc', 
                                       json=create_realm_request("realm_a", "search query"))
        
        # Test realm B
        response_b = await session.post('http://localhost:8080/mcp/jsonrpc',
                                       json=create_realm_request("realm_b", "search query"))
        
        # Verify realm isolation
        assert response_a.data != response_b.data
```

### 3. Performance Tests
```python
# Test concurrent client handling
async def test_concurrent_clients():
    """Test server handling multiple concurrent clients"""
    tasks = []
    for i in range(100):
        task = asyncio.create_task(make_search_request(f"client_{i}"))
        tasks.append(task)
        
    results = await asyncio.gather(*tasks)
    assert len(results) == 100
    assert all(r['status'] == 'success' for r in results)
```

## Risk Mitigation

### 1. Backward Compatibility
- Maintain stdio transport during transition period
- Default realm parameter handling for legacy clients
- Gradual migration path with feature flags

### 2. Rollback Strategy
- Keep current environment-based implementation as fallback
- Feature flags for transport selection
- Database migrations are additive only

### 3. Monitoring and Observability
```python
# Comprehensive logging for troubleshooting
class MCPServerObservability:
    """Observability framework for MCP server"""
    
    def log_request(self, request: Dict[str, Any], realm_id: str):
        """Log request with realm context"""
        logger.info(f"MCP Request: {request['method']} for realm {realm_id}")
        
    def track_performance(self, operation: str, duration: float, realm_id: str):
        """Track performance metrics per realm"""
        self.metrics.record_duration(f"mcp.{operation}.{realm_id}", duration)
        
    def alert_on_errors(self, error: Exception, context: Dict[str, Any]):
        """Alert on critical errors"""
        self.alerting.send_alert(f"MCP Server Error: {error}", context)
```

## Success Criteria

### 1. Performance Metrics
- [ ] Server startup time: < 60 seconds (one-time)
- [ ] Request response time: < 2 seconds average
- [ ] Support 10+ concurrent clients
- [ ] Memory usage: < 50% increase vs single client

### 2. Functionality Requirements
- [ ] All existing MCP tools work with realm parameters
- [ ] Backward compatibility with existing clients
- [ ] Dynamic realm switching without server restart
- [ ] Proper realm isolation and security

### 3. Operational Requirements
- [ ] Container deployment works correctly
- [ ] Health check endpoints functional
- [ ] Logging and monitoring operational
- [ ] Graceful error handling and recovery

## Conclusion

This refactoring transforms the MegaMind MCP server from a problematic process-spawning architecture to a modern, scalable HTTP service. The key benefits include:

1. **Elimination of Environment Variable Issues**: No more complex subprocess environment passing
2. **Dramatic Performance Improvement**: ~30-90 second startup becomes ~1 second response time
3. **Resource Efficiency**: Shared services across multiple clients
4. **Scalability**: Support for concurrent clients and dynamic realm management
5. **Operational Excellence**: Standard containerized deployment patterns

The implementation follows a phased approach that maintains backward compatibility while introducing modern architecture patterns. This positions the MegaMind system for future growth and integration with additional AI development workflows.

---

**Next Steps**: Begin Phase 1 implementation with enhanced tool schema design and realm manager factory development.