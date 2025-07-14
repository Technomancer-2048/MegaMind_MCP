# Option 2 Execution Plan: Fully Dynamic Realm Configuration

## Overview

This plan implements **Option 2: Fully Dynamic Realm Configuration** from GitHub Issue #13, enabling true multi-tenant support where realm configuration is passed dynamically via request headers without requiring container restarts or static configuration.

## Architecture Changes

### Current Flow
```
Claude Code → STDIO Bridge → HTTP Server → Database
            (X-MCP-Realm-ID)   (Static Config)
```

### Target Flow  
```
Claude Code → STDIO Bridge → HTTP Server → Database
            (X-MCP-Realm-Config + X-MCP-Realm-ID)   (Dynamic Config)
```

## Phase 1: Enhanced STDIO Bridge Configuration Passing

### 1.1 Extend STDIO Bridge Headers
**File**: `mcp_server/stdio_http_bridge.py`

**Current Implementation** (lines 38-42):
```python
self.request_headers = {
    'Content-Type': 'application/json',
    'X-MCP-Realm-ID': self.project_realm,
    'X-MCP-Project-Name': self.project_name
}
```

**Enhanced Implementation**:
```python
self.request_headers = {
    'Content-Type': 'application/json',
    'X-MCP-Realm-ID': self.project_realm,
    'X-MCP-Project-Name': self.project_name,
    'X-MCP-Realm-Config': json.dumps({
        'project_realm': self.project_realm,
        'project_name': self.project_name,
        'default_target': self.default_target,
        'global_realm': 'GLOBAL',
        'cross_realm_search_enabled': True,
        'project_priority_weight': 1.2,
        'global_priority_weight': 1.0
    })
}
```

### 1.2 Configuration Validation
Add validation method to ensure complete configuration:

```python
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
        return False
    
    return True
```

## Phase 2: HTTP Server Dynamic Configuration Processing

### 2.1 Enhanced Realm Context Extraction
**File**: `mcp_server/http_transport.py`

**Current Method** (lines 95-141): `extract_realm_context()`

**Enhanced Method**:
```python
def extract_realm_context(self, data: Dict[str, Any], request: Request) -> RealmContext:
    """Extract realm context with dynamic configuration support"""
    try:
        # 1. Try to get full realm configuration from header
        realm_config_header = request.headers.get('X-MCP-Realm-Config')
        if realm_config_header:
            try:
                realm_config = json.loads(realm_config_header)
                logger.debug(f"Using dynamic realm configuration: {realm_config}")
                
                # Validate required fields
                required_fields = ['project_realm', 'project_name', 'default_target']
                if all(field in realm_config for field in required_fields):
                    return RealmContext.from_dynamic_config(realm_config)
                else:
                    logger.warning(f"Incomplete realm configuration in header, missing: {[f for f in required_fields if f not in realm_config]}")
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON in X-MCP-Realm-Config header: {e}")
        
        # 2. Fallback to existing realm ID extraction logic
        realm_id = self._extract_realm_id(data, request)
        
        # 3. Create minimal context for backward compatibility
        return RealmContext(
            realm_id=realm_id,
            project_name=f"Dynamic Project {realm_id}",
            default_target="PROJECT"
        )
        
    except Exception as e:
        logger.error(f"Failed to extract realm context: {e}")
        return self._get_default_realm_context()

def _extract_realm_id(self, data: Dict[str, Any], request: Request) -> str:
    """Extract realm ID using existing priority logic"""
    # Existing logic from current extract_realm_context method
    # (tool arguments → params → headers → query → default)
    
def _get_default_realm_context(self) -> RealmContext:
    """Get safe default realm context"""
    return RealmContext(
        realm_id="PROJECT",
        project_name="Default HTTP Project", 
        default_target="PROJECT"
    )
```

### 2.2 Per-Request Realm Manager Creation
**Enhancement to** `handle_jsonrpc()` method:

```python
async def handle_jsonrpc(self, request: Request) -> Response:
    """Handle JSON-RPC requests with dynamic realm configuration"""
    try:
        data = await request.json()
        
        # Extract dynamic realm context
        realm_context = self.extract_realm_context(data, request)
        
        # Create dynamic realm manager for this request
        realm_manager = await self.realm_factory.create_dynamic_realm_manager(realm_context)
        
        # Create MCP server instance with dynamic configuration
        mcp_server = MCPServer(realm_manager)
        
        # Process request with realm-specific configuration
        response = await mcp_server.handle_request(data)
        
        # Add realm metadata to response
        response['_meta'] = {
            'realm_id': realm_context.realm_id,
            'realm_config': realm_context.to_dict(),
            'processing_time_ms': processing_time
        }
        
        return web.json_response(response, headers={
            'X-MCP-Realm-ID': realm_context.realm_id,
            'X-MCP-Realm-Active': 'dynamic'
        })
        
    except Exception as e:
        # Enhanced error handling with realm context
        logger.error(f"Dynamic realm request failed: {e}")
        return self._create_error_response(e, data)
```

## Phase 3: Dynamic Realm Context System

### 3.1 Enhanced RealmContext Class
**File**: `mcp_server/realm_manager_factory.py`

```python
@dataclass
class RealmContext:
    """Enhanced realm context supporting dynamic configuration"""
    realm_id: str
    project_name: str
    default_target: str
    global_realm: str = 'GLOBAL'
    cross_realm_search_enabled: bool = True
    project_priority_weight: float = 1.2
    global_priority_weight: float = 1.0
    
    @classmethod
    def from_dynamic_config(cls, config: Dict[str, Any]) -> 'RealmContext':
        """Create RealmContext from dynamic configuration"""
        return cls(
            realm_id=config['project_realm'],
            project_name=config['project_name'],
            default_target=config['default_target'],
            global_realm=config.get('global_realm', 'GLOBAL'),
            cross_realm_search_enabled=config.get('cross_realm_search_enabled', True),
            project_priority_weight=config.get('project_priority_weight', 1.2),
            global_priority_weight=config.get('global_priority_weight', 1.0)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'realm_id': self.realm_id,
            'project_name': self.project_name,
            'default_target': self.default_target,
            'global_realm': self.global_realm,
            'cross_realm_search_enabled': self.cross_realm_search_enabled,
            'project_priority_weight': self.project_priority_weight,
            'global_priority_weight': self.global_priority_weight
        }
    
    def create_realm_config(self) -> 'RealmConfig':
        """Create RealmConfig instance for access control"""
        from realm_config import RealmConfig
        return RealmConfig(
            project_realm=self.realm_id,
            project_name=self.project_name,
            default_target=self.default_target,
            global_realm=self.global_realm,
            cross_realm_search_enabled=self.cross_realm_search_enabled,
            project_priority_weight=self.project_priority_weight,
            global_priority_weight=self.global_priority_weight
        )
```

### 3.2 Dynamic Realm Manager Factory
**Enhancement to** `DynamicRealmManagerFactory`:

```python
class DynamicRealmManagerFactory(RealmManagerFactory):
    """Factory supporting per-request dynamic realm configuration"""
    
    async def create_dynamic_realm_manager(self, realm_context: RealmContext) -> RealmAwareMegaMindDatabase:
        """Create realm manager with dynamic configuration"""
        try:
            # Create dynamic realm configuration
            realm_config = realm_context.create_realm_config()
            
            # Create temporary configuration manager with dynamic config
            config_manager = DynamicRealmConfigurationManager(realm_config)
            
            # Create access controller with dynamic configuration
            access_controller = RealmAccessController(config_manager)
            
            # Create realm-aware database with dynamic configuration
            realm_manager = RealmAwareMegaMindDatabase(
                config_manager=config_manager,
                access_controller=access_controller
            )
            
            # Initialize with shared services (database connections, embeddings)
            await realm_manager.initialize_with_shared_services(self.shared_services)
            
            logger.debug(f"Created dynamic realm manager for {realm_context.realm_id}")
            return realm_manager
            
        except Exception as e:
            logger.error(f"Failed to create dynamic realm manager: {e}")
            raise
```

## Phase 4: Security and Validation Layer

### 4.1 Dynamic Configuration Validation
```python
class DynamicRealmValidator:
    """Validates dynamic realm configurations for security"""
    
    def __init__(self, allowed_realms: Set[str] = None):
        self.allowed_realms = allowed_realms or {'PROJECT', 'GLOBAL'}
        self.restricted_patterns = {
            'admin', 'root', 'system', 'internal'
        }
    
    def validate_realm_config(self, config: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate dynamic realm configuration"""
        # 1. Check required fields
        required = ['project_realm', 'project_name', 'default_target']
        missing = [f for f in required if f not in config]
        if missing:
            return False, f"Missing required fields: {missing}"
        
        # 2. Validate realm ID format
        realm_id = config['project_realm']
        if not re.match(r'^[A-Za-z0-9_-]+$', realm_id):
            return False, f"Invalid realm ID format: {realm_id}"
        
        # 3. Check for restricted patterns
        if any(pattern in realm_id.lower() for pattern in self.restricted_patterns):
            return False, f"Realm ID contains restricted pattern: {realm_id}"
        
        # 4. Validate default target
        if config['default_target'] not in ['PROJECT', 'GLOBAL']:
            return False, f"Invalid default_target: {config['default_target']}"
        
        return True, "Valid configuration"
    
    def sanitize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize configuration values"""
        sanitized = config.copy()
        
        # Sanitize project name
        sanitized['project_name'] = re.sub(r'[<>"]', '', sanitized['project_name'])[:100]
        
        # Clamp numeric values
        sanitized['project_priority_weight'] = max(0.1, min(5.0, 
            float(sanitized.get('project_priority_weight', 1.2))))
        sanitized['global_priority_weight'] = max(0.1, min(5.0,
            float(sanitized.get('global_priority_weight', 1.0))))
        
        return sanitized
```

### 4.2 Audit Logging
```python
class DynamicRealmAuditLogger:
    """Audit logging for dynamic realm operations"""
    
    def log_realm_creation(self, realm_context: RealmContext, request_ip: str):
        """Log dynamic realm creation"""
        logger.info(f"AUDIT: Dynamic realm created - ID: {realm_context.realm_id}, "
                   f"Name: {realm_context.project_name}, IP: {request_ip}")
    
    def log_access_attempt(self, realm_id: str, operation: str, success: bool, reason: str = ""):
        """Log access attempts with outcomes"""
        status = "SUCCESS" if success else "DENIED"
        logger.info(f"AUDIT: {status} - Realm: {realm_id}, Operation: {operation}, "
                   f"Reason: {reason}")
```

## Phase 5: Testing Strategy

### 5.1 Unit Tests
```python
class TestDynamicRealmConfiguration:
    """Unit tests for dynamic realm configuration"""
    
    def test_realm_context_from_config(self):
        config = {
            'project_realm': 'TestRealm',
            'project_name': 'Test Project',
            'default_target': 'PROJECT'
        }
        context = RealmContext.from_dynamic_config(config)
        assert context.realm_id == 'TestRealm'
        assert context.project_name == 'Test Project'
    
    def test_invalid_realm_config_validation(self):
        validator = DynamicRealmValidator()
        valid, message = validator.validate_realm_config({})
        assert not valid
        assert "Missing required fields" in message
    
    def test_dynamic_realm_manager_creation(self):
        # Test creation of dynamic realm manager
        pass
```

### 5.2 Integration Tests
```bash
#!/bin/bash
# Integration test script for dynamic realm configuration

echo "Testing dynamic realm configuration..."

# Test 1: Dynamic realm creation
curl -X POST http://10.255.250.22:8080/mcp/jsonrpc \
  -H "Content-Type: application/json" \
  -H "X-MCP-Realm-Config: {\"project_realm\":\"TestRealm\",\"project_name\":\"Test Project\",\"default_target\":\"PROJECT\"}" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"mcp__megamind__create_chunk","arguments":{"content":"Dynamic realm test","source_document":"test.md","section_path":"/test","session_id":"dynamic_test"}}}'

# Test 2: Fallback to realm ID only
curl -X POST http://10.255.250.22:8080/mcp/jsonrpc \
  -H "Content-Type: application/json" \
  -H "X-MCP-Realm-ID: FallbackRealm" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"mcp__megamind__search_chunks","arguments":{"query":"test","limit":5}}}'

# Test 3: Invalid configuration handling
curl -X POST http://10.255.250.22:8080/mcp/jsonrpc \
  -H "Content-Type: application/json" \
  -H "X-MCP-Realm-Config: {\"invalid\":\"config\"}" \
  -d '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"mcp__megamind__search_chunks","arguments":{"query":"test","limit":5}}}'
```

## Phase 6: Migration and Rollback Plan

### 6.1 Backward Compatibility
- Maintain support for existing `X-MCP-Realm-ID` header
- Fallback to minimal realm context when configuration is missing
- No breaking changes to existing STDIO bridge configurations

### 6.2 Feature Flags
```python
# Environment variable to enable/disable dynamic configuration
ENABLE_DYNAMIC_REALM_CONFIG = os.getenv('ENABLE_DYNAMIC_REALM_CONFIG', 'true').lower() == 'true'

if ENABLE_DYNAMIC_REALM_CONFIG and realm_config_header:
    # Use dynamic configuration
    return RealmContext.from_dynamic_config(realm_config)
else:
    # Use existing static approach
    return self._extract_legacy_realm_context(data, request)
```

### 6.3 Rollback Strategy
1. Set `ENABLE_DYNAMIC_REALM_CONFIG=false`
2. Restart HTTP container to disable dynamic features
3. Falls back to existing realm ID extraction logic
4. No data loss or configuration corruption

## Implementation Timeline

### Week 1: Core Infrastructure
- [ ] Enhance RealmContext class with dynamic configuration support
- [ ] Implement DynamicRealmValidator and security layer
- [ ] Create unit tests for new components

### Week 2: STDIO Bridge Enhancement  
- [ ] Extend STDIO bridge to pass realm configuration headers
- [ ] Add configuration validation to bridge initialization
- [ ] Test bridge configuration passing

### Week 3: HTTP Server Integration
- [ ] Implement enhanced realm context extraction
- [ ] Add dynamic realm manager factory support
- [ ] Integrate security validation layer

### Week 4: Testing and Validation
- [ ] Comprehensive integration testing
- [ ] Security testing with malicious configurations
- [ ] Performance testing with multiple concurrent realms
- [ ] Documentation and rollback procedures

## Success Criteria

- ✅ **Multi-tenant Support**: Multiple realms can be served by single HTTP container
- ✅ **Zero Downtime**: Realm configuration changes without container restart
- ✅ **Security Maintained**: Dynamic configuration doesn't bypass access controls
- ✅ **Backward Compatible**: Existing STDIO bridge configurations continue working
- ✅ **Audit Trail**: All dynamic realm operations logged for security monitoring
- ✅ **Performance**: <5% performance impact compared to static configuration

## Risk Mitigation

1. **Configuration Injection**: Strict validation and sanitization of dynamic config
2. **Memory Leaks**: Proper cleanup of per-request realm managers
3. **Performance Impact**: Caching and connection pooling for frequently used realms
4. **Security Bypass**: Comprehensive audit logging and access control validation
5. **Rollback Capability**: Feature flags for safe rollback to static configuration

This plan provides a comprehensive path to implement fully dynamic realm configuration while maintaining security, performance, and backward compatibility.