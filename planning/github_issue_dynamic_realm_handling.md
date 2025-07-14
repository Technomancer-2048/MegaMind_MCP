# GitHub Issue: Complete Dynamic Realm Context Configuration for HTTP MCP Server

## Issue Summary

**Title**: HTTP MCP Server Access Control Requires Realm Configuration Environment Variables for Write Operations

**Priority**: High  
**Component**: HTTP Transport, Realm Access Control  
**Related Issues**: #12 (STDIO Bridge Implementation - Resolved)

## Problem Description

Following the successful resolution of GitHub Issue #12 (STDIO-HTTP bridge connectivity), a critical access control configuration gap has been identified. The HTTP MCP Server correctly receives dynamic realm context via headers but lacks the base realm configuration needed for write access validation.

### Current Behavior
- ✅ **STDIO Bridge**: Correctly passes realm ID via `X-MCP-Realm-ID` header
- ✅ **HTTP Server**: Successfully extracts realm context from request headers  
- ✅ **Dynamic Routing**: Realm-specific requests properly routed
- ❌ **Write Access**: All write operations denied due to missing realm configuration

### Error Messages
```
Cannot create chunk in realm MegaMind_MCP: Write access denied for realm MegaMind_MCP
Cannot create chunk in realm PROJECT: Write access denied for realm PROJECT  
Cannot create chunk in realm GLOBAL: Write access denied for realm GLOBAL
```

## Root Cause Analysis

### Architecture Design (Correct)
The system is correctly designed for dynamic realm handling:

1. **STDIO Bridge** (`stdio_http_bridge.py`):
   ```python
   self.request_headers = {
       'Content-Type': 'application/json',
       'X-MCP-Realm-ID': self.project_realm,  # From MEGAMIND_PROJECT_REALM env var
       'X-MCP-Project-Name': self.project_name
   }
   ```

2. **HTTP Server** (`http_transport.py`):
   ```python
   def extract_realm_context(self, data: Dict[str, Any], request: Request) -> RealmContext:
       # Priority order: tool_arguments → params → headers → query → default
       realm_id = request.headers.get('X-MCP-Realm-ID')  # Correctly extracts header
   ```

3. **Realm Access Control** (`realm_config.py`):
   ```python
   def can_write_realm(self, realm_id: str) -> bool:
       if realm_id == self.config_manager.config.project_realm:  # ❌ Not configured
           return True
   ```

### Configuration Gap (Problem)
The HTTP server container lacks realm configuration environment variables:

**Missing Environment Variables in HTTP Container:**
- `MEGAMIND_PROJECT_REALM` 
- `MEGAMIND_PROJECT_NAME`
- `MEGAMIND_DEFAULT_TARGET`

**Current Container Environment** (from `docker-compose.yml`):
```yaml
megamind-mcp-server-http:
  environment:
    # ✅ Database configuration present
    MEGAMIND_DB_HOST: megamind-mysql
    MEGAMIND_DB_USER: megamind_user
    # ❌ Realm configuration missing
    # MEGAMIND_PROJECT_REALM: ???
    # MEGAMIND_PROJECT_NAME: ???  
    # MEGAMIND_DEFAULT_TARGET: ???
```

**Result**: `RealmConfigurationManager` defaults to `project_realm = 'PROJ_DEFAULT'`, causing access denial for all actual realm requests.

## Technical Analysis

### Request Flow (Working)
```
Claude Code 
  ↓ (STDIO with env: MEGAMIND_PROJECT_REALM=MegaMind_MCP)
STDIO Bridge (stdio_http_bridge.py)
  ↓ (HTTP with header: X-MCP-Realm-ID=MegaMind_MCP) 
HTTP Server (http_transport.py)
  ↓ (realm_context.realm_id = "MegaMind_MCP")
RealmAwareMegaMindDatabase
  ↓ (Access Control Check)
❌ DENIED: realm_id != config.project_realm ("PROJ_DEFAULT")
```

### Access Control Logic (Correct Design)
From `realm_config.py:150-159`:
```python
def can_write_realm(self, realm_id: str) -> bool:
    # Can write to project realm
    if realm_id == self.config_manager.config.project_realm:
        return True
    elif realm_id == self.config_manager.config.global_realm:
        # Global realm write requires explicit targeting  
        return self.config_manager.config.default_target == 'GLOBAL'
    else:
        return False
```

**The logic is correct** - it just needs proper configuration to function.

## Proposed Solutions

### Option 1: Static Container Configuration (Simple)
Add realm configuration to `docker-compose.yml`:

```yaml
megamind-mcp-server-http:
  environment:
    # Realm Configuration (Required for Write Access)
    MEGAMIND_PROJECT_REALM: ${MEGAMIND_PROJECT_REALM:-MegaMind_MCP}
    MEGAMIND_PROJECT_NAME: ${MEGAMIND_PROJECT_NAME:-MegaMind Context Database}
    MEGAMIND_DEFAULT_TARGET: ${MEGAMIND_DEFAULT_TARGET:-PROJECT}
```

**Pros**: Simple, immediate fix  
**Cons**: Single-realm HTTP server, requires restart for realm changes

### Option 2: Fully Dynamic Realm Configuration (Advanced)
Enhance realm context extraction to include full configuration:

1. **Extended Headers**: Pass complete realm configuration via headers
   ```python
   # STDIO Bridge enhancement
   self.request_headers = {
       'X-MCP-Realm-ID': self.project_realm,
       'X-MCP-Realm-Config': json.dumps({
           'project_realm': self.project_realm,
           'project_name': self.project_name,
           'default_target': self.default_target
       })
   }
   ```

2. **Dynamic Realm Manager**: Create per-request realm managers
   ```python
   # HTTP Server enhancement  
   def extract_realm_context(self, data, request):
       realm_config = json.loads(request.headers.get('X-MCP-Realm-Config', '{}'))
       return RealmContext.from_dynamic_config(realm_config)
   ```

**Pros**: True multi-tenant support, no container restart needed  
**Cons**: More complex implementation, potential security considerations

### Option 3: Hybrid Approach (Recommended)
Combine static defaults with dynamic overrides:

1. **Container Defaults**: Basic realm configuration in container environment
2. **Header Overrides**: Allow dynamic realm context for specific requests  
3. **Validation**: Ensure dynamic contexts don't violate security policies

## Testing Plan

### Validation Steps
1. **Environment Variable Test**:
   ```bash
   docker exec megamind-mcp-server-http env | grep MEGAMIND
   # Should show realm configuration variables
   ```

2. **Write Access Test**:
   ```bash
   curl -X POST http://10.255.250.22:8080/mcp/jsonrpc \
     -H "Content-Type: application/json" \
     -H "X-MCP-Realm-ID: MegaMind_MCP" \
     -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"mcp__megamind__create_chunk","arguments":{"content":"Test write access","source_document":"test.md","section_path":"/test","session_id":"access_test"}}}'
   # Should return success, not "Write access denied"
   ```

3. **Cross-Realm Test**:
   ```bash
   # Test PROJECT realm access
   curl ... -H "X-MCP-Realm-ID: PROJECT" ...
   # Test GLOBAL realm access (should be restricted)
   curl ... -H "X-MCP-Realm-ID: GLOBAL" ...
   ```

## Implementation Checklist

- [ ] **Phase 1**: Add realm environment variables to `docker-compose.yml`
- [ ] **Phase 2**: Test write access for configured realm
- [ ] **Phase 3**: Validate security controls for cross-realm operations
- [ ] **Phase 4**: Document dynamic realm configuration patterns
- [ ] **Phase 5**: Consider multi-tenant enhancement for future scalability

## Security Considerations

1. **Realm Isolation**: Ensure dynamic realm context doesn't bypass security controls
2. **Access Validation**: Maintain proper access control even with dynamic configuration  
3. **Audit Logging**: Log all realm context changes for security monitoring
4. **Default Deny**: Unknown realms should default to restrictive access

## Success Criteria

- ✅ HTTP container has proper realm configuration environment variables
- ✅ Write operations succeed for properly configured realm contexts
- ✅ Security controls maintain realm isolation  
- ✅ STDIO bridge connectivity continues to work correctly
- ✅ Access denied errors resolved for legitimate write operations

## Related Files

- `docker-compose.yml` - Container environment configuration
- `mcp_server/http_transport.py` - Realm context extraction  
- `mcp_server/stdio_http_bridge.py` - Header configuration
- `mcp_server/realm_config.py` - Access control logic
- `mcp_server/realm_aware_database.py` - Database operations with realm support

## Additional Context

This issue builds on the successful resolution of GitHub Issue #12, which established the STDIO-HTTP bridge connectivity. The architecture is correctly designed for dynamic realm handling - this issue addresses the final configuration gap to enable full functionality.

The solution should maintain the existing security model while enabling the write operations that are currently being denied due to missing realm configuration.