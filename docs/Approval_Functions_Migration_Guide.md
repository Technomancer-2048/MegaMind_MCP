# Approval Functions Migration Guide

This guide provides step-by-step instructions for migrating existing MegaMind Context Database systems to the new approval architecture implemented in **GitHub Issue #26**.

## Overview

The approval functions introduce a new governance layer for content management while maintaining backward compatibility with existing systems. This migration guide covers:

- Database schema updates
- Function name changes
- Client code migration
- Testing and validation
- Rollback procedures

## Migration Timeline

### Phase 1: Database Migration (Completed)
- Database schema updates with approval columns
- Index creation for performance optimization
- Default value assignment for existing chunks

### Phase 2: Function Implementation (Completed)
- New approval functions implementation
- MCP server integration
- Function name standardization

### Phase 3: Testing and Validation (Completed)
- Comprehensive testing of all approval functions
- Error handling validation
- Performance testing

### Phase 4: Documentation and Migration (Current)
- Migration guide creation
- Best practices documentation
- Client migration assistance

## Breaking Changes

### Function Name Changes

The approval functions were renamed to follow the standardized naming convention:

| Old Name | New Name | Status |
|----------|----------|--------|
| `mcp__megamind__get_pending_chunks` | `mcp__megamind__approval_get_pending` | ⚠️ **BREAKING** |
| `mcp__megamind__approve_chunk` | `mcp__megamind__approval_approve` | ⚠️ **BREAKING** |
| `mcp__megamind__reject_chunk` | `mcp__megamind__approval_reject` | ⚠️ **BREAKING** |
| `mcp__megamind__bulk_approve_chunks` | `mcp__megamind__approval_bulk_approve` | ⚠️ **BREAKING** |

### Database Schema Changes

New columns added to `megamind_chunks` table:

```sql
-- New approval columns (automatically added)
approval_status ENUM('pending', 'approved', 'rejected') DEFAULT 'pending',
approved_by VARCHAR(255) DEFAULT NULL,
approved_at TIMESTAMP NULL DEFAULT NULL,
rejected_by VARCHAR(255) DEFAULT NULL,
rejected_at TIMESTAMP NULL DEFAULT NULL,
approval_notes TEXT DEFAULT NULL,
rejection_reason TEXT DEFAULT NULL
```

## Pre-Migration Checklist

Before starting the migration, ensure you have:

- [ ] **Database Backup**: Complete backup of the MegaMind database
- [ ] **System Documentation**: Current system architecture documentation
- [ ] **Client Code Inventory**: List of all clients using approval functions
- [ ] **Test Environment**: Isolated environment for migration testing
- [ ] **Rollback Plan**: Detailed rollback procedures
- [ ] **Downtime Window**: Scheduled maintenance window if required

## Migration Steps

### Step 1: Database Schema Verification

Verify the database schema has been updated with approval columns:

```sql
-- Check if approval columns exist
DESCRIBE megamind_chunks;

-- Verify indexes are created
SHOW INDEX FROM megamind_chunks WHERE Key_name LIKE 'idx_approval%';

-- Check existing data migration
SELECT 
    approval_status,
    COUNT(*) as count
FROM megamind_chunks 
GROUP BY approval_status;
```

**Expected Results**:
- All approval columns should be present
- Indexes should be created for performance
- Existing chunks should have `approval_status = 'pending'`

### Step 2: Function Availability Verification

Verify the new approval functions are available:

```python
# Test function availability
import sys
sys.path.append('mcp_server')
from consolidated_mcp_server import ConsolidatedMCPServer

server = ConsolidatedMCPServer(None)
tools = server.get_tools_list()
tool_names = [tool['name'] for tool in tools]

# Check new function names
approval_functions = [
    'mcp__megamind__approval_get_pending',
    'mcp__megamind__approval_approve',
    'mcp__megamind__approval_reject',
    'mcp__megamind__approval_bulk_approve'
]

for func in approval_functions:
    if func in tool_names:
        print(f"✓ {func} - Available")
    else:
        print(f"✗ {func} - Missing")
```

### Step 3: Client Code Migration

#### 3.1 Identify Client Usage

Search for usage of old function names in your codebase:

```bash
# Search for old function names
grep -r "get_pending_chunks" /path/to/your/code/
grep -r "approve_chunk" /path/to/your/code/
grep -r "reject_chunk" /path/to/your/code/
grep -r "bulk_approve_chunks" /path/to/your/code/
```

#### 3.2 Update Function Calls

Replace old function calls with new standardized names:

**Before (Old)**:
```python
# Old function names
pending_chunks = await mcp_client.call_tool("mcp__megamind__get_pending_chunks", {
    "limit": 20
})

approval_result = await mcp_client.call_tool("mcp__megamind__approve_chunk", {
    "chunk_id": "chunk_123",
    "approved_by": "user@example.com"
})
```

**After (New)**:
```python
# New standardized function names
pending_chunks = await mcp_client.call_tool("mcp__megamind__approval_get_pending", {
    "limit": 20
})

approval_result = await mcp_client.call_tool("mcp__megamind__approval_approve", {
    "chunk_id": "chunk_123",
    "approved_by": "user@example.com"
})
```

#### 3.3 Update Error Handling

Update error handling for new response structures:

```python
# Error handling for new functions
try:
    result = await mcp_client.call_tool("mcp__megamind__approval_get_pending", {
        "limit": 50
    })
    
    if result.get('success'):
        chunks = result.get('chunks', [])
        print(f"Found {len(chunks)} pending chunks")
    else:
        print(f"Error: {result.get('error')}")
        
except Exception as e:
    print(f"Exception: {e}")
```

### Step 4: Configuration Updates

#### 4.1 Update MCP Configuration

If you have custom MCP configurations, update them:

```json
{
  "approval_functions": {
    "get_pending": "mcp__megamind__approval_get_pending",
    "approve": "mcp__megamind__approval_approve",
    "reject": "mcp__megamind__approval_reject",
    "bulk_approve": "mcp__megamind__approval_bulk_approve"
  }
}
```

#### 4.2 Update Documentation

Update any internal documentation with new function names:

- API documentation
- User guides
- Integration examples
- Testing procedures

### Step 5: Testing and Validation

#### 5.1 Unit Tests

Update unit tests with new function names:

```python
# Updated test cases
class TestApprovalFunctions:
    async def test_get_pending_chunks(self):
        # Test new function name
        result = await self.server.handle_request({
            'method': 'tools/call',
            'params': {
                'name': 'mcp__megamind__approval_get_pending',
                'arguments': {'limit': 10}
            }
        })
        assert 'result' in result
    
    async def test_approve_chunk(self):
        # Test new function name
        result = await self.server.handle_request({
            'method': 'tools/call',
            'params': {
                'name': 'mcp__megamind__approval_approve',
                'arguments': {
                    'chunk_id': 'test_chunk',
                    'approved_by': 'test_user'
                }
            }
        })
        assert 'result' in result
```

### Step 6: Deployment

#### 6.1 Staged Deployment

Deploy changes in stages:

1. **Development Environment**: Test all changes
2. **Staging Environment**: Full integration testing
3. **Production Environment**: Phased rollout

#### 6.2 Monitoring

Monitor the system after deployment:

```python
# Monitoring script
import time
import asyncio

async def monitor_approval_functions():
    while True:
        try:
            # Test function availability
            result = await client.call_tool("mcp__megamind__approval_get_pending", {
                "limit": 1
            })
            
            if result.get('success'):
                print(f"✓ Approval functions working - {time.now()}")
            else:
                print(f"✗ Error: {result.get('error')}")
                
        except Exception as e:
            print(f"✗ Exception: {e}")
        
        await asyncio.sleep(60)  # Check every minute
```

## Rollback Procedures

If issues arise during migration, follow these rollback steps:

### Step 1: Code Rollback

```bash
# Rollback to previous version
git checkout previous_version_tag

# Redeploy previous version
docker compose down
docker compose up -d
```

### Step 2: Database Rollback

```sql
-- If needed, rollback database schema (USE WITH CAUTION)
-- This will remove all approval data
ALTER TABLE megamind_chunks 
DROP COLUMN approval_status,
DROP COLUMN approved_by,
DROP COLUMN approved_at,
DROP COLUMN rejected_by,
DROP COLUMN rejected_at,
DROP COLUMN approval_notes,
DROP COLUMN rejection_reason;

-- Remove indexes
DROP INDEX idx_approval_status ON megamind_chunks;
DROP INDEX idx_approved_by ON megamind_chunks;
DROP INDEX idx_rejected_by ON megamind_chunks;
DROP INDEX idx_approved_at ON megamind_chunks;
DROP INDEX idx_rejected_at ON megamind_chunks;
```

### Step 3: Client Rollback

Revert client code to use old function names if necessary.

## Post-Migration Checklist

After successful migration:

- [ ] **Function Tests**: All approval functions working correctly
- [ ] **Performance Tests**: No performance degradation
- [ ] **Client Integration**: All clients updated and working
- [ ] **Documentation Updated**: All documentation reflects new function names
- [ ] **Monitoring Active**: System monitoring in place
- [ ] **Backup Verified**: Recent backup available
- [ ] **Team Notification**: All team members informed of changes

## Common Migration Issues

### Issue 1: Old Function Names Still Used

**Problem**: Client code still using old function names
**Solution**: 
```bash
# Search and replace old function names
sed -i 's/get_pending_chunks/approval_get_pending/g' *.py
sed -i 's/approve_chunk/approval_approve/g' *.py
sed -i 's/reject_chunk/approval_reject/g' *.py
sed -i 's/bulk_approve_chunks/approval_bulk_approve/g' *.py
```

### Issue 2: Database Schema Not Updated

**Problem**: Approval columns missing from database
**Solution**: 
```sql
-- Manually add approval columns
ALTER TABLE megamind_chunks ADD COLUMN approval_status ENUM('pending', 'approved', 'rejected') DEFAULT 'pending';
ALTER TABLE megamind_chunks ADD COLUMN approved_by VARCHAR(255) DEFAULT NULL;
-- Add other columns as needed
```

### Issue 3: Performance Issues

**Problem**: Slow approval queries
**Solution**: 
```sql
-- Ensure indexes are created
CREATE INDEX idx_approval_status ON megamind_chunks(approval_status);
CREATE INDEX idx_approved_by ON megamind_chunks(approved_by);
-- Add other indexes as needed
```

## Support and Resources

### Documentation
- [Approval Functions Documentation](./Approval_Functions_Documentation.md)
- [Best Practices Guide](./Approval_Workflow_Best_Practices.md)
- [GitHub Issue #26](https://github.com/Technomancer-2048/MegaMind_MCP/issues/26)

### Getting Help
- Review the troubleshooting section in the main documentation
- Check the GitHub issue for common problems and solutions
- Contact the development team for migration assistance

### Testing Resources
- Use the comprehensive test suite to validate migration
- Run performance tests to ensure no degradation
- Validate all client integrations after migration

## Conclusion

The approval functions migration introduces powerful governance capabilities while maintaining system stability. Following this migration guide ensures a smooth transition to the new architecture with minimal disruption to existing systems.

For additional support or questions about the migration process, please refer to the documentation links above or contact the development team.