# Approval Functions Documentation

## Overview

The MegaMind Context Database System includes a comprehensive approval architecture that allows for governed content management through chunk-level approval workflows. This system was implemented as part of **GitHub Issue #26** to provide enterprise-grade content governance capabilities.

## Architecture

### Design Principles
- **Decoupled from Sessions**: Approval workflow operates independently of session management
- **Dual-Realm Support**: Works across both PROJECT and GLOBAL realms
- **Batch Operations**: Supports both individual and bulk approval operations
- **Audit Trail**: Complete tracking of approval decisions and history
- **Standardized Naming**: Follows `mcp__megamind__[CLASS]_[PURPOSE]` convention

### Database Schema

The approval system adds the following columns to the `megamind_chunks` table:

```sql
-- Approval status tracking
approval_status ENUM('pending', 'approved', 'rejected') DEFAULT 'pending',
approved_by VARCHAR(255) DEFAULT NULL,
approved_at TIMESTAMP NULL DEFAULT NULL,
rejected_by VARCHAR(255) DEFAULT NULL,
rejected_at TIMESTAMP NULL DEFAULT NULL,
approval_notes TEXT DEFAULT NULL,
rejection_reason TEXT DEFAULT NULL,

-- Indexes for performance
KEY idx_approval_status (approval_status),
KEY idx_approved_by (approved_by),
KEY idx_rejected_by (rejected_by),
KEY idx_approved_at (approved_at),
KEY idx_rejected_at (rejected_at)
```

## Functions Reference

### 1. `mcp__megamind__approval_get_pending`

**Purpose**: Retrieve all pending chunks across the system with optional filtering.

**Parameters**:
- `limit` (integer, optional): Maximum number of chunks to return (default: 20)
- `realm_filter` (string, optional): Filter by realm ("PROJECT", "GLOBAL", or null for all)

**Returns**:
```json
{
  "success": true,
  "chunks": [
    {
      "chunk_id": "chunk_123",
      "content": "Chunk content preview...",
      "source_document": "document.md",
      "realm": "PROJECT",
      "created_at": "2023-01-01T00:00:00Z",
      "approval_status": "pending"
    }
  ],
  "count": 15,
  "realm_filter": "PROJECT",
  "limit": 20
}
```

**Example Usage**:
```python
# Get all pending chunks
result = await search_query("approval_get_pending", limit=50)

# Get pending chunks from specific realm
result = await search_query("approval_get_pending", realm_filter="PROJECT")
```

### 2. `mcp__megamind__approval_approve`

**Purpose**: Approve a chunk by updating its approval status to 'approved'.

**Parameters**:
- `chunk_id` (string, required): ID of the chunk to approve
- `approved_by` (string, required): Identifier of the user performing the approval
- `approval_notes` (string, optional): Optional notes about the approval

**Returns**:
```json
{
  "success": true,
  "chunk_id": "chunk_123",
  "approval_status": "approved",
  "approved_by": "user@example.com",
  "approved_at": "2023-01-01T00:00:00Z",
  "message": "Chunk approved successfully"
}
```

**Example Usage**:
```python
# Approve a chunk
result = await approval_approve(
    chunk_id="chunk_123",
    approved_by="reviewer@company.com",
    approval_notes="Content reviewed and approved for production"
)
```

### 3. `mcp__megamind__approval_reject`

**Purpose**: Reject a chunk by updating its approval status to 'rejected'.

**Parameters**:
- `chunk_id` (string, required): ID of the chunk to reject
- `rejected_by` (string, required): Identifier of the user performing the rejection
- `rejection_reason` (string, required): Reason for rejection

**Returns**:
```json
{
  "success": true,
  "chunk_id": "chunk_123",
  "approval_status": "rejected",
  "rejected_by": "user@example.com",
  "rejected_at": "2023-01-01T00:00:00Z",
  "rejection_reason": "Content needs revision"
}
```

**Example Usage**:
```python
# Reject a chunk
result = await approval_reject(
    chunk_id="chunk_123",
    rejected_by="reviewer@company.com",
    rejection_reason="Content contains outdated information and needs revision"
)
```

### 4. `mcp__megamind__approval_bulk_approve`

**Purpose**: Approve multiple chunks in a single batch operation.

**Parameters**:
- `chunk_ids` (array of strings, required): List of chunk IDs to approve
- `approved_by` (string, required): Identifier of the user performing the bulk approval

**Returns**:
```json
{
  "success": true,
  "approved_count": 8,
  "failed_count": 2,
  "approved_chunks": ["chunk_123", "chunk_124", "chunk_125"],
  "failed_chunks": [
    {
      "chunk_id": "chunk_126",
      "error": "Chunk not found"
    }
  ],
  "approved_by": "user@example.com",
  "approved_at": "2023-01-01T00:00:00Z"
}
```

**Example Usage**:
```python
# Bulk approve multiple chunks
chunk_list = ["chunk_123", "chunk_124", "chunk_125", "chunk_126"]
result = await approval_bulk_approve(
    chunk_ids=chunk_list,
    approved_by="reviewer@company.com"
)

print(f"Approved {result['approved_count']} chunks")
print(f"Failed {result['failed_count']} chunks")
```

## Integration with Search Functions

The approval system integrates with the existing search functions to provide filtered results:

### Search with Approval Status
```python
# Search for approved content only
approved_content = await search_query(
    "production documentation",
    approval_filter="approved"
)

# Search for pending content needing review
pending_content = await search_query(
    "new documentation",
    approval_filter="pending"
)
```

## Error Handling

### Common Error Scenarios

1. **Chunk Not Found**:
```json
{
  "success": false,
  "error": "Chunk not found: chunk_123",
  "chunk_id": "chunk_123"
}
```

2. **Missing Required Parameters**:
```json
{
  "success": false,
  "error": "Missing required parameter: approved_by"
}
```

3. **Database Connection Issues**:
```json
{
  "success": false,
  "error": "Database connection failed",
  "chunks": [],
  "count": 0
}
```

## Performance Considerations

### Indexing Strategy
The approval system includes optimized indexes for:
- `approval_status` - Fast filtering by approval state
- `approved_by` / `rejected_by` - User-based queries
- `approved_at` / `rejected_at` - Time-based queries

### Batch Operations
- Use `approval_bulk_approve` for multiple chunks to reduce database calls
- Limit batch sizes to 100 chunks per operation for optimal performance
- Consider pagination for large approval queues

### Caching Strategy
- Approval status is cached at the chunk level
- Cache invalidation occurs on status changes
- Use realm-aware caching for multi-tenant scenarios

## Security Considerations

### Access Control
- Approval functions require authenticated users
- User identifiers are logged for audit trail
- Role-based access control can be implemented at the application level

### Data Validation
- All user inputs are validated and sanitized
- Chunk IDs are validated against existing chunks
- Approval status changes are atomic operations

### Audit Trail
- Complete history of approval decisions
- User identification and timestamps
- Reason tracking for rejections
- Immutable audit logs

## Best Practices

### Approval Workflow
1. **Content Creation**: New chunks default to 'pending' status
2. **Review Process**: Use `approval_get_pending` to get review queue
3. **Decision Making**: Use `approval_approve` or `approval_reject` with clear reasons
4. **Bulk Operations**: Use `approval_bulk_approve` for efficient batch processing
5. **Audit**: Maintain clear audit trail with user identification

### Performance Optimization
- Use pagination for large approval queues
- Implement caching for frequently accessed approval data
- Consider background processing for bulk operations
- Monitor database performance with approval indexes

### Integration Patterns
- Integrate with existing content management workflows
- Use approval status in search filters
- Implement notification systems for approval decisions
- Create dashboard interfaces for approval queue management

## Troubleshooting

### Common Issues

1. **Slow Approval Queries**:
   - Check database indexes are properly created
   - Consider query optimization for large datasets
   - Implement proper pagination

2. **Bulk Operation Timeouts**:
   - Reduce batch sizes
   - Implement background processing
   - Add progress tracking

3. **Audit Trail Gaps**:
   - Ensure all approval operations include user identification
   - Validate timestamp consistency
   - Check database transaction integrity

### Debug Mode
Enable debug logging to trace approval operations:
```python
import logging
logging.getLogger('consolidated_functions').setLevel(logging.DEBUG)
```

## Migration Notes

For systems upgrading to the approval architecture, see the [Migration Guide](./Migration_Guide.md) for detailed migration steps and compatibility considerations.