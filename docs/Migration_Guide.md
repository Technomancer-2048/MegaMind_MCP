# Migration Guide: Original to Consolidated Functions
## Phase 2: Function Consolidation Cleanup Plan

### Overview
This guide helps developers migrate from the original 20 MCP functions to the new consolidated 19 functions. All original functions are now deprecated and will be removed in v2.0.

## Migration Timeline

### Phase 2: Deprecation Period (Current)
- **Status**: All original functions show deprecation warnings
- **Backward Compatibility**: ✅ All original functions still work
- **Action Required**: Update code to use new consolidated functions

### Phase 3: Removal Period (Future)
- **Timeline**: 4-6 weeks after Phase 2
- **Status**: Original functions will be removed
- **Action Required**: Complete migration to consolidated functions

## Function Migration Guide

### 🔍 SEARCH Functions (5 → 3)

#### 1. search_chunks → search_query
```python
# OLD (deprecated)
results = mcp__megamind__search_chunks("query text", limit=10)

# NEW (consolidated)
results = mcp__megamind__search_query("query text", limit=10, search_type="hybrid")
```

#### 2. get_chunk → search_retrieve
```python
# OLD (deprecated)
chunk = mcp__megamind__get_chunk("chunk_123", include_relationships=True)

# NEW (consolidated)
chunk = mcp__megamind__search_retrieve("chunk_123", include_relationships=True)
```

#### 3. get_related_chunks → search_related
```python
# OLD (deprecated)
related = mcp__megamind__get_related_chunks("chunk_123", max_depth=2)

# NEW (consolidated)
related = mcp__megamind__search_related("chunk_123", max_depth=2)
```

#### 4. search_chunks_semantic → search_query (semantic)
```python
# OLD (deprecated)
results = mcp__megamind__search_chunks_semantic("query", limit=10, threshold=0.7)

# NEW (consolidated)
results = mcp__megamind__search_query("query", search_type="semantic", limit=10, threshold=0.7)
```

#### 5. search_chunks_by_similarity → search_query (similarity)
```python
# OLD (deprecated)
results = mcp__megamind__search_chunks_by_similarity("chunk_ref", limit=10, threshold=0.7)

# NEW (consolidated)
results = mcp__megamind__search_query("", search_type="similarity", 
                                     reference_chunk_id="chunk_ref", limit=10, threshold=0.7)
```

### 📝 CONTENT Functions (4 → 4)

#### 1. create_chunk → content_create
```python
# OLD (deprecated)
result = mcp__megamind__create_chunk(content, "doc.md", "/section", "session_123")

# NEW (consolidated)
result = mcp__megamind__content_create(content, "doc.md", "session_123", create_relationships=True)
```

#### 2. update_chunk → content_update
```python
# OLD (deprecated)
result = mcp__megamind__update_chunk("chunk_123", new_content, "session_123")

# NEW (consolidated)
result = mcp__megamind__content_update("chunk_123", new_content, "session_123", update_embeddings=True)
```

#### 3. add_relationship → content_process
```python
# OLD (deprecated)
result = mcp__megamind__add_relationship("chunk_1", "chunk_2", "related", "session_123")

# NEW (consolidated)
result = mcp__megamind__content_process(action="add_relationship", 
                                       chunk_id_1="chunk_1", chunk_id_2="chunk_2",
                                       relationship_type="related", session_id="session_123")
```

#### 4. batch_generate_embeddings → content_manage
```python
# OLD (deprecated)
result = mcp__megamind__batch_generate_embeddings(["chunk_1", "chunk_2"], "realm_123")

# NEW (consolidated)
result = mcp__megamind__content_manage(action="batch_generate_embeddings", 
                                      chunk_ids=["chunk_1", "chunk_2"], realm_id="realm_123")
```

### 🚀 PROMOTION Functions (6 → 3)

#### 1. create_promotion_request → promotion_request
```python
# OLD (deprecated)
result = mcp__megamind__create_promotion_request("chunk_123", "GLOBAL", "justification", "session_123")

# NEW (consolidated)
result = mcp__megamind__promotion_request("chunk_123", "GLOBAL", "justification", "session_123")
```

#### 2. get_promotion_requests → promotion_monitor
```python
# OLD (deprecated)
requests = mcp__megamind__get_promotion_requests("pending", "GLOBAL", 20)

# NEW (consolidated)
requests = mcp__megamind__promotion_monitor(filter_status="pending", filter_realm="GLOBAL", limit=20)
```

#### 3. approve_promotion_request → promotion_review
```python
# OLD (deprecated)
result = mcp__megamind__approve_promotion_request("promo_123", "reason", "session_123")

# NEW (consolidated)
result = mcp__megamind__promotion_review("promo_123", action="approve", reason="reason", session_id="session_123")
```

#### 4. reject_promotion_request → promotion_review
```python
# OLD (deprecated)
result = mcp__megamind__reject_promotion_request("promo_123", "reason", "session_123")

# NEW (consolidated)
result = mcp__megamind__promotion_review("promo_123", action="reject", reason="reason", session_id="session_123")
```

#### 5. get_promotion_impact → promotion_review
```python
# OLD (deprecated)
impact = mcp__megamind__get_promotion_impact("promo_123")

# NEW (consolidated)
impact = mcp__megamind__promotion_review("promo_123", action="analyze", analyze_before=True)
```

#### 6. get_promotion_queue_summary → promotion_monitor
```python
# OLD (deprecated)
summary = mcp__megamind__get_promotion_queue_summary("GLOBAL")

# NEW (consolidated)
summary = mcp__megamind__promotion_monitor(filter_realm="GLOBAL", include_summary=True)
```

### 📊 SESSION Functions (3 → 4)

#### 1. get_session_primer → session_create
```python
# OLD (deprecated)
primer = mcp__megamind__get_session_primer("last_session_data")

# NEW (consolidated)
session = mcp__megamind__session_create("operational", "claude-code", "Session primer migration", auto_prime=True)
```

#### 2. get_pending_changes → session_manage
```python
# OLD (deprecated)
changes = mcp__megamind__get_pending_changes("session_123")

# NEW (consolidated)
changes = mcp__megamind__session_manage("session_123", action="get_pending")
```

#### 3. commit_session_changes → session_commit
```python
# OLD (deprecated)
result = mcp__megamind__commit_session_changes("session_123", ["change_1", "change_2"])

# NEW (consolidated)
result = mcp__megamind__session_commit("session_123", approved_changes=["change_1", "change_2"])
```

### 📈 ANALYTICS Functions (2 → 2)

#### 1. track_access → analytics_track
```python
# OLD (deprecated)
result = mcp__megamind__track_access("chunk_123", "query_context")

# NEW (consolidated)
result = mcp__megamind__analytics_track("chunk_123", track_type="access", metadata={"query_context": "query_context"})
```

#### 2. get_hot_contexts → analytics_insights
```python
# OLD (deprecated)
contexts = mcp__megamind__get_hot_contexts("sonnet", 20)

# NEW (consolidated)
contexts = mcp__megamind__analytics_insights(insight_type="hot_contexts", model_type="sonnet", limit=20)
```

## Benefits of Consolidated Functions

### 1. Intelligent Routing
- **Search Functions**: Single `search_query` function routes to appropriate search type
- **Promotion Functions**: Single `promotion_review` function handles approve/reject/analyze actions
- **Session Functions**: Single `session_manage` function handles various session operations

### 2. Enhanced Parameters
- **Optional Parameters**: Better defaults and more flexible parameter handling
- **Batch Operations**: Support for batch processing where applicable
- **Metadata Support**: Enhanced metadata and configuration options

### 3. Better Error Handling
- **Consistent Responses**: All functions return standardized response format
- **Validation**: Better parameter validation and error messages
- **Logging**: Enhanced logging and debugging capabilities

### 4. Future-Ready Architecture
- **Extensible**: Easy to add new functionality without breaking changes
- **Scalable**: Designed for high-performance operations
- **Maintainable**: Cleaner codebase with reduced duplication

## Deprecation Warning Examples

When using deprecated functions, you'll see warnings like:
```
WARNING: Function 'mcp__megamind__search_chunks' is deprecated and will be removed in v2.0. Use 'mcp__megamind__search_query' instead.
```

## Migration Checklist

### For Developers
- [ ] **Identify Usage**: Find all deprecated function calls in your code
- [ ] **Update Calls**: Replace with consolidated function equivalents
- [ ] **Test Functionality**: Verify all functionality works with new functions
- [ ] **Update Documentation**: Update any documentation or examples
- [ ] **Remove Deprecated Calls**: Clean up old function calls

### For System Administrators
- [ ] **Monitor Usage**: Track deprecated function usage in logs
- [ ] **Plan Migration**: Schedule migration work before v2.0 release
- [ ] **Update Configurations**: Update any configuration files using old names
- [ ] **Training**: Train team on new consolidated functions

## Support and Resources

### Documentation
- **Function Mapping**: `/docs/Function_Mapping_Documentation.md`
- **Phase 2 Implementation**: This migration guide
- **API Reference**: Updated function signatures and parameters

### Migration Tools
- **Deprecation Warnings**: Automatic warnings when using deprecated functions
- **Usage Statistics**: Track deprecated function usage
- **Backward Compatibility**: All old functions still work during transition

### Getting Help
- **GitHub Issues**: Report migration issues on GitHub
- **Documentation**: Complete API documentation available
- **Examples**: Migration examples for each function category

---

**Migration Status**: Phase 2 Active - All functions deprecated with warnings  
**Removal Timeline**: 4-6 weeks after Phase 2 deployment  
**Backward Compatibility**: Maintained until v2.0 release