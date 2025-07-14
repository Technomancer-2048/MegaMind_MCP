# Megamind MCP Integration Policies

## Session Startup Protocol

```
1. Call `mcp__megamind__get_session_primer(last_session_data)` on conversation start
2. If active session found: Load procedural context (workflow state, current focus)  
3. If no active session: Prompt user to create/select session
4. Session primer = procedural workflow context, NOT knowledge chunks
```

## Pre-Task Context Retrieval

```
Before project-level tasks requiring domain knowledge:
1. Primary: `mcp__megamind__search_chunks(query, limit=10, search_type="hybrid")`
   - Semantic search with keyword fallback for safety
2. Deep context: `mcp__megamind__get_related_chunks(chunk_id, max_depth=2)` 
   - Semantic relationship traversal for discovered chunks
3. Track usage: `mcp__megamind__track_access(chunk_id, query_context)`
4. Always include chunk IDs in responses for traceability
```

## Knowledge Capture Workflow

```
During debugging/development when significant findings emerge:
1. Buffer discoveries:
   - New knowledge: `mcp__megamind__create_chunk(content, source_doc, section_path, session_id)`
   - Updates: `mcp__megamind__update_chunk(chunk_id, new_content, session_id)`
   - Cross-references: `mcp__megamind__add_relationship(chunk_id_1, chunk_id_2, type, session_id)`
2. Generate summary: `mcp__megamind__get_pending_changes(session_id)`
3. Present summary with impact assessment to user
4. On approval: `mcp__megamind__commit_session_changes(session_id, approved_changes)`
```

## Knowledge Promotion Protocol

```
For discoveries with broader applicability beyond current project:
1. Request promotion: `mcp__megamind__create_promotion_request(chunk_id, "GLOBAL", justification, session_id)`
2. Impact analysis: `mcp__megamind__get_promotion_impact(promotion_id)`
3. User manages queue via promotion management functions
```

## Available MCP Functions Reference

### Search & Retrieval Functions (5)
- `mcp__megamind__search_chunks(query, limit=10, search_type="hybrid")` - Enhanced dual-realm search with hybrid semantic capabilities
- `mcp__megamind__get_chunk(chunk_id, include_relationships=true)` - Get specific chunk by ID with relationships
- `mcp__megamind__get_related_chunks(chunk_id, max_depth=2)` - Get chunks related to specified chunk
- `mcp__megamind__search_chunks_semantic(query, limit=10, threshold=0.7)` - Pure semantic search across Global + Project realms
- `mcp__megamind__search_chunks_by_similarity(reference_chunk_id, limit=10, threshold=0.7)` - Find chunks similar to a reference chunk using embeddings

### Content Management Functions (4)
- `mcp__megamind__create_chunk(content, source_document, section_path, session_id, target_realm="PROJECT")` - Buffer new knowledge creation with realm targeting and embedding generation
- `mcp__megamind__update_chunk(chunk_id, new_content, session_id)` - Buffer chunk modifications for review
- `mcp__megamind__add_relationship(chunk_id_1, chunk_id_2, relationship_type, session_id)` - Create cross-references between chunks
- `mcp__megamind__batch_generate_embeddings(chunk_ids=[], realm_id="")` - Generate embeddings for existing chunks in batch

### Knowledge Promotion Functions (6)
- `mcp__megamind__create_promotion_request(chunk_id, target_realm, justification, session_id)` - Request promotion of knowledge chunk to different realm
- `mcp__megamind__get_promotion_requests(filter_status="", filter_realm="", limit=20)` - Retrieve list of promotion requests with filtering
- `mcp__megamind__approve_promotion_request(promotion_id, approval_reason, session_id)` - Approve pending promotion request
- `mcp__megamind__reject_promotion_request(promotion_id, rejection_reason, session_id)` - Reject promotion request with reason
- `mcp__megamind__get_promotion_impact(promotion_id)` - Analyze potential impact of promotion on target realm
- `mcp__megamind__get_promotion_queue_summary(filter_realm="")` - Get summary statistics of promotion queue

### Session Management Functions (3)
- `mcp__megamind__get_session_primer(last_session_data="")` - Generate lightweight context for session continuity
- `mcp__megamind__get_pending_changes(session_id)` - Get pending changes with smart highlighting
- `mcp__megamind__commit_session_changes(session_id, approved_changes)` - Commit approved changes and track contributions

### Analytics & Optimization Functions (2)
- `mcp__megamind__track_access(chunk_id, query_context="")` - Update access analytics for optimization
- `mcp__megamind__get_hot_contexts(model_type="sonnet", limit=20)` - Get frequently accessed chunks prioritized by usage patterns

## Implementation Notes

### Hybrid Search Behavior
- **Primary Strategy:** Semantic search with automatic keyword fallback
- **Safety First:** Hybrid search ensures results even when semantic search fails
- **Relationship Traversal:** `get_related_chunks()` uses semantic search for connection discovery

### Session Context vs Knowledge Chunks
- **Session Primer:** Procedural workflow state (what we were working on, next steps)
- **Knowledge Chunks:** Factual content retrieved via search functions
- **Separation of Concerns:** Primer provides workflow continuity, chunks provide domain knowledge

### Knowledge Capture Best Practices
- **Buffer Changes:** All modifications remain pending until user approval
- **Impact Assessment:** Review summaries highlight critical vs. routine changes
- **Traceability:** Include chunk IDs in all responses for reference tracking
- **Promotion Path:** Move project-specific discoveries to global knowledge when applicable