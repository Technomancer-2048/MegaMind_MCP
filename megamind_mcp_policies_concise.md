# Megamind MCP Behavioral Policies

## Session Startup
1. Call `mcp__megamind__get_session_primer(last_session_data)` on conversation start
2. If active session: Load procedural context (workflow state, not knowledge chunks)
3. If no session: Prompt user to create/select session

## Context Retrieval
Before project-level tasks:
1. Use `mcp__megamind__search_chunks(query, limit=10, search_type="hybrid")`
2. For deeper context: `mcp__megamind__get_related_chunks(chunk_id, max_depth=2)`
3. Track access: `mcp__megamind__track_access(chunk_id, query_context)`
4. Include chunk IDs in responses for traceability

## Knowledge Capture
During development when significant findings emerge:
1. Buffer discoveries with appropriate MCP functions
2. Generate summary: `mcp__megamind__get_pending_changes(session_id)`
3. Present summary with impact assessment to user
4. On approval: `mcp__megamind__commit_session_changes(session_id, approved_changes)`

## Knowledge Promotion
For discoveries with broader applicability:
1. Create promotion request to GLOBAL realm with justification
2. User manages promotion queue via promotion functions