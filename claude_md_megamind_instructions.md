## MegaMind Database Integration

### MegaMind MCP Server - Intelligent Context Retrieval System
**Purpose:** Replace inefficient markdown file loading with semantic chunking database system to eliminate context exhaustion
**Context Reduction:** 70-80% token savings through precise, relevant context retrieval
**Model Optimization:** Enables Opus 4 usage for strategic analysis through concentrated context delivery

#### Core MegaMind Functions

**Context Retrieval (Primary Usage):**
- `mcp__megamind_db__search_chunks(query, limit=10, model_type="sonnet")` - Search for relevant context chunks with model-specific optimization
- `mcp__megamind_db__get_chunk(chunk_id, include_relationships=true)` - Retrieve specific chunk with metadata and relationships
- `mcp__megamind_db__get_related_chunks(chunk_id, max_depth=2)` - Find connected chunks through relationship graph
- `mcp__megamind_db__get_hot_contexts(model_type, limit=20)` - Get frequently accessed chunks prioritized by usage patterns

**Session Management:**
- `mcp__megamind_db__get_session_primer(last_session_data)` - Generate lightweight context for session continuity
- `mcp__megamind_db__track_access(chunk_id, query_context)` - Update access analytics for optimization

**Knowledge Enhancement (Bidirectional Flow):**
- `mcp__megamind_db__update_chunk(chunk_id, new_content, session_id)` - Buffer chunk modifications for review
- `mcp__megamind_db__create_chunk(content, source_document, section_path, session_id)` - Buffer new knowledge creation
- `mcp__megamind_db__add_relationship(chunk_id_1, chunk_id_2, relationship_type, session_id)` - Create cross-references

#### MegaMind Update Reference Mechanism

**Critical Workflow:** When AI discovers patterns, optimizations, or corrections during development work:

**Step 1: Reference Context with Chunk IDs**
```
Query: "SQL trigger creation standards"
Response includes: chunk_id "sql_standards_trigger_001" with content
AI can reference: "Based on chunk sql_standards_trigger_001, I recommend..."
```

**Step 2: Buffer Knowledge Updates**
```
When AI finds improvements:
mcp__megamind_db__update_chunk(
    chunk_id="sql_standards_trigger_001", 
    new_content="Original content + discovered edge case handling",
    session_id="current_session"
)

When AI discovers new patterns:
mcp__megamind_db__create_chunk(
    content="New MySQL 8.0 trigger optimization pattern",
    source_document="DISCOVERED_PATTERNS.md",
    section_path="/sql/triggers/optimizations",
    session_id="current_session"
)

When AI finds relationships:
mcp__megamind_db__add_relationship(
    chunk_id_1="sql_standards_trigger_001",
    chunk_id_2="mysql_performance_002", 
    relationship_type="enhances",
    session_id="current_session"
)
```

**Step 3: Session-End Review Process**
```
Review pending changes:
mcp__megamind_db__get_pending_changes(session_id="current_session")

Returns summary with smart highlighting:
🔴 CRITICAL: Modified sql_standards_trigger_001 (47 accesses) - Core trigger patterns
🟡 IMPORTANT: Created new optimization chunk with 3 relationships
🟢 STANDARD: Documentation improvement to error handling examples

Commit approved changes:
mcp__megamind_db__commit_session_changes(
    session_id="current_session",
    approved_changes=["change_id_1", "change_id_3"]  // Selective approval
)
```

#### MegaMind Usage Patterns

**For Daily Development (Sonnet 4):**
1. Query MegaMind for specific context: `search_chunks("SQL table creation", limit=8)`
2. Receive precise, relevant chunks (200-400 tokens vs 14,600+ tokens previously)
3. Complete development work with targeted context
4. Buffer any discoveries/improvements during session
5. Review and commit knowledge enhancements at session end

**For Strategic Analysis (Opus 4):**
1. Use curated context: `get_hot_contexts(model_type="opus", limit=15)`
2. Get concentrated, battle-tested knowledge from hundreds of previous sessions
3. Perform complex architectural analysis with precision context
4. Generate high-level insights and system improvements

**For Session Continuity:**
1. Check session primer: `get_session_primer(last_session_data)`
2. AI presents: "Resume previous session on spatial system development? Last session completed trigger optimization work."
3. User confirms, receives targeted context for seamless continuation

#### Integration with Existing Workflow

**MegaMind Replaces:**
- Loading entire markdown files (SQL_STANDARDS.md, MASTER_FUNCTIONS.sql, etc.)
- Context exhaustion from broad document loading
- Manual cross-reference discovery across multiple files

**MegaMind Enhances:**
- Existing session tracking (maintains compatibility with current session documents)
- Development workflow automation (integrates with existing scripts)
- Knowledge evolution (AI contributions improve system over time)

**Compatibility:**
- Works alongside existing MCP servers (textsmith, filesystem, etc.)
- No conflicts with current project knowledge search
- Maintains all existing development tools and patterns

#### MegaMind Success Indicators

**Immediate Benefits:**
- Context consumption drops 70-80% for typical development tasks
- Opus 4 becomes viable for regular strategic analysis  
- Cross-contextual queries find relationships across multiple source documents
- Session startup provides targeted context without loading entire documents

**Long-term Evolution:**
- Knowledge base becomes more interconnected through AI relationship discovery
- Cold context cleanup identifies obsolete patterns for system optimization
- AI contributions enhance utility beyond original markdown content
- Developer productivity increases through precise context delivery

#### Important Notes

**Always Use MegaMind For:**
- Any development task requiring standards, patterns, or technical references
- Cross-system analysis (spatial + entity + SQL requirements)
- Session continuity when resuming complex development work
- Knowledge discovery and pattern enhancement during development

**MegaMind Session Buffer:**
- All knowledge updates are session-scoped and require manual approval
- Changes are NOT committed automatically - review process prevents knowledge corruption
- Rollback capability ensures system reliability and knowledge quality
- Smart highlighting prioritizes review of high-impact changes

**Model Optimization:**
- Sonnet 4: Efficient daily development with hot context prioritization
- Opus 4: Strategic analysis with curated, concentrated knowledge delivery
- Context limits enforced per model type to prevent context exhaustion
- Usage analytics drive continuous optimization of retrieval patterns