# Context Database System - Claude Code Execution Plan

## Overview

This execution plan provides step-by-step implementation guidance for Claude Code to build the Context Database System. Each phase includes specific deliverables, validation criteria, and integration points.

## Phase 1: Core Infrastructure (Weeks 1-2)

### 1.1 Database Schema Design
**Directory:** `database/context_system/`

**Tasks:**
- Create `context_chunks` table with columns:
  - `chunk_id` (PRIMARY KEY, VARCHAR(50))
  - `content` (TEXT)
  - `source_document` (VARCHAR(255))
  - `section_path` (VARCHAR(500))
  - `chunk_type` (ENUM: 'rule', 'function', 'section', 'example')
  - `line_count` (INT)
  - `created_at` (TIMESTAMP)
  - `last_accessed` (TIMESTAMP)
  - `access_count` (INT DEFAULT 0)

- Create `chunk_relationships` table:
  - `relationship_id` (PRIMARY KEY, AUTO_INCREMENT)
  - `chunk_id` (VARCHAR(50), FOREIGN KEY)
  - `related_chunk_id` (VARCHAR(50), FOREIGN KEY)
  - `relationship_type` (ENUM: 'references', 'depends_on', 'contradicts', 'enhances')
  - `strength` (DECIMAL(3,2))
  - `discovered_by` (ENUM: 'manual', 'ai_analysis', 'usage_pattern')

- Create `chunk_tags` table:
  - `tag_id` (PRIMARY KEY, AUTO_INCREMENT)
  - `chunk_id` (VARCHAR(50), FOREIGN KEY)
  - `tag_type` (ENUM: 'subsystem', 'function_type', 'applies_to', 'language')
  - `tag_value` (VARCHAR(100))

**Validation:** Schema loads without errors, foreign key constraints work

### 1.2 Markdown Ingestion Tool
**File:** `tools/markdown_ingester.py`

**Functionality:**
- Parse markdown files by semantic boundaries (headers, code blocks, logical sections)
- Generate chunk IDs based on source + section (e.g., `sql_standards_triggers_001`)
- Extract metadata (line count, section hierarchy, content type)
- Preserve original formatting and cross-references
- Batch insert chunks with transaction safety

**Input:** Directory path to existing markdown documentation
**Output:** Populated `context_chunks` table with initial metadata

**Validation:** Process existing documentation without data loss, chunk boundaries make semantic sense

### 1.3 MCP Server Foundation
**File:** `mcp_server/context_database_server.py`

**Core MCP Functions:**
- `mcp__context_db__search_chunks(query, limit=10)` - Basic similarity search returning chunk content + metadata
- `mcp__context_db__get_chunk(chunk_id)` - Retrieve specific chunk with full metadata
- `mcp__context_db__track_access(chunk_id, query_context)` - Update access statistics

**Database Integration:**
- Direct database connection (no file system dependencies)
- Transaction handling for data consistency
- Error handling and connection management
- Logging for debugging and monitoring

**Validation:** MCP functions return correct data, access tracking updates database, error handling works

## Phase 2: Intelligence Layer (Weeks 3-4)

### 2.1 Semantic Analysis Engine
**File:** `analysis/semantic_analyzer.py`

**Functionality:**
- Generate embeddings for chunk content using sentence transformers
- Discover relationships between chunks based on semantic similarity
- Identify cross-references and dependencies automatically
- Tag chunks with subsystem/function type classifications

**Process:**
- Batch process all chunks to generate embeddings
- Calculate similarity matrices for relationship discovery
- Auto-populate `chunk_relationships` table with discovered connections
- Generate and apply semantic tags

**Validation:** Relationship discovery finds logical connections, tagging accuracy >80%

### 2.2 Context Analytics Dashboard
**File:** `dashboard/context_analytics.py` (Flask app)

**Features:**
- Usage heatmap showing hot/cold chunks
- Relationship visualization (network graph)
- Search pattern analysis
- Context efficiency metrics (tokens saved vs. baseline)

**Metrics Tracked:**
- Most/least accessed chunks
- Average context size per query type
- Cross-reference utilization
- Token consumption trends

**Validation:** Dashboard accurately reflects usage patterns, identifies optimization opportunities

### 2.3 Enhanced MCP Functions
**File:** `mcp_server/context_database_server.py` (expansion)

**Advanced MCP Functions:**
- `mcp__context_db__get_related_chunks(chunk_id, max_depth=2)` - Traverse relationship graph
- `mcp__context_db__get_session_primer(last_session_data)` - Generate lightweight context for session continuity
- `mcp__context_db__search_by_tags(tag_type, tag_value, limit=10)` - Tag-based retrieval

**Integration with CLAUDE.md:**
- Parse session metadata from CLAUDE.md when available
- Generate primer context based on recent project activity
- Return structured context for session restoration decisions
- No file system modification - read-only session state detection

**Validation:** Primer context reduces cold-start confusion, maintains session continuity through MCP interface

## Phase 3: Bidirectional Flow (Weeks 5-6)

### 3.1 Bidirectional MCP Functions
**File:** `mcp_server/context_database_server.py` (expansion)

**Knowledge Update MCP Functions:**
- `mcp__context_db__update_chunk(chunk_id, new_content, session_id)` - Buffer chunk modifications
- `mcp__context_db__create_chunk(content, source_document, section_path, session_id)` - Buffer new chunk creation
- `mcp__context_db__add_relationship(chunk_id_1, chunk_id_2, relationship_type, session_id)` - Buffer relationship changes
- `mcp__context_db__get_pending_changes(session_id)` - Retrieve buffered changes for review

**Database additions:**
- `session_changes` table:
  - `session_id` (VARCHAR(50))
  - `change_type` (ENUM: 'update', 'create', 'relate')
  - `chunk_id` (VARCHAR(50))
  - `change_data` (JSON)
  - `timestamp` (TIMESTAMP)

**Functionality:**
- All changes buffered in database, not committed until approval
- Session-scoped change tracking through MCP interface
- Validate change consistency before allowing commit
- Generate change summaries accessible via MCP

**Validation:** Changes properly buffered through MCP interface, no data loss, consistency checks work

### 3.2 Manual Review Interface
**File:** `review/change_reviewer.py` (Web interface)

**Features:**
- Summary view: modified/created/related counts with impact assessment
- Detailed view: diff-style changes with before/after content
- Smart highlighting: critical changes (high-traffic chunks) vs. standard updates
- Selective approval: approve/reject individual changes or batches

**Priority Matrix:**
- 🔴 Critical: High-traffic chunks (>50 accesses), core system patterns
- 🟡 Important: Medium-traffic (10-50 accesses), cross-system updates  
- 🟢 Standard: Low-traffic (<10 accesses), documentation improvements

**Validation:** Review interface correctly prioritizes changes, approval process works

### 3.3 MCP Change Management
**File:** `mcp_server/context_database_server.py` (expansion)

**Change Management MCP Functions:**
- `mcp__context_db__commit_session_changes(session_id, approved_changes)` - Commit specific approved changes
- `mcp__context_db__rollback_session_changes(session_id)` - Discard all pending changes
- `mcp__context_db__get_change_summary(session_id)` - Generate review summary with impact analysis

**Database additions:**
- `knowledge_contributions` table:
  - `contribution_id` (PRIMARY KEY)
  - `session_id` (VARCHAR(50))
  - `chunks_modified` (INT)
  - `chunks_created` (INT)
  - `relationships_added` (INT)
  - `commit_timestamp` (TIMESTAMP)
  - `rollback_available` (BOOLEAN)

**Functionality:**
- Track AI contributions to knowledge base through MCP interface
- Maintain rollback capability for committed changes
- Generate contribution reports and impact analysis via MCP
- Monitor knowledge evolution patterns

**Validation:** Contributions properly tracked via MCP, rollback functionality works through MCP interface

## Phase 4: Advanced Optimization (Weeks 7-8)

### 4.1 Model-Optimized MCP Functions
**File:** `mcp_server/context_database_server.py` (expansion)

**Model-Specific MCP Functions:**
- `mcp__context_db__search_chunks(query, limit=10, model_type="sonnet")` - Model-optimized context delivery
- `mcp__context_db__get_hot_contexts(model_type, limit=20)` - Priority chunks for Opus with usage-based sorting
- `mcp__context_db__get_curated_context(query, model_type="opus", max_tokens=1000)` - Token-budgeted context assembly

**Functionality:**
- Sonnet 4: Efficient context delivery with hot chunk prioritization via MCP
- Opus 4: Curated, concentrated context with relationship emphasis via MCP
- Context limit enforcement per model type through MCP parameters
- Usage pattern analysis for model-specific optimization accessible via MCP

**Features:**
- Model-specific context assembly strategies executed through MCP calls
- Token budget management and enforcement via MCP function parameters
- Performance metrics per model tier tracked in database
- Automatic context refinement based on success patterns

**Validation:** Context consumption meets 70-80% reduction targets via MCP interface, model tier optimization works

### 4.2 Automated Curation System
**File:** `curation/auto_curator.py`

**Functionality:**
- Threshold-based identification of cold chunks (configurable, e.g., 60 days no access)
- Opus 4 integration for intelligent curation decisions
- Automated relationship consolidation for related chunks
- Cleanup recommendations with impact analysis

**Process:**
- Identify underutilized chunks based on access patterns
- Generate curation recommendations using Opus 4 analysis
- Present consolidation/removal suggestions with rationale
- Execute approved cleanup operations

**Validation:** Curation maintains knowledge quality while reducing bloat

### 4.3 System Health Monitoring
**File:** `monitoring/system_health.py`

**Metrics:**
- Context efficiency: average tokens per query vs. baseline
- Retrieval accuracy: relevant chunks returned per query
- Knowledge growth: new chunks and relationships over time
- System performance: query response times, database size

**Alerting:**
- Degraded retrieval performance
- Excessive knowledge bloat
- Relationship integrity issues
- Context efficiency below targets

**Validation:** Monitoring accurately reflects system health, alerts fire appropriately

## Integration Points

### 4.4 CLAUDE.md Integration via MCP
- MCP functions read session status from CLAUDE.md when available (read-only)
- Include context database connection status accessible via MCP
- Reference chunk IDs in development workflow through MCP responses
- Maintain compatibility with existing workflow tools through MCP interface
- **No file modification** - MCP operates as pure database interface

### 4.5 Freestanding MCP Server
**Server:** `context_database_mcp_server`
**Purpose:** Standalone MCP interface for direct database interaction

**MCP Functions to implement:**
- `mcp__context_db__search_chunks(query, limit=10, model_type="sonnet")`
- `mcp__context_db__get_chunk(chunk_id, include_relationships=true)`
- `mcp__context_db__get_related_chunks(chunk_id, max_depth=2)`
- `mcp__context_db__update_chunk(chunk_id, new_content, session_id)`
- `mcp__context_db__create_chunk(content, source_document, section_path, session_id)`
- `mcp__context_db__add_relationship(chunk_id_1, chunk_id_2, relationship_type, session_id)`
- `mcp__context_db__get_session_primer(last_session_data)`
- `mcp__context_db__commit_session_changes(session_id, approved_changes)`
- `mcp__context_db__get_pending_changes(session_id)`
- `mcp__context_db__track_access(chunk_id, query_context)`

**No file system dependencies** - operates entirely through database connections
**Direct agent interaction** - AI calls MCP functions directly for context retrieval and updates
**Session management** - handles pending changes buffer through MCP interface
**Independent operation** - does not rely on textsmith or other file-based MCP servers

## Success Criteria

**Phase 1 Complete:** Core infrastructure operational, basic retrieval working
**Phase 2 Complete:** Intelligent retrieval with analytics, session primer functional  
**Phase 3 Complete:** Bidirectional flow operational with review system
**Phase 4 Complete:** Multi-tier optimization and automated curation working

**Overall Success:** 70-80% context reduction achieved, Opus 4 viable for regular use, knowledge quality maintained or improved.

## Deployment Strategy

1. **Development Environment:** Test with subset of documentation
2. **Validation:** Parallel operation with existing system for comparison
3. **Migration:** Gradual replacement of markdown loading with database retrieval
4. **Production:** Full deployment with monitoring and rollback capability

This execution plan provides Claude Code with concrete, implementable steps while maintaining flexibility for iteration and optimization based on testing results.