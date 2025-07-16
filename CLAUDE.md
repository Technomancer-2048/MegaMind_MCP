# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ⚠️ HIGH IMPORTANCE - MegaMind MCP Behavioral Policies

### 🔄 Session Startup Protocol
1. **ALWAYS** call `mcp__megamind__get_session_primer(last_session_data)` on conversation start
2. If active session exists: Load procedural context (workflow state, not knowledge chunks)
3. If no session exists: Prompt user to create/select session

### 🔍 Context Retrieval Protocol
**Before ANY project-level tasks**:
1. Use `mcp__megamind__search_chunks(query, limit=10, search_type="hybrid")` for initial context
2. For deeper relationships: `mcp__megamind__get_related_chunks(chunk_id, max_depth=2)`
3. **ALWAYS** track access: `mcp__megamind__track_access(chunk_id, query_context)`
4. Include chunk IDs in responses for complete traceability

### 💾 Knowledge Capture Protocol  
**During development when significant findings emerge**:
1. Buffer discoveries using appropriate MCP functions (`create_chunk`, `update_chunk`, `add_relationship`)
2. Generate session summary: `mcp__megamind__get_pending_changes(session_id)`
3. Present summary with impact assessment to user for review
4. On user approval: `mcp__megamind__commit_session_changes(session_id, approved_changes)`

### 🚀 Knowledge Promotion Protocol
**For discoveries with broader applicability**:
1. Create promotion request to GLOBAL realm with clear justification using `mcp__megamind__create_promotion_request`
2. User manages promotion queue via promotion system functions
3. Track promotion impact and maintain audit trail

**CRITICAL**: These protocols ensure proper knowledge management, session continuity, and systematic capture of development insights.

---

## Project Overview

This is the **Context Database System** - an MCP server designed to eliminate context exhaustion in AI development workflows by replacing large markdown file loading with precise, semantically-chunked database retrieval.

### Core Problem
Current markdown-based knowledge systems consume 14,600+ tokens for simple tasks, making high-capability models like Opus 4 practically unusable due to context limitations.

### Solution Architecture
- **Semantic Chunking**: Break documentation into 20-150 line coherent chunks
- **Database Storage**: Metadata-rich storage with cross-references and usage tracking
- **Intelligent Retrieval**: AI-driven context assembly with relevance scoring
- **Bidirectional Flow**: AI contributions enhance the knowledge base through review cycles

## Project Structure

This is a **production-ready MCP server** with the following structure:

### Core MCP Server (`mcp_server/`)
- `megamind_database_server.py` - Main MCP server implementation with all 20 functions
- `realm_aware_database.py` - Realm-aware database operations and dual-access patterns
- `stdio_http_bridge.py` - **STDIO-to-HTTP bridge for Claude Code connectivity** ✨
- `http_transport.py` - HTTP transport for direct API access
- `realm_manager_factory.py` - Dynamic realm management and configuration
- `services/` - Embedding, caching, and vector search services

### Database & Schema (`database/`, `mcp_server/`)
- `complete_schema.sql` - Full production database schema with all 13 tables
- `database/realm_system/` - Realm inheritance and promotion system schemas
- `database/context_system/` - Legacy context system schemas

### Tools & Utilities (`tools/`)
- `realm_aware_markdown_ingester.py` - Bulk knowledge ingestion with realm support
- `bulk_semantic_ingester.py` - Advanced semantic processing pipeline
- `markdown_ingester.py` - Basic markdown processing

### Configuration & Deployment
- `docker-compose.yml` - Production container orchestration
- `.env` - Environment configuration for database credentials and realm settings
- `.mcp.json` - **Claude Code MCP configuration with STDIO bridge** ✨
- `scripts/` - Deployment, migration, and testing scripts

### Planning & Documentation (`planning/`, `guides/`)
- `context_db_project_mission.md` - Project mission and success metrics
- `context_db_execution_plan.md` - Implementation phases and architecture
- `guides/claude-code-quickstart.md` - Claude Code integration guide

## Development Phases

### Phase 1: Core Infrastructure (Weeks 1-2)
- Database schema design (`context_chunks`, `chunk_relationships`, `chunk_tags` tables)
- Markdown ingestion tool (`tools/markdown_ingester.py`)
- Basic MCP server foundation (`mcp_server/context_database_server.py`)

### Phase 2: Intelligence Layer (Weeks 3-4)
- Semantic analysis engine (`analysis/semantic_analyzer.py`)
- Context analytics dashboard (`dashboard/context_analytics.py`)
- Enhanced MCP functions with relationship traversal

### Phase 3: Bidirectional Flow (Weeks 5-6)
- Session-scoped change buffering through MCP interface
- Manual review interface (`review/change_reviewer.py`)
- Change management and rollback capabilities

### Phase 4: Advanced Optimization (Weeks 7-8)
- Model-specific optimization (Sonnet vs Opus context strategies)
- Automated curation system (`curation/auto_curator.py`)
- System health monitoring (`monitoring/system_health.py`)

## Key Implementation Requirements

### MCP Server Functions
The core MCP server implements these functions with **realm-aware dual-access**:

#### **Search & Retrieval Functions (5)**
- `mcp__megamind__search_chunks(query, limit=10, search_type="hybrid")` - Enhanced dual-realm search with hybrid semantic capabilities
- `mcp__megamind__get_chunk(chunk_id, include_relationships=true)` - Get specific chunk by ID with relationships
- `mcp__megamind__get_related_chunks(chunk_id, max_depth=2)` - Get chunks related to specified chunk
- `mcp__megamind__search_chunks_semantic(query, limit=10, threshold=0.7)` - Pure semantic search across Global + Project realms
- `mcp__megamind__search_chunks_by_similarity(reference_chunk_id, limit=10, threshold=0.7)` - Find chunks similar to a reference chunk using embeddings

#### **Content Management Functions (4)**
- `mcp__megamind__create_chunk(content, source_document, section_path, session_id, target_realm="PROJECT")` - Buffer new knowledge creation with realm targeting and embedding generation
- `mcp__megamind__update_chunk(chunk_id, new_content, session_id)` - Buffer chunk modifications for review
- `mcp__megamind__add_relationship(chunk_id_1, chunk_id_2, relationship_type, session_id)` - Create cross-references between chunks
- `mcp__megamind__batch_generate_embeddings(chunk_ids=[], realm_id="")` - Generate embeddings for existing chunks in batch

#### **Knowledge Promotion Functions (6) - NEW**
- `mcp__megamind__create_promotion_request(chunk_id, target_realm, justification, session_id)` - Request promotion of knowledge chunk to different realm
- `mcp__megamind__get_promotion_requests(filter_status="", filter_realm="", limit=20)` - Retrieve list of promotion requests with filtering
- `mcp__megamind__approve_promotion_request(promotion_id, approval_reason, session_id)` - Approve pending promotion request
- `mcp__megamind__reject_promotion_request(promotion_id, rejection_reason, session_id)` - Reject promotion request with reason
- `mcp__megamind__get_promotion_impact(promotion_id)` - Analyze potential impact of promotion on target realm
- `mcp__megamind__get_promotion_queue_summary(filter_realm="")` - Get summary statistics of promotion queue

#### **Session Management Functions (3)**
- `mcp__megamind__get_session_primer(last_session_data="")` - Generate lightweight context for session continuity
- `mcp__megamind__get_pending_changes(session_id)` - Get pending changes with smart highlighting
- `mcp__megamind__commit_session_changes(session_id, approved_changes)` - Commit approved changes and track contributions

#### **Analytics & Optimization Functions (2)**
- `mcp__megamind__track_access(chunk_id, query_context="")` - Update access analytics for optimization
- `mcp__megamind__get_hot_contexts(model_type="sonnet", limit=20)` - Get frequently accessed chunks prioritized by usage patterns

#### **Phase 2 Enhanced Embedding Functions (7) - NEW**
- `mcp__megamind__content_analyze_document(content, document_name, session_id, metadata)` - Analyze document structure with semantic boundary detection
- `mcp__megamind__content_create_chunks(content, document_name, session_id, strategy, max_tokens, target_realm)` - Create optimized chunks with intelligent strategies
- `mcp__megamind__content_assess_quality(chunk_ids, session_id, include_context)` - 8-dimensional quality assessment of chunks
- `mcp__megamind__content_optimize_embeddings(chunk_ids, session_id, model, cleaning_level, batch_size)` - Optimize chunks for embedding generation
- `mcp__megamind__session_create(session_type, created_by, description, metadata)` - Create new embedding processing sessions
- `mcp__megamind__session_get_state(session_id)` - Get current session state and progress tracking
- `mcp__megamind__session_complete(session_id)` - Complete and finalize processing sessions

**Total MCP Functions**: 37 (5 Search + 4 Content + 6 Promotion + 3 Session + 2 Analytics + 7 Phase 2 Enhanced Embedding + 10 Phase 3 Knowledge & Session) (5 Search + 4 Content + 6 Promotion + 3 Session + 2 Analytics + 7 Phase 2 Enhanced Embedding)

### Database Schema Design (MegaMind Naming Convention)
- **Primary Tables**: `megamind_chunks`, `megamind_chunk_relationships`, `megamind_chunk_tags`
- **Realm Management**: `megamind_realms`, `megamind_realm_inheritance`
- **Change Management**: `megamind_session_changes`, `megamind_knowledge_contributions`
- **Promotion System**: `megamind_promotion_queue`, `megamind_promotion_history`, `megamind_promotion_impact`
- **Intelligence Layer**: `megamind_embeddings` (with realm support)
- **Analytics**: `megamind_performance_metrics`, `megamind_system_health`
- **Usage Tracking**: Access tracking, usage patterns, relationship discovery

### Integration Strategy
- **No File System Dependencies**: Pure database interface through MCP
- **Read-Only CLAUDE.md Integration**: Session state detection without modification
- **Independent Operation**: Standalone MCP server for direct AI interaction

## Success Criteria
- **Context Reduction**: 70-80% reduction in token consumption
- **Model Accessibility**: Enable regular Opus 4 usage for strategic analysis
- **Knowledge Quality**: Measurable improvement in cross-contextual discovery
- **Performance**: Sub-second retrieval for interactive workflows

## MCP Usage Patterns

### Claude Code Connection (Primary Method)
**CURRENT**: Claude Code connects via STDIO-to-HTTP bridge with built-in security:

#### Connection Architecture
```
Claude Code (STDIO) → stdio_http_bridge.py → HTTP MCP Server (10.255.250.22:8080) → Database
```

#### Security Features
- **Realm Access Control**: GLOBAL realm access blocked, only PROJECT and MegaMind_MCP allowed
- **Request Sanitization**: All requests filtered before reaching HTTP backend  
- **Graceful Degradation**: Blocked requests forced to PROJECT realm (no failures)
- **Audit Logging**: All security violations logged with warnings

#### Configuration in `.mcp.json`
```json
"megamind-context-db": {
  "command": "python3",
  "args": ["/Data/MCP_Servers/MegaMind_MCP/mcp_server/stdio_http_bridge.py"],
  "env": {
    "MEGAMIND_PROJECT_REALM": "MegaMind_MCP",
    "MEGAMIND_PROJECT_NAME": "MegaMind Context Database", 
    "MEGAMIND_DEFAULT_TARGET": "PROJECT"
  }
}
```

### MegaMind Realm Configuration
**CRITICAL**: The MegaMind MCP server uses realm-aware database operations with environment-based configuration:

- **Realm Configuration**: The server's realm context is established at startup via environment variables in `.mcp.json`
- **NO JSON-RPC Realm Passing**: NEVER attempt to pass realm configuration through JSON-RPC protocol calls - this will break the realm system
- **Environment Variables Required**:
  - `MEGAMIND_ROOT` - Base path for all MegaMind resources (optional - can auto-detect)
  - `MEGAMIND_PROJECT_REALM` - Target project realm (e.g., "MegaMind_MCP")
  - `MEGAMIND_PROJECT_NAME` - Project display name
  - `MEGAMIND_DEFAULT_TARGET` - Default operation target ("PROJECT")
- **Intelligent Path Resolution**: The server automatically configures all cache and module paths from `MEGAMIND_ROOT`
- **Single Realm Per Server**: Each MCP server instance is bound to one realm configuration
- **Inheritance Aware**: The server automatically accesses inherited realms (GLOBAL + PROJECT) based on its configuration

### Textsmith MCP Integration
**IMPORTANT**: Use the `textsmith` MCP for all large file operations and code handling:

- **Large File Processing**: Use `mcp__textsmith__load_file_to_register` for files over 500 lines
- **Code Safety**: Use `mcp__textsmith__safe_replace_text` and `mcp__textsmith__safe_replace_block` for code modifications
- **Content Management**: Leverage textsmith registers for temporary content staging and processing
- **Multi-file Operations**: Use `mcp__textsmith__load_directory_to_registers` for batch processing

### Path Translation for Textsmith
**CRITICAL**: Textsmith MCP uses a different path mapping:
- **Local path**: `/Data/MCP_Servers/MegaMind_MCP`
- **Textsmith path**: `/app/workspace`

When using textsmith functions, translate paths:
```
Local: /Data/MCP_Servers/MegaMind_MCP/some/file.py
Textsmith: /app/workspace/some/file.py
```

Example usage:
```
# Load local file into textsmith register
mcp__textsmith__load_file_to_register(
    path="/app/workspace/mcp_server/context_database_server.py",
    register_name="server_code"
)
```

### Safe Code Handling Practices
When working with code in this project:

1. **Always use textsmith for code modifications** - Use `safe_replace_text` with `mode="literal"` for exact matches
2. **Use registers for staging** - Load code into textsmith registers before making changes
3. **Block-level replacements** - Use `safe_replace_block` for entire function/class replacements
4. **Validate before commit** - Use textsmith's analysis tools to verify changes before applying

### MCP Function Usage Priority
1. **MegaMind Context Database** - For semantic chunk retrieval and knowledge management (realm-aware)
2. **Textsmith** - For all file operations, code modifications, content processing
3. **SQL Files** - For database schema operations and query optimization
4. **Quick Data** - For analytics and usage pattern analysis
5. **Make a Mind MCP** - For brain-inspired knowledge organization and memory palace creation

### Make a Mind MCP Integration
**PURPOSE**: The "Make a Mind" MCP complements the MegaMind Context Database by providing brain-inspired knowledge organization patterns and memory palace techniques for enhanced knowledge retention and retrieval.

#### Core Capabilities
- **Memory Palace Creation**: Build spatial memory structures for complex knowledge domains
- **Knowledge Chunking Optimization**: Apply cognitive science principles to improve chunk boundaries
- **Associative Linking**: Create brain-inspired relationship patterns between concepts
- **Spaced Repetition Integration**: Optimize knowledge access patterns based on cognitive research
- **Concept Hierarchy Mapping**: Organize knowledge using natural cognitive categorization

#### Integration with MegaMind Context Database
**Workflow Pattern**:
1. **MegaMind Search**: Use `mcp__megamind__search_chunks` for initial knowledge retrieval
2. **Make a Mind Organization**: Apply cognitive structuring to search results for better comprehension
3. **Enhanced Relationships**: Use Make a Mind patterns to suggest improved `mcp__megamind__add_relationship` connections
4. **Memory Palace Storage**: Create spatial representations of complex knowledge clusters
5. **Optimized Retrieval**: Use cognitive principles to improve future search strategies

#### Use Cases
- **Complex System Understanding**: Build memory palaces for large codebases or architectural patterns
- **Learning Path Optimization**: Apply spaced repetition to knowledge chunk access patterns
- **Concept Relationship Discovery**: Find non-obvious connections between disparate knowledge areas
- **Knowledge Retention Enhancement**: Structure information for better long-term retention
- **Cognitive Load Management**: Organize complex information to reduce mental overhead

#### Function Integration Examples
```python
# Example workflow combining both systems
search_results = mcp__megamind__search_chunks("authentication patterns")
memory_palace = make_a_mind_create_palace(search_results, domain="security")
enhanced_chunks = make_a_mind_optimize_chunking(search_results)

# Apply cognitive insights back to MegaMind
for chunk in enhanced_chunks:
    mcp__megamind__add_relationship(
        chunk['id'], 
        chunk['cognitive_anchor'], 
        "memory_palace_association"
    )
```

#### Cognitive Enhancement Features
- **Spatial Memory Mapping**: Convert abstract concepts into spatial relationships
- **Chunking Optimization**: Apply Miller's Rule and cognitive load theory to chunk sizing
- **Associative Networks**: Build memory networks based on psychological association principles
- **Retrieval Practice**: Implement testing effects for knowledge reinforcement
- **Elaborative Encoding**: Enhance chunk content with cognitive elaboration techniques

### MegaMind MCP Function Naming Schema
**CRITICAL**: The MegaMind MCP server uses a consistent naming convention to avoid confusion:

#### **Function Names** (Exposed MCP Tools)
- **Format**: `mcp__megamind__[function_name]`
- **Examples**: `mcp__megamind__search_chunks`, `mcp__megamind__get_chunk`, `mcp__megamind__create_promotion_request`
- **Total Count**: 20 functions across 5 categories (Search, Content, Promotion, Session, Analytics)

#### **NEW MCP Function Naming Convention** ⚠️
**CRITICAL**: When creating new MCP function names, the first component MUST be the function type/category:

- **Format**: `mcp__megamind__[TYPE]_[specific_function]`
- **Function Type Prefixes**:
  - `search_` - Search and retrieval operations
  - `content_` - Content management operations  
  - `promotion_` - Knowledge promotion operations
  - `session_` - Session management operations
  - `analytics_` - Analytics and optimization operations

**Examples of Correct New Function Names**:
- `mcp__megamind__session_new_session` - Create new session
- `mcp__megamind__search_advanced_filter` - Advanced search with filters
- `mcp__megamind__content_bulk_import` - Bulk content import
- `mcp__megamind__promotion_auto_approve` - Automated promotion approval
- `mcp__megamind__analytics_usage_report` - Generate usage analytics

**CRITICALLY DELINEATE**: Each function group MUST be clearly identified by its prefix to maintain architectural clarity and prevent function categorization confusion.

#### **Quick Reference - All 37 Functions**
```
Search & Retrieval (5):
├── mcp__megamind__search_chunks
├── mcp__megamind__get_chunk  
├── mcp__megamind__get_related_chunks
├── mcp__megamind__search_chunks_semantic
└── mcp__megamind__search_chunks_by_similarity

Content Management (4):
├── mcp__megamind__create_chunk
├── mcp__megamind__update_chunk
├── mcp__megamind__add_relationship
└── mcp__megamind__batch_generate_embeddings

Knowledge Promotion (6):
├── mcp__megamind__create_promotion_request
├── mcp__megamind__get_promotion_requests
├── mcp__megamind__approve_promotion_request
├── mcp__megamind__reject_promotion_request
├── mcp__megamind__get_promotion_impact
└── mcp__megamind__get_promotion_queue_summary

Session Management (3):
├── mcp__megamind__get_session_primer
├── mcp__megamind__get_pending_changes
└── mcp__megamind__commit_session_changes

Analytics & Optimization (2):
├── mcp__megamind__track_access
└── mcp__megamind__get_hot_contexts

Phase 2 Enhanced Embedding (7):
├── mcp__megamind__content_analyze_document
├── mcp__megamind__content_create_chunks
├── mcp__megamind__content_assess_quality
├── mcp__megamind__content_optimize_embeddings
├── mcp__megamind__session_create
├── mcp__megamind__session_get_state
└── mcp__megamind__session_complete

Phase 3 Knowledge Management (4):
├── mcp__megamind__knowledge_ingest_document
├── mcp__megamind__knowledge_discover_relationships
├── mcp__megamind__knowledge_optimize_retrieval
└── mcp__megamind__knowledge_get_related

Phase 3 Session Tracking (6):
├── mcp__megamind__session_create_operational
├── mcp__megamind__session_track_action
├── mcp__megamind__session_get_recap
├── mcp__megamind__session_prime_context
├── mcp__megamind__session_list_recent
└── mcp__megamind__session_close
```

#### **Database Tables** (Internal Storage)
- **Format**: `megamind_[table_name]`
- **Examples**: `megamind_chunks`, `megamind_realms`, `megamind_promotion_queue`
- **Total Count**: 13 tables with complete realm inheritance and promotion system support

#### **Internal Method Names** (Code Implementation)
- **Format**: `[method_name]` or `[method_name]_dual_realm`
- **Examples**: `search_chunks_dual_realm`, `get_chunk_dual_realm`, `track_access`
- **Note**: Methods must exist in both base database class AND RealmAwareMegaMindDatabase

#### **Architecture Alignment**
- **MCP Layer**: `mcp__context_db__*` functions → **MCPServer** class
- **Database Layer**: `megamind_*` tables → **RealmAwareMegaMindDatabase** class  
- **Method Resolution**: MCPServer calls must match available database methods
- **Missing Methods**: Will cause "object has no attribute" errors during function calls

### Realm-Aware MCP Operations
**IMPORTANT**: When using MegaMind MCP functions, understand that:

- **Automatic Realm Context**: All operations inherit the server's configured realm without needing explicit parameters
- **Dual-Realm Search**: Search operations automatically query both PROJECT and GLOBAL realms with proper inheritance
- **Cross-Realm Relationships**: The system handles cross-realm chunk relationships transparently
- **Access Control**: Realm-based permissions are enforced automatically based on inheritance rules
- **No Manual Realm Selection**: Do not attempt to specify realms in function calls - use the pre-configured server realm context

## Knowledge Promotion System Usage Guide

### Overview
The Knowledge Promotion System enables cross-realm knowledge transfer, allowing valuable insights to move between PROJECT and GLOBAL realms through a governed workflow.

### Basic Promotion Workflow

#### 1. Create a Promotion Request
```python
# Request to promote a valuable chunk to GLOBAL realm
result = mcp__megamind__create_promotion_request(
    chunk_id="chunk_12345",
    target_realm="GLOBAL",
    justification="This debugging technique is broadly applicable across projects",
    session_id="session_abc123"
)
```

#### 2. Review Promotion Queue
```python
# Get overview of all pending promotions
summary = mcp__megamind__get_promotion_queue_summary()
print(f"Pending promotions: {summary['total_pending']}")

# Get detailed list of promotion requests
requests = mcp__megamind__get_promotion_requests(
    filter_status="pending",
    limit=10
)
```

#### 3. Analyze Promotion Impact
```python
# Analyze impact before approving
impact = mcp__megamind__get_promotion_impact("promotion_67890")
print(f"Affected relationships: {impact['relationship_count']}")
print(f"Similar existing chunks: {impact['similarity_matches']}")
```

#### 4. Approve or Reject Promotion
```python
# Approve promotion
approval = mcp__megamind__approve_promotion_request(
    promotion_id="promotion_67890",
    approval_reason="High-value pattern applicable to multiple projects",
    session_id="session_abc123"
)

# Or reject with reason
rejection = mcp__megamind__reject_promotion_request(
    promotion_id="promotion_67890", 
    rejection_reason="Too project-specific, limited general applicability",
    session_id="session_abc123"
)
```

### Advanced Usage Patterns

#### Batch Promotion Review
```python
# Get all pending promotions for systematic review
pending = mcp__megamind__get_promotion_requests(filter_status="pending")

for request in pending:
    # Analyze each promotion's impact
    impact = mcp__megamind__get_promotion_impact(request['promotion_id'])
    
    # Auto-approve based on criteria
    if impact['confidence_score'] > 0.8 and impact['conflict_count'] == 0:
        mcp__megamind__approve_promotion_request(
            request['promotion_id'],
            "Auto-approved: High confidence, no conflicts",
            session_id
        )
```

#### Realm-Specific Queue Monitoring
```python
# Monitor promotions targeting specific realm
global_queue = mcp__megamind__get_promotion_queue_summary(filter_realm="GLOBAL")
print(f"GLOBAL promotions pending: {global_queue['total_pending']}")

# Track promotion activity over time
requests = mcp__megamind__get_promotion_requests(filter_realm="GLOBAL", limit=50)
recent_activity = [r for r in requests if r['created_date'] > '2025-07-01']
```

### Promotion System Database Schema

#### megamind_promotion_queue
- **promotion_id**: Unique identifier for promotion request
- **chunk_id**: Source chunk being promoted  
- **source_realm**: Origin realm of chunk
- **target_realm**: Destination realm for promotion
- **justification**: Reason for promotion request
- **status**: pending, approved, rejected, completed
- **created_by**: Session ID of requester
- **created_date**: Request timestamp

#### megamind_promotion_history  
- **history_id**: Unique identifier for history record
- **promotion_id**: Reference to promotion request
- **action**: Action taken (approved, rejected, completed)
- **action_reason**: Justification for action
- **action_by**: Session ID of decision maker
- **action_date**: Decision timestamp

#### megamind_promotion_impact
- **impact_id**: Unique identifier for impact analysis
- **promotion_id**: Reference to promotion request
- **impact_type**: Type of impact (relationship, similarity, conflict)
- **impact_target**: Affected chunk or relationship ID
- **impact_score**: Numerical impact assessment
- **impact_description**: Human-readable impact summary

### Best Practices

#### Promotion Request Guidelines
- **Clear Justification**: Provide specific reasons why knowledge should be promoted
- **Scope Assessment**: Consider whether knowledge applies beyond current project
- **Quality Review**: Ensure promoted content is well-structured and accurate
- **Relationship Impact**: Consider how promotion affects existing cross-references

#### Approval Workflow
- **Impact Analysis First**: Always review impact before approving promotions
- **Conflict Resolution**: Address any conflicts with existing GLOBAL knowledge
- **Documentation**: Provide clear approval/rejection reasons for audit trail
- **Batch Processing**: Process related promotions together for consistency

#### Monitoring and Maintenance
- **Regular Queue Review**: Monitor promotion queues to prevent backlogs
- **Success Metrics**: Track promotion success rates and impact assessments
- **Quality Feedback**: Use promotion outcomes to improve future requests
- **Cross-Realm Validation**: Verify promoted knowledge integrates well in target realm

## Development Guidelines

**IMPORTANT**: All coding for this system is designed for **fresh database deployment from the Docker container**. Modifying the running database is not the goal - all schema changes and fixes should be applied to the initialization scripts (`init_schema.sql`) for clean container deployments.

**CRITICAL - Container Rebuild Requirement**: 
⚠️ **If any Python code changes are made to the MCP server files (especially `megamind_database_server.py`, `realm_aware_database.py`, or other `mcp_server/` files), the HTTP container MUST be rebuilt before testing**, as Docker containers use cached file layers. Simply restarting the container will NOT pick up code changes.

**Required rebuild steps after code changes**:
```bash
# Stop and remove the container
docker compose down megamind-mcp-server-http

# Rebuild with updated code
docker compose build megamind-mcp-server-http

# Start the rebuilt container
docker compose up megamind-mcp-server-http -d
```

**IMPORTANT - Testing Requirements**:
⚠️ **All tests MUST be run inside the container** because:
- Database is not exposed externally (only accessible within Docker network)
- Test files and dependencies are available only within container environment
- JSON-RPC endpoint has been moved to root path (`/` instead of `/mcp/jsonrpc`)

**Required test execution pattern**:
```bash
# Run tests inside the HTTP container
docker exec megamind-mcp-server-http python3 tests/test_[phase]_functions.py

# Example: Run Phase 4 tests
docker exec megamind-mcp-server-http python3 tests/test_phase4_functions.py
```

**API Testing with Updated JSON-RPC Path**:
```bash
# Test MCP connectivity (JSON-RPC endpoint moved to root path)
curl -X POST http://10.255.250.22:8080 -H "Content-Type: application/json" -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'

# Legacy path no longer available
# curl -X POST http://10.255.250.22:8080/mcp/jsonrpc  # ❌ DEPRECATED
```

When implementing this system:
1. **Database First**: All operations through database, no file system dependencies
2. **MCP Interface**: All AI interactions through MCP function calls
3. **Textsmith for Code**: Use textsmith MCP for all file operations and safe code handling
4. **Session Safety**: Buffer all changes with manual review cycles
5. **Semantic Integrity**: Maintain meaningful context boundaries in chunks
6. **Performance Focus**: Sub-second response times for retrieval operations
7. **Relationship Preservation**: Maintain cross-reference validity through updates
8. **Clean Deployment**: All database changes must be in initialization scripts, not migration scripts

## Current Status - Production Deployment Complete ✅

**Deployment Status**: **PRODUCTION READY** - All 37 MCP functions deployed and operational with Claude Code connectivity

### Implementation Summary (as of 2025-07-16)
- ✅ **Core MCP Server**: 14 original functions fully operational
- ✅ **Knowledge Promotion System**: 6 new functions implemented and tested  
- ✅ **Database Schema**: Complete with all promotion system tables
- ✅ **Container Deployment**: Production container built and running
- ✅ **HTTP Transport**: Phase 2 JSON-RPC realm management implemented
- ✅ **Claude Code Integration**: STDIO-HTTP bridge with security controls deployed
- ✅ **Connectivity Issue Resolution**: GitHub Issue #12 resolved with secure bridge

### Function Availability
**All 37 MCP functions are now available for use:**
- **Search & Retrieval**: 5 functions (search_chunks, get_chunk, semantic search, etc.)
- **Content Management**: 4 functions (create_chunk, update_chunk, relationships, embeddings)
- **Knowledge Promotion**: 6 functions (create_request, approve/reject, impact analysis, queue management)
- **Session Management**: 3 functions (session primer, pending changes, commit changes)
- **Analytics**: 2 functions (access tracking, hot contexts)
- **Phase 2 Enhanced Embedding**: 7 functions (content analysis, chunking, quality assessment, embedding optimization, session management)
- **Phase 3 Knowledge Management**: 4 functions (document ingestion, relationship discovery, retrieval optimization, related chunks)
- **Phase 3 Session Tracking**: 6 functions (operational sessions, action tracking, recap, context priming, session management)

### Deployment Configuration
- **Container**: megamind-mcp-server-http (running on port 8080)
- **Database**: MySQL with complete schema including promotion tables
- **Environment**: Production-ready with realm-aware dual-access patterns
- **Transport**: HTTP server + STDIO-HTTP bridge for Claude Code connectivity
- **Security**: PROJECT-only realm access via STDIO bridge with GLOBAL access blocked

### File Structure Overview
```
MegaMind_MCP/
├── mcp_server/              # Core MCP implementation
│   ├── stdio_http_bridge.py # ✨ Claude Code STDIO bridge
│   ├── megamind_database_server.py # Main MCP server
│   ├── realm_aware_database.py     # Realm-aware operations
│   └── services/            # Embedding and search services
├── database/                # Schema and migrations
│   ├── realm_system/        # Realm inheritance schemas
│   └── context_system/      # Legacy context schemas
├── tools/                   # Ingestion and utilities
├── scripts/                 # Deployment and testing
├── .mcp.json               # ✨ Claude Code configuration
├── .env                    # Environment variables
└── docker-compose.yml      # Container orchestration
```

### Next Development Phase
The system is ready for Phase 3 enhancements:
- Advanced promotion scoring algorithms
- ML-based impact prediction
- Automated curation workflows
- Performance optimization for embedding service startup

## MCP Protocol Implementation Guidelines

### Connection Initialization Handshake Structure

**CRITICAL**: Claude Code expects a specific MCP protocol initialization sequence. This handshake MUST be implemented correctly in any STDIO bridge or MCP server implementation.

#### **Required Handshake Sequence**

**1. Initialize Request/Response (Required)**
```json
// Claude Code sends:
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": "2024-11-05",
    "capabilities": {
      "roots": {
        "listChanged": true
      },
      "sampling": {}
    },
    "clientInfo": {
      "name": "claude-code",
      "version": "1.0.0"
    }
  }
}

// Server MUST respond with:
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "protocolVersion": "2024-11-05",
    "capabilities": {
      "tools": {},           // Your tool capabilities
      "resources": {}        // Optional resources
    },
    "serverInfo": {
      "name": "megamind-mcp-server",
      "version": "1.0.0"
    }
  }
}
```

**2. Initialized Notification (Required)**
```json
// Claude Code sends (no response expected):
{
  "jsonrpc": "2.0",
  "method": "notifications/initialized"
}
```

#### **Implementation Requirements**

**Bridge Architecture**: For STDIO-HTTP bridges, handle MCP protocol locally:
- **Initialization Requests**: Handle `initialize` and `notifications/initialized` locally in the bridge
- **Tool Calls**: Forward `tools/list` and `tools/call` requests to HTTP backend
- **Protocol Compliance**: Must respond with exact JSON-RPC 2.0 format and required fields

**Timeout Configuration**: Claude Code uses configurable timeouts:
- **Default Timeout**: 30 seconds for MCP connection establishment
- **Extended Timeout**: Set `MCP_TIMEOUT` environment variable (e.g., `"60000"` for 60 seconds)
- **Accommodation**: Allow time for embedding service initialization in containers

#### **Common Implementation Issues**

**Missing Handlers**: 
- ❌ **Problem**: Bridge forwards ALL requests to HTTP backend including initialization
- ✅ **Solution**: Handle MCP protocol locally, forward only actual tool calls

**Incorrect Response Format**:
- ❌ **Problem**: Missing required fields in `initialize` response
- ✅ **Solution**: Include `protocolVersion`, `capabilities`, and `serverInfo`

**Timeout Issues**:
- ❌ **Problem**: Container services (embedding models) take time to initialize
- ✅ **Solution**: Set appropriate `MCP_TIMEOUT` in environment configuration

#### **Testing MCP Protocol Compliance**

**Manual Bridge Test**:
```bash
# Test initialization handshake
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}' | python3 stdio_http_bridge.py

# Test initialized notification
echo '{"jsonrpc":"2.0","method":"notifications/initialized"}' | python3 stdio_http_bridge.py
```

**Expected Results**:
- Initialize: Returns complete capabilities response with all 20 tools
- Initialized: Acknowledges notification (may return empty response)
- No EOF errors or connection termination

#### **Bridge Configuration Examples**

**Production STDIO Bridge Configuration**:
```json
"megamind-context-db": {
  "command": "python3",
  "args": ["/Data/MCP_Servers/MegaMind_MCP/mcp_server/stdio_http_bridge.py"],
  "env": {
    "MEGAMIND_PROJECT_REALM": "MegaMind_MCP",
    "MEGAMIND_PROJECT_NAME": "MegaMind Context Database",
    "MEGAMIND_DEFAULT_TARGET": "PROJECT",
    "LOG_LEVEL": "INFO",
    "MCP_TIMEOUT": "60000"
  }
}
```

This protocol implementation ensures reliable Claude Code connectivity and prevents common connection timeout and handshake failures.