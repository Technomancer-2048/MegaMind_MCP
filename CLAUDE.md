# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

This is currently a **planning and design repository** containing:
- `context_db_project_mission.md` - Project mission, goals, and success metrics
- `context_db_execution_plan.md` - Detailed implementation plan with phases

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

#### **Search & Retrieval Functions**
- `mcp__megamind__search_chunks(query, limit=10, search_type="hybrid")` - Enhanced dual-realm search with hybrid semantic capabilities
- `mcp__megamind__get_chunk(chunk_id, include_relationships=true)` - Get specific chunk by ID with relationships
- `mcp__megamind__get_related_chunks(chunk_id, max_depth=2)` - Get chunks related to specified chunk
- `mcp__megamind__search_chunks_semantic(query, limit=10, threshold=0.7)` - Pure semantic search across Global + Project realms
- `mcp__megamind__search_chunks_by_similarity(reference_chunk_id, limit=10, threshold=0.7)` - Find chunks similar to a reference chunk using embeddings

#### **Content Management Functions**
- `mcp__megamind__create_chunk(content, source_document, section_path, session_id, target_realm="PROJECT")` - Buffer new knowledge creation with realm targeting and embedding generation
- `mcp__megamind__update_chunk(chunk_id, new_content, session_id)` - Buffer chunk modifications for review
- `mcp__megamind__add_relationship(chunk_id_1, chunk_id_2, relationship_type, session_id)` - Create cross-references between chunks
- `mcp__megamind__batch_generate_embeddings(chunk_ids=[], realm_id="")` - Generate embeddings for existing chunks in batch

#### **Session Management Functions**
- `mcp__megamind__get_session_primer(last_session_data="")` - Generate lightweight context for session continuity
- `mcp__megamind__get_pending_changes(session_id)` - Get pending changes with smart highlighting
- `mcp__megamind__commit_session_changes(session_id, approved_changes)` - Commit approved changes and track contributions

#### **Analytics & Optimization Functions**
- `mcp__megamind__track_access(chunk_id, query_context="")` - Update access analytics for optimization
- `mcp__megamind__get_hot_contexts(model_type="sonnet", limit=20)` - Get frequently accessed chunks prioritized by usage patterns

### Database Schema Design (MegaMind Naming Convention)
- **Primary Tables**: `megamind_chunks`, `megamind_chunk_relationships`, `megamind_chunk_tags`
- **Realm Management**: `megamind_realms`, `megamind_realm_inheritance`
- **Change Management**: `megamind_session_changes`, `megamind_knowledge_contributions`
- **Intelligence Layer**: `megamind_embeddings` (with realm support)
- **Analytics**: `megamind_performance_metrics`, `megamind_system_health`
- **Analytics**: Access tracking, usage patterns, relationship discovery

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

### MCP Server Environment Variable Configuration
**CRITICAL FIX**: Due to Claude Code's subprocess environment inheritance issues with direct executable paths, the MegaMind MCP server uses a specialized configuration approach:

- **Environment Command Wrapper**: Uses `env` command as wrapper instead of direct Python execution
- **Explicit Variable Setting**: All environment variables are explicitly set in the `args` array using `env` command syntax
- **Per-Project Configuration**: Each project's `.mcp.json` contains its own realm-specific environment variables
- **Reliable Subprocess Inheritance**: The `env` command ensures proper environment variable passing to subprocesses

**Configuration Pattern**:
```json
"megamind-database": {
  "command": "env",
  "args": [
    "MEGAMIND_ROOT=/Data/MCP_Servers/MegaMind_MCP",
    "MEGAMIND_PROJECT_REALM=MegaMind_MCP",
    "MEGAMIND_PROJECT_NAME=MegaMind MCP Platform",
    "MEGAMIND_DEFAULT_TARGET=PROJECT",
    "MEGAMIND_DB_HOST=10.255.250.21",
    "MEGAMIND_DB_PORT=3309",
    "MEGAMIND_DB_NAME=megamind_database",
    "MEGAMIND_DB_USER=megamind_user",
    "MEGAMIND_DB_PASSWORD=...",
    "CONNECTION_POOL_SIZE=10",
    "MEGAMIND_DEBUG=false",
    "MEGAMIND_LOG_LEVEL=INFO",
    "/Data/MCP_Servers/MegaMind_MCP/venv/bin/python",
    "megamind_database_server.py"
  ],
  "cwd": "/Data/MCP_Servers/MegaMind_MCP/mcp_server"
}
```

**Why This Approach**:
- **Subprocess Compatibility**: Claude Code handles system commands (`env`, `docker`, `node`) more reliably than direct executable paths
- **Environment Inheritance**: The `env` command explicitly sets variables before executing the target command
- **Per-Project Isolation**: Each project maintains its own realm configuration without global contamination
- **Proven Pattern**: Follows the same pattern as other working MCP servers (docker, uv, npx)

### Environment Path Management
**STREAMLINED**: The server uses intelligent path resolution from a single root variable:

- **Root Path**: Set `MEGAMIND_ROOT=/Data/MCP_Servers/MegaMind_MCP` (or auto-detect from script location)
- **Automatic Derivation**: All paths are automatically configured:
  - Model cache: `{MEGAMIND_ROOT}/models`
  - Python modules: `{MEGAMIND_ROOT}/mcp_server`
  - HuggingFace cache: `{MEGAMIND_ROOT}/models`
- **No Redundant Variables**: Eliminates need for separate `HF_HOME`, `PYTHONPATH`, etc.
- **Deployment Flexibility**: Easy to relocate entire installation by changing one variable

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

### MegaMind MCP Function Naming Schema
**CRITICAL**: The MegaMind MCP server uses a consistent naming convention to avoid confusion:

#### **Function Names** (Exposed MCP Tools)
- **Format**: `mcp__megamind__[function_name]`
- **Examples**: `mcp__megamind__search_chunks`, `mcp__megamind__get_chunk`, `mcp__megamind__create_chunk`
- **Total Count**: 14 functions across 4 categories (Search, Content, Session, Analytics)

#### **Database Tables** (Internal Storage)
- **Format**: `megamind_[table_name]`
- **Examples**: `megamind_chunks`, `megamind_realms`, `megamind_session_changes`
- **Total Count**: 10 tables with complete realm inheritance support

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

## Development Guidelines

**IMPORTANT**: All coding for this system is designed for **fresh database deployment from the Docker container**. Modifying the running database is not the goal - all schema changes and fixes should be applied to the initialization scripts (`init_schema.sql`) for clean container deployments.

When implementing this system:
1. **Database First**: All operations through database, no file system dependencies
2. **MCP Interface**: All AI interactions through MCP function calls
3. **Textsmith for Code**: Use textsmith MCP for all file operations and safe code handling
4. **Session Safety**: Buffer all changes with manual review cycles
5. **Semantic Integrity**: Maintain meaningful context boundaries in chunks
6. **Performance Focus**: Sub-second response times for retrieval operations
7. **Relationship Preservation**: Maintain cross-reference validity through updates
8. **Clean Deployment**: All database changes must be in initialization scripts, not migration scripts

## Current Status
This repository contains planning documents only. Implementation should follow the detailed execution plan with focus on building a robust, standalone MCP server that operates entirely through database interactions.