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
The core MCP server will implement these functions:
- `mcp__context_db__search_chunks(query, limit=10, model_type="sonnet")`
- `mcp__context_db__get_chunk(chunk_id, include_relationships=true)`
- `mcp__context_db__get_related_chunks(chunk_id, max_depth=2)`
- `mcp__context_db__update_chunk(chunk_id, new_content, session_id)`
- `mcp__context_db__create_chunk(content, source_document, section_path, session_id)`
- `mcp__context_db__get_session_primer(last_session_data)`
- `mcp__context_db__commit_session_changes(session_id, approved_changes)`

### Database Schema Design
- **Primary Tables**: `context_chunks`, `chunk_relationships`, `chunk_tags`
- **Change Management**: `session_changes`, `knowledge_contributions`
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
1. **Textsmith** - For all file operations, code modifications, content processing
2. **Context Database** - For semantic chunk retrieval and knowledge management
3. **SQL Files** - For database schema operations and query optimization
4. **Quick Data** - For analytics and usage pattern analysis

## Development Guidelines

When implementing this system:
1. **Database First**: All operations through database, no file system dependencies
2. **MCP Interface**: All AI interactions through MCP function calls
3. **Textsmith for Code**: Use textsmith MCP for all file operations and safe code handling
4. **Session Safety**: Buffer all changes with manual review cycles
5. **Semantic Integrity**: Maintain meaningful context boundaries in chunks
6. **Performance Focus**: Sub-second response times for retrieval operations
7. **Relationship Preservation**: Maintain cross-reference validity through updates

## Current Status
This repository contains planning documents only. Implementation should follow the detailed execution plan with focus on building a robust, standalone MCP server that operates entirely through database interactions.