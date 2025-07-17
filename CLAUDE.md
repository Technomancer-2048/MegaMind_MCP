# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ⚠️ HIGH IMPORTANCE - MegaMind MCP Behavioral Policies

### 🔄 Session Startup Protocol
1. **ALWAYS** call `mcp__megamind__session_create(session_type, created_by)` on conversation start
2. If active session exists: Load procedural context (workflow state, not knowledge chunks)
3. If no session exists: Prompt user to create/select session

### 🔍 Context Retrieval Protocol
**Before ANY project-level tasks**:
1. Use `mcp__megamind__search_query(query, limit=10, search_type="hybrid")` for initial context
2. For deeper relationships: `mcp__megamind__search_related(chunk_id, max_depth=2)`
3. **ALWAYS** track access: `mcp__megamind__analytics_track(chunk_id, metadata={"query_context": query_context})`
4. Include chunk IDs in responses for complete traceability

### 💾 Knowledge Capture Protocol  
**During development when significant findings emerge**:
1. Buffer discoveries using appropriate MCP functions (`mcp__megamind__content_create`, `mcp__megamind__content_update`, `mcp__megamind__content_process`)
2. Generate session summary: `mcp__megamind__session_manage(session_id, action="get_state")`
3. Present summary with impact assessment to user for review
4. On user approval: `mcp__megamind__session_commit(session_id, approved_changes)`

### 🚀 Knowledge Promotion Protocol
**For discoveries with broader applicability**:
1. Create promotion request to GLOBAL realm with clear justification using `mcp__megamind__promotion_request`
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

## 📚 Documentation System

**IMPORTANT**: Most detailed documentation has been migrated to the MegaMind Context Database. Use these MCP functions to access comprehensive information:

### Quick Reference Commands
```python
# Get MCP function listings and documentation
mcp__megamind__search_query("MCP Server Functions Core Implementation")

# Get detailed behavioral policies 
mcp__megamind__search_query("MegaMind MCP Behavioral Policies")

# Get Claude Code connection info
mcp__megamind__search_query("Claude Code Connection Architecture STDIO bridge")

# Get development guidelines
mcp__megamind__search_query("Development Guidelines Container Testing")

# Get knowledge promotion system guide
mcp__megamind__search_query("Knowledge Promotion System Usage Guide")

# Get MCP protocol implementation details
mcp__megamind__search_query("MCP Protocol Implementation Guidelines")
```

### Key Information Available in Chunks
- **MCP Functions**: Complete 19-function consolidated API reference
- **Connection Architecture**: STDIO-HTTP bridge configuration 
- **Development Guidelines**: Container rebuild requirements and testing
- **Promotion System**: Complete workflow and best practices
- **Protocol Implementation**: MCP handshake and bridge requirements

## Essential Configuration

### Claude Code MCP Connection
**Quick Setup**: Claude Code connects via STDIO-to-HTTP bridge configured in `.mcp.json`:

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

### Realm Configuration 
**CRITICAL**: 
- **Environment Variables Only**: Realm configuration via `.mcp.json` environment variables
- **NO JSON-RPC Realm Passing**: NEVER pass realms through function calls
- **Single Realm Per Server**: Each MCP server instance bound to one realm configuration

**📋 For detailed connection architecture and security features:**
```python
mcp__megamind__search_query("Claude Code Connection Architecture STDIO bridge")
```

### Tool Usage Priority
1. **MegaMind Context Database** - Semantic chunk retrieval and knowledge management (realm-aware)  
2. **Textsmith** - File operations, code modifications, content processing
3. **SQL Files** - Database schema operations and query optimization
4. **Quick Data** - Analytics and usage pattern analysis

### Essential Code Handling
**Textsmith Path Translation**:
- **Local**: `/Data/MCP_Servers/MegaMind_MCP/some/file.py`
- **Textsmith**: `/app/workspace/some/file.py`

**Code Safety**: Always use `mcp__textsmith__safe_replace_text` for code modifications

## Essential Development Information

### Container Rebuild Requirement ⚠️
**CRITICAL**: After any Python code changes to MCP server files, rebuild container:
```bash
docker compose down megamind-mcp-server-http
docker compose build megamind-mcp-server-http  
docker compose up megamind-mcp-server-http -d
```

### Testing Requirements
**All tests MUST run inside container**:
```bash
docker exec megamind-mcp-server-http python3 tests/test_[phase]_functions.py
```

### Current Status
**PHASE 5 AGI READY** - All 56 Next-Generation AI functions deployed with AGI capabilities

**📋 For complete development guidelines and testing procedures:**
```python
mcp__megamind__search_query("Development Guidelines Container Testing")
```

## Implementation Guidelines

### Function Naming Convention
- **Format**: `mcp__megamind__[TYPE]_[specific_function]`
- **Prefixes**: `search_`, `content_`, `promotion_`, `session_`, `analytics_`
- **Database Tables**: `megamind_[table_name]`

### Realm Operations
**CRITICAL**: 
- **Automatic Realm Context**: Operations inherit server's configured realm
- **Dual-Realm Search**: Automatic PROJECT + GLOBAL realm querying
- **No Manual Realm Selection**: Use pre-configured server realm context only

## 🔧 Essential Commands

### GitHub Issue Management
```bash
gh issue create --title "🐛 Bug: Description" --label "bug,mcp-server"
gh issue list --label="bug"
gh issue close 123 --comment "Fixed in commit abc123"
```

---

## 📝 Access Comprehensive Documentation

**All detailed documentation has been migrated to chunks. Use these commands:**

```python
# Function reference and API documentation
mcp__megamind__search_query("MCP Server Functions Core Implementation")

# Promotion system workflows and best practices  
mcp__megamind__search_query("Knowledge Promotion System Usage Guide")

# MCP protocol implementation details
mcp__megamind__search_query("MCP Protocol Implementation Guidelines")
```

**This CLAUDE.md now serves as a quick reference. All comprehensive guides, examples, workflows, and detailed documentation are available through the MCP search system.**