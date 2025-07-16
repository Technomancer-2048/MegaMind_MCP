# MegaMind MCP Functions Usage Guide

## Overview

The MegaMind Context Database System provides 37 MCP functions across 4 major phases, enabling intelligent document processing, knowledge management, and AI-enhanced optimization. This guide provides comprehensive documentation for all exposed functions.

## Table of Contents

1. [Core Search & Retrieval Functions](#core-search--retrieval-functions)
2. [Content Management Functions](#content-management-functions)
3. [Knowledge Promotion Functions](#knowledge-promotion-functions)
4. [Session Management Functions](#session-management-functions)
5. [Analytics & Optimization Functions](#analytics--optimization-functions)
6. [Phase 2: Enhanced Embedding Functions](#phase-2-enhanced-embedding-functions)
7. [Phase 3: Knowledge Management Functions](#phase-3-knowledge-management-functions)
8. [Phase 3: Session Tracking Functions](#phase-3-session-tracking-functions)
9. [Phase 4: AI Enhancement Functions](#phase-4-ai-enhancement-functions)
10. [Workflow Patterns](#workflow-patterns)
11. [Best Practices](#best-practices)

---

## Core Search & Retrieval Functions

### 1. `mcp__megamind__search_chunks`

**Purpose**: Primary search function with hybrid semantic capabilities across Global + Project realms.

**Parameters**:
- `query` (string): Search query text
- `limit` (integer, default: 10): Maximum number of results
- `search_type` (string, default: "hybrid"): "semantic", "keyword", or "hybrid"

**Returns**: List of matching chunks with relevance scores and metadata

**Example**:
```python
# Basic search
results = mcp__megamind__search_chunks(
    query="authentication patterns",
    limit=5,
    search_type="hybrid"
)

# Semantic-only search
results = mcp__megamind__search_chunks(
    query="database connection handling",
    limit=10,
    search_type="semantic"
)
```

**Use Cases**:
- Initial context retrieval for development tasks
- Knowledge discovery across project documentation
- Finding related implementation patterns

### 2. `mcp__megamind__get_chunk`

**Purpose**: Retrieve specific chunk by ID with optional relationship data.

**Parameters**:
- `chunk_id` (string): Unique chunk identifier
- `include_relationships` (boolean, default: true): Include related chunks

**Returns**: Chunk data with content, metadata, and optional relationships

**Example**:
```python
# Get chunk with relationships
chunk = mcp__megamind__get_chunk(
    chunk_id="chunk_12345",
    include_relationships=True
)

# Get chunk content only
chunk = mcp__megamind__get_chunk(
    chunk_id="chunk_12345",
    include_relationships=False
)
```

**Use Cases**:
- Retrieving specific documentation sections
- Following up on search results
- Accessing related context

### 3. `mcp__megamind__get_related_chunks`

**Purpose**: Find chunks related to a specified chunk through semantic and explicit relationships.

**Parameters**:
- `chunk_id` (string): Reference chunk ID
- `max_depth` (integer, default: 2): Relationship traversal depth

**Returns**: List of related chunks with relationship types and distances

**Example**:
```python
# Get immediate relations
related = mcp__megamind__get_related_chunks(
    chunk_id="chunk_12345",
    max_depth=1
)

# Deep relationship traversal
related = mcp__megamind__get_related_chunks(
    chunk_id="chunk_12345",
    max_depth=3
)
```

**Use Cases**:
- Exploring knowledge connections
- Building context for complex topics
- Discovering implementation dependencies

### 4. `mcp__megamind__search_chunks_semantic`

**Purpose**: Pure semantic search using embedding similarity across realms.

**Parameters**:
- `query` (string): Natural language query
- `limit` (integer, default: 10): Maximum results
- `threshold` (float, default: 0.7): Similarity threshold (0.0-1.0)

**Returns**: Semantically similar chunks with similarity scores

**Example**:
```python
# High-precision semantic search
results = mcp__megamind__search_chunks_semantic(
    query="How to handle database connection errors",
    limit=5,
    threshold=0.8
)

# Broader semantic search
results = mcp__megamind__search_chunks_semantic(
    query="error handling patterns",
    limit=15,
    threshold=0.6
)
```

**Use Cases**:
- Finding conceptually similar content
- Discovering patterns and approaches
- Cross-domain knowledge transfer

### 5. `mcp__megamind__search_chunks_by_similarity`

**Purpose**: Find chunks similar to a reference chunk using embedding vectors.

**Parameters**:
- `reference_chunk_id` (string): Reference chunk for similarity
- `limit` (integer, default: 10): Maximum results
- `threshold` (float, default: 0.7): Similarity threshold

**Returns**: List of similar chunks with similarity scores

**Example**:
```python
# Find similar implementation patterns
similar = mcp__megamind__search_chunks_by_similarity(
    reference_chunk_id="chunk_auth_pattern",
    limit=8,
    threshold=0.75
)
```

**Use Cases**:
- Finding similar code patterns
- Discovering related documentation
- Building knowledge clusters

---

## Content Management Functions

### 6. `mcp__megamind__create_chunk`

**Purpose**: Create new knowledge chunks with realm targeting and automatic embedding generation.

**Parameters**:
- `content` (string): Chunk content
- `source_document` (string): Source document identifier
- `section_path` (string): Section path within document
- `session_id` (string): Session identifier for tracking
- `target_realm` (string, default: "PROJECT"): Target realm for storage

**Returns**: Created chunk information with ID and metadata

**Example**:
```python
# Create project-specific chunk
chunk = mcp__megamind__create_chunk(
    content="Authentication middleware implementation for Express.js",
    source_document="auth_guide.md",
    section_path="middleware/authentication",
    session_id="session_123",
    target_realm="PROJECT"
)

# Create global knowledge chunk
chunk = mcp__megamind__create_chunk(
    content="Best practices for API error handling",
    source_document="api_standards.md",
    section_path="error_handling/best_practices",
    session_id="session_123",
    target_realm="GLOBAL"
)
```

**Use Cases**:
- Adding new documentation
- Capturing development insights
- Creating knowledge base entries

### 7. `mcp__megamind__update_chunk`

**Purpose**: Update existing chunk content with session tracking for review.

**Parameters**:
- `chunk_id` (string): Chunk to update
- `new_content` (string): Updated content
- `session_id` (string): Session identifier

**Returns**: Update confirmation with change tracking

**Example**:
```python
# Update chunk content
result = mcp__megamind__update_chunk(
    chunk_id="chunk_12345",
    new_content="Updated authentication middleware with JWT support",
    session_id="session_123"
)
```

**Use Cases**:
- Updating documentation
- Refining knowledge content
- Correcting information

### 8. `mcp__megamind__add_relationship`

**Purpose**: Create cross-references between chunks for knowledge graph construction.

**Parameters**:
- `chunk_id_1` (string): First chunk ID
- `chunk_id_2` (string): Second chunk ID
- `relationship_type` (string): Type of relationship
- `session_id` (string): Session identifier

**Returns**: Relationship creation confirmation

**Example**:
```python
# Create implementation relationship
relationship = mcp__megamind__add_relationship(
    chunk_id_1="chunk_auth_theory",
    chunk_id_2="chunk_auth_implementation",
    relationship_type="implements",
    session_id="session_123"
)

# Create dependency relationship
relationship = mcp__megamind__add_relationship(
    chunk_id_1="chunk_database_setup",
    chunk_id_2="chunk_migration_scripts",
    relationship_type="depends_on",
    session_id="session_123"
)
```

**Relationship Types**:
- `"implements"`: Implementation of concept
- `"depends_on"`: Dependency relationship
- `"extends"`: Extension or enhancement
- `"related_to"`: General relationship
- `"conflicts_with"`: Conflicting approaches
- `"supersedes"`: Newer version/approach

### 9. `mcp__megamind__batch_generate_embeddings`

**Purpose**: Generate embeddings for existing chunks in batch for performance.

**Parameters**:
- `chunk_ids` (array): List of chunk IDs to process
- `realm_id` (string): Realm context for processing

**Returns**: Batch processing results with success/failure status

**Example**:
```python
# Generate embeddings for multiple chunks
result = mcp__megamind__batch_generate_embeddings(
    chunk_ids=["chunk_001", "chunk_002", "chunk_003"],
    realm_id="PROJECT"
)
```

**Use Cases**:
- Initial system setup
- Migrating existing content
- Performance optimization

---

## Knowledge Promotion Functions

### 10. `mcp__megamind__create_promotion_request`

**Purpose**: Request promotion of knowledge chunk to different realm with justification.

**Parameters**:
- `chunk_id` (string): Chunk to promote
- `target_realm` (string): Destination realm
- `justification` (string): Reason for promotion
- `session_id` (string): Session identifier

**Returns**: Promotion request ID and status

**Example**:
```python
# Promote debugging technique to global realm
promotion = mcp__megamind__create_promotion_request(
    chunk_id="chunk_debug_technique",
    target_realm="GLOBAL",
    justification="This debugging approach is applicable across all JavaScript projects",
    session_id="session_123"
)
```

**Use Cases**:
- Sharing valuable insights globally
- Promoting best practices
- Knowledge standardization

### 11. `mcp__megamind__get_promotion_requests`

**Purpose**: Retrieve list of promotion requests with filtering options.

**Parameters**:
- `filter_status` (string, optional): Filter by status
- `filter_realm` (string, optional): Filter by target realm
- `limit` (integer, default: 20): Maximum results

**Returns**: List of promotion requests with details

**Example**:
```python
# Get all pending promotions
requests = mcp__megamind__get_promotion_requests(
    filter_status="pending",
    limit=10
)

# Get promotions targeting GLOBAL realm
requests = mcp__megamind__get_promotion_requests(
    filter_realm="GLOBAL",
    limit=20
)
```

**Use Cases**:
- Reviewing promotion queue
- Managing knowledge governance
- Tracking promotion activity

### 12. `mcp__megamind__approve_promotion_request`

**Purpose**: Approve pending promotion request with reasoning.

**Parameters**:
- `promotion_id` (string): Promotion request ID
- `approval_reason` (string): Justification for approval
- `session_id` (string): Session identifier

**Returns**: Approval confirmation and processing status

**Example**:
```python
# Approve promotion
result = mcp__megamind__approve_promotion_request(
    promotion_id="promotion_456",
    approval_reason="High-value pattern with broad applicability",
    session_id="session_123"
)
```

### 13. `mcp__megamind__reject_promotion_request`

**Purpose**: Reject promotion request with explanation.

**Parameters**:
- `promotion_id` (string): Promotion request ID
- `rejection_reason` (string): Reason for rejection
- `session_id` (string): Session identifier

**Returns**: Rejection confirmation

**Example**:
```python
# Reject promotion
result = mcp__megamind__reject_promotion_request(
    promotion_id="promotion_456",
    rejection_reason="Too project-specific, limited general applicability",
    session_id="session_123"
)
```

### 14. `mcp__megamind__get_promotion_impact`

**Purpose**: Analyze potential impact of promotion on target realm.

**Parameters**:
- `promotion_id` (string): Promotion request ID

**Returns**: Impact analysis with affected relationships and conflicts

**Example**:
```python
# Analyze promotion impact
impact = mcp__megamind__get_promotion_impact("promotion_456")
print(f"Affected relationships: {impact['relationship_count']}")
print(f"Potential conflicts: {impact['conflict_count']}")
```

### 15. `mcp__megamind__get_promotion_queue_summary`

**Purpose**: Get summary statistics of promotion queue by realm.

**Parameters**:
- `filter_realm` (string, optional): Filter by realm

**Returns**: Queue statistics and metrics

**Example**:
```python
# Get overall queue summary
summary = mcp__megamind__get_promotion_queue_summary()

# Get GLOBAL realm queue summary
summary = mcp__megamind__get_promotion_queue_summary(filter_realm="GLOBAL")
```

---

## Session Management Functions

### 16. `mcp__megamind__get_session_primer`

**Purpose**: Generate lightweight context for session continuity without loading full knowledge chunks.

**Parameters**:
- `last_session_data` (string, optional): Previous session information

**Returns**: Session primer with procedural context and workflow state

**Example**:
```python
# Start new session
primer = mcp__megamind__get_session_primer()

# Resume existing session
primer = mcp__megamind__get_session_primer(
    last_session_data="session_abc123_state"
)
```

**Use Cases**:
- Session initialization
- Context priming for AI assistants
- Workflow continuity

### 17. `mcp__megamind__get_pending_changes`

**Purpose**: Retrieve pending changes for session with smart highlighting.

**Parameters**:
- `session_id` (string): Session identifier

**Returns**: List of pending changes with impact assessment

**Example**:
```python
# Get pending changes for review
changes = mcp__megamind__get_pending_changes("session_123")
```

**Use Cases**:
- Session review workflows
- Change approval processes
- Impact assessment

### 18. `mcp__megamind__commit_session_changes`

**Purpose**: Commit approved changes and track contributions.

**Parameters**:
- `session_id` (string): Session identifier
- `approved_changes` (array): List of approved change IDs

**Returns**: Commit confirmation with applied changes

**Example**:
```python
# Commit selected changes
result = mcp__megamind__commit_session_changes(
    session_id="session_123",
    approved_changes=["change_001", "change_002"]
)
```

**Use Cases**:
- Finalizing session work
- Applying approved changes
- Tracking contributions

---

## Analytics & Optimization Functions

### 19. `mcp__megamind__track_access`

**Purpose**: Update access analytics for optimization and usage tracking.

**Parameters**:
- `chunk_id` (string): Accessed chunk ID
- `query_context` (string, optional): Context of access

**Returns**: Access tracking confirmation

**Example**:
```python
# Track chunk access
mcp__megamind__track_access(
    chunk_id="chunk_12345",
    query_context="authentication implementation"
)
```

**Use Cases**:
- Usage analytics
- Performance optimization
- Access pattern analysis

### 20. `mcp__megamind__get_hot_contexts`

**Purpose**: Get frequently accessed chunks prioritized by usage patterns.

**Parameters**:
- `model_type` (string, default: "sonnet"): Model type for optimization
- `limit` (integer, default: 20): Maximum results

**Returns**: List of frequently accessed chunks with usage metrics

**Example**:
```python
# Get hot contexts for Sonnet model
hot_chunks = mcp__megamind__get_hot_contexts(
    model_type="sonnet",
    limit=15
)

# Get hot contexts for Opus model
hot_chunks = mcp__megamind__get_hot_contexts(
    model_type="opus",
    limit=10
)
```

**Use Cases**:
- Performance optimization
- Cache management
- Content prioritization

---

## Phase 2: Enhanced Embedding Functions

### 21. `mcp__megamind__content_analyze_document`

**Purpose**: Analyze document structure with semantic boundary detection.

**Parameters**:
- `content` (string): Document content to analyze
- `document_name` (string): Document identifier
- `session_id` (string): Session identifier
- `metadata` (object): Additional document metadata

**Returns**: Document structure analysis with semantic boundaries

**Example**:
```python
# Analyze document structure
analysis = mcp__megamind__content_analyze_document(
    content=document_text,
    document_name="api_documentation.md",
    session_id="session_123",
    metadata={"type": "technical_doc", "version": "1.0"}
)
```

**Use Cases**:
- Document preprocessing
- Structure analysis
- Boundary detection

### 22. `mcp__megamind__content_create_chunks`

**Purpose**: Create optimized chunks with intelligent strategies and configurable parameters.

**Parameters**:
- `content` (string): Content to chunk
- `document_name` (string): Document identifier
- `session_id` (string): Session identifier
- `strategy` (string): Chunking strategy
- `max_tokens` (integer): Maximum tokens per chunk
- `target_realm` (string): Target realm

**Returns**: Created chunks with metadata and quality metrics

**Example**:
```python
# Create chunks with semantic strategy
chunks = mcp__megamind__content_create_chunks(
    content=document_text,
    document_name="user_guide.md",
    session_id="session_123",
    strategy="semantic_aware",
    max_tokens=512,
    target_realm="PROJECT"
)
```

**Chunking Strategies**:
- `"semantic_aware"`: Respects semantic boundaries
- `"markdown_structure"`: Follows markdown hierarchy
- `"hybrid"`: Combines semantic and structural approaches

### 23. `mcp__megamind__content_assess_quality`

**Purpose**: Perform 8-dimensional quality assessment on chunks.

**Parameters**:
- `chunk_ids` (array): List of chunk IDs to assess
- `session_id` (string): Session identifier
- `include_context` (boolean): Include contextual assessment

**Returns**: Quality assessment with scores and improvement suggestions

**Example**:
```python
# Assess chunk quality
assessment = mcp__megamind__content_assess_quality(
    chunk_ids=["chunk_001", "chunk_002"],
    session_id="session_123",
    include_context=True
)
```

**Quality Dimensions**:
1. **Readability** (15%): Text clarity and structure
2. **Technical Accuracy** (25%): Correctness of technical content
3. **Completeness** (20%): Coverage of topic
4. **Relevance** (15%): Relevance to context
5. **Freshness** (10%): Currency of information
6. **Coherence** (10%): Logical flow and consistency
7. **Uniqueness** (3%): Distinctive value
8. **Authority** (2%): Source credibility

### 24. `mcp__megamind__content_optimize_embeddings`

**Purpose**: Optimize chunks for embedding generation with multiple cleaning levels.

**Parameters**:
- `chunk_ids` (array): Chunks to optimize
- `session_id` (string): Session identifier
- `model` (string): Target embedding model
- `cleaning_level` (string): Cleaning intensity
- `batch_size` (integer): Batch processing size

**Returns**: Optimization results with embedding generation status

**Example**:
```python
# Optimize for embedding generation
result = mcp__megamind__content_optimize_embeddings(
    chunk_ids=["chunk_001", "chunk_002", "chunk_003"],
    session_id="session_123",
    model="all-MiniLM-L6-v2",
    cleaning_level="standard",
    batch_size=50
)
```

**Cleaning Levels**:
- `"minimal"`: Basic formatting cleanup
- `"standard"`: Standard optimization
- `"aggressive"`: Maximum optimization

### 25. `mcp__megamind__session_create`

**Purpose**: Create new embedding processing sessions.

**Parameters**:
- `session_type` (string): Type of session
- `created_by` (string): Session creator
- `description` (string): Session description
- `metadata` (object): Session metadata

**Returns**: New session ID and configuration

**Example**:
```python
# Create analysis session
session = mcp__megamind__session_create(
    session_type="analysis",
    created_by="user_123",
    description="Document structure analysis",
    metadata={"project": "api_docs", "version": "1.0"}
)
```

**Session Types**:
- `"analysis"`: Document analysis
- `"ingestion"`: Content ingestion
- `"curation"`: Content curation
- `"mixed"`: Multiple operations

### 26. `mcp__megamind__session_get_state`

**Purpose**: Get current session state and progress tracking.

**Parameters**:
- `session_id` (string): Session identifier

**Returns**: Session state with progress metrics and status

**Example**:
```python
# Get session state
state = mcp__megamind__session_get_state("session_123")
print(f"Processed chunks: {state['processed_chunks']}")
print(f"Failed chunks: {state['failed_chunks']}")
```

### 27. `mcp__megamind__session_complete`

**Purpose**: Complete and finalize processing sessions.

**Parameters**:
- `session_id` (string): Session identifier

**Returns**: Final session statistics and completion status

**Example**:
```python
# Complete session
result = mcp__megamind__session_complete("session_123")
print(f"Total processed: {result['total_processed']}")
print(f"Success rate: {result['success_rate']}")
```

---

## Phase 3: Knowledge Management Functions

### 28. `mcp__megamind__knowledge_ingest_document`

**Purpose**: Ingest documents with automatic knowledge type detection and quality-driven chunk creation.

**Parameters**:
- `document_content` (string): Document content
- `document_name` (string): Document identifier
- `session_id` (string): Session identifier
- `knowledge_type` (string): Type of knowledge
- `metadata` (object): Document metadata

**Returns**: Ingestion results with created chunks and quality metrics

**Example**:
```python
# Ingest technical documentation
result = mcp__megamind__knowledge_ingest_document(
    document_content=doc_text,
    document_name="api_reference.md",
    session_id="session_123",
    knowledge_type="documentation",
    metadata={"category": "api", "version": "2.0"}
)
```

**Knowledge Types**:
- `"documentation"`: Technical documentation
- `"code_patterns"`: Code examples and patterns
- `"best_practices"`: Best practice guidelines
- `"troubleshooting"`: Problem-solving guides
- `"architecture"`: System architecture info
- `"procedures"`: Process documentation
- `"reference"`: Reference materials

### 29. `mcp__megamind__knowledge_discover_relationships`

**Purpose**: Discover cross-references with semantic similarity analysis.

**Parameters**:
- `chunk_ids` (array): Chunks to analyze for relationships
- `session_id` (string): Session identifier
- `similarity_threshold` (float): Minimum similarity for relationships

**Returns**: Discovered relationships with confidence scores

**Example**:
```python
# Discover relationships
relationships = mcp__megamind__knowledge_discover_relationships(
    chunk_ids=["chunk_001", "chunk_002", "chunk_003"],
    session_id="session_123",
    similarity_threshold=0.75
)
```

**Relationship Types Discovered**:
- Semantic similarity
- Conceptual relationships
- Implementation dependencies
- Complementary information

### 30. `mcp__megamind__knowledge_optimize_retrieval`

**Purpose**: Optimize retrieval patterns based on usage data.

**Parameters**:
- `usage_patterns` (object): Usage pattern data
- `session_id` (string): Session identifier
- `optimization_target` (string): Target for optimization

**Returns**: Optimization recommendations and applied changes

**Example**:
```python
# Optimize retrieval for performance
optimization = mcp__megamind__knowledge_optimize_retrieval(
    usage_patterns=usage_data,
    session_id="session_123",
    optimization_target="performance"
)
```

**Optimization Targets**:
- `"performance"`: Speed optimization
- `"relevance"`: Relevance optimization
- `"diversity"`: Result diversity
- `"accuracy"`: Accuracy improvement

### 31. `mcp__megamind__knowledge_get_related`

**Purpose**: Get related knowledge chunks with intelligent relationship traversal.

**Parameters**:
- `chunk_id` (string): Reference chunk ID
- `relationship_types` (array): Types of relationships to follow
- `max_depth` (integer): Maximum traversal depth
- `session_id` (string): Session identifier

**Returns**: Related chunks with relationship paths and scores

**Example**:
```python
# Get related chunks
related = mcp__megamind__knowledge_get_related(
    chunk_id="chunk_auth_impl",
    relationship_types=["implements", "depends_on"],
    max_depth=2,
    session_id="session_123"
)
```

---

## Phase 3: Session Tracking Functions

### 32. `mcp__megamind__session_create_operational`

**Purpose**: Create lightweight operational session for tracking development activities.

**Parameters**:
- `session_type` (string): Type of operational session
- `created_by` (string): Session creator
- `description` (string): Session description
- `metadata` (object): Session metadata

**Returns**: Operational session ID and configuration

**Example**:
```python
# Create development session
session = mcp__megamind__session_create_operational(
    session_type="development",
    created_by="developer_123",
    description="Authentication module development",
    metadata={"feature": "auth", "sprint": "sprint_5"}
)
```

**Operational Session Types**:
- `"development"`: Development activities
- `"debugging"`: Debugging sessions
- `"research"`: Research activities
- `"review"`: Code review sessions

### 33. `mcp__megamind__session_track_action`

**Purpose**: Track actions within operational sessions with priority levels.

**Parameters**:
- `session_id` (string): Session identifier
- `action_type` (string): Type of action
- `action_description` (string): Description of action
- `priority` (string): Action priority
- `metadata` (object): Action metadata

**Returns**: Action tracking confirmation

**Example**:
```python
# Track development action
action = mcp__megamind__session_track_action(
    session_id="session_123",
    action_type="implementation",
    action_description="Implemented JWT authentication middleware",
    priority="high",
    metadata={"file": "auth.middleware.js", "lines": 45}
)
```

**Action Types**:
- `"implementation"`: Code implementation
- `"testing"`: Testing activities
- `"debugging"`: Debugging actions
- `"documentation"`: Documentation updates
- `"refactoring"`: Code refactoring

**Priority Levels**:
- `"high"`: High priority
- `"medium"`: Medium priority
- `"low"`: Low priority

### 34. `mcp__megamind__session_get_recap`

**Purpose**: Generate session recap with automatic accomplishment detection.

**Parameters**:
- `session_id` (string): Session identifier
- `include_timeline` (boolean): Include timeline view
- `format` (string): Output format

**Returns**: Session recap with accomplishments and timeline

**Example**:
```python
# Get session recap
recap = mcp__megamind__session_get_recap(
    session_id="session_123",
    include_timeline=True,
    format="structured"
)
```

**Output Formats**:
- `"structured"`: Structured data format
- `"narrative"`: Natural language summary
- `"timeline"`: Timeline view

### 35. `mcp__megamind__session_prime_context`

**Purpose**: Prime context for session resumption with next-step suggestions.

**Parameters**:
- `session_id` (string): Session identifier
- `context_type` (string): Type of context to prime

**Returns**: Context primer with next-step recommendations

**Example**:
```python
# Prime context for resumption
context = mcp__megamind__session_prime_context(
    session_id="session_123",
    context_type="development"
)
```

**Context Types**:
- `"development"`: Development context
- `"debugging"`: Debugging context
- `"research"`: Research context

### 36. `mcp__megamind__session_list_recent`

**Purpose**: List recent operational sessions with filtering options.

**Parameters**:
- `limit` (integer): Maximum sessions to return
- `session_type` (string, optional): Filter by session type
- `created_by` (string, optional): Filter by creator

**Returns**: List of recent sessions with metadata

**Example**:
```python
# List recent sessions
sessions = mcp__megamind__session_list_recent(
    limit=10,
    session_type="development"
)
```

### 37. `mcp__megamind__session_close`

**Purpose**: Close operational session with final summary generation.

**Parameters**:
- `session_id` (string): Session identifier
- `completion_notes` (string): Final notes
- `generate_summary` (boolean): Generate final summary

**Returns**: Session closure confirmation with summary

**Example**:
```python
# Close session
result = mcp__megamind__session_close(
    session_id="session_123",
    completion_notes="Authentication module completed and tested",
    generate_summary=True
)
```

---

## Phase 4: AI Enhancement Functions

### 38. `mcp__megamind__ai_improve_chunk_quality`

**Purpose**: Analyze chunk quality and suggest/apply improvements using AI.

**Parameters**:
- `chunk_id` (string): Chunk to analyze
- `session_id` (string): Session identifier
- `improvement_type` (string): Type of improvement
- `apply_suggestions` (boolean): Apply improvements automatically

**Returns**: Quality analysis with improvement suggestions and applied changes

**Example**:
```python
# Analyze chunk quality
improvement = mcp__megamind__ai_improve_chunk_quality(
    chunk_id="chunk_12345",
    session_id="session_123",
    improvement_type="quality_analysis",
    apply_suggestions=False
)
```

**Improvement Types**:
- `"quality_analysis"`: Quality assessment only
- `"readability"`: Readability improvements
- `"completeness"`: Completeness enhancements
- `"accuracy"`: Accuracy corrections

### 39. `mcp__megamind__ai_record_user_feedback`

**Purpose**: Record user feedback and trigger adaptive learning.

**Parameters**:
- `feedback_type` (string): Type of feedback
- `target_id` (string): Target chunk/function ID
- `feedback_content` (string): Feedback content
- `rating` (integer): Numerical rating
- `session_id` (string): Session identifier

**Returns**: Feedback recording confirmation with learning trigger

**Example**:
```python
# Record positive feedback
feedback = mcp__megamind__ai_record_user_feedback(
    feedback_type="chunk_quality",
    target_id="chunk_12345",
    feedback_content="Very helpful implementation example",
    rating=5,
    session_id="session_123"
)
```

**Feedback Types**:
- `"chunk_quality"`: Chunk quality feedback
- `"search_relevance"`: Search result relevance
- `"boundary_accuracy"`: Chunking boundary accuracy
- `"relationship_accuracy"`: Relationship accuracy

### 40. `mcp__megamind__ai_get_adaptive_strategy`

**Purpose**: Get current adaptive strategy based on learned patterns.

**Parameters**:
- `context` (object): Context for strategy
- `session_id` (string): Session identifier

**Returns**: Adaptive strategy recommendations

**Example**:
```python
# Get adaptive strategy
strategy = mcp__megamind__ai_get_adaptive_strategy(
    context={"domain": "web_development", "task": "authentication"},
    session_id="session_123"
)
```

### 41. `mcp__megamind__ai_curate_chunks`

**Purpose**: Run automated curation workflow on chunks.

**Parameters**:
- `chunk_ids` (array): Chunks to curate
- `workflow_id` (string): Curation workflow
- `session_id` (string): Session identifier

**Returns**: Curation results with applied changes

**Example**:
```python
# Run quality curation
curation = mcp__megamind__ai_curate_chunks(
    chunk_ids=["chunk_001", "chunk_002"],
    workflow_id="standard_quality",
    session_id="session_123"
)
```

**Curation Workflows**:
- `"standard_quality"`: Standard quality curation
- `"technical_accuracy"`: Technical accuracy focus
- `"readability"`: Readability enhancement
- `"completeness"`: Completeness improvement

### 42. `mcp__megamind__ai_optimize_performance`

**Purpose**: Optimize system performance based on usage patterns.

**Parameters**:
- `optimization_type` (string): Type of optimization
- `parameters` (object): Optimization parameters
- `session_id` (string): Session identifier

**Returns**: Optimization results with performance metrics

**Example**:
```python
# Optimize search performance
optimization = mcp__megamind__ai_optimize_performance(
    optimization_type="search_performance",
    parameters={"target_latency": 100, "accuracy_threshold": 0.85},
    session_id="session_123"
)
```

**Optimization Types**:
- `"search_performance"`: Search optimization
- `"embedding_cache"`: Embedding cache optimization
- `"retrieval_accuracy"`: Retrieval accuracy improvement
- `"storage_efficiency"`: Storage optimization

### 43. `mcp__megamind__ai_get_performance_insights`

**Purpose**: Get current performance insights and recommendations.

**Parameters**:
- `session_id` (string): Session identifier

**Returns**: Performance insights with recommendations

**Example**:
```python
# Get performance insights
insights = mcp__megamind__ai_get_performance_insights(
    session_id="session_123"
)
```

### 44. `mcp__megamind__ai_generate_enhancement_report`

**Purpose**: Generate comprehensive AI enhancement report.

**Parameters**:
- `report_type` (string): Type of report
- `start_date` (string): Report start date
- `end_date` (string): Report end date
- `session_id` (string): Session identifier

**Returns**: Comprehensive enhancement report

**Example**:
```python
# Generate quality report
report = mcp__megamind__ai_generate_enhancement_report(
    report_type="quality_improvement",
    start_date="2024-01-01",
    end_date="2024-01-31",
    session_id="session_123"
)
```

**Report Types**:
- `"quality_improvement"`: Quality improvement report
- `"performance_optimization"`: Performance optimization report
- `"usage_analytics"`: Usage analytics report
- `"learning_progress"`: Learning progress report

---

## Workflow Patterns

### 1. Document Ingestion Workflow

```python
# 1. Create session
session = mcp__megamind__session_create(
    session_type="ingestion",
    created_by="system",
    description="Document processing workflow"
)

# 2. Analyze document
analysis = mcp__megamind__content_analyze_document(
    content=document_content,
    document_name="guide.md",
    session_id=session["session_id"]
)

# 3. Create optimized chunks
chunks = mcp__megamind__content_create_chunks(
    content=document_content,
    document_name="guide.md",
    session_id=session["session_id"],
    strategy="semantic_aware"
)

# 4. Assess quality
quality = mcp__megamind__content_assess_quality(
    chunk_ids=[chunk["id"] for chunk in chunks],
    session_id=session["session_id"]
)

# 5. Complete session
mcp__megamind__session_complete(session["session_id"])
```

### 2. Knowledge Search Workflow

```python
# 1. Get session primer
primer = mcp__megamind__get_session_primer()

# 2. Search for relevant chunks
results = mcp__megamind__search_chunks(
    query="authentication patterns",
    limit=10,
    search_type="hybrid"
)

# 3. Get related information
for result in results:
    related = mcp__megamind__get_related_chunks(
        chunk_id=result["chunk_id"],
        max_depth=2
    )
    
    # 4. Track access
    mcp__megamind__track_access(
        chunk_id=result["chunk_id"],
        query_context="authentication implementation"
    )
```

### 3. Knowledge Promotion Workflow

```python
# 1. Create promotion request
promotion = mcp__megamind__create_promotion_request(
    chunk_id="valuable_chunk",
    target_realm="GLOBAL",
    justification="Broadly applicable pattern",
    session_id="session_123"
)

# 2. Analyze impact
impact = mcp__megamind__get_promotion_impact(
    promotion["promotion_id"]
)

# 3. Review and approve/reject
if impact["conflict_count"] == 0:
    mcp__megamind__approve_promotion_request(
        promotion_id=promotion["promotion_id"],
        approval_reason="No conflicts, high value",
        session_id="session_123"
    )
```

### 4. AI Enhancement Workflow

```python
# 1. Analyze chunk quality
quality = mcp__megamind__ai_improve_chunk_quality(
    chunk_id="chunk_12345",
    session_id="session_123",
    improvement_type="quality_analysis"
)

# 2. Apply improvements if needed
if quality["quality_score"] < 0.8:
    mcp__megamind__ai_improve_chunk_quality(
        chunk_id="chunk_12345",
        session_id="session_123",
        improvement_type="readability",
        apply_suggestions=True
    )

# 3. Record feedback
mcp__megamind__ai_record_user_feedback(
    feedback_type="chunk_quality",
    target_id="chunk_12345",
    feedback_content="Much improved!",
    rating=5,
    session_id="session_123"
)
```

---

## Best Practices

### 1. Session Management
- Always create sessions for related operations
- Use descriptive session names and metadata
- Complete sessions properly to track contributions
- Use session primers for context continuity

### 2. Search Optimization
- Start with hybrid search for best results
- Use semantic search for conceptual queries
- Track access patterns for optimization
- Leverage relationship traversal for deep context

### 3. Content Quality
- Assess quality before and after processing
- Use appropriate chunking strategies
- Implement feedback loops for continuous improvement
- Monitor and optimize embeddings regularly

### 4. Knowledge Governance
- Use promotion system for valuable insights
- Implement proper review processes
- Track relationships and dependencies
- Maintain realm-appropriate content

### 5. Performance Optimization
- Use batch operations for bulk processing
- Monitor and optimize hot contexts
- Implement caching strategies
- Regular performance analysis and tuning

### 6. Error Handling
- Always check return values and status
- Implement retry logic for transient failures
- Use session tracking for debugging
- Monitor system health and performance

---

## Function Quick Reference

| Function | Purpose | Phase |
|----------|---------|--------|
| `search_chunks` | Primary search with hybrid capabilities | Core |
| `get_chunk` | Retrieve specific chunk with relationships | Core |
| `get_related_chunks` | Find related chunks through relationships | Core |
| `search_chunks_semantic` | Pure semantic search | Core |
| `search_chunks_by_similarity` | Find similar chunks by embedding | Core |
| `create_chunk` | Create new knowledge chunks | Core |
| `update_chunk` | Update existing chunk content | Core |
| `add_relationship` | Create chunk relationships | Core |
| `batch_generate_embeddings` | Generate embeddings in batch | Core |
| `create_promotion_request` | Request knowledge promotion | Core |
| `get_promotion_requests` | List promotion requests | Core |
| `approve_promotion_request` | Approve promotion | Core |
| `reject_promotion_request` | Reject promotion | Core |
| `get_promotion_impact` | Analyze promotion impact | Core |
| `get_promotion_queue_summary` | Get promotion queue stats | Core |
| `get_session_primer` | Get session context | Core |
| `get_pending_changes` | Get pending session changes | Core |
| `commit_session_changes` | Commit approved changes | Core |
| `track_access` | Track chunk access | Core |
| `get_hot_contexts` | Get frequently accessed chunks | Core |
| `content_analyze_document` | Analyze document structure | Phase 2 |
| `content_create_chunks` | Create optimized chunks | Phase 2 |
| `content_assess_quality` | Assess chunk quality | Phase 2 |
| `content_optimize_embeddings` | Optimize for embeddings | Phase 2 |
| `session_create` | Create embedding session | Phase 2 |
| `session_get_state` | Get session state | Phase 2 |
| `session_complete` | Complete session | Phase 2 |
| `knowledge_ingest_document` | Ingest documents | Phase 3 |
| `knowledge_discover_relationships` | Discover relationships | Phase 3 |
| `knowledge_optimize_retrieval` | Optimize retrieval | Phase 3 |
| `knowledge_get_related` | Get related knowledge | Phase 3 |
| `session_create_operational` | Create operational session | Phase 3 |
| `session_track_action` | Track session actions | Phase 3 |
| `session_get_recap` | Get session recap | Phase 3 |
| `session_prime_context` | Prime session context | Phase 3 |
| `session_list_recent` | List recent sessions | Phase 3 |
| `session_close` | Close operational session | Phase 3 |
| `ai_improve_chunk_quality` | AI quality improvement | Phase 4 |
| `ai_record_user_feedback` | Record user feedback | Phase 4 |
| `ai_get_adaptive_strategy` | Get adaptive strategy | Phase 4 |
| `ai_curate_chunks` | Automated curation | Phase 4 |
| `ai_optimize_performance` | Performance optimization | Phase 4 |
| `ai_get_performance_insights` | Get performance insights | Phase 4 |
| `ai_generate_enhancement_report` | Generate enhancement report | Phase 4 |

---

This usage guide provides comprehensive documentation for all 37 MCP functions in the Enhanced Multi-Embedding Entry System. Each function includes detailed parameters, return values, examples, and use cases to help developers effectively utilize the system's capabilities.