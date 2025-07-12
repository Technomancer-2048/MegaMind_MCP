# Semantic Search Implementation Plan
## Using sentence-transformers/all-MiniLM-L6-v2

**Created:** 2025-07-12  
**Status:** Planning Phase  
**Target Model:** sentence-transformers/all-MiniLM-L6-v2  

## Executive Summary

This plan outlines the implementation of semantic search capabilities in the MegaMind Context Database System with **realm-aware architecture**, replacing the current SQL LIKE-based text search with vector similarity search using sentence-transformers. The implementation will integrate embedding generation into all ingestion workflows and enhance search capabilities with semantic understanding across both Global and Project realms.

## Current State Analysis

### Existing Search Implementation
- **Method:** SQL `LIKE` queries on content, source_document, and section_path fields
- **Location:** `megamind_database_server.py:search_chunks()` (lines 56-92)
- **Limitations:** 
  - No semantic understanding (searches only exact text matches)
  - Poor recall for conceptually related content
  - No ranking by semantic relevance

### Existing Ingestion Flows
1. **Individual Chunk Creation:** `create_chunk()` in MCP server (line 352)
2. **Markdown Ingestion:** `tools/markdown_ingester.py` for structured document import
3. **Bulk Import:** Planned in `Guides/megamind_bulk_ingestion_guide.md`

### Database Schema Status (Updated for Realm Architecture)
- **Realm-Aware Chunks Table:** Fresh schema with built-in `realm_id` and `embedding` JSON field
- **Dual-Realm Access:** Environment-based configuration for Global + Project realm access
- **No Migration Required:** Fresh database creation with semantic search support from the start
- **Infrastructure:** Ready for realm-aware vector storage and retrieval

## Architecture Design

### Component Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Realm-Aware Semantic Search Architecture                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────────┐ │
│  │   Ingestion     │    │    Embedding     │    │   Dual-Realm Search    │ │
│  │   Pipeline      │───▶│    Generator     │───▶│   Engine                │ │
│  │   (Realm-Aware) │    │                  │    │                         │ │
│  └─────────────────┘    └──────────────────┘    └─────────────────────────┘ │
│           │                       │                          │              │
│           ▼                       ▼                          ▼              │
│  ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────────┐ │
│  │ Chunk Creation  │    │ Vector Storage   │    │ Similarity Ranking      │ │
│  │ • PROJECT (def) │    │ • JSON Embeddings│    │ • Cosine Similarity     │ │
│  │ • GLOBAL (opt)  │    │ • Realm Context  │    │ • Realm Priority        │ │
│  │ • Markdown      │    │ • Async Updates  │    │ • Global + Project      │ │
│  │ • Bulk Import   │    │ • Realm Indexes  │    │ • Threshold Filtering   │ │
│  └─────────────────┘    └──────────────────┘    └─────────────────────────┘ │
│                                                                             │
│  Environment Config: MEGAMIND_PROJECT_REALM, MEGAMIND_DEFAULT_TARGET        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. Embedding Service (`services/embedding_service.py`)
- **Model:** sentence-transformers/all-MiniLM-L6-v2
- **Functionality:** Generate 384-dimensional embeddings for text chunks
- **Features:**
  - Singleton pattern for model caching
  - Batch processing for efficiency
  - Async/await support for non-blocking operations
  - Text preprocessing and normalization

#### 2. Realm-Aware Vector Search Engine (`services/vector_search.py`)
- **Algorithm:** Cosine similarity between query and chunk embeddings
- **Realm Integration:** Dual-realm search (Global + Project) with priority ranking
- **Features:**
  - Hybrid search (semantic + keyword fallback)
  - Realm-aware result prioritization (Project > Global)
  - Configurable similarity thresholds
  - Cross-realm relationship discovery
  - Environment-based realm filtering

#### 3. Enhanced Realm-Aware Ingestion Pipeline
- **Integration Points:** All chunk creation flows with realm targeting
- **Process:** Text → Embedding → Realm Assignment → Storage
- **Realm Logic:** 
  - Default: PROJECT realm (environment configured)
  - Optional: GLOBAL realm (explicit targeting)
  - Automatic: Realm inheritance for relationships
- **Error Handling:** Graceful fallback without embeddings, maintains realm assignment

## Implementation Plan

### Phase 1: Foundation Infrastructure (Week 1)

#### 1.1 Embedding Service Implementation
**File:** `mcp_server/services/embedding_service.py`

```python
class EmbeddingService:
    """Singleton service for generating text embeddings"""
    
    def __init__(self):
        self.model = None
        self.device = 'cpu'  # CPU-only for MCP server compatibility
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for single text chunk"""
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple chunks efficiently"""
    
    def preprocess_text(self, text: str) -> str:
        """Clean and normalize text for embedding"""
```

**Dependencies Added:**
```python
# requirements.txt additions
sentence-transformers==2.2.2
torch>=1.9.0
transformers>=4.6.0
```

#### 1.2 Realm-Aware Database Embedding Storage
**Fresh Schema Integration:** Work with realm-aware `megamind_chunks` table

- Enhance realm-aware `create_chunk_with_target()` to include embedding generation
- Update `get_chunk()` to retrieve embeddings with realm context
- Create dual-realm embedding batch operations
- Add embedding validation and error handling with realm support
- Integrate with environment-based realm configuration

#### 1.3 Vector Search Implementation
**File:** `mcp_server/services/vector_search.py`

```python
class RealmAwareVectorSearchEngine:
    """Realm-aware semantic search using embedding similarity"""
    
    def __init__(self, project_realm: str, global_realm: str = 'GLOBAL'):
        self.project_realm = project_realm
        self.global_realm = global_realm
    
    def dual_realm_semantic_search(self, query: str, limit: int = 10, threshold: float = 0.7) -> List[Dict]:
        """Primary semantic search across Global + Project realms"""
    
    def realm_aware_hybrid_search(self, query: str, limit: int = 10) -> List[Dict]:
        """Combine semantic and keyword search with realm prioritization"""
    
    def calculate_realm_aware_similarity(self, query_embedding: List[float], 
                                        chunk_embedding: List[float], 
                                        chunk_realm: str) -> float:
        """Cosine similarity with realm-based weighting"""
```

### Phase 2: Search Integration (Week 2)

#### 2.1 Enhanced Realm-Aware Search Functions
**File:** `megamind_database_server.py`

**Update existing `search_chunks_dual_realm()` method for semantic support:**
```python
def search_chunks_dual_realm(self, query: str, limit: int = 10, search_type: str = "hybrid") -> List[Dict[str, Any]]:
    """Enhanced dual-realm search with semantic capabilities"""
    if search_type == "semantic":
        return self._realm_semantic_search(query, limit)
    elif search_type == "keyword":
        return self._realm_keyword_search(query, limit)  # Current multi-word implementation
    else:  # hybrid (default)
        return self._realm_hybrid_search(query, limit)
```

**Add new realm-aware semantic methods:**
```python
def _realm_semantic_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
    """Pure semantic search across Global + Project realms"""

def _realm_hybrid_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
    """Combine semantic and keyword results with realm prioritization"""
    
def _calculate_realm_semantic_score(self, query_embedding: List[float], 
                                   chunk_embedding: List[float], 
                                   chunk_realm: str) -> float:
    """Calculate similarity score with realm-based weighting"""
```

#### 2.2 MCP Function Enhancement  
**Update existing MCP search function for semantic support:**
```python
# Enhanced existing MCP function (maintains backward compatibility)
def search_chunks(self, query: str, limit: int = 10, search_type: str = "hybrid") -> List[Dict]:
    """Enhanced search with semantic capabilities (dual-realm by default)"""
    # Automatically searches both Global and Project realms
    # search_type options: "semantic", "keyword", "hybrid" (default)

# Optional new MCP functions for explicit semantic search
def search_chunks_semantic(self, query: str, limit: int = 10, threshold: float = 0.7) -> List[Dict]:
    """Pure semantic search across Global + Project realms"""

def search_chunks_by_similarity(self, reference_chunk_id: str, limit: int = 10) -> List[Dict]:
    """Find chunks similar to a reference chunk using embeddings"""
```

### Phase 3: Ingestion Integration (Week 3)

#### 3.1 Realm-Aware Individual Chunk Creation Enhancement
**Modify:** `create_chunk_with_target()` in `megamind_database_server.py`

```python
def create_chunk_with_target(self, content: str, source_document: str, section_path: str, 
                             session_id: str, target_realm: str = None) -> str:
    """Enhanced realm-aware chunk creation with embedding generation"""
    
    # Determine target realm (PROJECT default, GLOBAL optional)
    target = self._determine_target_realm(target_realm)
    
    # Generate embedding for new content
    embedding = self.embedding_service.generate_embedding(content)
    
    # Store embedding in change_data with realm context
    change_data = {
        "chunk_id": new_chunk_id,
        "content": content,
        "realm_id": target,  # Realm assignment
        "embedding": embedding,  # Add embedding to buffered changes
        # ... existing fields
    }
```

#### 3.2 Realm-Aware Markdown Ingestion Enhancement
**Modify:** `tools/markdown_ingester.py`

```python
class RealmAwareMarkdownIngester:
    def __init__(self, db_config: Dict[str, str], target_realm: str = None):
        self.embedding_service = EmbeddingService()
        self.target_realm = target_realm or os.getenv('MEGAMIND_DEFAULT_TARGET', 'PROJECT')
        self.project_realm = os.getenv('MEGAMIND_PROJECT_REALM', 'PROJECT_DEFAULT')
    
    def _process_chunk_with_realm(self, chunk: ChunkMetadata) -> ChunkMetadata:
        """Enhanced chunk processing with embedding generation and realm assignment"""
        
        # Determine realm based on content or configuration
        realm_id = self._determine_chunk_realm(chunk)
        
        # Generate embedding for chunk content
        embedding = self.embedding_service.generate_embedding(chunk.content)
        chunk.embedding = embedding
        chunk.realm_id = realm_id
        return chunk
    
    def _batch_insert_chunks_with_realm(self, chunks: List[ChunkMetadata]):
        """Batch insertion with embedding storage and realm context"""
        
        # Generate embeddings in batch for efficiency
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_service.generate_embeddings_batch(texts)
        
        # Insert chunks with embeddings and realm assignment
        for chunk, embedding in zip(chunks, embeddings):
            self._insert_chunk_with_embedding_and_realm(chunk, embedding)
    
    def _determine_chunk_realm(self, chunk: ChunkMetadata) -> str:
        """Determine appropriate realm for chunk based on content patterns"""
        # Logic to assign GLOBAL vs PROJECT based on content analysis
        # Default to configured target realm
        return self.project_realm if self.target_realm == 'PROJECT' else 'GLOBAL'
```

#### 3.3 Bulk Import Integration
**Create:** `tools/bulk_semantic_ingester.py`

```python
class BulkSemanticIngester(MarkdownIngester):
    """Bulk ingestion with optimized embedding generation"""
    
    def __init__(self, db_config: Dict[str, str], batch_size: int = 50):
        super().__init__(db_config)
        self.batch_size = batch_size
    
    def ingest_directory(self, directory_path: str) -> Dict[str, Any]:
        """Bulk ingest with semantic processing"""
        
        # Process files in batches
        # Generate embeddings efficiently
        # Store with semantic metadata
```

### Phase 4: Performance Optimization (Week 4)

#### 4.1 Embedding Caching Strategy
**File:** `services/embedding_cache.py`

```python
class EmbeddingCache:
    """LRU cache for frequently accessed embeddings"""
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
    
    def get_embedding(self, content_hash: str) -> Optional[List[float]]:
        """Retrieve cached embedding by content hash"""
    
    def store_embedding(self, content_hash: str, embedding: List[float]):
        """Store embedding with content hash key"""
```

#### 4.2 Realm-Aware Database Indexing Optimization
**File:** `database/context_system/05_semantic_indexes.sql`

```sql
-- Optimize embedding storage and retrieval with realm awareness
CREATE INDEX idx_chunks_embedding_exists ON megamind_chunks (realm_id, (JSON_VALID(embedding)));
CREATE INDEX idx_chunks_realm_embedding_search ON megamind_chunks (realm_id, access_count DESC) WHERE embedding IS NOT NULL;
CREATE INDEX idx_chunks_dual_realm_access ON megamind_chunks (realm_id, last_accessed DESC) WHERE realm_id IN ('GLOBAL', ?);

-- Realm-aware performance monitoring views
CREATE VIEW realm_embedding_coverage AS
SELECT 
    realm_id,
    COUNT(*) as total_chunks,
    SUM(CASE WHEN embedding IS NOT NULL THEN 1 ELSE 0 END) as chunks_with_embeddings,
    (SUM(CASE WHEN embedding IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*)) * 100 as coverage_percentage
FROM megamind_chunks
GROUP BY realm_id;

-- Cross-realm performance optimization
CREATE VIEW dual_realm_hot_contexts AS
SELECT 
    chunk_id, 
    content, 
    realm_id,
    access_count,
    CASE 
        WHEN realm_id = ? THEN access_count * 1.2  -- Project realm priority boost
        WHEN realm_id = 'GLOBAL' THEN access_count * 1.0
        ELSE access_count * 0.8
    END as prioritized_score
FROM megamind_chunks 
WHERE embedding IS NOT NULL 
    AND realm_id IN ('GLOBAL', ?)
ORDER BY prioritized_score DESC;
```

#### 4.3 Async Processing Pipeline
**File:** `services/async_embedding_processor.py`

```python
class AsyncEmbeddingProcessor:
    """Background embedding generation for existing chunks"""
    
    def process_missing_embeddings(self):
        """Generate embeddings for chunks without them"""
    
    def update_embeddings_batch(self, chunk_ids: List[str]):
        """Batch update embeddings for specified chunks"""
```

## Realm-Aware Configuration Management

### Environment Variables
```bash
# Realm configuration (from project-realms.md)
MEGAMIND_PROJECT_REALM=PROJ_ECOMMERCE
MEGAMIND_PROJECT_NAME="E-Commerce Platform"
MEGAMIND_DEFAULT_TARGET=PROJECT

# Semantic search configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu
EMBEDDING_BATCH_SIZE=50
SEMANTIC_SEARCH_THRESHOLD=0.7
EMBEDDING_CACHE_SIZE=1000
ASYNC_EMBEDDING_ENABLED=true

# Dual-realm search configuration
REALM_PRIORITY_PROJECT=1.2
REALM_PRIORITY_GLOBAL=1.0
CROSS_REALM_SEARCH_ENABLED=true
```

### MCP Server Configuration
**File:** `mcp.json` enhancement for realm-aware semantic search

```json
{
  "mcpServers": {
    "megamind-database": {
      "command": "python",
      "args": ["mcp_server/megamind_database_server.py"],
      "env": {
        "MEGAMIND_PROJECT_REALM": "PROJ_ECOMMERCE",
        "MEGAMIND_DEFAULT_TARGET": "PROJECT",
        "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
        "SEMANTIC_SEARCH_ENABLED": "true",
        "CROSS_REALM_SEARCH_ENABLED": "true"
      }
    }
  }
}
```

## Realm-Aware Testing Strategy

### Unit Tests
**File:** `tests/test_realm_semantic_search.py`

```python
class TestRealmSemanticSearch:
    def test_embedding_generation_with_realm(self):
        """Test embedding service functionality with realm context"""
    
    def test_dual_realm_search_accuracy(self):
        """Test search relevance across Global + Project realms"""
    
    def test_realm_priority_ranking(self):
        """Test Project realm priority in search results"""
    
    def test_cross_realm_relationship_discovery(self):
        """Test finding relationships across realm boundaries"""
    
    def test_environment_based_configuration(self):
        """Test realm configuration via environment variables"""
```

### Integration Tests
**File:** `tests/test_realm_ingestion_with_embeddings.py`

```python
class TestRealmIngestionWithEmbeddings:
    def test_chunk_creation_with_realm_and_embedding(self):
        """Test individual chunk creation includes embeddings and realm assignment"""
    
    def test_default_realm_assignment(self):
        """Test chunks default to PROJECT realm as configured"""
    
    def test_global_realm_explicit_targeting(self):
        """Test explicit GLOBAL realm targeting for organizational knowledge"""
    
    def test_dual_realm_search_after_ingestion(self):
        """Test search finds content across both realms after ingestion"""
```

### Performance Benchmarks
**File:** `tests/benchmark_realm_semantic_search.py`

```python
def benchmark_dual_realm_search_performance():
    """Compare single-realm vs dual-realm search performance"""
    
def benchmark_realm_priority_calculation():
    """Measure realm-aware scoring performance impact"""
    
def benchmark_cross_realm_memory_usage():
    """Monitor memory consumption with dual-realm access"""
    
def benchmark_realm_aware_embedding_storage():
    """Test embedding storage performance with realm indexing"""
```

## Deployment Strategy (Fresh Database)

### Fresh Database Deployment
1. **Dependencies Installation:** Add sentence-transformers to requirements.txt
2. **Database Creation:** Deploy fresh database with realm-aware schema including embedding support
3. **Initial Data:** Create sample chunks with embeddings from the start
4. **MCP Server Deployment:** Deploy realm-aware MCP server with semantic search capabilities
5. **Environment Configuration:** Set realm-specific environment variables
6. **Validation:** Test semantic search across Global and Project realms

### No Migration Required
- **Fresh Start:** Database created with embedding support from day one
- **Realm-Aware:** All chunks created with proper realm assignment and embeddings
- **Environment-Based:** Configuration through environment variables eliminates complex migration

### Deployment Validation
**Script:** `scripts/validate_realm_semantic_search.py`

```python
def validate_realm_semantic_search():
    """Validate semantic search functionality across realms"""
    
    # Environment configuration validation
    project_realm = os.getenv('MEGAMIND_PROJECT_REALM')
    if not project_realm:
        raise ValueError("MEGAMIND_PROJECT_REALM must be configured")
    
    # Test Global realm semantic search
    global_results = db.search_chunks_semantic("organizational standards", limit=5)
    assert any(chunk['realm_id'] == 'GLOBAL' for chunk in global_results)
    
    # Test Project realm semantic search
    project_results = db.search_chunks_semantic("project implementation", limit=5)
    assert any(chunk['realm_id'] == project_realm for chunk in project_results)
    
    # Test dual-realm search prioritization (Project > Global)
    mixed_results = db.search_chunks_semantic("design patterns", limit=10)
    project_chunks = [c for c in mixed_results if c['realm_id'] == project_realm]
    global_chunks = [c for c in mixed_results if c['realm_id'] == 'GLOBAL']
    
    # Validate Project realm chunks appear first when scores are similar
    if project_chunks and global_chunks:
        assert project_chunks[0]['similarity_score'] >= global_chunks[0]['similarity_score'] * 0.95
    
    # Validate embedding generation and storage
    test_chunk = db.create_chunk_with_target(
        content="Test semantic content",
        source_document="validation_test",
        section_path="test/validation",
        session_id="validation_session"
    )
    
    chunk_data = db.get_chunk(test_chunk)
    assert chunk_data['embedding'] is not None
    assert chunk_data['realm_id'] == project_realm  # Default target
    
    # Monitor search performance and accuracy
    import time
    start_time = time.time()
    results = db.search_chunks_semantic("performance test query", limit=20)
    end_time = time.time()
    
    search_time = end_time - start_time
    assert search_time < 0.5  # Sub-500ms requirement
    assert len(results) > 0   # Must return results
    
    print(f"✅ Realm-aware semantic search validation passed")
    print(f"   Search time: {search_time:.3f}s")
    print(f"   Results found: {len(results)}")
    print(f"   Project realm: {project_realm}")
```

## Success Metrics

### Technical Metrics
- **Search Relevance:** >85% user satisfaction with semantic results across both realms
- **Performance:** <300ms average dual-realm semantic search response time
- **Coverage:** >95% chunks have embeddings within 24 hours (both Global and Project realms)
- **Memory Usage:** <2GB additional memory for embedding service
- **Realm Isolation:** 100% proper realm assignment and access control

### User Experience Metrics
- **Search Success Rate:** >90% queries return relevant results from appropriate realms
- **Token Reduction:** Maintain 70-80% context token reduction with dual-realm access
- **Cross-Reference Discovery:** 50% improvement in related content finding across realm boundaries
- **Realm Prioritization:** Project-specific results ranked higher than global when relevance is similar

## Risk Assessment and Mitigation

### Technical Risks
1. **Memory Usage:** sentence-transformers model loading
   - **Mitigation:** CPU-only mode, model caching, lazy loading
2. **Performance Impact:** Dual-realm search latency
   - **Mitigation:** Realm-aware indexing, async processing, batch operations
3. **Storage Growth:** Embedding vectors increase database size across realms
   - **Mitigation:** JSON compression, realm-specific cleanup policies
4. **Realm Configuration Errors:** Incorrect environment setup
   - **Mitigation:** Configuration validation, clear error messages, fallback defaults

### Operational Risks
1. **Realm Isolation Breach:** Cross-realm data leakage
   - **Mitigation:** Strict environment-based configuration, access validation
2. **Cross-Realm Performance:** Searching both realms impacts speed
   - **Mitigation:** Optimized dual-realm indexes, result caching, priority weighting
3. **Fresh Database Deployment:** No migration path for existing data
   - **Mitigation:** Fresh database approach eliminates migration complexity

## Implementation Timeline

### Week 1: Realm-Aware Foundation
- [ ] Implement EmbeddingService class with realm context support
- [ ] Create RealmAwareVectorSearchEngine
- [ ] Add dependency management and environment configuration
- [ ] Basic unit tests for dual-realm functionality

### Week 2: Dual-Realm Search Integration
- [ ] Enhance search_chunks_dual_realm() with semantic capabilities
- [ ] Add realm-aware MCP functions for semantic search
- [ ] Integration testing with Global + Project realm access
- [ ] Performance baseline establishment for dual-realm search

### Week 3: Realm-Aware Ingestion Enhancement
- [ ] Update create_chunk_with_target() with embedding generation and realm assignment
- [ ] Enhance RealmAwareMarkdownIngester
- [ ] Create BulkSemanticIngester with realm support
- [ ] End-to-end testing across both realms

### Week 4: Optimization & Fresh Database Deployment
- [ ] Implement realm-aware caching strategies
- [ ] Database indexing optimization for dual-realm access
- [ ] Async processing pipeline with realm context
- [ ] Fresh database deployment with semantic search and realm support

## Future Enhancements

### Advanced Semantic Features
- **Multi-language Support:** Support non-English content
- **Domain-Specific Models:** Fine-tuned models for specific knowledge domains
- **Hierarchical Embeddings:** Different granularity levels (paragraph, section, document)

### AI-Powered Improvements
- **Query Expansion:** Automatic query enhancement using LLMs
- **Semantic Clustering:** Group related chunks for better organization
- **Adaptive Ranking:** Learn from user interactions to improve search results

### Performance Optimizations
- **Vector Databases:** Consider specialized vector databases (Pinecone, Weaviate)
- **Hardware Acceleration:** GPU support for larger deployments
- **Distributed Processing:** Scale embedding generation across multiple nodes

## Conclusion

This realm-aware semantic search implementation will transform the MegaMind Context Database from a simple keyword search system into an intelligent, multi-tenant knowledge retrieval platform. The integration with the project realms architecture enables scalable knowledge organization while maintaining semantic search capabilities across both Global and Project contexts.

The use of sentence-transformers/all-MiniLM-L6-v2 provides an optimal balance of accuracy, performance, and resource requirements for the MCP server environment. The dual-realm approach ensures project isolation while enabling cross-realm knowledge discovery, with Project realm content appropriately prioritized over Global realm content.

With proper implementation of the outlined realm-aware architecture, the system will achieve significant improvements in search relevance while maintaining sub-second performance requirements across both realms. The fresh database approach eliminates migration complexity and enables immediate deployment with full semantic search and realm support from day one.