# MegaMind Bulk Markdown Ingestion Guide

## Overview

This guide provides step-by-step instructions for ingesting existing markdown documentation into the MegaMind database system. The process converts large markdown files into semantically coherent chunks while preserving relationships and metadata.

## Prerequisites

### System Requirements
- MegaMind database server running (megamind_database)
- Python 3.8+ with required dependencies
- Access to existing markdown documentation directory
- Sufficient disk space (estimate 2-3x source file size for database storage)

### Required Python Packages
```bash
pip install sentence-transformers mysql-connector-python beautifulsoup4 markdown nltk
```

### Database Preparation
```sql
-- Ensure MegaMind database exists and is accessible
USE megamind_database;

-- Verify core tables exist
SHOW TABLES LIKE 'megamind_%';

-- Expected tables:
-- megamind_chunks
-- megamind_chunk_relationships  
-- megamind_chunk_tags
-- megamind_session_changes
-- megamind_knowledge_contributions
```

## Phase 1: Pre-Ingestion Analysis

### 1.1 Document Discovery and Assessment
```bash
# Analyze existing documentation structure
python tools/megamind_ingestion_analyzer.py \
    --source-dir /path/to/markdown/docs \
    --output-report pre_ingestion_analysis.json

# Expected output:
# - Total files and size
# - Average file length
# - Heading structure analysis
# - Code block distribution
# - Cross-reference patterns
```

**Analysis Report Includes:**
- **File inventory**: Count, total size, average length per file
- **Structure patterns**: Heading hierarchy, section boundaries
- **Content distribution**: Code blocks, tables, lists, plain text ratios
- **Cross-reference mapping**: Internal links and dependencies
- **Complexity assessment**: Estimated chunk count and processing time

### 1.2 Ingestion Strategy Planning
```bash
# Generate ingestion plan based on analysis
python tools/megamind_ingestion_planner.py \
    --analysis-file pre_ingestion_analysis.json \
    --strategy balanced \
    --output-plan ingestion_plan.json

# Strategy options:
# --strategy conservative  # Larger chunks (100-200 lines), fewer relationships
# --strategy balanced     # Medium chunks (50-150 lines), moderate relationships  
# --strategy aggressive   # Smaller chunks (20-100 lines), maximum relationships
```

**Ingestion Plan Contains:**
- **Chunking strategy**: Size targets, boundary detection rules
- **Processing order**: Dependency-aware file processing sequence
- **Relationship mapping**: Cross-reference discovery approach
- **Resource allocation**: Memory usage, processing time estimates
- **Quality targets**: Expected chunk count, relationship density

## Phase 2: Semantic Chunking and Processing

### 2.1 Batch Processing Configuration
```python
# tools/megamind_bulk_ingester.py configuration
INGESTION_CONFIG = {
    "chunk_size_min": 20,        # Minimum lines per chunk
    "chunk_size_max": 150,       # Maximum lines per chunk  
    "chunk_size_target": 75,     # Target lines per chunk
    "preserve_code_blocks": True, # Keep code blocks intact
    "preserve_tables": True,     # Keep tables intact
    "boundary_detection": "semantic", # "semantic", "heading", "hybrid"
    "cross_reference_depth": 2,  # Relationship discovery depth
    "batch_size": 100,          # Chunks per database transaction
    "embedding_batch_size": 50, # Embeddings per batch
    "parallel_workers": 4       # Concurrent processing threads
}
```

### 2.2 Execute Bulk Ingestion
```bash
# Start bulk ingestion process
python tools/megamind_bulk_ingester.py \
    --config ingestion_plan.json \
    --source-dir /path/to/markdown/docs \
    --database-config megamind_db_config.json \
    --log-level INFO \
    --resume-checkpoint ingestion_checkpoint.json

# Processing stages:
# 1. File parsing and semantic boundary detection
# 2. Chunk generation with metadata extraction
# 3. Embedding generation for semantic search
# 4. Cross-reference discovery and relationship mapping
# 5. Database insertion with transaction safety
# 6. Validation and quality assurance
```

**Ingestion Process Flow:**
1. **Parse markdown files** using semantic boundary detection
2. **Generate chunks** with preserved formatting and metadata
3. **Extract relationships** through cross-reference analysis
4. **Generate embeddings** using sentence-transformers model
5. **Batch insert** into database with transaction safety
6. **Validate integrity** and generate completion report

### 2.3 Progress Monitoring
```bash
# Monitor ingestion progress (separate terminal)
python tools/megamind_ingestion_monitor.py \
    --checkpoint-file ingestion_checkpoint.json \
    --refresh-interval 30

# Real-time metrics:
# - Files processed / total files
# - Chunks created / estimated chunks  
# - Relationships discovered
# - Database insertion rate
# - Estimated completion time
# - Error count and types
```

## Phase 3: Semantic Analysis and Relationship Discovery

### 3.1 Embedding Generation and Similarity Analysis
```bash
# Generate embeddings for all chunks (if not done during ingestion)
python tools/megamind_embedding_generator.py \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --batch-size 100 \
    --output-format json \
    --cache-embeddings True

# Expected processing time: ~5-10 chunks per second
# Memory usage: ~2GB for 10,000 chunks
```

### 3.2 Automatic Relationship Discovery
```bash
# Discover semantic relationships between chunks
python tools/megamind_relationship_discoverer.py \
    --similarity-threshold 0.75 \
    --max-relationships-per-chunk 10 \
    --relationship-types semantic,references,enhances \
    --confidence-threshold 0.8

# Relationship types discovered:
# - semantic: Content similarity above threshold
# - references: Explicit links and mentions
# - enhances: Complementary information patterns
# - depends_on: Prerequisite relationships
```

### 3.3 Cross-Reference Validation
```bash
# Validate and strengthen discovered relationships
python tools/megamind_relationship_validator.py \
    --validate-bidirectional True \
    --remove-weak-relationships True \
    --strength-threshold 0.6 \
    --update-database True

# Validation process:
# - Verify bidirectional relationship consistency
# - Remove relationships below strength threshold  
# - Consolidate duplicate relationships
# - Update relationship strength scores
```

## Phase 4: Tagging and Metadata Enhancement

### 4.1 Automatic Tagging System
```bash
# Generate semantic tags for all chunks
python tools/megamind_auto_tagger.py \
    --tag-types subsystem,function_type,language,applies_to \
    --confidence-threshold 0.7 \
    --max-tags-per-chunk 5

# Tag categories:
# - subsystem: entity, spatial, sql, deployment, etc.
# - function_type: validation, creation, optimization, debugging
# - language: sql, python, bash, javascript, markdown
# - applies_to: table_design, trigger_creation, performance_tuning
```

### 4.2 Content Classification
```bash
# Classify chunks by content type and complexity
python tools/megamind_content_classifier.py \
    --chunk-types rule,function,section,example,reference \
    --complexity-levels basic,intermediate,advanced \
    --update-metadata True

# Classification results:
# - Content type assignment for optimized retrieval
# - Complexity scoring for model-specific delivery
# - Priority scoring based on content importance
```

## Phase 5: Quality Assurance and Validation

### 5.1 Ingestion Quality Report
```bash
# Generate comprehensive quality assessment
python tools/megamind_quality_assessor.py \
    --source-docs /path/to/markdown/docs \
    --output-report ingestion_quality_report.html \
    --include-samples True

# Quality metrics:
# - Content preservation: 99.5%+ target
# - Relationship accuracy: 85%+ target  
# - Semantic coherence: 90%+ target
# - Cross-reference coverage: 80%+ target
```

**Quality Assessment Includes:**
- **Content preservation**: Verify no information loss during chunking
- **Boundary quality**: Check semantic coherence of chunk boundaries
- **Relationship accuracy**: Validate discovered relationships make sense
- **Search effectiveness**: Test retrieval quality with sample queries
- **Performance metrics**: Response times, memory usage, database size

### 5.2 Sample Query Testing
```bash
# Test retrieval effectiveness with real queries
python tools/megamind_retrieval_tester.py \
    --test-queries sample_queries.json \
    --expected-results expected_results.json \
    --relevance-threshold 0.8

# Sample test queries:
# - "SQL trigger creation standards"
# - "MySQL performance optimization"  
# - "Entity system relationship management"
# - "Deployment script error handling"
```

### 5.3 Performance Benchmarking
```bash
# Benchmark search and retrieval performance
python tools/megamind_performance_benchmark.py \
    --concurrent-users 50 \
    --queries-per-user 20 \
    --query-types search,get_chunk,get_related \
    --target-response-time 200ms

# Performance targets:
# - Search queries: <200ms (95th percentile)
# - Chunk retrieval: <50ms (95th percentile)
# - Relationship traversal: <100ms (95th percentile)
# - Concurrent handling: 50+ users without degradation
```

## Phase 6: System Integration and Deployment

### 6.1 MCP Server Integration
```bash
# Start MegaMind MCP server with ingested data
docker-compose -f docker-compose.megamind-db.yml up -d

# Verify MCP functions work with ingested data
python tools/megamind_mcp_tester.py \
    --test-all-functions True \
    --sample-queries 10 \
    --verify-responses True

# Test key MCP functions:
# - mcp__megamind_db__search_chunks
# - mcp__megamind_db__get_chunk  
# - mcp__megamind_db__get_related_chunks
# - mcp__megamind_db__track_access
```

### 6.2 CLAUDE.md Integration Verification
```bash
# Test integration with existing workflow
python tools/megamind_claude_integration_test.py \
    --claude-md-path /path/to/CLAUDE.md \
    --test-session-primer True \
    --test-context-delivery True \
    --compare-with-baseline True

# Integration tests:
# - Session primer generation
# - Context delivery vs. baseline markdown loading
# - Cross-reference discovery effectiveness
# - Token consumption comparison
```

## Common Issues and Troubleshooting

### Issue 1: Large File Processing Errors
**Symptoms:** Memory errors, timeouts during ingestion
**Solution:**
```bash
# Reduce batch sizes and increase timeouts
--chunk-batch-size 25 \
--embedding-batch-size 10 \
--database-timeout 300 \
--memory-limit 4GB
```

### Issue 2: Poor Relationship Discovery
**Symptoms:** Few relationships found, low cross-reference coverage
**Solution:**
```bash
# Lower similarity thresholds and increase relationship depth
--similarity-threshold 0.65 \
--max-relationships-per-chunk 15 \
--cross-reference-depth 3 \
--include-weak-relationships True
```

### Issue 3: Inconsistent Chunk Boundaries
**Symptoms:** Code blocks split, incomplete sections
**Solution:**
```bash
# Use conservative chunking strategy
--strategy conservative \
--preserve-code-blocks True \
--preserve-tables True \
--minimum-chunk-size 50
```

### Issue 4: Database Performance Degradation
**Symptoms:** Slow queries after ingestion
**Solution:**
```sql
-- Rebuild indexes after bulk ingestion
OPTIMIZE TABLE megamind_chunks;
OPTIMIZE TABLE megamind_chunk_relationships;
ANALYZE TABLE megamind_chunks;
ANALYZE TABLE megamind_chunk_relationships;
```

## Post-Ingestion Optimization

### Analytics and Usage Tracking Setup
```bash
# Enable usage analytics for optimization
python tools/megamind_analytics_setup.py \
    --enable-access-tracking True \
    --enable-performance-monitoring True \
    --retention-period 90days

# Set up automated curation
python tools/megamind_curation_scheduler.py \
    --schedule-interval weekly \
    --cold-chunk-threshold 30days \
    --cleanup-threshold 90days
```

### Continuous Improvement
```bash
# Schedule periodic optimization
crontab -e

# Add entries:
# Daily: Update access statistics and hot chunk identification
0 2 * * * /path/to/megamind_daily_optimization.sh

# Weekly: Relationship strength recalculation  
0 3 * * 0 /path/to/megamind_weekly_analysis.sh

# Monthly: Cold chunk analysis and cleanup recommendations
0 4 1 * * /path/to/megamind_monthly_curation.sh
```

## Success Validation

### Ingestion Complete Checklist:
- [ ] All source markdown files processed without errors
- [ ] Chunk count matches expected range (±10% of estimate)
- [ ] Relationship density above 3 relationships per chunk average
- [ ] Sample queries return relevant, coherent results
- [ ] Performance benchmarks meet targets (<200ms search)
- [ ] MCP functions operational and tested
- [ ] Integration with CLAUDE.md workflow verified
- [ ] Token consumption reduction achieved (70-80% target)

### Ready for Production:
- [ ] Quality assessment report shows >95% content preservation
- [ ] Performance benchmarks consistently meet targets
- [ ] Error handling and recovery procedures tested
- [ ] Backup and disaster recovery validated
- [ ] User acceptance testing completed
- [ ] Documentation and operational procedures finalized