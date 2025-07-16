# Phase 1: Enhanced Multi-Embedding Entry System - Foundation Components

## Overview

This directory contains the foundation libraries for the Enhanced Multi-Embedding Entry System, implementing intelligent content-aware chunking and quality assessment for the MegaMind Context Database.

## Components

### 1. Content Analyzer (`content_analyzer.py`)

Analyzes document structure and identifies semantic boundaries.

**Key Features:**
- Markdown structure parsing
- Content type detection (markdown, code, documentation, etc.)
- Semantic boundary identification
- Element extraction (headings, paragraphs, code blocks, lists)

**Usage:**
```python
from content_analyzer import ContentAnalyzer

analyzer = ContentAnalyzer()
structure = analyzer.analyze_document_structure(content)
print(f"Found {len(structure.elements)} elements")
```

### 2. Intelligent Chunker (`intelligent_chunker.py`)

Creates optimally-sized chunks that respect semantic boundaries.

**Key Features:**
- Multiple chunking strategies (semantic-aware, markdown-structure, hybrid)
- Token-aware chunk sizing
- Code block preservation
- Chunk optimization (merging/splitting)
- Quality scoring

**Usage:**
```python
from intelligent_chunker import IntelligentChunker, ChunkingConfig, ChunkingStrategy

config = ChunkingConfig(
    max_tokens=512,
    strategy=ChunkingStrategy.SEMANTIC_AWARE
)
chunker = IntelligentChunker(config)
chunks = chunker.chunk_document(document_structure)
```

### 3. Embedding Optimizer (`embedding_optimizer.py`)

Optimizes text for embedding generation while preserving semantic meaning.

**Key Features:**
- Multiple cleaning levels (minimal, standard, aggressive)
- Format stripping with semantic preservation
- Model-specific optimization
- Batch processing support
- Similarity calculation

**Usage:**
```python
from embedding_optimizer import EmbeddingOptimizer, EmbeddingModel

optimizer = EmbeddingOptimizer(model=EmbeddingModel.ALL_MINILM_L6_V2)
optimized_text = optimizer.prepare_text_for_embedding(chunk)
embeddings = optimizer.generate_optimized_embeddings(chunks)
```

### 4. AI Quality Assessor (`ai_quality_assessor.py`)

Performs 8-dimensional quality assessment on chunks.

**Key Features:**
- Multi-dimensional quality scoring
- Quality issue identification
- Improvement suggestions
- Technical accuracy validation
- Coherence measurement

### 5. Sentence Splitter (`sentence_splitter.py`)

Provides robust sentence splitting that handles edge cases.

**Key Features:**
- Handles abbreviations (Dr., Mr., U.S.A., etc.)
- Preserves decimal numbers and URLs
- Intelligent merging of incorrectly split sentences
- Context-aware splitting decisions

**Usage:**
```python
from sentence_splitter import SentenceSplitter

splitter = SentenceSplitter()
sentences = splitter.split_sentences("Dr. Smith went to the U.S.A. He arrived at 3.14 p.m.")
print(sentences)  # ['Dr. Smith went to the U.S.A.', 'He arrived at 3.14 p.m.']
```

**Quality Dimensions:**
1. **Readability** (15%) - Text clarity and comprehension
2. **Technical Accuracy** (25%) - Correctness and precision
3. **Completeness** (20%) - Sufficient detail and coverage
4. **Relevance** (15%) - Context and topic alignment
5. **Freshness** (10%) - Recency and currency
6. **Coherence** (10%) - Logical flow and consistency
7. **Uniqueness** (3%) - Non-redundant information
8. **Authority** (2%) - Source credibility

**Usage:**
```python
from ai_quality_assessor import AIQualityAssessor

assessor = AIQualityAssessor()
quality_score = assessor.assess_chunk_quality(chunk, context=surrounding_chunks)
print(f"Overall quality: {quality_score.overall_score:.2f}")
```

## Database Schema

The Phase 1 implementation includes new database tables:

- `megamind_document_structures` - Document metadata and structure
- `megamind_chunk_metadata` - Enhanced chunk information
- `megamind_entry_embeddings` - Embedding storage
- `megamind_chunk_relationships` - Cross-references
- `megamind_processing_queue` - Processing pipeline
- `megamind_quality_assessments` - Quality history

## Configuration

Key configuration options:

```python
# Chunking Configuration
max_tokens = 512  # Maximum tokens per chunk
min_tokens = 50   # Minimum tokens per chunk
chunking_strategy = "semantic_aware"  # or "markdown_structure", "hybrid"

# Embedding Configuration
embedding_model = "all-MiniLM-L6-v2"  # 384 dimensions, 512 tokens
cleaning_level = "standard"  # or "minimal", "aggressive"

# Quality Configuration
quality_threshold = 0.7  # Minimum acceptable quality score
```

## Testing

Run the test suite:

```bash
python tests/test_phase1_foundation.py
```

## Integration with MCP Server

The Phase 1 components are designed to integrate with the existing MegaMind MCP server. New functions will be added in Phase 2:

```python
# Planned MCP functions
"mcp__megamind__content_analyze_document"
"mcp__megamind__content_create_chunks"
"mcp__megamind__content_assess_quality"
"mcp__megamind__content_optimize_embeddings"
```

## Enhanced Features

### Table Preservation
The system now includes enhanced table detection and preservation:
- **Markdown pipe tables**: `| col1 | col2 |`
- **Grid tables** (RST style): `+------+------+`
- **Simple space-aligned tables**: Column headers with separator lines
- Tables are kept intact during chunking and never split inappropriately

### Sentence Boundary Detection
Enhanced sentence splitting that handles:
- Abbreviations (Dr., U.S.A., etc.)
- Decimal numbers (3.14, 10.99)
- URLs and email patterns
- False positive detection and correction

### Intelligent Chunk Optimization
- Table-aware chunk merging and splitting
- Enhanced paragraph splitting using sentence boundaries
- Clause-based splitting for long sentences
- Context-preserving optimizations

## Next Steps (Phase 2)

- Implement session tracking system
- Create knowledge management system
- Add MCP function integration
- Deploy to production container

## Performance Metrics

Target performance:
- Document analysis: <1s per 1000 tokens
- Chunking: <2s per document
- Quality assessment: <100ms per chunk
- Embedding optimization: <50ms per chunk

## Dependencies

- Python 3.8+
- Standard library only (no external dependencies for Phase 1)
- Optional: numpy for similarity calculations
- Optional: embedding service for actual vector generation