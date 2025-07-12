-- Context Database System - Performance Indexes
-- Phase 1: Core Infrastructure Performance Optimization

-- Core performance indexes for megamind_chunks
CREATE INDEX idx_chunks_content_fulltext ON megamind_chunks (content) USING FULLTEXT;
CREATE INDEX idx_chunks_source_section ON megamind_chunks (source_document, section_path);
CREATE INDEX idx_chunks_access_count ON megamind_chunks (access_count DESC, last_accessed DESC);
CREATE INDEX idx_chunks_type_tags ON megamind_chunks (chunk_type, created_at);
CREATE INDEX idx_chunks_last_accessed ON megamind_chunks (last_accessed DESC);
CREATE INDEX idx_chunks_line_count ON megamind_chunks (line_count);

-- Relationship performance indexes  
CREATE INDEX idx_relationships_chunk_type ON megamind_chunk_relationships (chunk_id, relationship_type);
CREATE INDEX idx_relationships_related_type ON megamind_chunk_relationships (related_chunk_id, relationship_type);
CREATE INDEX idx_relationships_strength ON megamind_chunk_relationships (strength DESC, relationship_type);
CREATE INDEX idx_relationships_discovered_by ON megamind_chunk_relationships (discovered_by, created_at);

-- Tag-based search indexes
CREATE INDEX idx_tags_type_value ON megamind_chunk_tags (tag_type, tag_value);
CREATE INDEX idx_tags_chunk_lookup ON megamind_chunk_tags (chunk_id, tag_type);
CREATE INDEX idx_tags_value_confidence ON megamind_chunk_tags (tag_value, confidence DESC);
CREATE INDEX idx_tags_created_by ON megamind_chunk_tags (created_by, created_at);

-- Composite indexes for common query patterns
CREATE INDEX idx_chunks_type_access ON megamind_chunks (chunk_type, access_count DESC);
CREATE INDEX idx_chunks_source_type ON megamind_chunks (source_document, chunk_type);
CREATE INDEX idx_relationships_bidirectional ON megamind_chunk_relationships (chunk_id, related_chunk_id, strength DESC);