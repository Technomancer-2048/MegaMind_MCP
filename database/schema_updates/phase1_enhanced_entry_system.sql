-- Phase 1: Enhanced Multi-Embedding Entry System Schema Updates
-- This script adds new tables for the content-aware chunking system

-- Table for storing document structures
CREATE TABLE IF NOT EXISTS megamind_document_structures (
    document_id VARCHAR(36) PRIMARY KEY,
    source_path TEXT NOT NULL,
    content_type VARCHAR(50) NOT NULL, -- 'markdown', 'code', 'documentation', etc
    structure_metadata JSON, -- Contains parsed structure info
    total_elements INT DEFAULT 0,
    total_chunks INT DEFAULT 0,
    processing_status VARCHAR(50) DEFAULT 'pending', -- 'pending', 'processing', 'completed', 'failed'
    processed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    created_by VARCHAR(100),
    realm_id VARCHAR(50) DEFAULT 'PROJECT',
    INDEX idx_source_path (source_path(255)),
    INDEX idx_processing_status (processing_status),
    INDEX idx_realm_id (realm_id),
    INDEX idx_processed_date (processed_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Enhanced chunk metadata table
CREATE TABLE IF NOT EXISTS megamind_chunk_metadata (
    chunk_id VARCHAR(36) PRIMARY KEY,
    document_id VARCHAR(36),
    chunk_type VARCHAR(50) NOT NULL, -- 'heading', 'paragraph', 'code_block', etc
    chunk_sequence INT NOT NULL, -- Order within document
    line_start INT NOT NULL,
    line_end INT NOT NULL,
    token_count INT NOT NULL,
    quality_scores JSON, -- Multi-dimensional quality assessment
    semantic_boundaries JSON, -- Boundary information
    parent_chunk_id VARCHAR(36), -- For hierarchical chunks
    child_chunk_ids JSON, -- Array of child chunk IDs
    chunk_metadata JSON, -- Additional metadata
    semantic_coherence DECIMAL(3,2) DEFAULT 0.00,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES megamind_document_structures(document_id) ON DELETE CASCADE,
    FOREIGN KEY (parent_chunk_id) REFERENCES megamind_chunk_metadata(chunk_id) ON DELETE SET NULL,
    INDEX idx_document_id (document_id),
    INDEX idx_chunk_type (chunk_type),
    INDEX idx_chunk_sequence (chunk_sequence),
    INDEX idx_quality_score (semantic_coherence),
    INDEX idx_parent_chunk (parent_chunk_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Table for storing embeddings specific to the entry system
CREATE TABLE IF NOT EXISTS megamind_entry_embeddings (
    embedding_id VARCHAR(36) PRIMARY KEY,
    chunk_id VARCHAR(36) NOT NULL,
    embedding_model VARCHAR(100) NOT NULL, -- Model used for embedding
    embedding_vector JSON NOT NULL, -- Vector storage (consider using specialized vector DB in production)
    embedding_dimension INT NOT NULL, -- Vector dimension (384, 768, etc)
    processing_metadata JSON, -- Contains cleaning level, optimization details
    generation_time_ms INT DEFAULT 0,
    quality_score DECIMAL(3,2) DEFAULT 0.00,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (chunk_id) REFERENCES megamind_chunks(chunk_id) ON DELETE CASCADE,
    INDEX idx_chunk_id (chunk_id),
    INDEX idx_embedding_model (embedding_model),
    INDEX idx_quality_score (quality_score),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Table for tracking chunk relationships and cross-references
CREATE TABLE IF NOT EXISTS megamind_chunk_relationships (
    relationship_id VARCHAR(36) PRIMARY KEY,
    source_chunk_id VARCHAR(36) NOT NULL,
    target_chunk_id VARCHAR(36) NOT NULL,
    relationship_type VARCHAR(50) NOT NULL, -- 'semantic_similar', 'sequential', 'hierarchical', 'reference'
    confidence_score DECIMAL(3,2) DEFAULT 0.00,
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_chunk_id) REFERENCES megamind_chunks(chunk_id) ON DELETE CASCADE,
    FOREIGN KEY (target_chunk_id) REFERENCES megamind_chunks(chunk_id) ON DELETE CASCADE,
    UNIQUE KEY unique_relationship (source_chunk_id, target_chunk_id, relationship_type),
    INDEX idx_source_chunk (source_chunk_id),
    INDEX idx_target_chunk (target_chunk_id),
    INDEX idx_relationship_type (relationship_type),
    INDEX idx_confidence (confidence_score)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Table for content processing queue
CREATE TABLE IF NOT EXISTS megamind_processing_queue (
    queue_id VARCHAR(36) PRIMARY KEY,
    document_path TEXT NOT NULL,
    document_type VARCHAR(50),
    processing_type VARCHAR(50) NOT NULL, -- 'initial', 'update', 'reprocess'
    priority VARCHAR(20) DEFAULT 'medium', -- 'low', 'medium', 'high', 'critical'
    status VARCHAR(50) DEFAULT 'pending', -- 'pending', 'processing', 'completed', 'failed'
    retry_count INT DEFAULT 0,
    error_message TEXT,
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP NULL,
    completed_at TIMESTAMP NULL,
    INDEX idx_status (status),
    INDEX idx_priority (priority),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Table for quality assessment history
CREATE TABLE IF NOT EXISTS megamind_quality_assessments (
    assessment_id VARCHAR(36) PRIMARY KEY,
    chunk_id VARCHAR(36) NOT NULL,
    assessment_type VARCHAR(50) NOT NULL, -- 'manual', 'ai_automatic', 'user_feedback'
    quality_dimensions JSON NOT NULL, -- 8-dimensional quality scores
    overall_score DECIMAL(3,2) NOT NULL,
    assessor_id VARCHAR(100), -- Who/what performed the assessment
    assessment_metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (chunk_id) REFERENCES megamind_chunks(chunk_id) ON DELETE CASCADE,
    INDEX idx_chunk_id (chunk_id),
    INDEX idx_assessment_type (assessment_type),
    INDEX idx_overall_score (overall_score),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Add columns to existing megamind_chunks table for enhanced entry system
ALTER TABLE megamind_chunks 
ADD COLUMN IF NOT EXISTS document_id VARCHAR(36),
ADD COLUMN IF NOT EXISTS chunk_metadata JSON,
ADD COLUMN IF NOT EXISTS quality_assessment JSON,
ADD COLUMN IF NOT EXISTS embedding_status VARCHAR(50) DEFAULT 'pending',
ADD CONSTRAINT fk_document_id FOREIGN KEY (document_id) 
    REFERENCES megamind_document_structures(document_id) ON DELETE SET NULL,
ADD INDEX idx_embedding_status (embedding_status);

-- Create views for easier querying

-- View for document chunk hierarchy
CREATE OR REPLACE VIEW v_document_chunk_hierarchy AS
SELECT 
    ds.document_id,
    ds.source_path,
    ds.content_type,
    cm.chunk_id,
    cm.chunk_type,
    cm.chunk_sequence,
    cm.parent_chunk_id,
    cm.semantic_coherence,
    c.content,
    c.token_count
FROM megamind_document_structures ds
JOIN megamind_chunk_metadata cm ON ds.document_id = cm.document_id
JOIN megamind_chunks c ON cm.chunk_id = c.chunk_id
ORDER BY ds.document_id, cm.chunk_sequence;

-- View for chunk quality overview
CREATE OR REPLACE VIEW v_chunk_quality_overview AS
SELECT 
    c.chunk_id,
    c.source_document,
    cm.chunk_type,
    cm.semantic_coherence,
    qa.overall_score as latest_quality_score,
    qa.assessment_type,
    qa.created_at as assessment_date,
    ee.quality_score as embedding_quality,
    COUNT(cr.relationship_id) as relationship_count
FROM megamind_chunks c
LEFT JOIN megamind_chunk_metadata cm ON c.chunk_id = cm.chunk_id
LEFT JOIN megamind_quality_assessments qa ON c.chunk_id = qa.chunk_id
    AND qa.created_at = (
        SELECT MAX(created_at) 
        FROM megamind_quality_assessments 
        WHERE chunk_id = c.chunk_id
    )
LEFT JOIN megamind_entry_embeddings ee ON c.chunk_id = ee.chunk_id
LEFT JOIN megamind_chunk_relationships cr ON c.chunk_id = cr.source_chunk_id
GROUP BY c.chunk_id;

-- Stored procedures for common operations

DELIMITER //

-- Procedure to get document processing status
CREATE PROCEDURE IF NOT EXISTS sp_get_document_status(IN p_document_id VARCHAR(36))
BEGIN
    SELECT 
        ds.document_id,
        ds.source_path,
        ds.processing_status,
        ds.total_elements,
        ds.total_chunks,
        COUNT(DISTINCT cm.chunk_id) as processed_chunks,
        COUNT(DISTINCT ee.embedding_id) as generated_embeddings,
        AVG(cm.semantic_coherence) as avg_coherence,
        MAX(ds.last_updated) as last_updated
    FROM megamind_document_structures ds
    LEFT JOIN megamind_chunk_metadata cm ON ds.document_id = cm.document_id
    LEFT JOIN megamind_entry_embeddings ee ON cm.chunk_id = ee.chunk_id
    WHERE ds.document_id = p_document_id
    GROUP BY ds.document_id;
END//

-- Procedure to find similar chunks
CREATE PROCEDURE IF NOT EXISTS sp_find_similar_chunks(
    IN p_chunk_id VARCHAR(36),
    IN p_limit INT
)
BEGIN
    SELECT 
        cr.target_chunk_id as similar_chunk_id,
        cr.confidence_score,
        cr.relationship_type,
        c.content,
        c.source_document,
        cm.chunk_type
    FROM megamind_chunk_relationships cr
    JOIN megamind_chunks c ON cr.target_chunk_id = c.chunk_id
    JOIN megamind_chunk_metadata cm ON c.chunk_id = cm.chunk_id
    WHERE cr.source_chunk_id = p_chunk_id
        AND cr.relationship_type = 'semantic_similar'
    ORDER BY cr.confidence_score DESC
    LIMIT p_limit;
END//

DELIMITER ;

-- Insert initial configuration for enhanced entry system
INSERT INTO megamind_system_config (config_key, config_value, description) VALUES
('enhanced_entry_system.enabled', 'true', 'Enable enhanced multi-embedding entry system'),
('enhanced_entry_system.default_chunking_strategy', 'semantic_aware', 'Default chunking strategy'),
('enhanced_entry_system.max_chunk_tokens', '512', 'Maximum tokens per chunk'),
('enhanced_entry_system.min_chunk_tokens', '50', 'Minimum tokens per chunk'),
('enhanced_entry_system.quality_threshold', '0.7', 'Minimum quality score for chunks'),
('enhanced_entry_system.embedding_model', 'all-MiniLM-L6-v2', 'Default embedding model'),
('enhanced_entry_system.auto_process_documents', 'true', 'Automatically process new documents')
ON DUPLICATE KEY UPDATE config_value = VALUES(config_value);

-- Grant permissions (adjust as needed for your user)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON megamind_document_structures TO 'megamind_user'@'%';
-- GRANT SELECT, INSERT, UPDATE, DELETE ON megamind_chunk_metadata TO 'megamind_user'@'%';
-- GRANT SELECT, INSERT, UPDATE, DELETE ON megamind_entry_embeddings TO 'megamind_user'@'%';
-- GRANT SELECT, INSERT, UPDATE, DELETE ON megamind_chunk_relationships TO 'megamind_user'@'%';
-- GRANT SELECT, INSERT, UPDATE, DELETE ON megamind_processing_queue TO 'megamind_user'@'%';
-- GRANT SELECT, INSERT, UPDATE, DELETE ON megamind_quality_assessments TO 'megamind_user'@'%';
-- GRANT EXECUTE ON PROCEDURE sp_get_document_status TO 'megamind_user'@'%';
-- GRANT EXECUTE ON PROCEDURE sp_find_similar_chunks TO 'megamind_user'@'%';