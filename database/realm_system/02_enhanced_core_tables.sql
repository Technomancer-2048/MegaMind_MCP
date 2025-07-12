-- Enhanced Core Tables with Realm Support
-- Phase 1: Realm-Aware Core Infrastructure
-- Database: megamind_database (MySQL 8.0+)

-- Drop tables if they exist (for development)
DROP TABLE IF EXISTS megamind_chunk_tags;
DROP TABLE IF EXISTS megamind_chunk_relationships;
DROP TABLE IF EXISTS megamind_chunks;

-- Enhanced chunks table with realm support
CREATE TABLE megamind_chunks (
    chunk_id VARCHAR(50) PRIMARY KEY,
    content TEXT NOT NULL,
    source_document VARCHAR(255) NOT NULL,
    section_path VARCHAR(500) NOT NULL,
    chunk_type ENUM('rule', 'function', 'section', 'example') NOT NULL,
    line_count INT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    access_count INT DEFAULT 1,  -- Creation counts as first access
    
    -- Realm assignment (core feature)
    realm_id VARCHAR(50) NOT NULL DEFAULT 'GLOBAL',
    
    -- Embedding storage for semantic search
    embedding JSON DEFAULT NULL,
    
    -- Metadata fields
    token_count INT DEFAULT NULL,
    complexity_score DECIMAL(3,2) DEFAULT NULL,
    
    -- Content hash for deduplication
    content_hash VARCHAR(64) DEFAULT NULL,
    
    -- Foreign key constraints
    FOREIGN KEY (realm_id) REFERENCES megamind_realms(realm_id) ON DELETE RESTRICT,
    
    -- Indexes for realm-aware operations
    INDEX idx_chunks_realm (realm_id, last_accessed DESC),
    INDEX idx_chunks_realm_type (realm_id, chunk_type),
    INDEX idx_chunks_realm_access (realm_id, access_count DESC),
    INDEX idx_chunks_source (source_document, section_path),
    INDEX idx_chunks_content_hash (content_hash),
    INDEX idx_chunks_embedding_exists (realm_id, (JSON_VALID(embedding)))
) ENGINE=InnoDB;

-- Enhanced chunk relationships table with realm context
CREATE TABLE megamind_chunk_relationships (
    relationship_id INT PRIMARY KEY AUTO_INCREMENT,
    chunk_id VARCHAR(50) NOT NULL,
    related_chunk_id VARCHAR(50) NOT NULL,
    relationship_type ENUM('references', 'depends_on', 'contradicts', 'enhances', 'implements', 'supersedes') NOT NULL,
    strength DECIMAL(3,2) NOT NULL CHECK (strength >= 0.0 AND strength <= 1.0),
    discovered_by ENUM('manual', 'ai_analysis', 'usage_pattern', 'semantic_similarity') NOT NULL DEFAULT 'manual',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Realm context for relationship
    source_realm_id VARCHAR(50) NOT NULL,
    target_realm_id VARCHAR(50) NOT NULL,
    is_cross_realm BOOLEAN GENERATED ALWAYS AS (source_realm_id != target_realm_id) STORED,
    
    -- Foreign key constraints
    FOREIGN KEY (chunk_id) REFERENCES megamind_chunks(chunk_id) ON DELETE CASCADE,
    FOREIGN KEY (related_chunk_id) REFERENCES megamind_chunks(chunk_id) ON DELETE CASCADE,
    FOREIGN KEY (source_realm_id) REFERENCES megamind_realms(realm_id) ON DELETE RESTRICT,
    FOREIGN KEY (target_realm_id) REFERENCES megamind_realms(realm_id) ON DELETE RESTRICT,
    
    -- Prevent duplicate relationships
    UNIQUE KEY unique_relationship (chunk_id, related_chunk_id, relationship_type),
    
    -- Prevent self-references
    CHECK (chunk_id != related_chunk_id),
    
    -- Indexes for realm-aware relationship queries
    INDEX idx_relationships_chunk_realm (chunk_id, source_realm_id),
    INDEX idx_relationships_related_realm (related_chunk_id, target_realm_id),
    INDEX idx_relationships_cross_realm (is_cross_realm, strength DESC),
    INDEX idx_relationships_type_strength (relationship_type, strength DESC)
) ENGINE=InnoDB;

-- Enhanced chunk tags table with realm context
CREATE TABLE megamind_chunk_tags (
    tag_id INT PRIMARY KEY AUTO_INCREMENT,
    chunk_id VARCHAR(50) NOT NULL,
    tag_type ENUM('subsystem', 'function_type', 'applies_to', 'language', 'difficulty', 'status', 'realm_scope') NOT NULL,
    tag_value VARCHAR(100) NOT NULL,
    confidence DECIMAL(3,2) DEFAULT 1.0 CHECK (confidence >= 0.0 AND confidence <= 1.0),
    created_by ENUM('manual', 'ai_analysis', 'automatic') NOT NULL DEFAULT 'manual',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Realm context for tag
    realm_id VARCHAR(50) NOT NULL,
    
    -- Foreign key constraints
    FOREIGN KEY (chunk_id) REFERENCES megamind_chunks(chunk_id) ON DELETE CASCADE,
    FOREIGN KEY (realm_id) REFERENCES megamind_realms(realm_id) ON DELETE RESTRICT,
    
    -- Prevent duplicate tags
    UNIQUE KEY unique_tag (chunk_id, tag_type, tag_value),
    
    -- Indexes for realm-aware tag queries
    INDEX idx_tags_chunk_realm (chunk_id, realm_id),
    INDEX idx_tags_type_value (tag_type, tag_value),
    INDEX idx_tags_realm_type (realm_id, tag_type)
) ENGINE=InnoDB;