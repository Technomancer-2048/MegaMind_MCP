-- Context Database System - Core Tables
-- Phase 1: Core Infrastructure
-- Database: megamind_database (MySQL 8.0+)

-- Drop tables if they exist (for development)
DROP TABLE IF EXISTS megamind_chunk_tags;
DROP TABLE IF EXISTS megamind_chunk_relationships;
DROP TABLE IF EXISTS megamind_chunks;

-- Core chunks table
CREATE TABLE megamind_chunks (
    chunk_id VARCHAR(50) PRIMARY KEY,
    content TEXT NOT NULL,
    source_document VARCHAR(255) NOT NULL,
    section_path VARCHAR(500) NOT NULL,
    chunk_type ENUM('rule', 'function', 'section', 'example') NOT NULL,
    line_count INT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    access_count INT DEFAULT 0,
    
    -- Add embedding storage for future semantic search
    embedding JSON DEFAULT NULL,
    
    -- Metadata fields
    token_count INT DEFAULT NULL,
    complexity_score DECIMAL(3,2) DEFAULT NULL
);

-- Chunk relationships table
CREATE TABLE megamind_chunk_relationships (
    relationship_id INT PRIMARY KEY AUTO_INCREMENT,
    chunk_id VARCHAR(50) NOT NULL,
    related_chunk_id VARCHAR(50) NOT NULL,
    relationship_type ENUM('references', 'depends_on', 'contradicts', 'enhances', 'implements', 'supersedes') NOT NULL,
    strength DECIMAL(3,2) NOT NULL CHECK (strength >= 0.0 AND strength <= 1.0),
    discovered_by ENUM('manual', 'ai_analysis', 'usage_pattern', 'semantic_similarity') NOT NULL DEFAULT 'manual',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign key constraints
    FOREIGN KEY (chunk_id) REFERENCES megamind_chunks(chunk_id) ON DELETE CASCADE,
    FOREIGN KEY (related_chunk_id) REFERENCES megamind_chunks(chunk_id) ON DELETE CASCADE,
    
    -- Prevent duplicate relationships
    UNIQUE KEY unique_relationship (chunk_id, related_chunk_id, relationship_type),
    
    -- Prevent self-references
    CHECK (chunk_id != related_chunk_id)
);

-- Chunk tags table
CREATE TABLE megamind_chunk_tags (
    tag_id INT PRIMARY KEY AUTO_INCREMENT,
    chunk_id VARCHAR(50) NOT NULL,
    tag_type ENUM('subsystem', 'function_type', 'applies_to', 'language', 'difficulty', 'status') NOT NULL,
    tag_value VARCHAR(100) NOT NULL,
    confidence DECIMAL(3,2) DEFAULT 1.0 CHECK (confidence >= 0.0 AND confidence <= 1.0),
    created_by ENUM('manual', 'ai_analysis', 'automatic') NOT NULL DEFAULT 'manual',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign key constraint
    FOREIGN KEY (chunk_id) REFERENCES megamind_chunks(chunk_id) ON DELETE CASCADE,
    
    -- Prevent duplicate tags
    UNIQUE KEY unique_tag (chunk_id, tag_type, tag_value)
);