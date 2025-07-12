-- Realm Management Tables for MegaMind Context Database
-- Phase 1: Core Realm Infrastructure
-- Database: megamind_database (MySQL 8.0+)

-- Drop tables if they exist (for development)
DROP TABLE IF EXISTS megamind_realm_inheritance;
DROP TABLE IF EXISTS megamind_realms;

-- Core realms table
CREATE TABLE megamind_realms (
    realm_id VARCHAR(50) PRIMARY KEY,
    realm_name VARCHAR(255) NOT NULL,
    realm_type ENUM('global', 'project', 'team', 'personal') NOT NULL,
    parent_realm_id VARCHAR(50) DEFAULT NULL,
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- Configuration settings stored as JSON
    realm_config JSON DEFAULT NULL,
    
    -- Metadata
    created_by VARCHAR(100) DEFAULT 'system',
    access_level ENUM('read_only', 'read_write', 'admin') DEFAULT 'read_write',
    
    -- Self-referencing foreign key for hierarchical realms
    FOREIGN KEY (parent_realm_id) REFERENCES megamind_realms(realm_id) ON DELETE SET NULL,
    
    -- Indexes
    INDEX idx_realms_type (realm_type, is_active),
    INDEX idx_realms_parent (parent_realm_id),
    INDEX idx_realms_active (is_active, created_at DESC)
) ENGINE=InnoDB;

-- Realm inheritance relationships table
CREATE TABLE megamind_realm_inheritance (
    inheritance_id INT PRIMARY KEY AUTO_INCREMENT,
    child_realm_id VARCHAR(50) NOT NULL,
    parent_realm_id VARCHAR(50) NOT NULL,
    inheritance_type ENUM('full', 'selective', 'read_only') NOT NULL DEFAULT 'full',
    priority_order INT DEFAULT 1,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Inheritance configuration
    inheritance_config JSON DEFAULT NULL,
    
    -- Foreign key constraints
    FOREIGN KEY (child_realm_id) REFERENCES megamind_realms(realm_id) ON DELETE CASCADE,
    FOREIGN KEY (parent_realm_id) REFERENCES megamind_realms(realm_id) ON DELETE CASCADE,
    
    -- Prevent circular inheritance and duplicate relationships
    UNIQUE KEY unique_inheritance (child_realm_id, parent_realm_id),
    CHECK (child_realm_id != parent_realm_id),
    
    -- Indexes
    INDEX idx_inheritance_child (child_realm_id, is_active),
    INDEX idx_inheritance_parent (parent_realm_id, priority_order),
    INDEX idx_inheritance_active (is_active, created_at)
) ENGINE=InnoDB;