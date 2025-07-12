# Project Realms Implementation Plan
## Multi-Tenant Knowledge Organization for MegaMind Context Database

**Created:** 2025-07-12  
**Status:** Planning Phase  
**Target:** Organize knowledge by project-specific realms + global organizational realm  

## Executive Summary

This plan outlines the implementation of a **Realm-based knowledge organization system** for the MegaMind Context Database. The system will support project-specific knowledge isolation while maintaining a global organizational realm for cross-project rules, standards, and constraints. This architecture enables scalable multi-project knowledge management with proper data separation and controlled knowledge sharing.

## Business Requirements

### Core Use Cases

1. **Project Isolation:** Each project maintains its own knowledge base with complete data separation
2. **Global Standards:** Organization-wide rules, coding standards, and policies apply across all projects
3. **Selective Sharing:** Controlled knowledge inheritance from global realm to project realms
4. **Access Control:** Role-based access with realm-specific permissions
5. **Knowledge Evolution:** Project learnings can be promoted to global realm when appropriate

### Organizational Benefits

- **Scalability:** Support unlimited projects without knowledge contamination
- **Compliance:** Ensure project-specific regulatory requirements and constraints
- **Standardization:** Maintain consistent organizational practices across projects
- **Knowledge Reuse:** Share proven patterns and solutions organization-wide
- **Security:** Project-sensitive information remains isolated

## Current State Analysis

### Existing Schema Capabilities
- **Chunks Table:** Contains `source_document` and `section_path` for organization
- **Session Management:** Has `project_context` field in `megamind_session_metadata`
- **Relationships:** Cross-references between chunks (currently global)
- **Tags:** Categorization system with `subsystem` and other types

### Limitations for Multi-Project Support
- No realm/project isolation at database level
- All chunks exist in global namespace
- No inheritance mechanism for global → project knowledge
- Search and retrieval don't respect project boundaries
- Session context limited to string field

## Architecture Design

### Simplified Realm Hierarchy Structure

```
Organization
├── Global Realm (GLOBAL)
│   ├── Organizational Standards
│   ├── Coding Guidelines  
│   ├── Security Policies
│   ├── Architecture Patterns
│   └── Best Practices
│
└── Project Realm (Configured via Environment)
    ├── Auto-inherits from Global (read-only)
    ├── Project-specific Rules (read/write)
    ├── Domain Knowledge (read/write)
    └── Implementation Details (read/write)
```

### Environment-Based Configuration
Each MCP server instance is configured for a specific project realm via environment variables:
- **MEGAMIND_PROJECT_REALM:** Project realm identifier (e.g., "PROJ_ECOMMERCE")
- **MEGAMIND_PROJECT_NAME:** Human-readable project name
- **MEGAMIND_DEFAULT_TARGET:** Default realm for new chunks ("PROJECT" or "GLOBAL")

Sessions automatically have access to:
1. **Global Realm (Read-Only):** Organization-wide knowledge
2. **Project Realm (Read/Write):** Project-specific knowledge

### Realm Types and Characteristics

#### 1. Global Realm (`GLOBAL`)
- **Purpose:** Organization-wide knowledge that applies to all projects
- **Content:** Standards, policies, architectural patterns, security guidelines
- **Access:** Read-only for most users, write access for knowledge administrators
- **Inheritance:** Source realm for all project realms
- **Examples:** 
  - Code review standards
  - Security compliance requirements
  - Database design patterns
  - API design guidelines

#### 2. Project Realms (`PROJECT_*`)
- **Purpose:** Project-specific knowledge with automatic global inheritance
- **Content:** Domain rules, implementation details, project-specific patterns
- **Access:** Full read/write for all project team members via environment configuration
- **Inheritance:** Automatic read-only access to global realm knowledge
- **Default Target:** New chunks created here unless explicitly targeting global
- **Examples:**
  - Domain-specific business rules
  - Project architecture decisions
  - Custom implementation patterns
  - Client-specific requirements

### Session Access Pattern (Simplified)
Each MCP server instance provides automatic access to:
- **Global knowledge (inherited):** Always available, read-only
- **Project knowledge (direct):** Full read/write access
- **Creation target:** Project realm by default, optional global targeting

## Database Schema Implementation

### Fresh Database Schema with Realm Support

#### Realm Management Tables

```sql
-- Realms Definition Table
CREATE TABLE megamind_realms (
    realm_id VARCHAR(50) PRIMARY KEY,
    realm_name VARCHAR(100) NOT NULL,
    realm_type ENUM('global', 'project') NOT NULL,
    parent_realm_id VARCHAR(50),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    
    -- Inheritance settings (simplified for environment-based model)
    inherit_from_parent BOOLEAN DEFAULT TRUE,
    
    -- Access control
    default_access_level ENUM('read', 'write') DEFAULT 'read',
    
    FOREIGN KEY (parent_realm_id) REFERENCES megamind_realms(realm_id),
    UNIQUE KEY unique_realm_name (realm_name),
    INDEX idx_realm_type (realm_type, is_active),
    INDEX idx_realm_parent (parent_realm_id, realm_type)
);

-- Realm Access Control (simplified for environment-based configuration)
CREATE TABLE megamind_realm_permissions (
    permission_id INT PRIMARY KEY AUTO_INCREMENT,
    realm_id VARCHAR(50) NOT NULL,
    user_id VARCHAR(100) NOT NULL,
    role ENUM('viewer', 'contributor', 'admin') NOT NULL,
    granted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    granted_by VARCHAR(100),
    
    FOREIGN KEY (realm_id) REFERENCES megamind_realms(realm_id) ON DELETE CASCADE,
    UNIQUE KEY unique_user_realm (user_id, realm_id),
    INDEX idx_user_permissions (user_id, role),
    INDEX idx_realm_permissions (realm_id, role)
);

-- Simplified inheritance (project realms automatically inherit from global)
CREATE TABLE megamind_realm_inheritance (
    inheritance_id INT PRIMARY KEY AUTO_INCREMENT,
    child_realm_id VARCHAR(50) NOT NULL,
    parent_realm_id VARCHAR(50) NOT NULL,
    inheritance_type ENUM('full') NOT NULL DEFAULT 'full',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (child_realm_id) REFERENCES megamind_realms(realm_id) ON DELETE CASCADE,
    FOREIGN KEY (parent_realm_id) REFERENCES megamind_realms(realm_id) ON DELETE CASCADE,
    UNIQUE KEY unique_child_parent (child_realm_id, parent_realm_id),
    INDEX idx_child_inheritance (child_realm_id),
    INDEX idx_parent_inheritance (parent_realm_id)
);
```

#### Core Tables with Built-in Realm Support

```sql
-- Enhanced chunks table with realm support built-in
CREATE TABLE megamind_chunks (
    chunk_id VARCHAR(50) PRIMARY KEY,
    content TEXT NOT NULL,
    source_document VARCHAR(255) NOT NULL,
    section_path VARCHAR(500) NOT NULL,
    chunk_type ENUM('rule', 'function', 'section', 'example') NOT NULL,
    line_count INT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    access_count INT DEFAULT 1, -- Start with 1 (creation counts as first access)
    
    -- Realm support
    realm_id VARCHAR(50) NOT NULL DEFAULT 'GLOBAL',
    
    -- Add embedding storage for future semantic search
    embedding JSON DEFAULT NULL,
    
    -- Metadata fields
    token_count INT DEFAULT NULL,
    complexity_score DECIMAL(3,2) DEFAULT NULL,
    
    -- Foreign key constraints
    FOREIGN KEY (realm_id) REFERENCES megamind_realms(realm_id),
    
    -- Indexes for realm-aware operations
    INDEX idx_chunks_realm (realm_id, chunk_type, access_count DESC),
    INDEX idx_chunks_search_realm (realm_id, created_at DESC),
    INDEX idx_chunks_access (realm_id, access_count DESC, last_accessed DESC)
);

-- Enhanced relationships table with realm awareness
CREATE TABLE megamind_chunk_relationships (
    relationship_id INT PRIMARY KEY AUTO_INCREMENT,
    chunk_id VARCHAR(50) NOT NULL,
    related_chunk_id VARCHAR(50) NOT NULL,
    relationship_type ENUM('references', 'depends_on', 'contradicts', 'enhances', 'implements', 'supersedes') NOT NULL,
    strength DECIMAL(3,2) NOT NULL CHECK (strength >= 0.0 AND strength <= 1.0),
    discovered_by ENUM('manual', 'ai_analysis', 'usage_pattern', 'semantic_similarity') NOT NULL DEFAULT 'manual',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Realm tracking for cross-realm relationships
    source_realm_id VARCHAR(50),
    target_realm_id VARCHAR(50),
    
    -- Foreign key constraints
    FOREIGN KEY (chunk_id) REFERENCES megamind_chunks(chunk_id) ON DELETE CASCADE,
    FOREIGN KEY (related_chunk_id) REFERENCES megamind_chunks(chunk_id) ON DELETE CASCADE,
    FOREIGN KEY (source_realm_id) REFERENCES megamind_realms(realm_id),
    FOREIGN KEY (target_realm_id) REFERENCES megamind_realms(realm_id),
    
    -- Prevent duplicate relationships
    UNIQUE KEY unique_relationship (chunk_id, related_chunk_id, relationship_type),
    
    -- Prevent self-references
    CHECK (chunk_id != related_chunk_id),
    
    -- Realm-aware indexes
    INDEX idx_relationships_realm (source_realm_id, target_realm_id),
    INDEX idx_relationships_source_realm (source_realm_id, relationship_type),
    INDEX idx_relationships_target_realm (target_realm_id, relationship_type)
);

-- Enhanced tags table with realm awareness
CREATE TABLE megamind_chunk_tags (
    tag_id INT PRIMARY KEY AUTO_INCREMENT,
    chunk_id VARCHAR(50) NOT NULL,
    tag_type ENUM('subsystem', 'function_type', 'applies_to', 'language', 'difficulty', 'status', 'realm_scope') NOT NULL,
    tag_value VARCHAR(100) NOT NULL,
    confidence DECIMAL(3,2) DEFAULT 1.0 CHECK (confidence >= 0.0 AND confidence <= 1.0),
    created_by ENUM('manual', 'ai_analysis', 'automatic') NOT NULL DEFAULT 'manual',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Realm context for tag
    realm_id VARCHAR(50),
    
    -- Foreign key constraints
    FOREIGN KEY (chunk_id) REFERENCES megamind_chunks(chunk_id) ON DELETE CASCADE,
    FOREIGN KEY (realm_id) REFERENCES megamind_realms(realm_id),
    
    -- Prevent duplicate tags
    UNIQUE KEY unique_tag (chunk_id, tag_type, tag_value),
    
    -- Realm-aware indexes
    INDEX idx_tags_realm (realm_id, tag_type, tag_value),
    INDEX idx_tags_chunk_realm (chunk_id, realm_id)
);
```

#### Session Management with Realm Support

```sql
-- Enhanced session changes table with realm support
CREATE TABLE megamind_session_changes (
    change_id VARCHAR(50) PRIMARY KEY,
    session_id VARCHAR(50) NOT NULL,
    change_type ENUM('update', 'create', 'relate', 'tag') NOT NULL,
    chunk_id VARCHAR(50),  -- NULL for create operations until commit
    target_chunk_id VARCHAR(50),  -- For relationship operations
    change_data JSON NOT NULL,
    impact_score DECIMAL(3,2) DEFAULT 0.00,  -- 0.00 to 1.00 based on chunk access patterns
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status ENUM('pending', 'approved', 'rejected') DEFAULT 'pending',
    
    -- Realm context for the change
    realm_id VARCHAR(50) NOT NULL,
    
    -- Foreign key constraints
    FOREIGN KEY (chunk_id) REFERENCES megamind_chunks(chunk_id) ON DELETE SET NULL,
    FOREIGN KEY (realm_id) REFERENCES megamind_realms(realm_id),
    
    -- Indexes
    INDEX idx_session_changes_session (session_id, timestamp),
    INDEX idx_session_changes_status (status, impact_score DESC),
    INDEX idx_session_changes_chunk (chunk_id, change_type),
    INDEX idx_changes_realm (realm_id, status, impact_score DESC)
) ENGINE=InnoDB;

-- Enhanced session metadata with realm support
CREATE TABLE megamind_session_metadata (
    session_id VARCHAR(50) PRIMARY KEY,
    user_context VARCHAR(255),
    project_context VARCHAR(255),
    start_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    pending_changes_count INT DEFAULT 0,
    
    -- Realm context for session (environment-based)
    realm_id VARCHAR(50) NOT NULL DEFAULT 'GLOBAL',
    
    -- Foreign key constraints
    FOREIGN KEY (realm_id) REFERENCES megamind_realms(realm_id),
    
    -- Indexes
    INDEX idx_session_active (is_active, last_activity),
    INDEX idx_session_pending (pending_changes_count DESC, last_activity),
    INDEX idx_session_realm (realm_id, is_active)
) ENGINE=InnoDB;

-- Enhanced knowledge contributions with realm tracking
CREATE TABLE megamind_knowledge_contributions (
    contribution_id VARCHAR(50) PRIMARY KEY,
    session_id VARCHAR(50) NOT NULL,
    chunks_modified INT DEFAULT 0,
    chunks_created INT DEFAULT 0,
    relationships_added INT DEFAULT 0,
    tags_added INT DEFAULT 0,
    commit_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    rollback_available BOOLEAN DEFAULT TRUE,
    rollback_data JSON,  -- Stores original state for rollback
    
    -- Realm context for contribution
    realm_id VARCHAR(50) NOT NULL,
    
    -- Foreign key constraints
    FOREIGN KEY (realm_id) REFERENCES megamind_realms(realm_id),
    
    -- Indexes
    INDEX idx_contributions_session (session_id, commit_timestamp),
    INDEX idx_contributions_timestamp (commit_timestamp DESC),
    INDEX idx_contributions_rollback (rollback_available, commit_timestamp),
    INDEX idx_contributions_realm (realm_id, commit_timestamp DESC)
) ENGINE=InnoDB;
```

#### Complete Schema SQL File
The complete realm-aware database schema should be created as:
- `database/realm_system/01_realm_tables.sql` - Realm management tables
- `database/realm_system/02_enhanced_core_tables.sql` - Core tables with realm support  
- `database/realm_system/03_realm_session_tables.sql` - Session management with realm support
- `database/realm_system/04_indexes_and_views.sql` - Performance indexes and inheritance views
- `database/realm_system/05_initial_data.sql` - Default realms and sample data

This replaces the current `database/context_system/` files with a fresh, realm-aware schema design.

### Phase 2: Inheritance and Virtual Views

#### Inheritance Resolution System

```sql
-- Virtual view for realm-aware chunk access with inheritance
CREATE VIEW megamind_chunks_with_inheritance AS
SELECT 
    c.chunk_id,
    c.content,
    c.source_document,
    c.section_path,
    c.chunk_type,
    c.realm_id AS source_realm_id,
    c.access_count,
    c.last_accessed,
    c.created_at,
    CASE 
        WHEN c.realm_id = @current_realm THEN 'direct'
        ELSE 'inherited'
    END AS access_type,
    r.realm_name AS source_realm_name
FROM megamind_chunks c
JOIN megamind_realms r ON c.realm_id = r.realm_id
WHERE c.realm_id = @current_realm
   OR c.realm_id IN (
       SELECT parent_realm_id 
       FROM megamind_realm_inheritance 
       WHERE child_realm_id = @current_realm 
         AND inheritance_type IN ('full', 'selective')
   );

-- Function to get effective chunks for a realm
DELIMITER //
CREATE FUNCTION get_realm_chunks(p_realm_id VARCHAR(50))
RETURNS JSON
READS SQL DATA
DETERMINISTIC
BEGIN
    DECLARE result JSON;
    
    SELECT JSON_ARRAYAGG(
        JSON_OBJECT(
            'chunk_id', chunk_id,
            'content', content,
            'source_realm_id', source_realm_id,
            'access_type', access_type
        )
    ) INTO result
    FROM megamind_chunks_with_inheritance
    WHERE @current_realm = p_realm_id;
    
    RETURN result;
END //
DELIMITER ;
```

## MCP Server Implementation

### Environment-Based Configuration

```python
class RealmAwareMegaMindDatabase:
    """Enhanced database manager with environment-based realm configuration"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Environment-based realm configuration (no runtime switching)
        self.project_realm = os.getenv('MEGAMIND_PROJECT_REALM', 'PROJECT_DEFAULT')
        self.project_name = os.getenv('MEGAMIND_PROJECT_NAME', 'Default Project')
        self.default_target = os.getenv('MEGAMIND_DEFAULT_TARGET', 'PROJECT').upper()
        
        # Fixed realms for this instance
        self.global_realm = 'GLOBAL'
        self.accessible_realms = [self.global_realm, self.project_realm]
        
        logger.info(f"MCP Server configured for project realm: {self.project_realm}")
        logger.info(f"Default creation target: {self.default_target}")
        
        self._setup_connection_pool()
        self._validate_realm_configuration()
    
    def search_chunks_dual_realm(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search chunks across both global and project realms (simplified access pattern)"""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            # Always search both global and project realms
            realm_filter = "c.realm_id IN (%s, %s)"
            params = [self.global_realm, self.project_realm]
            
            # Multi-word search with realm awareness
            query_words = [word.strip().lower() for word in query.split() if word.strip()]
            where_conditions = []
            
            for word in query_words:
                like_pattern = f"%{word}%"
                where_conditions.append(
                    "(LOWER(c.content) LIKE %s OR LOWER(c.source_document) LIKE %s OR LOWER(c.section_path) LIKE %s)"
                )
                params.extend([like_pattern, like_pattern, like_pattern])
            
            word_filter = " AND ".join(where_conditions) if where_conditions else "1=1"
            
            search_query = f"""
            SELECT c.chunk_id, c.content, c.source_document, c.section_path, 
                   c.chunk_type, c.realm_id, c.access_count, c.last_accessed,
                   r.realm_name,
                   CASE WHEN c.realm_id = %s THEN 'project' ELSE 'global' END as realm_type
            FROM megamind_chunks c
            JOIN megamind_realms r ON c.realm_id = r.realm_id
            WHERE ({realm_filter}) AND ({word_filter})
            ORDER BY 
                CASE WHEN c.realm_id = %s THEN 0 ELSE 1 END,  -- Prioritize project realm
                c.access_count DESC
            LIMIT %s
            """
            
            params.extend([self.project_realm, self.project_realm, limit])
            cursor.execute(search_query, params)
            return cursor.fetchall()
            
        except Exception as e:
            logger.error(f"Realm-aware search failed: {e}")
            return []
        finally:
            if connection:
                connection.close()
    
    def create_chunk_with_target(self, content: str, source_document: str, 
                                 section_path: str, session_id: str,
                                 target_realm: str = None) -> str:
        """Create chunk with optional realm targeting (PROJECT default, GLOBAL optional)"""
        
        # Determine target realm
        if target_realm and target_realm.upper() == 'GLOBAL':
            target = self.global_realm
        elif target_realm and target_realm.upper() == 'PROJECT':
            target = self.project_realm
        else:
            # Use environment default
            target = self.project_realm if self.default_target == 'PROJECT' else self.global_realm
        
        # Validate write access (global requires special permission)
        if target == self.global_realm:
            if not self._validate_global_write_access():
                raise PermissionError("Global realm write access requires elevated permissions")
        
        logger.info(f"Creating chunk in {target} realm (requested: {target_realm}, default: {self.default_target})")
        
        # Enhanced create_chunk with realm support
        return self._create_chunk_with_realm(content, source_document, 
                                           section_path, session_id, target)
    
    def get_realm_hot_contexts(self, model_type: str = "sonnet", 
                               limit: int = 20, include_inherited: bool = True) -> List[Dict[str, Any]]:
        """Get hot contexts for current realm with inheritance support"""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            # Adjust thresholds by model type
            threshold = 2 if model_type.lower() == "opus" else 1
            
            realm_filter = "c.realm_id = %s"
            params = [self.current_realm]
            
            if include_inherited:
                realm_filter += """ OR c.realm_id IN (
                    SELECT ri.parent_realm_id 
                    FROM megamind_realm_inheritance ri 
                    WHERE ri.child_realm_id = %s 
                      AND ri.inheritance_type IN ('full', 'selective')
                )"""
                params.append(self.current_realm)
            
            hot_query = f"""
            SELECT c.chunk_id, c.content, c.source_document, c.section_path,
                   c.chunk_type, c.realm_id, c.access_count, c.last_accessed,
                   r.realm_name,
                   CASE WHEN c.realm_id = %s THEN 'direct' ELSE 'inherited' END as access_type
            FROM megamind_chunks c
            JOIN megamind_realms r ON c.realm_id = r.realm_id
            WHERE ({realm_filter}) AND c.access_count >= %s
            ORDER BY 
                CASE WHEN c.realm_id = %s THEN 0 ELSE 1 END,
                c.access_count DESC, c.last_accessed DESC
            LIMIT %s
            """
            
            params.extend([self.current_realm, threshold, self.current_realm, limit])
            cursor.execute(hot_query, params)
            return cursor.fetchall()
            
        except Exception as e:
            logger.error(f"Realm hot contexts failed: {e}")
            return []
        finally:
            if connection:
                connection.close()
```

### Simplified MCP Functions

```python
# Simplified MCP function set with environment-based realm configuration
{
    "name": "mcp__context_db__search_chunks",
    "description": "Search chunks across global and project realms (automatic dual-realm access)",
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "limit": {"type": "integer", "default": 10, "description": "Maximum results"}
        },
        "required": ["query"]
    }
},
{
    "name": "mcp__context_db__create_chunk",
    "description": "Create chunk with optional realm targeting (PROJECT default)",
    "inputSchema": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "Chunk content"},
            "source_document": {"type": "string", "description": "Source document"},
            "section_path": {"type": "string", "description": "Section path"},
            "session_id": {"type": "string", "description": "Session identifier"},
            "target_realm": {"type": "string", "enum": ["PROJECT", "GLOBAL"], "description": "Target realm (optional, defaults to environment setting)"}
        },
        "required": ["content", "source_document", "section_path", "session_id"]
    }
},
{
    "name": "mcp__context_db__get_realm_info",
    "description": "Get information about current realm configuration",
    "inputSchema": {
        "type": "object",
        "properties": {},
        "required": []
    }
},
{
    "name": "mcp__context_db__promote_to_global",
    "description": "Promote knowledge from project realm to global realm",
    "inputSchema": {
        "type": "object",
        "properties": {
            "chunk_ids": {"type": "array", "items": {"type": "string"}, "description": "Chunks to promote"},
            "source_realm_id": {"type": "string", "description": "Source realm"},
            "review_session_id": {"type": "string", "description": "Review session for approval workflow"}
        },
        "required": ["chunk_ids", "source_realm_id"]
    }
}
```

### MCP Configuration Examples

#### Environment Variables Setup
```bash
# E-commerce project MCP server configuration
export MEGAMIND_PROJECT_REALM="PROJ_ECOMMERCE"
export MEGAMIND_PROJECT_NAME="E-commerce Platform"
export MEGAMIND_DEFAULT_TARGET="PROJECT"

# Analytics project MCP server configuration  
export MEGAMIND_PROJECT_REALM="PROJ_ANALYTICS"
export MEGAMIND_PROJECT_NAME="Data Analytics Pipeline"
export MEGAMIND_DEFAULT_TARGET="PROJECT"

# Global administration MCP server configuration
export MEGAMIND_PROJECT_REALM="GLOBAL"
export MEGAMIND_PROJECT_NAME="Global Standards"
export MEGAMIND_DEFAULT_TARGET="GLOBAL"
```

#### MCP Server JSON Configuration
```json
{
  "mcpServers": {
    "megamind-ecommerce": {
      "command": "python",
      "args": ["mcp_server/megamind_database_server.py"],
      "env": {
        "MEGAMIND_PROJECT_REALM": "PROJ_ECOMMERCE",
        "MEGAMIND_PROJECT_NAME": "E-commerce Platform", 
        "MEGAMIND_DEFAULT_TARGET": "PROJECT",
        "DATABASE_HOST": "localhost",
        "DATABASE_PORT": "3306",
        "DATABASE_NAME": "megamind_database"
      }
    },
    "megamind-analytics": {
      "command": "python", 
      "args": ["mcp_server/megamind_database_server.py"],
      "env": {
        "MEGAMIND_PROJECT_REALM": "PROJ_ANALYTICS",
        "MEGAMIND_PROJECT_NAME": "Data Analytics Pipeline",
        "MEGAMIND_DEFAULT_TARGET": "PROJECT"
      }
    },
    "megamind-global": {
      "command": "python",
      "args": ["mcp_server/megamind_database_server.py"], 
      "env": {
        "MEGAMIND_PROJECT_REALM": "GLOBAL",
        "MEGAMIND_PROJECT_NAME": "Global Standards",
        "MEGAMIND_DEFAULT_TARGET": "GLOBAL"
      }
    }
  }
}
```

## Implementation Phases

### Phase 1: Fresh Database with Realm Infrastructure (Weeks 1-2)

#### Week 1: Complete Schema Creation
- [ ] Create fresh database schema with all realm-aware tables from scratch
- [ ] Implement realm management tables (`megamind_realms`, `megamind_realm_permissions`, `megamind_realm_inheritance`)
- [ ] Create enhanced core tables with built-in realm support (`megamind_chunks`, `megamind_chunk_relationships`, `megamind_chunk_tags`)
- [ ] Create realm-aware session management tables
- [ ] Add comprehensive indexes for realm-aware operations

#### Week 2: Environment-Based MCP Server
- [ ] Implement environment-based realm configuration in MCP server
- [ ] Add dual-realm search functionality (Global + Project)
- [ ] Create realm-aware chunk creation with optional targeting
- [ ] Implement basic realm information and status functions

### Phase 2: Inheritance and Search (Weeks 3-4)

#### Week 3: Inheritance System
- [ ] Implement inheritance resolution algorithms
- [ ] Create selective inheritance filtering
- [ ] Add inheritance conflict resolution
- [ ] Implement inheritance caching for performance

#### Week 4: Realm-Aware Search
- [ ] Enhance search functions with realm filtering
- [ ] Implement inheritance-aware search results
- [ ] Add realm priority in search ranking
- [ ] Create cross-realm relationship traversal

### Phase 3: Advanced Features (Weeks 5-6)

#### Week 5: Knowledge Promotion
- [ ] Implement chunk promotion workflows
- [ ] Create promotion approval processes
- [ ] Add promotional relationship tracking
- [ ] Implement promotion conflict detection

#### Week 6: Access Control and Security
- [ ] Implement role-based access control
- [ ] Add audit logging for realm operations
- [ ] Create realm-specific session management
- [ ] Implement secure realm isolation

### Phase 4: Production Deployment and Optimization (Weeks 7-8)

#### Week 7: Initial Deployment
- [ ] Deploy fresh database with realm schema
- [ ] Create initial global realm with organizational standards
- [ ] Set up project realms for initial projects
- [ ] Configure environment-based MCP server instances

#### Week 8: Monitoring and Analytics
- [ ] Optimize realm-aware queries for production load
- [ ] Implement realm usage analytics and reporting
- [ ] Add realm health monitoring and alerting
- [ ] Create realm performance dashboards

## Database Initialization Strategy

### Fresh Database Creation with Default Realms

#### Initial Realm Setup
```sql
-- Create default realms during initial database setup
INSERT INTO megamind_realms (realm_id, realm_name, realm_type, parent_realm_id, description, is_active) VALUES
('GLOBAL', 'Global Organization Standards', 'global', NULL, 'Organization-wide rules, standards, and best practices', TRUE);

-- Create inheritance for any project realms (done automatically when projects are created)
-- Example project realm creation (done via environment configuration)
INSERT INTO megamind_realms (realm_id, realm_name, realm_type, parent_realm_id, description, is_active) VALUES
('PROJ_ECOMMERCE', 'E-commerce Platform', 'project', 'GLOBAL', 'Customer-facing e-commerce application', TRUE),
('PROJ_ANALYTICS', 'Data Analytics Pipeline', 'project', 'GLOBAL', 'Internal analytics and reporting system', TRUE),
('PROJ_MOBILE', 'Mobile Application', 'project', 'GLOBAL', 'iOS and Android mobile applications', TRUE);

-- Set up automatic inheritance relationships
INSERT INTO megamind_realm_inheritance (child_realm_id, parent_realm_id, inheritance_type) VALUES
('PROJ_ECOMMERCE', 'GLOBAL', 'full'),
('PROJ_ANALYTICS', 'GLOBAL', 'full'),
('PROJ_MOBILE', 'GLOBAL', 'full');
```

#### Sample Data with Realm Assignment
```sql
-- Insert sample global data
INSERT INTO megamind_chunks (chunk_id, content, source_document, section_path, chunk_type, line_count, token_count, access_count, realm_id) VALUES
('global_security_001', 'All API endpoints must implement authentication using OAuth 2.0 or API keys with rate limiting', 'security_standards.md', '/api/authentication', 'rule', 2, 25, 1, 'GLOBAL'),
('global_database_001', 'Always use transactions for multi-table operations to ensure data consistency', 'database_standards.md', '/database/transactions', 'rule', 1, 18, 1, 'GLOBAL'),
('global_error_001', 'Implement structured error responses with error codes, messages, and correlation IDs', 'error_handling_standards.md', '/errors/structure', 'rule', 2, 22, 1, 'GLOBAL');

-- Insert sample project data
INSERT INTO megamind_chunks (chunk_id, content, source_document, section_path, chunk_type, line_count, token_count, access_count, realm_id) VALUES
('ecom_cart_001', 'Shopping cart items expire after 24 hours of inactivity, send reminder emails at 2h and 12h marks', 'ecommerce_business_rules.md', '/cart/expiration', 'rule', 2, 28, 1, 'PROJ_ECOMMERCE'),
('analytics_etl_001', 'ETL jobs must process data in 4-hour windows with overlap handling for late-arriving data', 'analytics_pipeline_rules.md', '/etl/scheduling', 'rule', 2, 24, 1, 'PROJ_ANALYTICS'),
('mobile_offline_001', 'Cache critical user data locally with automatic sync when connection is restored', 'mobile_patterns.md', '/offline/sync', 'section', 2, 20, 1, 'PROJ_MOBILE');
```

### Environment-Based Deployment
Since each MCP server instance is configured for a specific project realm via environment variables, the migration strategy is simplified:

1. **Fresh Database Creation:** Build new database with realm-aware schema from the start
2. **Default Global Realm:** Always created with essential organizational standards
3. **Project Realm Creation:** Created as needed when new MCP server instances are deployed
4. **No Data Migration:** Fresh start with proper realm organization from day one

## Usage Patterns and Examples

### Typical Organization Setup

#### Initial Realm Structure
```sql
-- Create global realm for organization standards
INSERT INTO megamind_realms VALUES 
('ORG_GLOBAL', 'Acme Corp Standards', 'global', NULL, 'Organization-wide standards and policies', NOW(), TRUE, TRUE, 'full', 'read');

-- Create project realms
INSERT INTO megamind_realms VALUES 
('PROJ_ECOMMERCE', 'E-commerce Platform', 'project', 'ORG_GLOBAL', 'Customer-facing e-commerce application', NOW(), TRUE, TRUE, 'full', 'write'),
('PROJ_ANALYTICS', 'Data Analytics Pipeline', 'project', 'ORG_GLOBAL', 'Internal analytics and reporting system', NOW(), TRUE, TRUE, 'selective', 'write'),
('PROJ_MOBILE', 'Mobile Application', 'project', 'ORG_GLOBAL', 'iOS and Android mobile applications', NOW(), TRUE, TRUE, 'full', 'write');

-- Set up inheritance relationships
INSERT INTO megamind_realm_inheritance VALUES 
(1, 'PROJ_ECOMMERCE', 'ORG_GLOBAL', 'full', NULL, 100, NOW()),
(2, 'PROJ_ANALYTICS', 'ORG_GLOBAL', 'selective', '{"include_tags": ["security", "database"], "exclude_tags": ["frontend"]}', 100, NOW()),
(3, 'PROJ_MOBILE', 'ORG_GLOBAL', 'full', NULL, 100, NOW());
```

#### Knowledge Content Examples

**Global Realm Content:**
```
- Security policies (authentication, encryption standards)
- Code review guidelines 
- Database design patterns
- API design standards
- Testing frameworks and practices
- Deployment procedures
- Incident response protocols
```

**Project Realm Content:**
```
E-commerce Project:
- Payment processing rules
- Customer data handling procedures
- Inventory management patterns
- Shopping cart business logic

Analytics Project:
- Data pipeline architectures
- Privacy compliance requirements
- ETL best practices
- Reporting template standards

Mobile Project:
- UI/UX design patterns
- Platform-specific guidelines
- App store submission procedures
- Push notification strategies
```

### Real-World Usage Scenarios

#### Scenario 1: New Developer Onboarding (E-commerce Project)
```python
# Developer connects to e-commerce MCP server (environment pre-configured)
# MEGAMIND_PROJECT_REALM="PROJ_ECOMMERCE" 
# MEGAMIND_DEFAULT_TARGET="PROJECT"

# Search for authentication patterns (automatic dual-realm search)
search_results = search_chunks("authentication OAuth setup")
# Returns (automatically searches both realms):
# - Global: OAuth 2.0 implementation standards (global realm)
# - Global: Security token management (global realm)
# - Project: E-commerce specific auth flows (project realm) 
# - Project: Customer session management (project realm)
```

#### Scenario 2: Creating Project-Specific Knowledge
```python
# Create project-specific business rule (uses PROJECT default from environment)
create_chunk(
    content="Shopping cart abandonment: automatically send reminder email after 2 hours, max 3 reminders per week",
    source_document="ecommerce_business_rules.md",
    section_path="/cart/abandonment_handling",
    session_id="ecommerce_session_456"
    # target_realm not specified → uses PROJECT default
)
```

#### Scenario 3: Contributing to Global Standards
```python
# Create global standard (explicit targeting required)
create_chunk(
    content="All API endpoints must include rate limiting headers: X-RateLimit-Limit, X-RateLimit-Remaining",
    source_document="api_standards.md", 
    section_path="/api/rate_limiting",
    session_id="standards_session_789",
    target_realm="GLOBAL"  # Explicit global targeting
)
```

#### Scenario 4: Knowledge Promotion Workflow
```python
# Project team discovers valuable pattern in project realm
create_chunk(
    content="Efficient bulk data loading pattern for PostgreSQL with conflict resolution using UPSERT",
    source_document="analytics_db_patterns.md",
    section_path="/database/bulk_loading", 
    session_id="analytics_session_123"
    # Created in PROJECT realm by default
)

# After validation and approval, promote to global realm
promote_to_global(
    chunk_ids=["chunk_analytics_bulk_001"],
    source_realm_id="PROJ_ANALYTICS",
    review_session_id="promotion_review_456"
)
```

## Security and Access Control

### Permission Levels

#### Realm-Level Permissions
- **Viewer:** Read-only access to chunks and relationships
- **Contributor:** Create and modify chunks, cannot manage realm settings
- **Admin:** Full realm management, user permissions, inheritance configuration

#### Cross-Realm Permissions
- **Global Admin:** Can manage all realms and promote knowledge
- **Project Admin:** Can manage specific project realm and child realms
- **Standard User:** Access based on explicit realm permissions

### Security Features

#### Data Isolation
- Chunks are strictly isolated by realm unless explicitly inherited
- Relationships cannot cross realms without proper inheritance setup
- Search results are automatically filtered by realm context

#### Audit Trail
- All realm operations logged with user, timestamp, and operation details
- Knowledge promotion tracked with approval workflows
- Access pattern monitoring for security analysis

#### Inheritance Security
- Selective inheritance with tag-based filtering
- Override protection for critical global standards
- Inheritance dependency validation

## Performance Considerations

### Database Optimization

#### Indexing Strategy
```sql
-- Realm-aware performance indexes
CREATE INDEX idx_chunks_realm_performance ON megamind_chunks (realm_id, access_count DESC, chunk_type);
CREATE INDEX idx_chunks_search_realm ON megamind_chunks (realm_id, created_at DESC);
CREATE INDEX idx_inheritance_lookup ON megamind_realm_inheritance (child_realm_id, inheritance_type, priority);
CREATE INDEX idx_realm_permissions_user ON megamind_realm_permissions (user_id, realm_id, role);
```

#### Query Optimization
- Materialized views for frequently accessed inheritance chains
- Query result caching with realm-specific cache keys
- Optimized realm context switching with session state management

#### Scaling Considerations
- Horizontal partitioning by realm for large installations
- Read replicas with realm-aware routing
- Caching layer with realm-specific invalidation

### Memory and Storage

#### Storage Patterns
- Realm-based data archival policies
- Inheritance cache precomputation
- Compressed storage for inherited chunks

#### Memory Management
- Realm context caching in application layer
- Permission cache with TTL
- Search result caching per realm

## Monitoring and Analytics

### Realm Usage Metrics

#### Key Performance Indicators
- Chunks per realm and growth trends
- Cross-realm inheritance effectiveness
- Knowledge promotion success rates
- User engagement by realm
- Search performance by realm complexity

#### Health Monitoring
- Inheritance chain depth and complexity
- Realm isolation integrity validation
- Permission consistency checks
- Performance impact of realm overhead

### Operational Dashboards

#### Realm Management Dashboard
- Realm hierarchy visualization
- Permission matrix overview
- Inheritance flow diagrams
- Knowledge promotion pipeline status

#### Usage Analytics Dashboard
- Popular chunks by realm
- Search pattern analysis
- Knowledge gap identification
- Promotion candidate recommendations

## Testing Strategy

### Unit Testing

#### Realm Operations
- Realm creation and configuration
- Permission validation and enforcement
- Inheritance resolution accuracy
- Data isolation verification

#### Search and Retrieval
- Realm-filtered search results
- Inheritance-aware chunk access
- Cross-realm relationship traversal
- Performance under various realm configurations

### Integration Testing

#### End-to-End Workflows
- Complete project onboarding flow
- Knowledge creation and promotion pipeline
- Multi-realm session management
- Permission inheritance validation

#### Performance Testing
- Large-scale realm hierarchies
- Complex inheritance scenarios
- Concurrent multi-realm operations
- Cache effectiveness under load

### User Acceptance Testing

#### Real-World Scenarios
- Developer workflow simulation
- Knowledge sharing effectiveness
- Security boundary validation
- Migration from existing systems

## Future Enhancements

### Advanced Features

#### Realm Templates
- Preconfigured realm structures for common project types
- Template-based realm inheritance setup
- Standard permission models by project type

#### Dynamic Inheritance
- Rule-based inheritance with conditional logic
- Time-based inheritance expiration
- Context-aware inheritance activation

#### Knowledge Lifecycle Management
- Automated archival of stale project knowledge
- Version control for realm configurations
- Automated promotion suggestion based on usage patterns

### Integration Capabilities

#### External System Integration
- LDAP/Active Directory integration for user management
- Project management tool integration for realm creation
- CI/CD pipeline integration for automated knowledge updates

#### API Extensions
- GraphQL API for complex realm queries
- Webhook support for realm events
- Real-time collaboration features

### AI-Powered Features

#### Intelligent Inheritance
- AI-suggested inheritance configurations
- Automatic conflict resolution
- Smart promotion recommendations

#### Content Analysis
- Automatic realm classification for new content
- Duplicate detection across realms
- Knowledge gap analysis and recommendations

## Conclusion

The Project Realms implementation provides a robust foundation for multi-project knowledge organization while maintaining the flexibility and power of the existing MegaMind Context Database. The phased approach ensures minimal disruption to existing workflows while progressively introducing powerful new capabilities.

Key benefits include:
- **Scalable Architecture:** Support for unlimited projects with proper isolation
- **Knowledge Reuse:** Efficient sharing of organizational standards and patterns
- **Security:** Project-level data isolation with controlled sharing
- **Flexibility:** Configurable inheritance and permission models
- **Performance:** Optimized queries and caching for realm-aware operations

The system will transform how organizations manage knowledge across multiple projects, enabling both standardization and specialization while maintaining security and performance requirements.

Implementation should begin with Phase 1 (Core Realm Infrastructure) to establish the foundation, followed by gradual rollout of advanced features based on organizational priorities and user feedback.