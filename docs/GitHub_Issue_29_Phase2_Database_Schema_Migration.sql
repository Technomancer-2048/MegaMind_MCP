-- ================================================================
-- GITHUB ISSUE #29 - PHASE 2: DATABASE SCHEMA MIGRATION
-- Environment Primer Function - Database Schema Extensions
-- ================================================================

-- Date: 2025-07-19
-- Purpose: Add database schema extensions for Environment Primer function
-- Affects: megamind_chunks table (extensions) + new global elements tables

USE megamind_database;

-- ================================================================
-- STEP 1: EXTEND EXISTING MEGAMIND_CHUNKS TABLE FOR GLOBAL ELEMENTS
-- ================================================================

-- Add columns to existing megamind_chunks table for global element support
ALTER TABLE megamind_chunks 
ADD COLUMN IF NOT EXISTS element_category VARCHAR(50) DEFAULT NULL COMMENT 'Primary category: development|security|process|quality|naming|dependencies|architecture',
ADD COLUMN IF NOT EXISTS element_subcategory VARCHAR(100) DEFAULT NULL COMMENT 'Optional subcategory for finer classification',
ADD COLUMN IF NOT EXISTS priority_score DECIMAL(3,2) DEFAULT 0.5 COMMENT 'Priority score 0.0-1.0 for global elements',
ADD COLUMN IF NOT EXISTS enforcement_level ENUM('required','recommended','optional') DEFAULT NULL COMMENT 'Enforcement level for global guidelines',
ADD COLUMN IF NOT EXISTS criticality ENUM('critical','high','medium','low') DEFAULT NULL COMMENT 'Criticality level for global elements',
ADD COLUMN IF NOT EXISTS impact_scope ENUM('system','project','team','individual') DEFAULT NULL COMMENT 'Impact scope of global elements',
ADD COLUMN IF NOT EXISTS applies_to JSON DEFAULT NULL COMMENT 'Array of technologies/contexts this applies to',
ADD COLUMN IF NOT EXISTS excludes JSON DEFAULT NULL COMMENT 'Array of explicit exclusions where not applicable',
ADD COLUMN IF NOT EXISTS prerequisites JSON DEFAULT NULL COMMENT 'Array of required conditions or dependencies',
ADD COLUMN IF NOT EXISTS technology_stack JSON DEFAULT NULL COMMENT 'Array of specific technologies this applies to',
ADD COLUMN IF NOT EXISTS project_phases JSON DEFAULT NULL COMMENT 'Array of development phases where applicable',
ADD COLUMN IF NOT EXISTS effective_date TIMESTAMP NULL COMMENT 'When this guideline became effective',
ADD COLUMN IF NOT EXISTS review_date TIMESTAMP NULL COMMENT 'Next scheduled review date',
ADD COLUMN IF NOT EXISTS expiry_date TIMESTAMP NULL COMMENT 'Optional expiry date for temporary guidelines',
ADD COLUMN IF NOT EXISTS tags JSON DEFAULT NULL COMMENT 'Array of searchable tags for flexible categorization',
ADD COLUMN IF NOT EXISTS keywords JSON DEFAULT NULL COMMENT 'Array of SEO and search optimization keywords';

-- ================================================================
-- STEP 2: CREATE NEW GLOBAL ELEMENTS METADATA TABLE
-- ================================================================

CREATE TABLE IF NOT EXISTS megamind_global_elements (
    element_id VARCHAR(50) PRIMARY KEY COMMENT 'Unique identifier for global element',
    chunk_id VARCHAR(50) NOT NULL COMMENT 'Reference to megamind_chunks.chunk_id',
    title VARCHAR(200) NOT NULL COMMENT 'Human-readable title',
    summary VARCHAR(500) DEFAULT NULL COMMENT 'Brief summary of the element',
    
    -- Core categorization (duplicated from chunks for performance)
    category VARCHAR(50) NOT NULL COMMENT 'Primary category',
    subcategory VARCHAR(100) DEFAULT NULL COMMENT 'Optional subcategory',
    priority_score DECIMAL(3,2) DEFAULT 0.5 COMMENT 'Priority score 0.0-1.0',
    enforcement_level ENUM('required','recommended','optional') DEFAULT 'recommended',
    criticality ENUM('critical','high','medium','low') DEFAULT 'medium',
    impact_scope ENUM('system','project','team','individual') DEFAULT 'project',
    
    -- Extended applicability
    applies_to JSON DEFAULT NULL COMMENT 'Project types, technologies, contexts',
    excludes JSON DEFAULT NULL COMMENT 'Explicit exclusions',
    prerequisites JSON DEFAULT NULL COMMENT 'Required conditions or dependencies',
    technology_stack JSON DEFAULT NULL COMMENT 'Specific technologies',
    project_phases JSON DEFAULT NULL COMMENT 'Development phases where applicable',
    
    -- Governance and ownership
    author VARCHAR(100) DEFAULT NULL COMMENT 'Creator or maintainer',
    maintainer VARCHAR(100) DEFAULT NULL COMMENT 'Current maintainer',
    version VARCHAR(20) DEFAULT '1.0' COMMENT 'Version of the guideline',
    effective_date TIMESTAMP NULL COMMENT 'When guideline became effective',
    review_date TIMESTAMP NULL COMMENT 'Next scheduled review date',
    expiry_date TIMESTAMP NULL COMMENT 'Optional expiry date',
    
    -- Compliance and automation
    compliance_check VARCHAR(200) DEFAULT NULL COMMENT 'Automated compliance check method',
    violation_severity ENUM('critical','high','medium','low','info') DEFAULT 'medium',
    exemption_process TEXT DEFAULT NULL COMMENT 'Process for requesting exemptions',
    approval_required BOOLEAN DEFAULT FALSE COMMENT 'Whether changes require approval',
    change_control_level ENUM('low','medium','high','critical') DEFAULT 'medium',
    
    -- Usage tracking and analytics
    access_count INT DEFAULT 0 COMMENT 'Number of times accessed',
    last_accessed TIMESTAMP NULL COMMENT 'Last access timestamp',
    feedback_score DECIMAL(2,1) DEFAULT 0.0 COMMENT 'User feedback rating 1.0-5.0',
    feedback_count INT DEFAULT 0 COMMENT 'Number of feedback submissions',
    usage_frequency ENUM('daily','weekly','monthly','rarely','unknown') DEFAULT 'unknown',
    
    -- Implementation support
    tooling_support JSON DEFAULT NULL COMMENT 'Tools that support this guideline',
    automation_available BOOLEAN DEFAULT FALSE COMMENT 'Whether automated checking is available',
    automation_config JSON DEFAULT NULL COMMENT 'Configuration for automated tools',
    metrics_tracked JSON DEFAULT NULL COMMENT 'What metrics are tracked for compliance',
    ci_cd_integration BOOLEAN DEFAULT FALSE COMMENT 'Whether integrated into CI/CD pipeline',
    
    -- Business context
    business_justification TEXT DEFAULT NULL COMMENT 'Why this guideline exists',
    cost_impact TEXT DEFAULT NULL COMMENT 'Cost implications',
    risk_mitigation TEXT DEFAULT NULL COMMENT 'What risks this guideline mitigates',
    regulatory_requirement BOOLEAN DEFAULT FALSE COMMENT 'Whether required by regulation',
    industry_standard VARCHAR(100) DEFAULT NULL COMMENT 'Related industry standard',
    
    -- Implementation guidance
    implementation_notes TEXT DEFAULT NULL COMMENT 'Specific implementation guidance',
    common_pitfalls JSON DEFAULT NULL COMMENT 'Known issues and how to avoid them',
    best_practices JSON DEFAULT NULL COMMENT 'Additional best practice recommendations',
    examples JSON DEFAULT NULL COMMENT 'Code examples or implementation examples',
    anti_patterns JSON DEFAULT NULL COMMENT 'What NOT to do',
    
    -- External references
    documentation_urls JSON DEFAULT NULL COMMENT 'Links to detailed documentation',
    example_urls JSON DEFAULT NULL COMMENT 'Links to examples and templates',
    tool_urls JSON DEFAULT NULL COMMENT 'Links to supporting tools',
    reference_standards JSON DEFAULT NULL COMMENT 'Industry standards or regulations',
    
    -- Change management
    change_log JSON DEFAULT NULL COMMENT 'History of changes to this element',
    notification_list JSON DEFAULT NULL COMMENT 'Who to notify of changes',
    review_cycle VARCHAR(50) DEFAULT 'annually' COMMENT 'Review frequency',
    stakeholders JSON DEFAULT NULL COMMENT 'Key stakeholders for this guideline',
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- Foreign key
    FOREIGN KEY (chunk_id) REFERENCES megamind_chunks(chunk_id) ON DELETE CASCADE,
    
    -- Indexes for performance
    INDEX idx_ge_category (category, subcategory),
    INDEX idx_ge_priority (priority_score DESC),
    INDEX idx_ge_enforcement (enforcement_level, criticality),
    INDEX idx_ge_effective_date (effective_date DESC),
    INDEX idx_ge_review_date (review_date ASC),
    INDEX idx_ge_usage (access_count DESC, last_accessed DESC),
    INDEX idx_ge_automation (automation_available, ci_cd_integration),
    INDEX idx_ge_compliance (regulatory_requirement, change_control_level),
    INDEX idx_ge_feedback (feedback_score DESC, feedback_count DESC),
    INDEX idx_ge_title (title),
    INDEX idx_ge_maintainer (maintainer),
    INDEX idx_ge_version (version)
    
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT='Extended metadata for global environment primer elements';

-- ================================================================
-- STEP 3: CREATE GLOBAL ELEMENT RELATIONSHIPS TABLE
-- ================================================================

CREATE TABLE IF NOT EXISTS megamind_global_element_relationships (
    relationship_id VARCHAR(50) PRIMARY KEY,
    source_element_id VARCHAR(50) NOT NULL COMMENT 'Source global element',
    target_element_id VARCHAR(50) NOT NULL COMMENT 'Target global element',
    relationship_type ENUM('supersedes','conflicts_with','depends_on','related_to','implements','enhances','validates') NOT NULL,
    relationship_strength DECIMAL(3,2) DEFAULT 1.0 COMMENT 'Strength of relationship 0.0-1.0',
    bidirectional BOOLEAN DEFAULT FALSE COMMENT 'Whether relationship applies both ways',
    description TEXT DEFAULT NULL COMMENT 'Description of the relationship',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100) DEFAULT NULL COMMENT 'Who created this relationship',
    validated_at TIMESTAMP NULL COMMENT 'When relationship was last validated',
    validation_confidence DECIMAL(3,2) DEFAULT 1.0 COMMENT 'Confidence in relationship accuracy',
    
    FOREIGN KEY (source_element_id) REFERENCES megamind_global_elements(element_id) ON DELETE CASCADE,
    FOREIGN KEY (target_element_id) REFERENCES megamind_global_elements(element_id) ON DELETE CASCADE,
    
    INDEX idx_ger_source (source_element_id, relationship_type),
    INDEX idx_ger_target (target_element_id, relationship_type),
    INDEX idx_ger_type (relationship_type),
    INDEX idx_ger_strength (relationship_strength DESC),
    INDEX idx_ger_created (created_at DESC),
    UNIQUE KEY unique_relationship (source_element_id, target_element_id, relationship_type)
    
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT='Relationships between global elements for dependency and conflict tracking';

-- ================================================================
-- STEP 4: CREATE GLOBAL ELEMENT CHANGE LOG TABLE
-- ================================================================

CREATE TABLE IF NOT EXISTS megamind_global_element_changelog (
    change_id VARCHAR(50) PRIMARY KEY,
    element_id VARCHAR(50) NOT NULL COMMENT 'Global element being changed',
    change_type ENUM('created','updated','deleted','superseded','activated','deactivated','promoted') NOT NULL,
    field_name VARCHAR(100) DEFAULT NULL COMMENT 'Specific field changed (for updates)',
    old_value TEXT DEFAULT NULL COMMENT 'Previous value',
    new_value TEXT DEFAULT NULL COMMENT 'New value',
    change_reason TEXT DEFAULT NULL COMMENT 'Reason for the change',
    change_impact_assessment TEXT DEFAULT NULL COMMENT 'Assessment of change impact',
    changed_by VARCHAR(100) NOT NULL COMMENT 'Who made the change',
    approved_by VARCHAR(100) DEFAULT NULL COMMENT 'Who approved the change',
    change_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    approval_timestamp TIMESTAMP NULL COMMENT 'When change was approved',
    notification_sent BOOLEAN DEFAULT FALSE COMMENT 'Whether stakeholders were notified',
    rollback_data JSON DEFAULT NULL COMMENT 'Data needed to rollback change',
    
    FOREIGN KEY (element_id) REFERENCES megamind_global_elements(element_id) ON DELETE CASCADE,
    
    INDEX idx_gecl_element (element_id, change_timestamp DESC),
    INDEX idx_gecl_type (change_type, change_timestamp DESC),
    INDEX idx_gecl_changed_by (changed_by, change_timestamp DESC),
    INDEX idx_gecl_approved_by (approved_by, approval_timestamp DESC),
    INDEX idx_gecl_notifications (notification_sent, change_timestamp DESC)
    
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT='Complete audit trail for all changes to global elements';

-- ================================================================
-- STEP 5: CREATE GLOBAL ELEMENT USAGE ANALYTICS TABLE
-- ================================================================

CREATE TABLE IF NOT EXISTS megamind_global_element_analytics (
    analytics_id VARCHAR(50) PRIMARY KEY,
    element_id VARCHAR(50) NOT NULL COMMENT 'Global element being tracked',
    access_type ENUM('primer_request','direct_access','relationship_traversal','search_result') NOT NULL,
    user_session_id VARCHAR(50) DEFAULT NULL COMMENT 'User session for correlation',
    access_context JSON DEFAULT NULL COMMENT 'Context of the access (query, filters, etc.)',
    relevance_score DECIMAL(3,2) DEFAULT NULL COMMENT 'How relevant was this element to user query',
    user_action ENUM('viewed','applied','bookmarked','shared','feedback') DEFAULT 'viewed',
    user_feedback JSON DEFAULT NULL COMMENT 'User feedback on usefulness',
    access_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processing_time_ms INT DEFAULT NULL COMMENT 'Time to retrieve and process',
    client_info JSON DEFAULT NULL COMMENT 'Client information (realm, project, etc.)',
    
    FOREIGN KEY (element_id) REFERENCES megamind_global_elements(element_id) ON DELETE CASCADE,
    
    INDEX idx_gea_element (element_id, access_timestamp DESC),
    INDEX idx_gea_type (access_type, access_timestamp DESC),
    INDEX idx_gea_session (user_session_id, access_timestamp DESC),
    INDEX idx_gea_relevance (relevance_score DESC, access_timestamp DESC),
    INDEX idx_gea_action (user_action, access_timestamp DESC),
    INDEX idx_gea_processing (processing_time_ms, access_timestamp DESC)
    
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT='Detailed analytics for global element usage patterns';

-- ================================================================
-- STEP 6: SPECIALIZED INDEXES FOR ENVIRONMENT PRIMER QUERIES
-- ================================================================

-- Composite index for most common primer query pattern
CREATE INDEX IF NOT EXISTS idx_global_primer_category_priority 
ON megamind_chunks(realm_id, element_category, priority_score DESC, enforcement_level) 
WHERE realm_id = 'GLOBAL';

-- Index for enforcement level filtering
CREATE INDEX IF NOT EXISTS idx_global_primer_enforcement_priority 
ON megamind_chunks(realm_id, enforcement_level, priority_score DESC) 
WHERE realm_id = 'GLOBAL';

-- Index for date-based queries (review dates, effective dates)
CREATE INDEX IF NOT EXISTS idx_global_primer_dates 
ON megamind_chunks(realm_id, review_date ASC, effective_date DESC) 
WHERE realm_id = 'GLOBAL';

-- Index for criticality-based filtering
CREATE INDEX IF NOT EXISTS idx_global_primer_criticality 
ON megamind_chunks(realm_id, criticality, priority_score DESC) 
WHERE realm_id = 'GLOBAL';

-- Composite index for complex filtering scenarios
CREATE INDEX IF NOT EXISTS idx_global_primer_composite 
ON megamind_chunks(realm_id, element_category, enforcement_level, criticality, priority_score DESC) 
WHERE realm_id = 'GLOBAL';

-- Full-text search index for content and summary
CREATE FULLTEXT INDEX IF NOT EXISTS idx_global_primer_fulltext 
ON megamind_global_elements(title, summary, implementation_notes);

-- Index for technology stack filtering (JSON array search optimization)
CREATE INDEX IF NOT EXISTS idx_global_primer_tech_stack 
ON megamind_global_elements((CAST(technology_stack AS CHAR(500))));

-- ================================================================
-- STEP 7: CREATE VIEWS FOR ENVIRONMENT PRIMER QUERIES
-- ================================================================

-- Main environment primer view combining chunks and metadata
CREATE OR REPLACE VIEW megamind_environment_primer_view AS
SELECT 
    c.chunk_id,
    c.realm_id,
    c.content,
    c.source_document,
    c.section_path,
    c.element_category as category,
    c.element_subcategory as subcategory,
    c.priority_score,
    c.enforcement_level,
    c.criticality,
    c.impact_scope,
    c.applies_to,
    c.technology_stack,
    c.effective_date,
    c.review_date,
    c.tags,
    c.created_at,
    c.updated_at,
    c.access_count,
    
    -- Extended metadata from global elements table
    ge.element_id,
    ge.title,
    ge.summary,
    ge.author,
    ge.maintainer,
    ge.version,
    ge.business_justification,
    ge.implementation_notes,
    ge.automation_available,
    ge.ci_cd_integration,
    ge.tooling_support,
    ge.examples,
    ge.documentation_urls,
    ge.feedback_score,
    ge.usage_frequency
    
FROM megamind_chunks c
LEFT JOIN megamind_global_elements ge ON c.chunk_id = ge.chunk_id
WHERE c.realm_id = 'GLOBAL' 
  AND c.element_category IS NOT NULL
ORDER BY c.priority_score DESC, c.updated_at DESC;

-- Category summary view for quick overview
CREATE OR REPLACE VIEW megamind_primer_category_summary_view AS
SELECT 
    c.element_category as category,
    COUNT(*) as total_elements,
    COUNT(CASE WHEN c.enforcement_level = 'required' THEN 1 END) as required_count,
    COUNT(CASE WHEN c.enforcement_level = 'recommended' THEN 1 END) as recommended_count,
    COUNT(CASE WHEN c.enforcement_level = 'optional' THEN 1 END) as optional_count,
    COUNT(CASE WHEN c.criticality = 'critical' THEN 1 END) as critical_count,
    COUNT(CASE WHEN c.criticality = 'high' THEN 1 END) as high_count,
    COUNT(CASE WHEN c.criticality = 'medium' THEN 1 END) as medium_count,
    COUNT(CASE WHEN c.criticality = 'low' THEN 1 END) as low_count,
    AVG(c.priority_score) as avg_priority_score,
    MAX(c.updated_at) as last_updated,
    COUNT(CASE WHEN ge.automation_available = TRUE THEN 1 END) as automation_available_count
FROM megamind_chunks c
LEFT JOIN megamind_global_elements ge ON c.chunk_id = ge.chunk_id
WHERE c.realm_id = 'GLOBAL' 
  AND c.element_category IS NOT NULL
GROUP BY c.element_category
ORDER BY total_elements DESC;

-- ================================================================
-- STEP 8: CREATE STORED PROCEDURES FOR ENVIRONMENT PRIMER
-- ================================================================

DELIMITER //

-- Procedure to search global elements with all filtering options
CREATE PROCEDURE IF NOT EXISTS sp_search_environment_primer(
    IN p_include_categories JSON,
    IN p_limit INT,
    IN p_priority_threshold DECIMAL(3,2),
    IN p_enforcement_level VARCHAR(20),
    IN p_criticality VARCHAR(20),
    IN p_technology_stack JSON,
    IN p_sort_by VARCHAR(50),
    IN p_session_id VARCHAR(50)
)
BEGIN
    DECLARE v_sql TEXT;
    DECLARE v_where_conditions TEXT DEFAULT '';
    DECLARE v_order_clause TEXT DEFAULT '';
    
    -- Build WHERE conditions
    SET v_where_conditions = 'c.realm_id = "GLOBAL" AND c.element_category IS NOT NULL';
    
    -- Add category filtering
    IF p_include_categories IS NOT NULL AND JSON_LENGTH(p_include_categories) > 0 THEN
        SET v_where_conditions = CONCAT(v_where_conditions, 
            ' AND c.element_category IN (', 
            REPLACE(REPLACE(JSON_UNQUOTE(p_include_categories), '[', ''), ']', ''), 
            ')');
    END IF;
    
    -- Add priority threshold
    IF p_priority_threshold IS NOT NULL AND p_priority_threshold > 0.0 THEN
        SET v_where_conditions = CONCAT(v_where_conditions, 
            ' AND c.priority_score >= ', p_priority_threshold);
    END IF;
    
    -- Add enforcement level
    IF p_enforcement_level IS NOT NULL AND p_enforcement_level != '' THEN
        SET v_where_conditions = CONCAT(v_where_conditions, 
            ' AND c.enforcement_level = "', p_enforcement_level, '"');
    END IF;
    
    -- Add criticality
    IF p_criticality IS NOT NULL AND p_criticality != '' THEN
        SET v_where_conditions = CONCAT(v_where_conditions, 
            ' AND c.criticality = "', p_criticality, '"');
    END IF;
    
    -- Build ORDER BY clause
    CASE p_sort_by
        WHEN 'priority_asc' THEN SET v_order_clause = 'ORDER BY c.priority_score ASC, c.updated_at ASC';
        WHEN 'updated_desc' THEN SET v_order_clause = 'ORDER BY c.updated_at DESC';
        WHEN 'updated_asc' THEN SET v_order_clause = 'ORDER BY c.updated_at ASC';
        WHEN 'category' THEN SET v_order_clause = 'ORDER BY c.element_category, c.priority_score DESC';
        WHEN 'enforcement' THEN SET v_order_clause = 'ORDER BY FIELD(c.enforcement_level, "required", "recommended", "optional"), c.priority_score DESC';
        ELSE SET v_order_clause = 'ORDER BY c.priority_score DESC, c.updated_at DESC';
    END CASE;
    
    -- Build and execute query
    SET v_sql = CONCAT(
        'SELECT * FROM megamind_environment_primer_view c WHERE ', 
        v_where_conditions, ' ', 
        v_order_clause, 
        ' LIMIT ', COALESCE(p_limit, 100)
    );
    
    SET @sql = v_sql;
    PREPARE stmt FROM @sql;
    EXECUTE stmt;
    DEALLOCATE PREPARE stmt;
    
    -- Track analytics if session provided
    IF p_session_id IS NOT NULL THEN
        INSERT INTO megamind_global_element_analytics (
            analytics_id, element_id, access_type, user_session_id, 
            access_context, access_timestamp
        ) 
        SELECT 
            CONCAT('analytics_', MD5(CONCAT(ge.element_id, p_session_id, NOW()))),
            ge.element_id,
            'primer_request',
            p_session_id,
            JSON_OBJECT(
                'categories', p_include_categories,
                'priority_threshold', p_priority_threshold,
                'enforcement_level', p_enforcement_level,
                'sort_by', p_sort_by
            ),
            NOW()
        FROM megamind_global_elements ge
        WHERE ge.element_id IN (
            SELECT ge2.element_id 
            FROM megamind_environment_primer_view c2
            JOIN megamind_global_elements ge2 ON c2.chunk_id = ge2.chunk_id
            WHERE c2.realm_id = 'GLOBAL'
        )
        LIMIT COALESCE(p_limit, 100);
    END IF;
    
END//

-- Procedure to get element relationships
CREATE PROCEDURE IF NOT EXISTS sp_get_global_element_relationships(
    IN p_element_id VARCHAR(50),
    IN p_relationship_types JSON
)
BEGIN
    SELECT 
        r.relationship_id,
        r.source_element_id,
        r.target_element_id,
        r.relationship_type,
        r.relationship_strength,
        r.description,
        
        -- Source element info
        ge_source.title as source_title,
        ge_source.category as source_category,
        
        -- Target element info
        ge_target.title as target_title,
        ge_target.category as target_category
        
    FROM megamind_global_element_relationships r
    JOIN megamind_global_elements ge_source ON r.source_element_id = ge_source.element_id
    JOIN megamind_global_elements ge_target ON r.target_element_id = ge_target.element_id
    
    WHERE (r.source_element_id = p_element_id OR r.target_element_id = p_element_id)
      AND (p_relationship_types IS NULL 
           OR r.relationship_type IN (
               SELECT JSON_UNQUOTE(JSON_EXTRACT(p_relationship_types, CONCAT('$[', idx, ']')))
               FROM (SELECT 0 as idx UNION SELECT 1 UNION SELECT 2 UNION SELECT 3 UNION SELECT 4) t
               WHERE idx < JSON_LENGTH(p_relationship_types)
           ))
    ORDER BY r.relationship_strength DESC, r.created_at DESC;
END//

DELIMITER ;

-- ================================================================
-- STEP 9: INITIAL SAMPLE DATA FOR TESTING
-- ================================================================

-- Insert sample global elements for testing environment primer function
INSERT IGNORE INTO megamind_chunks (
    chunk_id, realm_id, content, source_document, section_path, 
    chunk_type, element_category, element_subcategory, priority_score, 
    enforcement_level, criticality, impact_scope, applies_to, 
    technology_stack, tags, created_at, updated_at
) VALUES 
(
    'global_dev_001', 
    'GLOBAL', 
    'All functions must have comprehensive docstrings following the Google docstring format. Include parameters, return values, exceptions, and usage examples.',
    'Development_Standards.md',
    '/Documentation/Function_Documentation',
    'rule',
    'development',
    'documentation_standards',
    0.9,
    'required',
    'high',
    'project',
    JSON_ARRAY('all_projects', 'code_documentation'),
    JSON_ARRAY('python', 'javascript', 'typescript', 'java'),
    JSON_ARRAY('documentation', 'code_quality', 'maintainability'),
    NOW(),
    NOW()
),
(
    'global_sec_001',
    'GLOBAL',
    'Never store secrets, API keys, or passwords in source code. Use environment variables or dedicated secret management systems.',
    'Security_Guidelines.md',
    '/Security/Secret_Management',
    'rule',
    'security',
    'data_protection',
    1.0,
    'required',
    'critical',
    'system',
    JSON_ARRAY('all_projects', 'api_development', 'web_applications'),
    JSON_ARRAY('all_technologies'),
    JSON_ARRAY('security', 'secrets', 'api_keys', 'critical'),
    NOW(),
    NOW()
),
(
    'global_proc_001',
    'GLOBAL',
    'All code changes must pass automated tests and code review before merging to main branch. Minimum 80% test coverage required.',
    'Process_Rules.md',
    '/CI_CD/Code_Quality_Gates',
    'rule',
    'process',
    'ci_cd_pipelines',
    0.8,
    'required',
    'high',
    'project',
    JSON_ARRAY('software_development', 'team_projects'),
    JSON_ARRAY('git', 'github', 'gitlab', 'jenkins', 'ci_cd'),
    JSON_ARRAY('ci_cd', 'testing', 'code_review', 'quality'),
    NOW(),
    NOW()
);

-- Insert corresponding global element metadata
INSERT IGNORE INTO megamind_global_elements (
    element_id, chunk_id, title, summary, category, subcategory,
    priority_score, enforcement_level, criticality, impact_scope,
    author, maintainer, version, effective_date, review_date,
    business_justification, implementation_notes, automation_available,
    tooling_support, examples, documentation_urls
) VALUES 
(
    'ge_dev_001',
    'global_dev_001',
    'Function Documentation Standards',
    'Comprehensive docstring requirements for all functions using Google format',
    'development',
    'documentation_standards',
    0.9,
    'required',
    'high',
    'project',
    'Development Team',
    'Tech Lead',
    '2.1',
    NOW(),
    DATE_ADD(NOW(), INTERVAL 6 MONTH),
    'Improves code maintainability and reduces onboarding time for new developers',
    'Use IDE plugins for docstring templates. Review during code review process.',
    TRUE,
    JSON_ARRAY('pylint', 'pydocstyle', 'sphinx', 'jsdoc'),
    JSON_ARRAY('def calculate_total(items: List[Item]) -> float:\\n    """Calculate total price of items.\\n    \\n    Args:\\n        items: List of items to calculate total for\\n        \\n    Returns:\\n        Total price as float\\n        \\n    Raises:\\n        ValueError: If items list is empty\\n    """'),
    JSON_ARRAY('https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings')
),
(
    'ge_sec_001',
    'global_sec_001',
    'Secret Management Security Requirements',
    'Critical security requirement for proper secret and API key management',
    'security',
    'data_protection',
    1.0,
    'required',
    'critical',
    'system',
    'Security Team',
    'Security Officer',
    '3.0',
    NOW(),
    DATE_ADD(NOW(), INTERVAL 3 MONTH),
    'Prevents credential exposure and security breaches. Legal compliance requirement.',
    'Use .env files for development, cloud secret managers for production. Scan code for exposed secrets.',
    TRUE,
    JSON_ARRAY('git-secrets', 'truffleHog', 'AWS Secrets Manager', 'Azure Key Vault', 'HashiCorp Vault'),
    JSON_ARRAY('# Good:\\napi_key = os.getenv("API_KEY")\\n\\n# Bad:\\napi_key = "sk-1234567890abcdef"'),
    JSON_ARRAY('https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html')
),
(
    'ge_proc_001',
    'global_proc_001',
    'Code Quality Gates and CI/CD Requirements',
    'Mandatory quality gates for code changes including testing and review',
    'process',
    'ci_cd_pipelines',
    0.8,
    'required',
    'high',
    'project',
    'DevOps Team',
    'Lead DevOps Engineer',
    '1.5',
    NOW(),
    DATE_ADD(NOW(), INTERVAL 12 MONTH),
    'Ensures code quality, reduces bugs in production, maintains team code standards',
    'Configure branch protection rules. Set up automated test runs. Use code coverage tools.',
    TRUE,
    JSON_ARRAY('GitHub Actions', 'Jenkins', 'SonarQube', 'Codecov', 'pytest', 'jest'),
    JSON_ARRAY('# GitHub branch protection rule:\\nrequire_pull_request_reviews: true\\nrequired_status_checks: [tests, coverage]'),
    JSON_ARRAY('https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/defining-the-mergeability-of-pull-requests/about-protected-branches')
);

-- Insert sample relationships
INSERT IGNORE INTO megamind_global_element_relationships (
    relationship_id, source_element_id, target_element_id, 
    relationship_type, relationship_strength, description, created_by
) VALUES 
(
    'rel_dev_proc_001',
    'ge_dev_001',
    'ge_proc_001',
    'enhances',
    0.8,
    'Documentation standards enhance code review quality in CI/CD process',
    'system'
),
(
    'rel_sec_proc_001',
    'ge_sec_001',
    'ge_proc_001',
    'validates',
    0.9,
    'Secret management security is validated through CI/CD quality gates',
    'system'
);

-- ================================================================
-- STEP 10: CONFIGURATION FOR ENVIRONMENT PRIMER
-- ================================================================

-- Add configuration for environment primer function
INSERT IGNORE INTO megamind_system_config (
    config_id, config_key, config_value, config_type, description, is_system_config
) VALUES 
('cfg_primer_001', 'environment_primer.cache_ttl_seconds', '3600', 'number', 'Cache TTL for primer responses', TRUE),
('cfg_primer_002', 'environment_primer.max_elements_per_request', '500', 'number', 'Maximum elements per primer request', TRUE),
('cfg_primer_003', 'environment_primer.default_limit', '100', 'number', 'Default limit if not specified', TRUE),
('cfg_primer_004', 'environment_primer.enable_analytics', 'true', 'boolean', 'Enable usage analytics tracking', TRUE),
('cfg_primer_005', 'environment_primer.enable_caching', 'true', 'boolean', 'Enable response caching', TRUE),
('cfg_primer_006', 'environment_primer.enable_relationships', 'true', 'boolean', 'Enable relationship traversal', TRUE),
('cfg_primer_007', 'environment_primer.performance_threshold_ms', '2000', 'number', 'Performance threshold in milliseconds', TRUE);

-- ================================================================
-- MIGRATION COMPLETION STATUS
-- ================================================================

SELECT 'GitHub Issue #29 Phase 2 Database Schema Migration completed successfully' as migration_status,
       'Tables created: megamind_global_elements, megamind_global_element_relationships, megamind_global_element_changelog, megamind_global_element_analytics' as tables_created,
       'Views created: megamind_environment_primer_view, megamind_primer_category_summary_view' as views_created,
       'Stored procedures: sp_search_environment_primer, sp_get_global_element_relationships' as procedures_created,
       'Specialized indexes: 5 indexes for optimized primer queries' as indexes_created,
       'Sample data: 3 global elements with relationships for testing' as sample_data,
       NOW() as completion_timestamp;