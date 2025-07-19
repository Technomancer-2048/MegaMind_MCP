-- ================================================================
-- GITHUB ISSUE #29 - PHASE 2: DATABASE SCHEMA MIGRATION (FIXED)
-- Environment Primer Function - Database Schema Extensions  
-- ================================================================

USE megamind_database;

-- ================================================================
-- STEP 1: EXTEND EXISTING MEGAMIND_CHUNKS TABLE FOR GLOBAL ELEMENTS
-- ================================================================

-- Check and add columns to existing megamind_chunks table
-- Note: Using separate statements for MySQL compatibility

-- Add element_category column
SET @sql = (SELECT IF(
    (SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS 
     WHERE TABLE_SCHEMA = 'megamind_database' 
     AND TABLE_NAME = 'megamind_chunks' 
     AND COLUMN_NAME = 'element_category') = 0,
    'ALTER TABLE megamind_chunks ADD COLUMN element_category VARCHAR(50) DEFAULT NULL COMMENT "Primary category"',
    'SELECT "element_category already exists" as message'
));
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- Add element_subcategory column
SET @sql = (SELECT IF(
    (SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS 
     WHERE TABLE_SCHEMA = 'megamind_database' 
     AND TABLE_NAME = 'megamind_chunks' 
     AND COLUMN_NAME = 'element_subcategory') = 0,
    'ALTER TABLE megamind_chunks ADD COLUMN element_subcategory VARCHAR(100) DEFAULT NULL COMMENT "Optional subcategory"',
    'SELECT "element_subcategory already exists" as message'
));
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- Add priority_score column
SET @sql = (SELECT IF(
    (SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS 
     WHERE TABLE_SCHEMA = 'megamind_database' 
     AND TABLE_NAME = 'megamind_chunks' 
     AND COLUMN_NAME = 'priority_score') = 0,
    'ALTER TABLE megamind_chunks ADD COLUMN priority_score DECIMAL(3,2) DEFAULT 0.5 COMMENT "Priority score 0.0-1.0"',
    'SELECT "priority_score already exists" as message'
));
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- Add enforcement_level column
SET @sql = (SELECT IF(
    (SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS 
     WHERE TABLE_SCHEMA = 'megamind_database' 
     AND TABLE_NAME = 'megamind_chunks' 
     AND COLUMN_NAME = 'enforcement_level') = 0,
    'ALTER TABLE megamind_chunks ADD COLUMN enforcement_level ENUM("required","recommended","optional") DEFAULT NULL COMMENT "Enforcement level"',
    'SELECT "enforcement_level already exists" as message'
));
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- Add criticality column
SET @sql = (SELECT IF(
    (SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS 
     WHERE TABLE_SCHEMA = 'megamind_database' 
     AND TABLE_NAME = 'megamind_chunks' 
     AND COLUMN_NAME = 'criticality') = 0,
    'ALTER TABLE megamind_chunks ADD COLUMN criticality ENUM("critical","high","medium","low") DEFAULT NULL COMMENT "Criticality level"',
    'SELECT "criticality already exists" as message'
));
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- Add applies_to column
SET @sql = (SELECT IF(
    (SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS 
     WHERE TABLE_SCHEMA = 'megamind_database' 
     AND TABLE_NAME = 'megamind_chunks' 
     AND COLUMN_NAME = 'applies_to') = 0,
    'ALTER TABLE megamind_chunks ADD COLUMN applies_to JSON DEFAULT NULL COMMENT "Technologies this applies to"',
    'SELECT "applies_to already exists" as message'
));
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- Add effective_date column
SET @sql = (SELECT IF(
    (SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS 
     WHERE TABLE_SCHEMA = 'megamind_database' 
     AND TABLE_NAME = 'megamind_chunks' 
     AND COLUMN_NAME = 'effective_date') = 0,
    'ALTER TABLE megamind_chunks ADD COLUMN effective_date TIMESTAMP NULL COMMENT "When guideline became effective"',
    'SELECT "effective_date already exists" as message'
));
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- ================================================================
-- STEP 2: CREATE NEW GLOBAL ELEMENTS METADATA TABLE
-- ================================================================

CREATE TABLE IF NOT EXISTS megamind_global_elements (
    element_id VARCHAR(50) PRIMARY KEY COMMENT 'Unique identifier for global element',
    chunk_id VARCHAR(50) NOT NULL COMMENT 'Reference to megamind_chunks.chunk_id',
    title VARCHAR(200) NOT NULL COMMENT 'Human-readable title',
    summary VARCHAR(500) DEFAULT NULL COMMENT 'Brief summary of the element',
    
    -- Core categorization
    category VARCHAR(50) NOT NULL COMMENT 'Primary category',
    subcategory VARCHAR(100) DEFAULT NULL COMMENT 'Optional subcategory',
    priority_score DECIMAL(3,2) DEFAULT 0.5 COMMENT 'Priority score 0.0-1.0',
    enforcement_level ENUM('required','recommended','optional') DEFAULT 'recommended',
    criticality ENUM('critical','high','medium','low') DEFAULT 'medium',
    
    -- Extended applicability
    applies_to JSON DEFAULT NULL COMMENT 'Project types, technologies, contexts',
    technology_stack JSON DEFAULT NULL COMMENT 'Specific technologies',
    
    -- Governance
    author VARCHAR(100) DEFAULT NULL COMMENT 'Creator or maintainer',
    maintainer VARCHAR(100) DEFAULT NULL COMMENT 'Current maintainer',
    version VARCHAR(20) DEFAULT '1.0' COMMENT 'Version of the guideline',
    effective_date TIMESTAMP NULL COMMENT 'When guideline became effective',
    review_date TIMESTAMP NULL COMMENT 'Next scheduled review date',
    
    -- Usage tracking
    access_count INT DEFAULT 0 COMMENT 'Number of times accessed',
    last_accessed TIMESTAMP NULL COMMENT 'Last access timestamp',
    feedback_score DECIMAL(2,1) DEFAULT 0.0 COMMENT 'User feedback rating 1.0-5.0',
    
    -- Implementation support
    automation_available BOOLEAN DEFAULT FALSE COMMENT 'Whether automated checking is available',
    tooling_support JSON DEFAULT NULL COMMENT 'Tools that support this guideline',
    
    -- Business context
    business_justification TEXT DEFAULT NULL COMMENT 'Why this guideline exists',
    
    -- Implementation guidance
    implementation_notes TEXT DEFAULT NULL COMMENT 'Specific implementation guidance',
    examples JSON DEFAULT NULL COMMENT 'Code examples or implementation examples',
    documentation_urls JSON DEFAULT NULL COMMENT 'Links to detailed documentation',
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- Foreign key
    FOREIGN KEY (chunk_id) REFERENCES megamind_chunks(chunk_id) ON DELETE CASCADE,
    
    -- Indexes for performance
    INDEX idx_ge_category (category, subcategory),
    INDEX idx_ge_priority (priority_score DESC),
    INDEX idx_ge_enforcement (enforcement_level, criticality),
    INDEX idx_ge_usage (access_count DESC, last_accessed DESC)
    
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT='Extended metadata for global environment primer elements';

-- ================================================================
-- STEP 3: CREATE SPECIALIZED INDEXES FOR ENVIRONMENT PRIMER QUERIES
-- ================================================================

-- Create indexes only if they don't exist
SET @sql = (SELECT IF(
    (SELECT COUNT(*) FROM INFORMATION_SCHEMA.STATISTICS 
     WHERE TABLE_SCHEMA = 'megamind_database' 
     AND TABLE_NAME = 'megamind_chunks' 
     AND INDEX_NAME = 'idx_global_primer_category') = 0,
    'CREATE INDEX idx_global_primer_category ON megamind_chunks(realm_id, element_category, priority_score DESC) WHERE realm_id = "GLOBAL"',
    'SELECT "idx_global_primer_category already exists" as message'
));

-- Note: MySQL doesn't support WHERE clauses in CREATE INDEX, so we'll create without WHERE
CREATE INDEX IF NOT EXISTS idx_global_primer_category 
ON megamind_chunks(realm_id, element_category, priority_score DESC);

CREATE INDEX IF NOT EXISTS idx_global_primer_enforcement 
ON megamind_chunks(realm_id, enforcement_level, priority_score DESC);

CREATE INDEX IF NOT EXISTS idx_global_primer_composite 
ON megamind_chunks(realm_id, element_category, enforcement_level, criticality, priority_score DESC);

-- ================================================================
-- STEP 4: CREATE VIEWS FOR ENVIRONMENT PRIMER QUERIES
-- ================================================================

-- Main environment primer view
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
    c.applies_to,
    c.effective_date,
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
    ge.tooling_support,
    ge.examples,
    ge.documentation_urls,
    ge.feedback_score
    
FROM megamind_chunks c
LEFT JOIN megamind_global_elements ge ON c.chunk_id = ge.chunk_id
WHERE c.realm_id = 'GLOBAL' 
  AND c.element_category IS NOT NULL
ORDER BY c.priority_score DESC, c.updated_at DESC;

-- Category summary view
CREATE OR REPLACE VIEW megamind_primer_category_summary_view AS
SELECT 
    c.element_category as category,
    COUNT(*) as total_elements,
    COUNT(CASE WHEN c.enforcement_level = 'required' THEN 1 END) as required_count,
    COUNT(CASE WHEN c.enforcement_level = 'recommended' THEN 1 END) as recommended_count,
    COUNT(CASE WHEN c.enforcement_level = 'optional' THEN 1 END) as optional_count,
    COUNT(CASE WHEN c.criticality = 'critical' THEN 1 END) as critical_count,
    COUNT(CASE WHEN c.criticality = 'high' THEN 1 END) as high_count,
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
-- STEP 5: INSERT SAMPLE DATA FOR TESTING
-- ================================================================

-- Insert sample global elements for testing
INSERT IGNORE INTO megamind_chunks (
    chunk_id, realm_id, content, source_document, section_path, 
    chunk_type, element_category, element_subcategory, priority_score, 
    enforcement_level, criticality, applies_to, created_at, updated_at
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
    JSON_ARRAY('python', 'javascript', 'typescript'),
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
    JSON_ARRAY('all_technologies'),
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
    JSON_ARRAY('git', 'github', 'ci_cd'),
    NOW(),
    NOW()
);

-- Insert corresponding global element metadata
INSERT IGNORE INTO megamind_global_elements (
    element_id, chunk_id, title, summary, category, subcategory,
    priority_score, enforcement_level, criticality,
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
    'Development Team',
    'Tech Lead',
    '2.1',
    NOW(),
    DATE_ADD(NOW(), INTERVAL 6 MONTH),
    'Improves code maintainability and reduces onboarding time for new developers',
    'Use IDE plugins for docstring templates. Review during code review process.',
    TRUE,
    JSON_ARRAY('pylint', 'pydocstyle', 'sphinx'),
    JSON_ARRAY('def calculate_total(items):\\n    """Calculate total price."""'),
    JSON_ARRAY('https://google.github.io/styleguide/pyguide.html')
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
    'Security Team',
    'Security Officer',
    '3.0',
    NOW(),
    DATE_ADD(NOW(), INTERVAL 3 MONTH),
    'Prevents credential exposure and security breaches. Legal compliance requirement.',
    'Use .env files for development, cloud secret managers for production.',
    TRUE,
    JSON_ARRAY('git-secrets', 'truffleHog', 'AWS Secrets Manager'),
    JSON_ARRAY('api_key = os.getenv("API_KEY")'),
    JSON_ARRAY('https://owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html')
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
    'DevOps Team',
    'Lead DevOps Engineer',
    '1.5',
    NOW(),
    DATE_ADD(NOW(), INTERVAL 12 MONTH),
    'Ensures code quality, reduces bugs in production, maintains team standards',
    'Configure branch protection rules. Set up automated test runs.',
    TRUE,
    JSON_ARRAY('GitHub Actions', 'Jenkins', 'SonarQube'),
    JSON_ARRAY('require_pull_request_reviews: true'),
    JSON_ARRAY('https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository')
);

-- ================================================================
-- STEP 6: ADD CONFIGURATION FOR ENVIRONMENT PRIMER
-- ================================================================

-- Add configuration entries for environment primer function
INSERT IGNORE INTO megamind_system_config (
    config_id, config_key, config_value, config_type, description, is_system_config
) VALUES 
('cfg_primer_001', 'environment_primer.cache_ttl_seconds', '3600', 'number', 'Cache TTL for primer responses', TRUE),
('cfg_primer_002', 'environment_primer.max_elements_per_request', '500', 'number', 'Maximum elements per primer request', TRUE),
('cfg_primer_003', 'environment_primer.default_limit', '100', 'number', 'Default limit if not specified', TRUE),
('cfg_primer_004', 'environment_primer.enable_analytics', 'true', 'boolean', 'Enable usage analytics tracking', TRUE);

-- ================================================================
-- MIGRATION COMPLETION STATUS
-- ================================================================

SELECT 'GitHub Issue #29 Phase 2 Database Schema Migration completed successfully' as migration_status,
       'Schema extended: megamind_chunks table with new columns' as chunks_extended,
       'New table created: megamind_global_elements' as elements_table,
       'Indexes created: 3 specialized primer indexes' as indexes_created,
       'Views created: 2 primer views for querying' as views_created,
       'Sample data: 3 global elements for testing' as sample_data,
       'Configuration: 4 primer config entries' as configuration,
       NOW() as completion_timestamp;