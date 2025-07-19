# Global Element Data Structures - GitHub Issue #29

## 📋 Overview
Comprehensive data structure design for the Environment Primer global elements system.

**Related**: GitHub Issue #29 - Add function environment primer  
**Phase**: 1.2 - Global Element Categories and Data Structure Design  
**Date**: 2025-07-19  

---

## 🏗️ Core Data Structures

### **Primary Global Element Schema**
```python
GlobalElement = {
    # === CORE IDENTIFICATION ===
    "element_id": "str",                    # Unique identifier (UUID format)
    "chunk_id": "str",                      # Reference to megamind_chunks.chunk_id
    "title": "str",                         # Human-readable title (max 200 chars)
    "content": "str",                       # Full element content (markdown formatted)
    "summary": "str",                       # Brief summary (max 500 chars)
    
    # === CATEGORIZATION ===
    "category": "str",                      # Primary category (enum)
    "subcategory": "str",                   # Optional subcategory for finer classification
    "tags": ["str"],                        # Searchable tags for flexible categorization
    "keywords": ["str"],                    # SEO and search optimization keywords
    
    # === PRIORITY AND ENFORCEMENT ===
    "priority_score": "float",              # 0.0-1.0 importance rating
    "enforcement_level": "str",             # "required"|"recommended"|"optional"
    "criticality": "str",                   # "critical"|"high"|"medium"|"low"
    "impact_scope": "str",                  # "system"|"project"|"team"|"individual"
    
    # === APPLICABILITY ===
    "applies_to": ["str"],                  # Project types, technologies, contexts
    "excludes": ["str"],                    # Explicit exclusions where not applicable
    "prerequisites": ["str"],               # Required conditions or dependencies
    "technology_stack": ["str"],            # Specific technologies this applies to
    "project_phases": ["str"],              # Development phases where applicable
    
    # === METADATA ===
    "source_document": "str",               # Original source document name
    "section_path": "str",                  # Section within source document
    "author": "str",                        # Creator or maintainer
    "maintainer": "str",                    # Current maintainer
    "version": "str",                       # Version of the guideline
    "effective_date": "datetime",           # When this guideline became effective
    "review_date": "datetime",              # Next scheduled review date
    "expiry_date": "datetime",              # Optional expiry date for temporary guidelines
    "last_updated": "datetime",             # Last modification timestamp
    "created_at": "datetime",               # Creation timestamp
    
    # === RELATIONSHIPS ===
    "related_elements": ["str"],            # Related guideline element IDs
    "supersedes": ["str"],                  # Element IDs that this supersedes
    "superseded_by": "str",                 # Element ID that supersedes this (if any)
    "conflicts_with": ["str"],              # Element IDs that conflict with this
    "dependencies": ["str"],                # Element IDs that this depends on
    "dependents": ["str"],                  # Element IDs that depend on this
    
    # === USAGE TRACKING ===
    "access_count": "int",                  # Number of times accessed
    "last_accessed": "datetime",            # Last access timestamp
    "feedback_score": "float",              # User feedback rating (1.0-5.0)
    "feedback_count": "int",                # Number of feedback submissions
    "usage_frequency": "str",               # "daily"|"weekly"|"monthly"|"rarely"
    
    # === COMPLIANCE AND GOVERNANCE ===
    "compliance_check": "str",              # Automated compliance check method
    "violation_severity": "str",            # Severity level for violations
    "exemption_process": "str",             # Process for requesting exemptions
    "approval_required": "bool",            # Whether changes require approval
    "change_control_level": "str",          # "low"|"medium"|"high"|"critical"
    
    # === IMPLEMENTATION GUIDANCE ===
    "implementation_notes": "str",          # Specific implementation guidance
    "common_pitfalls": ["str"],             # Known issues and how to avoid them
    "best_practices": ["str"],              # Additional best practice recommendations
    "examples": ["str"],                    # Code examples or implementation examples
    "anti_patterns": ["str"],               # What NOT to do
    
    # === EXTERNAL REFERENCES ===
    "documentation_urls": ["str"],          # Links to detailed documentation
    "example_urls": ["str"],                # Links to examples and templates
    "tool_urls": ["str"],                   # Links to supporting tools
    "reference_standards": ["str"],         # Industry standards or regulations
    
    # === AUTOMATION AND TOOLING ===
    "tooling_support": ["str"],             # Tools that support this guideline
    "automation_available": "bool",         # Whether automated checking is available
    "automation_config": "dict",            # Configuration for automated tools
    "metrics_tracked": ["str"],             # What metrics are tracked for compliance
    "ci_cd_integration": "bool",            # Whether integrated into CI/CD pipeline
    
    # === CHANGE MANAGEMENT ===
    "change_log": ["dict"],                 # History of changes to this element
    "notification_list": ["str"],           # Who to notify of changes
    "review_cycle": "str",                  # Review frequency ("quarterly"|"annually"|etc.)
    "stakeholders": ["str"],                # Key stakeholders for this guideline
    
    # === BUSINESS CONTEXT ===
    "business_justification": "str",        # Why this guideline exists
    "cost_impact": "str",                   # Cost implications of following/not following
    "risk_mitigation": "str",               # What risks this guideline mitigates
    "regulatory_requirement": "bool",       # Whether required by regulation
    "industry_standard": "str"              # Related industry standard (if any)
}
```

---

## 📂 Global Element Categories

### **1. 🔧 Development Standards**
```python
DEVELOPMENT_SUBCATEGORIES = [
    "coding_conventions",          # Naming, formatting, style guidelines
    "architecture_patterns",      # Design patterns, architectural principles
    "code_organization",          # File structure, module organization
    "documentation_standards",    # Code comments, API documentation
    "version_control",            # Git workflows, branching strategies
    "database_design",            # Schema design, query optimization
    "api_design",                 # RESTful design, GraphQL standards
    "error_handling",             # Exception handling, error codes
    "logging_standards",          # Log levels, formats, destinations
    "performance_guidelines"      # Optimization practices, benchmarks
]

DEVELOPMENT_ENFORCEMENT_EXAMPLES = {
    "required": [
        "All functions must have docstrings",
        "Database migrations must be reversible",
        "API endpoints must follow RESTful conventions"
    ],
    "recommended": [
        "Use type hints in Python code",
        "Implement comprehensive error handling",
        "Follow established naming conventions"
    ],
    "optional": [
        "Consider using design patterns where appropriate",
        "Document complex algorithms with comments",
        "Optimize for readability over cleverness"
    ]
}
```

### **2. 🛡️ Security Guidelines**
```python
SECURITY_SUBCATEGORIES = [
    "authentication",             # Login, session management
    "authorization",              # Access control, permissions
    "data_protection",            # Encryption, data handling
    "input_validation",           # Sanitization, injection prevention
    "vulnerability_management",   # Scanning, patching, updates
    "secure_coding",              # Security-first development practices
    "compliance",                 # GDPR, SOX, HIPAA requirements
    "incident_response",          # Security incident procedures
    "threat_modeling",            # Risk assessment, threat analysis
    "cryptography"                # Encryption standards, key management
]

SECURITY_CRITICALITY_MATRIX = {
    "critical": [
        "SQL injection prevention",
        "Authentication bypass protection",
        "Data encryption requirements"
    ],
    "high": [
        "Input validation standards",
        "Session security requirements",
        "Access control implementation"
    ],
    "medium": [
        "Security logging requirements",
        "Dependency vulnerability scanning",
        "Security testing standards"
    ],
    "low": [
        "Security awareness training",
        "Documentation security guidelines",
        "Security tool recommendations"
    ]
}
```

### **3. 📋 Process Rules**
```python
PROCESS_SUBCATEGORIES = [
    "ci_cd_pipelines",            # Build, test, deploy automation
    "testing_standards",          # Unit, integration, e2e testing
    "code_review",                # Review process, approval requirements
    "release_management",         # Versioning, deployment procedures
    "change_management",          # Change approval, rollback procedures
    "monitoring_alerting",        # Observability, incident detection
    "backup_recovery",            # Data backup, disaster recovery
    "environment_management",     # Dev, staging, production environments
    "dependency_management",      # Library updates, security patches
    "documentation_process"       # Documentation requirements, updates
]

PROCESS_AUTOMATION_LEVELS = {
    "fully_automated": [
        "Unit test execution",
        "Code formatting checks",
        "Dependency vulnerability scanning"
    ],
    "partially_automated": [
        "Code review assignment",
        "Deployment approvals",
        "Security testing"
    ],
    "manual_process": [
        "Architecture design reviews",
        "Business requirement validation",
        "Post-incident reviews"
    ]
}
```

### **4. 🎯 Quality Standards**
```python
QUALITY_SUBCATEGORIES = [
    "code_metrics",               # Complexity, coverage, maintainability
    "performance_benchmarks",     # Response times, throughput
    "reliability_requirements",   # Uptime, error rates
    "usability_standards",        # User experience, accessibility
    "maintainability_criteria",   # Code readability, modularity
    "scalability_requirements",   # Load handling, growth planning
    "compatibility_standards",    # Browser, OS, device support
    "localization_requirements", # Internationalization, accessibility
    "documentation_quality",      # Completeness, accuracy, clarity
    "test_coverage_requirements"  # Coverage thresholds, test quality
]

QUALITY_METRICS_THRESHOLDS = {
    "code_coverage": {"minimum": 80, "target": 90, "excellent": 95},
    "cyclomatic_complexity": {"maximum": 10, "warning": 7, "good": 5},
    "response_time": {"maximum": "2000ms", "target": "1000ms", "excellent": "500ms"},
    "uptime": {"minimum": "99.5%", "target": "99.9%", "excellent": "99.99%"}
}
```

### **5. 🏷️ Naming Conventions**
```python
NAMING_SUBCATEGORIES = [
    "variables_functions",        # Variable and function naming
    "classes_modules",            # Class and module naming
    "files_directories",          # File system organization
    "database_objects",           # Tables, columns, indexes
    "api_endpoints",              # URL patterns, parameter naming
    "configuration_keys",         # Config file and environment variables
    "git_branches_commits",       # Version control naming
    "docker_containers",          # Container and image naming
    "cloud_resources",            # AWS/Azure/GCP resource naming
    "documentation_files"         # Documentation structure and naming
]

NAMING_PATTERN_EXAMPLES = {
    "python_functions": "snake_case",
    "python_classes": "PascalCase",
    "javascript_variables": "camelCase",
    "constants": "UPPER_SNAKE_CASE",
    "database_tables": "snake_case",
    "api_endpoints": "/kebab-case/resources",
    "git_branches": "feature/issue-number-description",
    "docker_images": "kebab-case:version-tag"
}
```

### **6. 📦 Dependency Guidelines**
```python
DEPENDENCY_SUBCATEGORIES = [
    "approved_libraries",         # Whitelist of approved dependencies
    "version_constraints",        # Version pinning, compatibility
    "security_scanning",          # Vulnerability assessment
    "license_compliance",         # Legal requirements, license types
    "update_policies",            # When and how to update dependencies
    "deprecation_management",     # Handling deprecated dependencies
    "private_packages",           # Internal package management
    "build_tools",                # Approved build and development tools
    "runtime_dependencies",       # Production runtime requirements
    "development_dependencies"    # Development-only dependencies
]

DEPENDENCY_APPROVAL_CRITERIA = {
    "security": ["No known high/critical vulnerabilities", "Active security maintenance"],
    "maintenance": ["Active development", "Regular releases", "Community support"],
    "license": ["Compatible with organization license policy", "Clear license terms"],
    "performance": ["Acceptable performance impact", "Minimal resource usage"],
    "compatibility": ["Compatible with target platforms", "Stable API"]
}
```

### **7. 🏛️ Architecture Standards**
```python
ARCHITECTURE_SUBCATEGORIES = [
    "system_design",              # Overall system architecture
    "service_patterns",           # Microservices, SOA patterns
    "data_architecture",          # Data flow, storage patterns
    "integration_patterns",       # API integration, messaging
    "scalability_patterns",       # Load balancing, caching
    "resilience_patterns",        # Circuit breakers, retries
    "observability_patterns",     # Logging, monitoring, tracing
    "deployment_patterns",        # Blue-green, canary deployments
    "security_architecture",      # Security by design principles
    "event_driven_architecture"   # Event sourcing, CQRS patterns
]

ARCHITECTURE_DECISION_TEMPLATE = {
    "decision_title": "str",
    "context": "str",
    "decision": "str",
    "rationale": "str",
    "consequences": "str",
    "alternatives_considered": ["str"],
    "decision_date": "datetime",
    "stakeholders": ["str"],
    "related_decisions": ["str"]
}
```

---

## 🗃️ Database Schema Extensions

### **Enhanced megamind_chunks Table**
```sql
-- Add columns to existing megamind_chunks table
ALTER TABLE megamind_chunks 
ADD COLUMN IF NOT EXISTS element_category VARCHAR(50),
ADD COLUMN IF NOT EXISTS element_subcategory VARCHAR(100),
ADD COLUMN IF NOT EXISTS priority_score DECIMAL(3,2) DEFAULT 0.5,
ADD COLUMN IF NOT EXISTS enforcement_level ENUM('required','recommended','optional') DEFAULT 'recommended',
ADD COLUMN IF NOT EXISTS criticality ENUM('critical','high','medium','low') DEFAULT 'medium',
ADD COLUMN IF NOT EXISTS applies_to JSON,
ADD COLUMN IF NOT EXISTS excludes JSON,
ADD COLUMN IF NOT EXISTS prerequisites JSON,
ADD COLUMN IF NOT EXISTS technology_stack JSON,
ADD COLUMN IF NOT EXISTS effective_date TIMESTAMP NULL,
ADD COLUMN IF NOT EXISTS review_date TIMESTAMP NULL,
ADD COLUMN IF NOT EXISTS expiry_date TIMESTAMP NULL;

-- Indexes for efficient global element queries
CREATE INDEX IF NOT EXISTS idx_global_category 
ON megamind_chunks(realm_id, element_category, priority_score DESC) 
WHERE realm_id = 'GLOBAL';

CREATE INDEX IF NOT EXISTS idx_global_enforcement 
ON megamind_chunks(realm_id, enforcement_level, criticality) 
WHERE realm_id = 'GLOBAL';

CREATE INDEX IF NOT EXISTS idx_global_effective_date 
ON megamind_chunks(realm_id, effective_date DESC) 
WHERE realm_id = 'GLOBAL';
```

### **New megamind_global_elements Table**
```sql
CREATE TABLE IF NOT EXISTS megamind_global_elements (
    element_id VARCHAR(50) PRIMARY KEY,
    chunk_id VARCHAR(50) NOT NULL,
    
    -- Core categorization
    category VARCHAR(50) NOT NULL,
    subcategory VARCHAR(100),
    priority_score DECIMAL(3,2) DEFAULT 0.5,
    enforcement_level ENUM('required','recommended','optional') DEFAULT 'recommended',
    criticality ENUM('critical','high','medium','low') DEFAULT 'medium',
    impact_scope ENUM('system','project','team','individual') DEFAULT 'project',
    
    -- Applicability
    applies_to JSON,
    excludes JSON,
    prerequisites JSON,
    technology_stack JSON,
    project_phases JSON,
    
    -- Governance
    author VARCHAR(100),
    maintainer VARCHAR(100),
    version VARCHAR(20),
    effective_date TIMESTAMP NULL,
    review_date TIMESTAMP NULL,
    expiry_date TIMESTAMP NULL,
    
    -- Compliance
    compliance_check VARCHAR(200),
    violation_severity ENUM('critical','high','medium','low','info') DEFAULT 'medium',
    exemption_process TEXT,
    approval_required BOOLEAN DEFAULT FALSE,
    change_control_level ENUM('low','medium','high','critical') DEFAULT 'medium',
    
    -- Usage tracking
    access_count INT DEFAULT 0,
    last_accessed TIMESTAMP NULL,
    feedback_score DECIMAL(2,1) DEFAULT 0.0,
    feedback_count INT DEFAULT 0,
    usage_frequency ENUM('daily','weekly','monthly','rarely','unknown') DEFAULT 'unknown',
    
    -- Implementation
    tooling_support JSON,
    automation_available BOOLEAN DEFAULT FALSE,
    automation_config JSON,
    metrics_tracked JSON,
    ci_cd_integration BOOLEAN DEFAULT FALSE,
    
    -- Business context
    business_justification TEXT,
    cost_impact TEXT,
    risk_mitigation TEXT,
    regulatory_requirement BOOLEAN DEFAULT FALSE,
    industry_standard VARCHAR(100),
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- Foreign key
    FOREIGN KEY (chunk_id) REFERENCES megamind_chunks(chunk_id) ON DELETE CASCADE,
    
    -- Indexes
    INDEX idx_category (category, subcategory),
    INDEX idx_priority (priority_score DESC),
    INDEX idx_enforcement (enforcement_level, criticality),
    INDEX idx_effective_date (effective_date DESC),
    INDEX idx_review_date (review_date ASC),
    INDEX idx_usage (access_count DESC, last_accessed DESC),
    INDEX idx_automation (automation_available, ci_cd_integration),
    INDEX idx_compliance (regulatory_requirement, change_control_level)
);
```

### **Global Element Relationships Table**
```sql
CREATE TABLE IF NOT EXISTS megamind_global_element_relationships (
    relationship_id VARCHAR(50) PRIMARY KEY,
    source_element_id VARCHAR(50) NOT NULL,
    target_element_id VARCHAR(50) NOT NULL,
    relationship_type ENUM('supersedes','conflicts_with','depends_on','related_to','implements') NOT NULL,
    relationship_strength DECIMAL(3,2) DEFAULT 1.0,
    bidirectional BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100),
    
    FOREIGN KEY (source_element_id) REFERENCES megamind_global_elements(element_id) ON DELETE CASCADE,
    FOREIGN KEY (target_element_id) REFERENCES megamind_global_elements(element_id) ON DELETE CASCADE,
    
    INDEX idx_source (source_element_id, relationship_type),
    INDEX idx_target (target_element_id, relationship_type),
    INDEX idx_relationship_type (relationship_type),
    UNIQUE KEY unique_relationship (source_element_id, target_element_id, relationship_type)
);
```

### **Global Element Change Log Table**
```sql
CREATE TABLE IF NOT EXISTS megamind_global_element_changelog (
    change_id VARCHAR(50) PRIMARY KEY,
    element_id VARCHAR(50) NOT NULL,
    change_type ENUM('created','updated','deleted','superseded','activated','deactivated') NOT NULL,
    field_name VARCHAR(100),
    old_value TEXT,
    new_value TEXT,
    change_reason TEXT,
    changed_by VARCHAR(100) NOT NULL,
    approved_by VARCHAR(100),
    change_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (element_id) REFERENCES megamind_global_elements(element_id) ON DELETE CASCADE,
    
    INDEX idx_element_changes (element_id, change_timestamp DESC),
    INDEX idx_change_type (change_type, change_timestamp DESC),
    INDEX idx_changed_by (changed_by, change_timestamp DESC)
);
```

---

## 🔧 Response Format Structures

### **Structured JSON Format**
```python
StructuredPrimerResponse = {
    "success": "bool",
    "global_elements": [
        {
            "element_id": "str",
            "title": "str",
            "summary": "str",
            "category": "str",
            "subcategory": "str",
            "priority_score": "float",
            "enforcement_level": "str",
            "criticality": "str",
            "content": "str",
            "applies_to": ["str"],
            "implementation_notes": "str",
            "examples": ["str"],
            "related_elements": ["str"],
            "source_info": {
                "document": "str",
                "section": "str",
                "author": "str",
                "version": "str",
                "last_updated": "datetime"
            },
            "compliance": {
                "automated_check": "str",
                "tools_available": ["str"],
                "ci_cd_integrated": "bool"
            },
            "metadata": {
                "access_count": "int",
                "feedback_score": "float",
                "usage_frequency": "str"
            }
        }
    ],
    "summary": {
        "total_count": "int",
        "categories_included": ["str"],
        "enforcement_breakdown": {
            "required": "int",
            "recommended": "int", 
            "optional": "int"
        },
        "priority_distribution": {
            "critical": "int",
            "high": "int",
            "medium": "int",
            "low": "int"
        }
    },
    "query_info": {
        "categories_requested": ["str"],
        "priority_threshold": "float",
        "enforcement_filter": "str",
        "sort_order": "str",
        "retrieved_at": "datetime"
    }
}
```

### **Markdown Format Structure**
```markdown
# Environment Primer - Global Development Guidelines

**Generated**: {timestamp}  
**Categories**: {categories}  
**Elements**: {count}  

## Executive Summary

This primer contains {count} global guidelines across {category_count} categories:
- **Required**: {required_count} guidelines
- **Recommended**: {recommended_count} guidelines  
- **Optional**: {optional_count} guidelines

---

## {Category Name} Guidelines

### {Element Title} [{Enforcement Level}] (Priority: {priority})

{Element Content}

**Applies to**: {applies_to}  
**Tools**: {tools_available}  
**Automation**: {automation_status}  

**Implementation Notes**:
{implementation_notes}

**Examples**:
{examples}

---
```

### **Condensed Format Structure**
```python
CondensedPrimerResponse = {
    "success": "bool",
    "summary": "str",                       # Executive summary paragraph
    "categories": [
        {
            "name": "str",
            "element_count": "int",
            "required_count": "int",
            "key_guidelines": ["str"]           # Top 3-5 most important
        }
    ],
    "critical_requirements": ["str"],        # All critical/required items
    "quick_checklist": ["str"],             # Actionable checklist items
    "automation_available": ["str"],        # Items with automated checking
    "next_review_dates": [                  # Upcoming review dates
        {
            "element": "str",
            "review_date": "datetime"
        }
    ],
    "query_info": {
        "total_elements": "int",
        "filtered_from": "int",
        "retrieval_time_ms": "int"
    }
}
```

---

## 📊 Usage Analytics Schema

### **Primer Access Analytics**
```python
PrimerAccessEvent = {
    "event_id": "str",
    "session_id": "str",
    "user_id": "str",
    "timestamp": "datetime",
    "query_parameters": {
        "categories": ["str"],
        "priority_threshold": "float",
        "enforcement_level": "str",
        "format": "str",
        "limit": "int"
    },
    "results": {
        "element_count": "int",
        "categories_returned": ["str"],
        "response_time_ms": "int",
        "cache_hit": "bool"
    },
    "user_context": {
        "project_realm": "str",
        "user_role": "str",
        "access_method": "str"            # "claude_code"|"api"|"web"
    }
}
```

---

**Document Status**: ✅ **COMPLETED**  
**Next Phase**: Phase 1.3 - Technical Requirements Document  
**Integration Ready**: Database schema and structures fully defined

---

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>