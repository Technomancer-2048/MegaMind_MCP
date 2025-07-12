-- =====================================================================
-- MegaMind Context Database - Global Realm Standards and Content
-- =====================================================================
-- Initial organizational standards and knowledge for global realm
-- Created: 2025-07-12
-- Purpose: Populate global realm with production-ready organizational knowledge

-- =====================================================================
-- Global Realm Content Categories
-- =====================================================================

-- Insert comprehensive organizational standards into global realm
INSERT INTO megamind_chunks (chunk_id, content, source_document, section_path, chunk_type, line_count, token_count, access_count, realm_id, complexity_score) VALUES

-- Security Standards
('global_security_001', 
'All API endpoints must implement authentication using OAuth 2.0 or API keys with rate limiting. Include security headers: X-Content-Type-Options, X-Frame-Options, X-XSS-Protection, Content-Security-Policy. Validate all input parameters and sanitize outputs to prevent injection attacks.',
'security_standards.md', '/api/authentication', 'rule', 3, 45, 1, 'GLOBAL', 0.85),

('global_security_002',
'Sensitive data must be encrypted at rest using AES-256 and in transit using TLS 1.3+. Database connections must use encrypted channels. Store secrets in secure vaults (AWS Secrets Manager, HashiCorp Vault) - never in code or environment variables in production.',
'security_standards.md', '/data/encryption', 'rule', 3, 42, 1, 'GLOBAL', 0.90),

('global_security_003',
'Implement comprehensive audit logging for all security-relevant operations: authentication, authorization failures, data access, configuration changes. Include correlation IDs, timestamps, user identifiers, and action details.',
'security_standards.md', '/audit/logging', 'rule', 2, 35, 1, 'GLOBAL', 0.75),

('global_security_004',
'Follow principle of least privilege for all system access. Regular access reviews quarterly. Multi-factor authentication required for production systems. Session timeout after 8 hours of inactivity. Password policies: minimum 12 characters, complexity requirements.',
'security_standards.md', '/access_control/policies', 'rule', 3, 48, 1, 'GLOBAL', 0.80),

-- Database Standards
('global_database_001',
'Always use database transactions for multi-table operations to ensure data consistency. Implement proper rollback mechanisms for failure scenarios. Use connection pooling with appropriate timeout settings. Monitor connection pool utilization.',
'database_standards.md', '/database/transactions', 'rule', 2, 32, 1, 'GLOBAL', 0.70),

('global_database_002',
'Database schema changes must be versioned and deployed through migration scripts. Test all migrations in staging environment. Include rollback procedures. Document breaking changes and impact assessment.',
'database_standards.md', '/database/migrations', 'rule', 2, 28, 1, 'GLOBAL', 0.75),

('global_database_003',
'Optimize database queries for performance: use appropriate indexes, avoid N+1 queries, implement query result caching where appropriate. Monitor slow queries and optimize regularly. Set query timeout limits.',
'database_standards.md', '/database/performance', 'rule', 2, 30, 1, 'GLOBAL', 0.80),

('global_database_004',
'Implement database backup strategy: daily full backups, transaction log backups every 15 minutes. Test restore procedures monthly. Maintain backups for 30 days minimum. Document disaster recovery procedures.',
'database_standards.md', '/database/backup', 'rule', 2, 33, 1, 'GLOBAL', 0.85),

-- API Design Standards
('global_api_001',
'RESTful API design: use HTTP verbs correctly (GET for retrieval, POST for creation, PUT for updates, DELETE for removal). Implement proper HTTP status codes. Use consistent URL patterns and resource naming.',
'api_standards.md', '/api/rest_design', 'rule', 2, 35, 1, 'GLOBAL', 0.70),

('global_api_002',
'API versioning through URL path (/v1/, /v2/) or Accept headers. Maintain backward compatibility for at least 2 major versions. Provide deprecation notices 6 months before removal. Document migration paths.',
'api_standards.md', '/api/versioning', 'rule', 2, 32, 1, 'GLOBAL', 0.75),

('global_api_003',
'Implement comprehensive API rate limiting: per-user limits, burst protection, graceful degradation. Return rate limit headers: X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset. Document limits in API documentation.',
'api_standards.md', '/api/rate_limiting', 'rule', 2, 38, 1, 'GLOBAL', 0.80),

('global_api_004',
'API responses must include correlation IDs for request tracing. Implement structured error responses with error codes, messages, and debugging information. Use consistent response formats across all endpoints.',
'api_standards.md', '/api/responses', 'rule', 2, 31, 1, 'GLOBAL', 0.70),

-- Error Handling Standards
('global_error_001',
'Implement structured error responses with error codes, human-readable messages, and correlation IDs for debugging. Include appropriate HTTP status codes. Never expose internal system details in error messages.',
'error_handling_standards.md', '/errors/structure', 'rule', 2, 35, 1, 'GLOBAL', 0.75),

('global_error_002',
'Log all errors with appropriate severity levels: DEBUG, INFO, WARN, ERROR, FATAL. Include context information: user ID, request ID, timestamp, stack traces for exceptions. Use structured logging (JSON format).',
'error_handling_standards.md', '/errors/logging', 'rule', 2, 37, 1, 'GLOBAL', 0.80),

('global_error_003',
'Implement circuit breaker pattern for external service calls. Set appropriate timeouts and retry policies with exponential backoff. Provide fallback mechanisms for critical services. Monitor failure rates.',
'error_handling_standards.md', '/errors/resilience', 'rule', 2, 34, 1, 'GLOBAL', 0.85),

-- Code Quality Standards
('global_code_001',
'All code must pass automated tests with minimum 80% code coverage. Implement unit tests, integration tests, and end-to-end tests. Use test-driven development practices. Maintain test documentation.',
'code_standards.md', '/code/testing', 'rule', 2, 30, 1, 'GLOBAL', 0.75),

('global_code_002',
'Follow consistent code formatting and style guidelines. Use automated linting tools (ESLint, Pylint, etc.). Implement pre-commit hooks for code quality checks. Document coding conventions for each language.',
'code_standards.md', '/code/formatting', 'rule', 2, 28, 1, 'GLOBAL', 0.70),

('global_code_003',
'Code reviews required for all changes: minimum 2 reviewers for production code, focus on security, performance, maintainability. Use pull request templates. Document review criteria and approval processes.',
'code_standards.md', '/code/reviews', 'rule', 2, 32, 1, 'GLOBAL', 0.75),

('global_code_004',
'Implement comprehensive code documentation: inline comments for complex logic, README files for projects, API documentation for public interfaces. Keep documentation current with code changes.',
'code_standards.md', '/code/documentation', 'rule', 2, 29, 1, 'GLOBAL', 0.70),

-- Deployment Standards
('global_deploy_001',
'Use infrastructure as code (Terraform, CloudFormation) for all environment provisioning. Version control infrastructure definitions. Implement automated deployment pipelines with proper testing stages.',
'deployment_standards.md', '/deployment/infrastructure', 'rule', 2, 31, 1, 'GLOBAL', 0.80),

('global_deploy_002',
'Implement blue-green or canary deployment strategies for production releases. Include rollback procedures and health checks. Monitor application metrics during deployments. Automate deployment validation.',
'deployment_standards.md', '/deployment/strategies', 'rule', 2, 33, 1, 'GLOBAL', 0.85),

('global_deploy_003',
'Container security: use official base images, scan for vulnerabilities, implement least-privilege user access, avoid running as root. Regularly update base images and dependencies.',
'deployment_standards.md', '/deployment/container_security', 'rule', 2, 27, 1, 'GLOBAL', 0.80),

-- Monitoring Standards
('global_monitor_001',
'Implement comprehensive application monitoring: metrics collection, log aggregation, distributed tracing, alerting. Use consistent monitoring across all services. Define SLAs and SLOs.',
'monitoring_standards.md', '/monitoring/application', 'rule', 2, 30, 1, 'GLOBAL', 0.75),

('global_monitor_002',
'Alert thresholds and escalation procedures: warning alerts for trending issues, critical alerts for immediate action required. Include runbooks for common alerts. Avoid alert fatigue with proper tuning.',
'monitoring_standards.md', '/monitoring/alerting', 'rule', 2, 32, 1, 'GLOBAL', 0.80),

('global_monitor_003',
'Performance monitoring: track response times, throughput, error rates, resource utilization. Implement performance budgets and regression detection. Use APM tools for detailed analysis.',
'monitoring_standards.md', '/monitoring/performance', 'rule', 2, 28, 1, 'GLOBAL', 0.75),

-- Architecture Patterns
('global_arch_001',
'Microservices communication patterns: use async messaging for loose coupling, implement circuit breakers, design for failure. Service discovery and load balancing. API gateway for external access.',
'architecture_patterns.md', '/architecture/microservices', 'section', 2, 35, 1, 'GLOBAL', 0.85),

('global_arch_002',
'Event-driven architecture: use message queues for decoupling, implement event sourcing for audit trails, design idempotent operations. Handle duplicate events and out-of-order processing.',
'architecture_patterns.md', '/architecture/event_driven', 'section', 2, 33, 1, 'GLOBAL', 0.90),

('global_arch_003',
'Caching strategies: implement multi-level caching (application, database, CDN), use appropriate cache invalidation, monitor cache hit rates. Design cache-aside and write-through patterns.',
'architecture_patterns.md', '/architecture/caching', 'section', 2, 30, 1, 'GLOBAL', 0.80),

-- Data Governance
('global_data_001',
'Data classification and handling: categorize data by sensitivity (public, internal, confidential, restricted), implement appropriate access controls, data retention policies, and disposal procedures.',
'data_governance.md', '/data/classification', 'rule', 2, 28, 1, 'GLOBAL', 0.80),

('global_data_002',
'Privacy compliance: implement GDPR, CCPA requirements, data subject rights (access, rectification, erasure), consent management, privacy by design principles. Regular compliance audits.',
'data_governance.md', '/data/privacy', 'rule', 2, 26, 1, 'GLOBAL', 0.85),

('global_data_003',
'Data quality standards: implement data validation, cleansing procedures, quality metrics, data lineage tracking. Establish data ownership and stewardship roles. Monitor data quality continuously.',
'data_governance.md', '/data/quality', 'rule', 2, 27, 1, 'GLOBAL', 0.75);

-- =====================================================================
-- Global Realm Tags
-- =====================================================================

-- Add standardized tags for global content
INSERT INTO megamind_chunk_tags (chunk_id, tag_type, tag_value, confidence, created_by, realm_id) VALUES

-- Security tags
('global_security_001', 'subsystem', 'security', 1.0, 'automatic', 'GLOBAL'),
('global_security_001', 'function_type', 'authentication', 1.0, 'automatic', 'GLOBAL'),
('global_security_001', 'applies_to', 'all_projects', 1.0, 'automatic', 'GLOBAL'),
('global_security_002', 'subsystem', 'security', 1.0, 'automatic', 'GLOBAL'),
('global_security_002', 'function_type', 'encryption', 1.0, 'automatic', 'GLOBAL'),
('global_security_003', 'subsystem', 'security', 1.0, 'automatic', 'GLOBAL'),
('global_security_003', 'function_type', 'audit', 1.0, 'automatic', 'GLOBAL'),
('global_security_004', 'subsystem', 'security', 1.0, 'automatic', 'GLOBAL'),
('global_security_004', 'function_type', 'access_control', 1.0, 'automatic', 'GLOBAL'),

-- Database tags
('global_database_001', 'subsystem', 'database', 1.0, 'automatic', 'GLOBAL'),
('global_database_001', 'function_type', 'transactions', 1.0, 'automatic', 'GLOBAL'),
('global_database_002', 'subsystem', 'database', 1.0, 'automatic', 'GLOBAL'),
('global_database_002', 'function_type', 'migrations', 1.0, 'automatic', 'GLOBAL'),
('global_database_003', 'subsystem', 'database', 1.0, 'automatic', 'GLOBAL'),
('global_database_003', 'function_type', 'performance', 1.0, 'automatic', 'GLOBAL'),
('global_database_004', 'subsystem', 'database', 1.0, 'automatic', 'GLOBAL'),
('global_database_004', 'function_type', 'backup', 1.0, 'automatic', 'GLOBAL'),

-- API tags
('global_api_001', 'subsystem', 'api', 1.0, 'automatic', 'GLOBAL'),
('global_api_001', 'function_type', 'rest_design', 1.0, 'automatic', 'GLOBAL'),
('global_api_002', 'subsystem', 'api', 1.0, 'automatic', 'GLOBAL'),
('global_api_002', 'function_type', 'versioning', 1.0, 'automatic', 'GLOBAL'),
('global_api_003', 'subsystem', 'api', 1.0, 'automatic', 'GLOBAL'),
('global_api_003', 'function_type', 'rate_limiting', 1.0, 'automatic', 'GLOBAL'),
('global_api_004', 'subsystem', 'api', 1.0, 'automatic', 'GLOBAL'),
('global_api_004', 'function_type', 'responses', 1.0, 'automatic', 'GLOBAL'),

-- Error handling tags
('global_error_001', 'subsystem', 'error_handling', 1.0, 'automatic', 'GLOBAL'),
('global_error_001', 'function_type', 'error_structure', 1.0, 'automatic', 'GLOBAL'),
('global_error_002', 'subsystem', 'error_handling', 1.0, 'automatic', 'GLOBAL'),
('global_error_002', 'function_type', 'logging', 1.0, 'automatic', 'GLOBAL'),
('global_error_003', 'subsystem', 'error_handling', 1.0, 'automatic', 'GLOBAL'),
('global_error_003', 'function_type', 'resilience', 1.0, 'automatic', 'GLOBAL'),

-- Code quality tags
('global_code_001', 'subsystem', 'code_quality', 1.0, 'automatic', 'GLOBAL'),
('global_code_001', 'function_type', 'testing', 1.0, 'automatic', 'GLOBAL'),
('global_code_002', 'subsystem', 'code_quality', 1.0, 'automatic', 'GLOBAL'),
('global_code_002', 'function_type', 'formatting', 1.0, 'automatic', 'GLOBAL'),
('global_code_003', 'subsystem', 'code_quality', 1.0, 'automatic', 'GLOBAL'),
('global_code_003', 'function_type', 'reviews', 1.0, 'automatic', 'GLOBAL'),
('global_code_004', 'subsystem', 'code_quality', 1.0, 'automatic', 'GLOBAL'),
('global_code_004', 'function_type', 'documentation', 1.0, 'automatic', 'GLOBAL'),

-- Deployment tags
('global_deploy_001', 'subsystem', 'deployment', 1.0, 'automatic', 'GLOBAL'),
('global_deploy_001', 'function_type', 'infrastructure', 1.0, 'automatic', 'GLOBAL'),
('global_deploy_002', 'subsystem', 'deployment', 1.0, 'automatic', 'GLOBAL'),
('global_deploy_002', 'function_type', 'strategies', 1.0, 'automatic', 'GLOBAL'),
('global_deploy_003', 'subsystem', 'deployment', 1.0, 'automatic', 'GLOBAL'),
('global_deploy_003', 'function_type', 'container_security', 1.0, 'automatic', 'GLOBAL'),

-- Monitoring tags
('global_monitor_001', 'subsystem', 'monitoring', 1.0, 'automatic', 'GLOBAL'),
('global_monitor_001', 'function_type', 'application', 1.0, 'automatic', 'GLOBAL'),
('global_monitor_002', 'subsystem', 'monitoring', 1.0, 'automatic', 'GLOBAL'),
('global_monitor_002', 'function_type', 'alerting', 1.0, 'automatic', 'GLOBAL'),
('global_monitor_003', 'subsystem', 'monitoring', 1.0, 'automatic', 'GLOBAL'),
('global_monitor_003', 'function_type', 'performance', 1.0, 'automatic', 'GLOBAL'),

-- Architecture tags
('global_arch_001', 'subsystem', 'architecture', 1.0, 'automatic', 'GLOBAL'),
('global_arch_001', 'function_type', 'microservices', 1.0, 'automatic', 'GLOBAL'),
('global_arch_002', 'subsystem', 'architecture', 1.0, 'automatic', 'GLOBAL'),
('global_arch_002', 'function_type', 'event_driven', 1.0, 'automatic', 'GLOBAL'),
('global_arch_003', 'subsystem', 'architecture', 1.0, 'automatic', 'GLOBAL'),
('global_arch_003', 'function_type', 'caching', 1.0, 'automatic', 'GLOBAL'),

-- Data governance tags
('global_data_001', 'subsystem', 'data_governance', 1.0, 'automatic', 'GLOBAL'),
('global_data_001', 'function_type', 'classification', 1.0, 'automatic', 'GLOBAL'),
('global_data_002', 'subsystem', 'data_governance', 1.0, 'automatic', 'GLOBAL'),
('global_data_002', 'function_type', 'privacy', 1.0, 'automatic', 'GLOBAL'),
('global_data_003', 'subsystem', 'data_governance', 1.0, 'automatic', 'GLOBAL'),
('global_data_003', 'function_type', 'quality', 1.0, 'automatic', 'GLOBAL'),

-- Universal applicability tags
('global_security_001', 'applies_to', 'web_services', 1.0, 'automatic', 'GLOBAL'),
('global_security_001', 'applies_to', 'apis', 1.0, 'automatic', 'GLOBAL'),
('global_database_001', 'applies_to', 'all_applications', 1.0, 'automatic', 'GLOBAL'),
('global_api_001', 'applies_to', 'public_apis', 1.0, 'automatic', 'GLOBAL'),
('global_code_001', 'applies_to', 'all_development', 1.0, 'automatic', 'GLOBAL'),

-- Priority and compliance tags
('global_security_001', 'status', 'mandatory', 1.0, 'automatic', 'GLOBAL'),
('global_security_002', 'status', 'mandatory', 1.0, 'automatic', 'GLOBAL'),
('global_security_003', 'status', 'mandatory', 1.0, 'automatic', 'GLOBAL'),
('global_data_002', 'status', 'compliance_required', 1.0, 'automatic', 'GLOBAL'),
('global_code_001', 'status', 'quality_gate', 1.0, 'automatic', 'GLOBAL');

-- =====================================================================
-- Sample Relationships Between Global Standards
-- =====================================================================

-- Create logical relationships between related standards
INSERT INTO megamind_chunk_relationships (chunk_id, related_chunk_id, relationship_type, strength, discovered_by, source_realm_id, target_realm_id) VALUES

-- Security relationships
('global_security_001', 'global_api_003', 'enhances', 0.85, 'manual', 'GLOBAL', 'GLOBAL'),
('global_security_002', 'global_database_001', 'depends_on', 0.80, 'manual', 'GLOBAL', 'GLOBAL'),
('global_security_003', 'global_error_002', 'enhances', 0.75, 'manual', 'GLOBAL', 'GLOBAL'),

-- Database relationships
('global_database_001', 'global_error_003', 'enhances', 0.70, 'manual', 'GLOBAL', 'GLOBAL'),
('global_database_003', 'global_monitor_003', 'depends_on', 0.80, 'manual', 'GLOBAL', 'GLOBAL'),
('global_database_004', 'global_deploy_001', 'enhances', 0.75, 'manual', 'GLOBAL', 'GLOBAL'),

-- API relationships
('global_api_001', 'global_error_001', 'enhances', 0.85, 'manual', 'GLOBAL', 'GLOBAL'),
('global_api_002', 'global_deploy_002', 'depends_on', 0.75, 'manual', 'GLOBAL', 'GLOBAL'),
('global_api_004', 'global_monitor_001', 'enhances', 0.80, 'manual', 'GLOBAL', 'GLOBAL'),

-- Code quality relationships
('global_code_001', 'global_deploy_002', 'depends_on', 0.85, 'manual', 'GLOBAL', 'GLOBAL'),
('global_code_003', 'global_security_001', 'enhances', 0.70, 'manual', 'GLOBAL', 'GLOBAL'),
('global_code_004', 'global_arch_001', 'enhances', 0.75, 'manual', 'GLOBAL', 'GLOBAL'),

-- Architecture relationships
('global_arch_001', 'global_monitor_001', 'depends_on', 0.80, 'manual', 'GLOBAL', 'GLOBAL'),
('global_arch_002', 'global_error_003', 'enhances', 0.85, 'manual', 'GLOBAL', 'GLOBAL'),
('global_arch_003', 'global_monitor_003', 'enhances', 0.75, 'manual', 'GLOBAL', 'GLOBAL'),

-- Data governance relationships
('global_data_001', 'global_security_002', 'depends_on', 0.90, 'manual', 'GLOBAL', 'GLOBAL'),
('global_data_002', 'global_security_003', 'enhances', 0.85, 'manual', 'GLOBAL', 'GLOBAL'),
('global_data_003', 'global_monitor_001', 'depends_on', 0.75, 'manual', 'GLOBAL', 'GLOBAL');

-- =====================================================================
-- Global Realm Summary
-- =====================================================================

-- Create a summary view of global standards
CREATE OR REPLACE VIEW megamind_global_standards_summary AS
SELECT 
    ct.tag_value as subsystem,
    COUNT(c.chunk_id) as standard_count,
    GROUP_CONCAT(DISTINCT ft.tag_value) as function_types,
    AVG(c.complexity_score) as avg_complexity,
    MAX(c.created_at) as last_updated
FROM megamind_chunks c
JOIN megamind_chunk_tags ct ON c.chunk_id = ct.chunk_id AND ct.tag_type = 'subsystem'
LEFT JOIN megamind_chunk_tags ft ON c.chunk_id = ft.chunk_id AND ft.tag_type = 'function_type'
WHERE c.realm_id = 'GLOBAL'
GROUP BY ct.tag_value
ORDER BY standard_count DESC;

-- Verification of global realm setup
SELECT 
    'Global Realm Standards Initialized' as status,
    COUNT(DISTINCT c.chunk_id) as total_standards,
    COUNT(DISTINCT ct.tag_value) as unique_subsystems,
    COUNT(DISTINCT cr.relationship_id) as total_relationships,
    NOW() as initialized_at
FROM megamind_chunks c
LEFT JOIN megamind_chunk_tags ct ON c.chunk_id = ct.chunk_id AND ct.tag_type = 'subsystem'
LEFT JOIN megamind_chunk_relationships cr ON c.chunk_id = cr.chunk_id
WHERE c.realm_id = 'GLOBAL';