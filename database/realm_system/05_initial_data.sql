-- Initial Data and Default Realms
-- Phase 1: Default Realm Setup and Sample Data
-- Database: megamind_database (MySQL 8.0+)

-- Create default realms during initial database setup
INSERT INTO megamind_realms (realm_id, realm_name, realm_type, parent_realm_id, description, is_active, created_by, access_level) VALUES
('GLOBAL', 'Global Organization Standards', 'global', NULL, 'Organization-wide rules, standards, and best practices that apply to all projects', TRUE, 'system', 'admin');

-- Example project realms (these would typically be created via environment configuration)
INSERT INTO megamind_realms (realm_id, realm_name, realm_type, parent_realm_id, description, is_active, created_by, access_level) VALUES
('PROJ_ECOMMERCE', 'E-commerce Platform', 'project', 'GLOBAL', 'Customer-facing e-commerce application with payment processing', TRUE, 'system', 'read_write'),
('PROJ_ANALYTICS', 'Data Analytics Pipeline', 'project', 'GLOBAL', 'Internal analytics and reporting system for business intelligence', TRUE, 'system', 'read_write'),
('PROJ_MOBILE', 'Mobile Application', 'project', 'GLOBAL', 'iOS and Android mobile applications for customer engagement', TRUE, 'system', 'read_write'),
('PROJ_DEFAULT', 'Default Project Realm', 'project', 'GLOBAL', 'Default project realm for new instances without specific configuration', TRUE, 'system', 'read_write');

-- Set up automatic inheritance relationships
INSERT INTO megamind_realm_inheritance (child_realm_id, parent_realm_id, inheritance_type, priority_order, is_active) VALUES
('PROJ_ECOMMERCE', 'GLOBAL', 'full', 1, TRUE),
('PROJ_ANALYTICS', 'GLOBAL', 'full', 1, TRUE),
('PROJ_MOBILE', 'GLOBAL', 'full', 1, TRUE),
('PROJ_DEFAULT', 'GLOBAL', 'full', 1, TRUE);

-- Insert sample global knowledge chunks
INSERT INTO megamind_chunks (chunk_id, content, source_document, section_path, chunk_type, line_count, realm_id, token_count, content_hash) VALUES
('global_001', 'All database queries must use parameterized statements to prevent SQL injection vulnerabilities. Never concatenate user input directly into SQL strings.', 'security_standards.md', 'database/sql_injection_prevention', 'rule', 2, 'GLOBAL', 45, SHA2('All database queries must use parameterized statements...', 256)),
('global_002', 'API responses must include appropriate HTTP status codes: 200 for success, 400 for client errors, 401 for authentication failures, 403 for authorization failures, 404 for not found, 500 for server errors.', 'api_standards.md', 'http/status_codes', 'rule', 3, 'GLOBAL', 62, SHA2('API responses must include appropriate HTTP status codes...', 256)),
('global_003', 'All user inputs must be validated both on client-side (for UX) and server-side (for security). Client-side validation is not sufficient for security purposes.', 'security_standards.md', 'input_validation/dual_validation', 'rule', 2, 'GLOBAL', 48, SHA2('All user inputs must be validated both on client-side...', 256)),
('global_004', 'Code reviews are mandatory for all changes to production systems. Reviews must check for: security vulnerabilities, performance implications, code quality, and business logic correctness.', 'development_process.md', 'quality_assurance/code_review', 'rule', 3, 'GLOBAL', 58, SHA2('Code reviews are mandatory for all changes...', 256)),
('global_005', 'Error messages shown to users must not reveal internal system details, file paths, database schemas, or stack traces. Log detailed errors server-side for debugging.', 'security_standards.md', 'error_handling/information_disclosure', 'rule', 2, 'GLOBAL', 52, SHA2('Error messages shown to users must not reveal...', 256));

-- Insert sample project-specific knowledge chunks
INSERT INTO megamind_chunks (chunk_id, content, source_document, section_path, chunk_type, line_count, realm_id, token_count, content_hash) VALUES
('ecom_001', 'Payment processing must handle PCI DSS compliance requirements. Never store credit card numbers, CVV codes, or full PAN data. Use tokenization for recurring payments.', 'ecommerce_requirements.md', 'payments/pci_compliance', 'rule', 3, 'PROJ_ECOMMERCE', 58, SHA2('Payment processing must handle PCI DSS compliance...', 256)),
('ecom_002', 'Shopping cart abandonment emails should be sent 1 hour, 24 hours, and 7 days after abandonment. Include personalized product recommendations and discount codes.', 'ecommerce_marketing.md', 'automation/cart_abandonment', 'function', 2, 'PROJ_ECOMMERCE', 48, SHA2('Shopping cart abandonment emails should be sent...', 256)),
('analytics_001', 'Data retention policy: Raw event data kept for 2 years, aggregated metrics kept for 5 years, personally identifiable information anonymized after 1 year per GDPR requirements.', 'analytics_governance.md', 'data_retention/policy', 'rule', 3, 'PROJ_ANALYTICS', 62, SHA2('Data retention policy: Raw event data kept for 2 years...', 256)),
('mobile_001', 'Push notifications must respect user preferences and time zones. Never send notifications between 10 PM and 8 AM in user local time unless explicitly opted in for urgent alerts.', 'mobile_ux_guidelines.md', 'notifications/timing_rules', 'rule', 2, 'PROJ_MOBILE', 55, SHA2('Push notifications must respect user preferences...', 256));

-- Create sample relationships between chunks
INSERT INTO megamind_chunk_relationships (chunk_id, related_chunk_id, relationship_type, strength, discovered_by, source_realm_id, target_realm_id) VALUES
('ecom_001', 'global_001', 'implements', 0.85, 'manual', 'PROJ_ECOMMERCE', 'GLOBAL'),
('ecom_001', 'global_003', 'implements', 0.90, 'manual', 'PROJ_ECOMMERCE', 'GLOBAL'),
('analytics_001', 'global_005', 'enhances', 0.75, 'manual', 'PROJ_ANALYTICS', 'GLOBAL'),
('mobile_001', 'global_003', 'implements', 0.80, 'manual', 'PROJ_MOBILE', 'GLOBAL'),
('global_001', 'global_003', 'enhances', 0.85, 'manual', 'GLOBAL', 'GLOBAL'),
('global_004', 'global_001', 'references', 0.70, 'manual', 'GLOBAL', 'GLOBAL');

-- Create sample tags for chunks
INSERT INTO megamind_chunk_tags (chunk_id, tag_type, tag_value, confidence, created_by, realm_id) VALUES
('global_001', 'subsystem', 'security', 1.0, 'manual', 'GLOBAL'),
('global_001', 'applies_to', 'database', 1.0, 'manual', 'GLOBAL'),
('global_002', 'subsystem', 'api', 1.0, 'manual', 'GLOBAL'),
('global_002', 'function_type', 'standard', 1.0, 'manual', 'GLOBAL'),
('global_003', 'subsystem', 'security', 1.0, 'manual', 'GLOBAL'),
('global_003', 'applies_to', 'validation', 1.0, 'manual', 'GLOBAL'),
('global_004', 'subsystem', 'process', 1.0, 'manual', 'GLOBAL'),
('global_004', 'function_type', 'requirement', 1.0, 'manual', 'GLOBAL'),
('global_005', 'subsystem', 'security', 1.0, 'manual', 'GLOBAL'),
('global_005', 'applies_to', 'error_handling', 1.0, 'manual', 'GLOBAL'),
('ecom_001', 'subsystem', 'payments', 1.0, 'manual', 'PROJ_ECOMMERCE'),
('ecom_001', 'applies_to', 'compliance', 1.0, 'manual', 'PROJ_ECOMMERCE'),
('ecom_002', 'subsystem', 'marketing', 1.0, 'manual', 'PROJ_ECOMMERCE'),
('ecom_002', 'function_type', 'automation', 1.0, 'manual', 'PROJ_ECOMMERCE'),
('analytics_001', 'subsystem', 'data_governance', 1.0, 'manual', 'PROJ_ANALYTICS'),
('analytics_001', 'applies_to', 'compliance', 1.0, 'manual', 'PROJ_ANALYTICS'),
('mobile_001', 'subsystem', 'notifications', 1.0, 'manual', 'PROJ_MOBILE'),
('mobile_001', 'applies_to', 'user_experience', 1.0, 'manual', 'PROJ_MOBILE');

-- Create sample session metadata for testing
INSERT INTO megamind_session_metadata (session_id, user_context, project_context, realm_id, session_config) VALUES
('test_session_global', 'admin', 'system_setup', 'GLOBAL', '{"session_type": "administrative", "permissions": ["read", "write", "admin"]}'),
('test_session_ecom', 'developer', 'ecommerce_development', 'PROJ_ECOMMERCE', '{"session_type": "development", "permissions": ["read", "write"]}'),
('test_session_analytics', 'analyst', 'data_analysis', 'PROJ_ANALYTICS', '{"session_type": "analysis", "permissions": ["read", "write"]}'),
('test_session_mobile', 'developer', 'mobile_development', 'PROJ_MOBILE', '{"session_type": "development", "permissions": ["read", "write"]}');

-- Set up user session variables for testing (these would be set programmatically)
-- SET @current_realm = 'PROJ_ECOMMERCE';  -- Example: would be set based on environment configuration