-- =====================================================================
-- MegaMind Context Database - Initial Project Realms Setup
-- =====================================================================
-- Create sample project realms with domain-specific knowledge
-- Created: 2025-07-12
-- Purpose: Establish initial project realms for common organizational scenarios

-- =====================================================================
-- Create Sample Project Realms
-- =====================================================================

-- E-commerce Platform Project
INSERT INTO megamind_realms (realm_id, realm_name, realm_type, parent_realm_id, description, is_active, inherit_from_parent, default_access_level) VALUES
('PROJ_ECOMMERCE', 'E-commerce Platform', 'project', 'GLOBAL', 'Customer-facing e-commerce application with payment processing and inventory management', TRUE, TRUE, 'write');

-- Data Analytics Pipeline Project  
INSERT INTO megamind_realms (realm_id, realm_name, realm_type, parent_realm_id, description, is_active, inherit_from_parent, default_access_level) VALUES
('PROJ_ANALYTICS', 'Data Analytics Pipeline', 'project', 'GLOBAL', 'Internal data processing, analytics, and business intelligence system', TRUE, TRUE, 'write');

-- Mobile Application Project
INSERT INTO megamind_realms (realm_id, realm_name, realm_type, parent_realm_id, description, is_active, inherit_from_parent, default_access_level) VALUES
('PROJ_MOBILE', 'Mobile Application Suite', 'project', 'GLOBAL', 'iOS and Android mobile applications with offline capabilities', TRUE, TRUE, 'write');

-- DevOps Infrastructure Project
INSERT INTO megamind_realms (realm_id, realm_name, realm_type, parent_realm_id, description, is_active, inherit_from_parent, default_access_level) VALUES
('PROJ_DEVOPS', 'DevOps Infrastructure', 'project', 'GLOBAL', 'Infrastructure automation, CI/CD pipelines, and monitoring systems', TRUE, TRUE, 'write');

-- =====================================================================
-- Set Up Inheritance Relationships
-- =====================================================================

-- Create full inheritance from global realm for all projects
INSERT INTO megamind_realm_inheritance (child_realm_id, parent_realm_id, inheritance_type) VALUES
('PROJ_ECOMMERCE', 'GLOBAL', 'full'),
('PROJ_ANALYTICS', 'GLOBAL', 'full'),
('PROJ_MOBILE', 'GLOBAL', 'full'),
('PROJ_DEVOPS', 'GLOBAL', 'full');

-- =====================================================================
-- E-commerce Project Knowledge
-- =====================================================================

INSERT INTO megamind_chunks (chunk_id, content, source_document, section_path, chunk_type, line_count, token_count, access_count, realm_id, complexity_score) VALUES

-- Business Rules
('ecom_cart_001', 
'Shopping cart items expire after 24 hours of inactivity. Send automated reminder emails at 2-hour and 12-hour marks. Maximum cart size is 100 items. Implement cart persistence across sessions for registered users.',
'ecommerce_business_rules.md', '/cart/expiration_handling', 'rule', 2, 35, 1, 'PROJ_ECOMMERCE', 0.70),

('ecom_payment_001',
'Payment processing flow: validate payment method, authorize amount, capture on shipment, handle payment failures with retry logic. Support multiple payment providers (Stripe, PayPal, Apple Pay). Implement PCI DSS compliance.',
'ecommerce_business_rules.md', '/payment/processing_flow', 'rule', 3, 45, 1, 'PROJ_ECOMMERCE', 0.85),

('ecom_inventory_001',
'Inventory management: real-time stock tracking, reserve items during checkout, release reservations after 15 minutes without payment. Handle backorders and pre-orders. Stock alerts at 10% threshold.',
'ecommerce_business_rules.md', '/inventory/stock_management', 'rule', 3, 42, 1, 'PROJ_ECOMMERCE', 0.80),

('ecom_shipping_001',
'Shipping calculation based on weight, dimensions, destination. Free shipping threshold at $75. Expedited shipping options. Integration with shipping providers (UPS, FedEx, USPS). Real-time tracking updates.',
'ecommerce_business_rules.md', '/shipping/calculation_rules', 'rule', 2, 38, 1, 'PROJ_ECOMMERCE', 0.75),

-- Technical Patterns
('ecom_session_001',
'User session management: 30-day persistent login with secure cookies, session validation on sensitive operations, concurrent session limits (3 per user), automatic logout on suspicious activity.',
'ecommerce_technical_patterns.md', '/session/user_management', 'section', 2, 32, 1, 'PROJ_ECOMMERCE', 0.70),

('ecom_search_001',
'Product search implementation: Elasticsearch integration, faceted search with filters (category, price, brand, ratings), autocomplete suggestions, search analytics tracking, personalized results.',
'ecommerce_technical_patterns.md', '/search/product_discovery', 'section', 2, 33, 1, 'PROJ_ECOMMERCE', 0.80),

-- =====================================================================
-- Analytics Project Knowledge  
-- =====================================================================

INSERT INTO megamind_chunks (chunk_id, content, source_document, section_path, chunk_type, line_count, token_count, access_count, realm_id, complexity_score) VALUES

-- Data Pipeline Rules
('analytics_etl_001',
'ETL pipeline processing: 4-hour batch windows with 1-hour overlap for late-arriving data. Implement idempotent operations for retry safety. Data validation at ingestion and transformation stages.',
'analytics_pipeline_rules.md', '/etl/batch_processing', 'rule', 2, 35, 1, 'PROJ_ANALYTICS', 0.80),

('analytics_data_001',
'Data retention policies: raw data 2 years, aggregated metrics 5 years, anonymized analytics 7 years. Implement automated archival to cold storage. GDPR compliance for user data deletion.',
'analytics_pipeline_rules.md', '/data/retention_policies', 'rule', 2, 32, 1, 'PROJ_ANALYTICS', 0.75),

('analytics_quality_001',
'Data quality validation: completeness checks (null values < 5%), accuracy validation against business rules, consistency across data sources, timeliness monitoring (data freshness < 6 hours).',
'analytics_pipeline_rules.md', '/data/quality_standards', 'rule', 2, 38, 1, 'PROJ_ANALYTICS', 0.85),

-- Technical Implementation
('analytics_streaming_001',
'Real-time data streaming: Apache Kafka for event ingestion, stream processing with Apache Flink, exactly-once delivery semantics, late event handling with watermarks.',
'analytics_technical_patterns.md', '/streaming/real_time_processing', 'section', 2, 30, 1, 'PROJ_ANALYTICS', 0.90),

('analytics_warehouse_001',
'Data warehouse design: star schema for reporting, slowly changing dimensions (SCD Type 2), partitioning by date for performance, columnstore indexes for analytics queries.',
'analytics_technical_patterns.md', '/warehouse/schema_design', 'section', 2, 31, 1, 'PROJ_ANALYTICS', 0.85),

-- =====================================================================
-- Mobile Project Knowledge
-- =====================================================================

INSERT INTO megamind_chunks (chunk_id, content, source_document, section_path, chunk_type, line_count, token_count, access_count, realm_id, complexity_score) VALUES

-- Mobile-Specific Rules
('mobile_offline_001',
'Offline functionality: cache critical user data locally with SQLite, implement sync conflict resolution, graceful degradation when offline, background sync when connection restored.',
'mobile_patterns.md', '/offline/data_synchronization', 'section', 2, 32, 1, 'PROJ_MOBILE', 0.80),

('mobile_performance_001',
'Performance optimization: lazy loading of images, pagination for large lists, memory management for large datasets, battery usage optimization, network request batching.',
'mobile_patterns.md', '/performance/optimization_strategies', 'section', 2, 28, 1, 'PROJ_MOBILE', 0.75),

('mobile_security_001',
'Mobile security: secure keychain/keystore for credentials, certificate pinning for API calls, biometric authentication, app transport security (ATS), obfuscation for sensitive code.',
'mobile_patterns.md', '/security/mobile_specific', 'rule', 2, 35, 1, 'PROJ_MOBILE', 0.85),

-- Platform-Specific Patterns
('mobile_ios_001',
'iOS development patterns: use Core Data for complex data models, implement background app refresh, handle iOS lifecycle events, App Store review guidelines compliance.',
'mobile_patterns.md', '/ios/platform_specifics', 'section', 2, 26, 1, 'PROJ_MOBILE', 0.70),

('mobile_android_001',
'Android development patterns: use Room database, implement WorkManager for background tasks, handle Android fragment lifecycle, Google Play Store policies compliance.',
'mobile_patterns.md', '/android/platform_specifics', 'section', 2, 25, 1, 'PROJ_MOBILE', 0.70),

-- =====================================================================
-- DevOps Project Knowledge
-- =====================================================================

INSERT INTO megamind_chunks (chunk_id, content, source_document, section_path, chunk_type, line_count, token_count, access_count, realm_id, complexity_score) VALUES

-- Infrastructure Patterns
('devops_cicd_001',
'CI/CD pipeline design: automated testing at each stage, security scanning, performance testing, automated rollback on failures, deployment approval gates for production.',
'devops_infrastructure_patterns.md', '/cicd/pipeline_design', 'section', 2, 30, 1, 'PROJ_DEVOPS', 0.80),

('devops_monitoring_001',
'Infrastructure monitoring: Prometheus for metrics collection, Grafana for visualization, AlertManager for notifications, distributed tracing with Jaeger, log aggregation with ELK stack.',
'devops_infrastructure_patterns.md', '/monitoring/observability_stack', 'section', 2, 32, 1, 'PROJ_DEVOPS', 0.85),

('devops_scaling_001',
'Auto-scaling strategies: horizontal pod autoscaling based on CPU/memory, vertical scaling for stateful services, predictive scaling using historical patterns, cost optimization with spot instances.',
'devops_infrastructure_patterns.md', '/scaling/auto_scaling', 'section', 2, 34, 1, 'PROJ_DEVOPS', 0.85),

-- Security and Compliance
('devops_security_001',
'Infrastructure security: network segmentation with VPCs, security groups configuration, secrets management with HashiCorp Vault, vulnerability scanning of container images.',
'devops_infrastructure_patterns.md', '/security/infrastructure_hardening', 'rule', 2, 31, 1, 'PROJ_DEVOPS', 0.80),

('devops_backup_001',
'Backup and disaster recovery: automated daily backups, cross-region replication, RTO/RPO requirements (4 hours/1 hour), disaster recovery testing quarterly.',
'devops_infrastructure_patterns.md', '/backup/disaster_recovery', 'rule', 2, 28, 1, 'PROJ_DEVOPS', 0.75);

-- =====================================================================
-- Project-Specific Tags
-- =====================================================================

-- E-commerce tags
INSERT INTO megamind_chunk_tags (chunk_id, tag_type, tag_value, confidence, created_by, realm_id) VALUES
('ecom_cart_001', 'subsystem', 'shopping_cart', 1.0, 'automatic', 'PROJ_ECOMMERCE'),
('ecom_cart_001', 'function_type', 'business_logic', 1.0, 'automatic', 'PROJ_ECOMMERCE'),
('ecom_payment_001', 'subsystem', 'payment_processing', 1.0, 'automatic', 'PROJ_ECOMMERCE'),
('ecom_payment_001', 'function_type', 'financial_operations', 1.0, 'automatic', 'PROJ_ECOMMERCE'),
('ecom_inventory_001', 'subsystem', 'inventory_management', 1.0, 'automatic', 'PROJ_ECOMMERCE'),
('ecom_shipping_001', 'subsystem', 'fulfillment', 1.0, 'automatic', 'PROJ_ECOMMERCE'),
('ecom_session_001', 'subsystem', 'user_management', 1.0, 'automatic', 'PROJ_ECOMMERCE'),
('ecom_search_001', 'subsystem', 'product_discovery', 1.0, 'automatic', 'PROJ_ECOMMERCE'),

-- Analytics tags
('analytics_etl_001', 'subsystem', 'data_pipeline', 1.0, 'automatic', 'PROJ_ANALYTICS'),
('analytics_etl_001', 'function_type', 'batch_processing', 1.0, 'automatic', 'PROJ_ANALYTICS'),
('analytics_data_001', 'subsystem', 'data_governance', 1.0, 'automatic', 'PROJ_ANALYTICS'),
('analytics_quality_001', 'subsystem', 'data_quality', 1.0, 'automatic', 'PROJ_ANALYTICS'),
('analytics_streaming_001', 'subsystem', 'real_time_processing', 1.0, 'automatic', 'PROJ_ANALYTICS'),
('analytics_warehouse_001', 'subsystem', 'data_warehouse', 1.0, 'automatic', 'PROJ_ANALYTICS'),

-- Mobile tags  
('mobile_offline_001', 'subsystem', 'offline_capabilities', 1.0, 'automatic', 'PROJ_MOBILE'),
('mobile_performance_001', 'subsystem', 'performance_optimization', 1.0, 'automatic', 'PROJ_MOBILE'),
('mobile_security_001', 'subsystem', 'mobile_security', 1.0, 'automatic', 'PROJ_MOBILE'),
('mobile_ios_001', 'subsystem', 'ios_development', 1.0, 'automatic', 'PROJ_MOBILE'),
('mobile_android_001', 'subsystem', 'android_development', 1.0, 'automatic', 'PROJ_MOBILE'),

-- DevOps tags
('devops_cicd_001', 'subsystem', 'continuous_integration', 1.0, 'automatic', 'PROJ_DEVOPS'),
('devops_monitoring_001', 'subsystem', 'observability', 1.0, 'automatic', 'PROJ_DEVOPS'),
('devops_scaling_001', 'subsystem', 'infrastructure_scaling', 1.0, 'automatic', 'PROJ_DEVOPS'),
('devops_security_001', 'subsystem', 'infrastructure_security', 1.0, 'automatic', 'PROJ_DEVOPS'),
('devops_backup_001', 'subsystem', 'disaster_recovery', 1.0, 'automatic', 'PROJ_DEVOPS');

-- =====================================================================
-- Cross-Project Relationships
-- =====================================================================

-- Relationships between project-specific knowledge and global standards
INSERT INTO megamind_chunk_relationships (chunk_id, related_chunk_id, relationship_type, strength, discovered_by, source_realm_id, target_realm_id) VALUES

-- E-commerce to global relationships
('ecom_payment_001', 'global_security_002', 'depends_on', 0.90, 'manual', 'PROJ_ECOMMERCE', 'GLOBAL'),
('ecom_session_001', 'global_security_001', 'implements', 0.85, 'manual', 'PROJ_ECOMMERCE', 'GLOBAL'),
('ecom_search_001', 'global_api_001', 'implements', 0.75, 'manual', 'PROJ_ECOMMERCE', 'GLOBAL'),

-- Analytics to global relationships  
('analytics_etl_001', 'global_database_001', 'depends_on', 0.80, 'manual', 'PROJ_ANALYTICS', 'GLOBAL'),
('analytics_data_001', 'global_data_002', 'implements', 0.90, 'manual', 'PROJ_ANALYTICS', 'GLOBAL'),
('analytics_quality_001', 'global_data_003', 'enhances', 0.85, 'manual', 'PROJ_ANALYTICS', 'GLOBAL'),

-- Mobile to global relationships
('mobile_security_001', 'global_security_001', 'enhances', 0.80, 'manual', 'PROJ_MOBILE', 'GLOBAL'),
('mobile_offline_001', 'global_error_003', 'implements', 0.75, 'manual', 'PROJ_MOBILE', 'GLOBAL'),

-- DevOps to global relationships
('devops_cicd_001', 'global_deploy_002', 'implements', 0.90, 'manual', 'PROJ_DEVOPS', 'GLOBAL'),
('devops_monitoring_001', 'global_monitor_001', 'enhances', 0.85, 'manual', 'PROJ_DEVOPS', 'GLOBAL'),
('devops_security_001', 'global_security_001', 'implements', 0.80, 'manual', 'PROJ_DEVOPS', 'GLOBAL');

-- =====================================================================
-- Project Realm Summary Views
-- =====================================================================

-- Create summary view for each project realm
CREATE OR REPLACE VIEW megamind_project_realm_summary AS
SELECT 
    r.realm_id,
    r.realm_name,
    r.realm_type,
    COUNT(c.chunk_id) as total_chunks,
    COUNT(DISTINCT ct.tag_value) as unique_subsystems,
    COUNT(cr.relationship_id) as internal_relationships,
    COUNT(cr_cross.relationship_id) as cross_realm_relationships,
    AVG(c.complexity_score) as avg_complexity,
    MAX(c.created_at) as last_updated
FROM megamind_realms r
LEFT JOIN megamind_chunks c ON r.realm_id = c.realm_id
LEFT JOIN megamind_chunk_tags ct ON c.chunk_id = ct.chunk_id AND ct.tag_type = 'subsystem'
LEFT JOIN megamind_chunk_relationships cr ON c.chunk_id = cr.chunk_id AND cr.source_realm_id = cr.target_realm_id
LEFT JOIN megamind_chunk_relationships cr_cross ON c.chunk_id = cr_cross.chunk_id AND cr_cross.source_realm_id != cr_cross.target_realm_id
WHERE r.realm_type = 'project'
GROUP BY r.realm_id, r.realm_name, r.realm_type
ORDER BY total_chunks DESC;

-- Project knowledge inheritance verification
CREATE OR REPLACE VIEW megamind_project_inheritance_status AS
SELECT 
    r.realm_id,
    r.realm_name,
    ri.inheritance_type,
    COUNT(DISTINCT gc.chunk_id) as inherited_global_chunks,
    COUNT(DISTINCT pc.chunk_id) as project_specific_chunks,
    ROUND((COUNT(DISTINCT pc.chunk_id) / (COUNT(DISTINCT gc.chunk_id) + COUNT(DISTINCT pc.chunk_id))) * 100, 2) as project_specific_ratio
FROM megamind_realms r
JOIN megamind_realm_inheritance ri ON r.realm_id = ri.child_realm_id
LEFT JOIN megamind_chunks gc ON ri.parent_realm_id = gc.realm_id
LEFT JOIN megamind_chunks pc ON r.realm_id = pc.realm_id
WHERE r.realm_type = 'project'
GROUP BY r.realm_id, r.realm_name, ri.inheritance_type
ORDER BY project_specific_chunks DESC;

-- Verification summary
SELECT 
    'Project Realms Initialized' as status,
    COUNT(DISTINCT r.realm_id) as project_realms_created,
    COUNT(DISTINCT c.chunk_id) as total_project_chunks,
    COUNT(DISTINCT ri.inheritance_id) as inheritance_relationships,
    COUNT(DISTINCT cr.relationship_id) as cross_realm_relationships,
    NOW() as initialized_at
FROM megamind_realms r
LEFT JOIN megamind_chunks c ON r.realm_id = c.realm_id
LEFT JOIN megamind_realm_inheritance ri ON r.realm_id = ri.child_realm_id
LEFT JOIN megamind_chunk_relationships cr ON c.chunk_id = cr.chunk_id AND cr.source_realm_id != cr.target_realm_id
WHERE r.realm_type = 'project';