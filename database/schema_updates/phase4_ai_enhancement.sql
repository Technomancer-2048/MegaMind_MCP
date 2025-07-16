-- Phase 4: AI Enhancement Schema
-- Tables for quality improvement, adaptive learning, automated curation, and performance optimization

-- 1. Quality Improvement History
CREATE TABLE IF NOT EXISTS megamind_quality_improvements (
    improvement_id VARCHAR(50) PRIMARY KEY,
    chunk_id VARCHAR(50) NOT NULL,
    improvement_type ENUM('readability', 'technical_accuracy', 'completeness', 'coherence', 'relevance') NOT NULL,
    original_score FLOAT NOT NULL,
    improved_score FLOAT,
    improvement_status ENUM('suggested', 'applied', 'rejected', 'pending') DEFAULT 'suggested',
    suggestion TEXT NOT NULL,
    implementation TEXT,
    automated BOOLEAN DEFAULT FALSE,
    confidence FLOAT DEFAULT 0.5,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    applied_date TIMESTAMP NULL,
    session_id VARCHAR(50),
    INDEX idx_chunk (chunk_id),
    INDEX idx_status (improvement_status),
    INDEX idx_created (created_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 2. User Feedback Table
CREATE TABLE IF NOT EXISTS megamind_user_feedback (
    feedback_id VARCHAR(50) PRIMARY KEY,
    feedback_type ENUM('chunk_quality', 'boundary_accuracy', 'retrieval_success', 'manual_correction') NOT NULL,
    target_id VARCHAR(50) NOT NULL,
    rating FLOAT NOT NULL CHECK (rating >= 0 AND rating <= 1),
    details JSON,
    user_id VARCHAR(100),
    session_id VARCHAR(50),
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_target (target_id),
    INDEX idx_type (feedback_type),
    INDEX idx_created (created_date),
    INDEX idx_user (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 3. Learning Patterns Table
CREATE TABLE IF NOT EXISTS megamind_learning_patterns (
    pattern_id VARCHAR(50) PRIMARY KEY,
    pattern_type ENUM('boundary', 'quality', 'chunking', 'retrieval') NOT NULL,
    pattern_data JSON NOT NULL,
    confidence FLOAT DEFAULT 0.5,
    occurrence_count INT DEFAULT 1,
    success_rate FLOAT DEFAULT 0.5,
    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_type (pattern_type),
    INDEX idx_confidence (confidence DESC),
    INDEX idx_success_rate (success_rate DESC)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 4. Adaptive Strategies Table
CREATE TABLE IF NOT EXISTS megamind_adaptive_strategies (
    strategy_id VARCHAR(50) PRIMARY KEY,
    strategy_name VARCHAR(100) NOT NULL,
    strategy_type ENUM('chunking', 'embedding', 'retrieval', 'curation') NOT NULL,
    preferred_chunk_size INT DEFAULT 512,
    boundary_patterns JSON,
    quality_weights JSON,
    confidence FLOAT DEFAULT 0.5,
    performance_metrics JSON,
    is_active BOOLEAN DEFAULT FALSE,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_type (strategy_type),
    INDEX idx_active (is_active),
    INDEX idx_confidence (confidence DESC)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 5. Curation Decisions Table
CREATE TABLE IF NOT EXISTS megamind_curation_decisions (
    decision_id VARCHAR(50) PRIMARY KEY,
    chunk_id VARCHAR(50) NOT NULL,
    action ENUM('approve', 'reject', 'improve', 'merge', 'split', 'archive', 'promote') NOT NULL,
    reason TEXT NOT NULL,
    confidence FLOAT DEFAULT 0.5,
    parameters JSON,
    workflow_id VARCHAR(50),
    automated BOOLEAN DEFAULT FALSE,
    applied BOOLEAN DEFAULT FALSE,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    applied_date TIMESTAMP NULL,
    session_id VARCHAR(50),
    INDEX idx_chunk (chunk_id),
    INDEX idx_action (action),
    INDEX idx_workflow (workflow_id),
    INDEX idx_created (created_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 6. Curation Workflows Table
CREATE TABLE IF NOT EXISTS megamind_curation_workflows (
    workflow_id VARCHAR(50) PRIMARY KEY,
    workflow_name VARCHAR(100) NOT NULL,
    workflow_type ENUM('standard_quality', 'fast_track', 'remediation', 'custom') DEFAULT 'standard_quality',
    stages JSON NOT NULL,
    rules JSON,
    is_active BOOLEAN DEFAULT TRUE,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_used TIMESTAMP NULL,
    INDEX idx_type (workflow_type),
    INDEX idx_active (is_active)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 7. Performance Metrics Table
CREATE TABLE IF NOT EXISTS megamind_performance_metrics (
    metric_id VARCHAR(50) PRIMARY KEY,
    metric_type ENUM('latency', 'throughput', 'memory_usage', 'cache_hit_rate', 'quality_score') NOT NULL,
    operation VARCHAR(100) NOT NULL,
    value FLOAT NOT NULL,
    configuration JSON,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    session_id VARCHAR(50),
    INDEX idx_type (metric_type),
    INDEX idx_operation (operation),
    INDEX idx_timestamp (timestamp)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 8. Optimization History Table
CREATE TABLE IF NOT EXISTS megamind_optimization_history (
    optimization_id VARCHAR(50) PRIMARY KEY,
    optimization_type ENUM('batch_size', 'cache_strategy', 'model_selection', 'preprocessing', 'indexing') NOT NULL,
    original_config JSON NOT NULL,
    optimized_config JSON NOT NULL,
    improvement_metrics JSON,
    confidence FLOAT DEFAULT 0.5,
    applied BOOLEAN DEFAULT FALSE,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    applied_date TIMESTAMP NULL,
    session_id VARCHAR(50),
    INDEX idx_type (optimization_type),
    INDEX idx_applied (applied),
    INDEX idx_created (created_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 9. Embedding Cache Table (for performance optimization)
CREATE TABLE IF NOT EXISTS megamind_embedding_cache (
    cache_id VARCHAR(50) PRIMARY KEY,
    chunk_id VARCHAR(50) NOT NULL,
    model VARCHAR(100) NOT NULL,
    embedding_vector JSON NOT NULL,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    access_count INT DEFAULT 1,
    expires_date TIMESTAMP NULL,
    UNIQUE KEY unique_chunk_model (chunk_id, model),
    INDEX idx_chunk (chunk_id),
    INDEX idx_model (model),
    INDEX idx_last_accessed (last_accessed),
    INDEX idx_expires (expires_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 10. AI Enhancement Reports Table
CREATE TABLE IF NOT EXISTS megamind_ai_reports (
    report_id VARCHAR(50) PRIMARY KEY,
    report_type ENUM('quality_improvement', 'learning_insights', 'curation_summary', 'optimization_report') NOT NULL,
    period_start TIMESTAMP NOT NULL,
    period_end TIMESTAMP NOT NULL,
    report_data JSON NOT NULL,
    summary TEXT,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100),
    INDEX idx_type (report_type),
    INDEX idx_period (period_start, period_end),
    INDEX idx_created (created_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 11. Create views for AI enhancement analytics

-- View: Quality improvement effectiveness
CREATE OR REPLACE VIEW megamind_quality_effectiveness_view AS
SELECT 
    qi.improvement_type,
    COUNT(*) as total_improvements,
    SUM(CASE WHEN qi.improvement_status = 'applied' THEN 1 ELSE 0 END) as applied_count,
    AVG(CASE WHEN qi.improvement_status = 'applied' THEN qi.improved_score - qi.original_score ELSE 0 END) as avg_score_increase,
    AVG(qi.confidence) as avg_confidence,
    SUM(CASE WHEN qi.automated = TRUE THEN 1 ELSE 0 END) as automated_count
FROM megamind_quality_improvements qi
GROUP BY qi.improvement_type;

-- View: Learning pattern effectiveness
CREATE OR REPLACE VIEW megamind_learning_effectiveness_view AS
SELECT 
    lp.pattern_type,
    COUNT(*) as pattern_count,
    AVG(lp.confidence) as avg_confidence,
    AVG(lp.success_rate) as avg_success_rate,
    SUM(lp.occurrence_count) as total_occurrences,
    MAX(lp.last_seen) as most_recent_use
FROM megamind_learning_patterns lp
GROUP BY lp.pattern_type;

-- View: Curation workflow performance
CREATE OR REPLACE VIEW megamind_curation_performance_view AS
SELECT 
    cw.workflow_id,
    cw.workflow_name,
    cw.workflow_type,
    COUNT(cd.decision_id) as total_decisions,
    SUM(CASE WHEN cd.action = 'approve' THEN 1 ELSE 0 END) as approvals,
    SUM(CASE WHEN cd.action = 'reject' THEN 1 ELSE 0 END) as rejections,
    SUM(CASE WHEN cd.automated = TRUE THEN 1 ELSE 0 END) as automated_decisions,
    AVG(cd.confidence) as avg_confidence
FROM megamind_curation_workflows cw
LEFT JOIN megamind_curation_decisions cd ON cw.workflow_id = cd.workflow_id
GROUP BY cw.workflow_id;

-- 12. Stored procedures for AI enhancement

DELIMITER //

-- Procedure: Apply quality improvement
CREATE PROCEDURE apply_quality_improvement(
    IN p_improvement_id VARCHAR(50),
    IN p_session_id VARCHAR(50)
)
BEGIN
    DECLARE v_chunk_id VARCHAR(50);
    DECLARE v_implementation TEXT;
    
    -- Get improvement details
    SELECT chunk_id, implementation INTO v_chunk_id, v_implementation
    FROM megamind_quality_improvements
    WHERE improvement_id = p_improvement_id AND improvement_status = 'suggested';
    
    IF v_chunk_id IS NOT NULL AND v_implementation IS NOT NULL THEN
        -- Update chunk content (simplified - in practice would apply the implementation)
        UPDATE megamind_knowledge_chunks
        SET quality_score = quality_score + 0.1,
            last_improved = CURRENT_TIMESTAMP
        WHERE chunk_id = v_chunk_id;
        
        -- Update improvement status
        UPDATE megamind_quality_improvements
        SET improvement_status = 'applied',
            applied_date = CURRENT_TIMESTAMP,
            session_id = p_session_id
        WHERE improvement_id = p_improvement_id;
    END IF;
END//

-- Procedure: Record user feedback and trigger learning
CREATE PROCEDURE record_feedback_and_learn(
    IN p_feedback_type VARCHAR(50),
    IN p_target_id VARCHAR(50),
    IN p_rating FLOAT,
    IN p_details JSON,
    IN p_user_id VARCHAR(100),
    IN p_session_id VARCHAR(50)
)
BEGIN
    DECLARE v_feedback_id VARCHAR(50);
    
    -- Generate feedback ID
    SET v_feedback_id = CONCAT('fb_', MD5(CONCAT(p_target_id, NOW(), RAND())));
    
    -- Insert feedback
    INSERT INTO megamind_user_feedback (
        feedback_id, feedback_type, target_id, rating, details, user_id, session_id
    ) VALUES (
        v_feedback_id, p_feedback_type, p_target_id, p_rating, p_details, p_user_id, p_session_id
    );
    
    -- Trigger learning if threshold reached
    IF (SELECT COUNT(*) FROM megamind_user_feedback WHERE feedback_type = p_feedback_type) % 10 = 0 THEN
        -- In practice, this would trigger the adaptive learning engine
        INSERT INTO megamind_ai_reports (report_id, report_type, period_start, period_end, report_data, summary)
        VALUES (
            CONCAT('report_', MD5(NOW())),
            'learning_insights',
            DATE_SUB(NOW(), INTERVAL 1 DAY),
            NOW(),
            JSON_OBJECT('trigger', 'feedback_threshold', 'type', p_feedback_type),
            'Learning triggered by feedback threshold'
        );
    END IF;
END//

DELIMITER ;

-- 13. Indexes for performance
CREATE INDEX idx_quality_chunk_status ON megamind_quality_improvements(chunk_id, improvement_status);
CREATE INDEX idx_feedback_rating ON megamind_user_feedback(rating);
CREATE INDEX idx_pattern_effectiveness ON megamind_learning_patterns(pattern_type, success_rate DESC);
CREATE INDEX idx_cache_lookup ON megamind_embedding_cache(chunk_id, model, last_accessed DESC);

-- 14. Initial configuration for Phase 4
INSERT INTO megamind_system_config (config_key, config_value, description) VALUES
    ('phase4.quality.auto_improvement_threshold', '0.8', 'Confidence threshold for automated improvements'),
    ('phase4.learning.min_feedback_count', '10', 'Minimum feedback count before learning triggers'),
    ('phase4.curation.default_workflow', 'standard_quality', 'Default curation workflow'),
    ('phase4.optimization.cache_size', '10000', 'Maximum embedding cache size'),
    ('phase4.optimization.batch_size', '32', 'Default batch size for embeddings'),
    ('phase4.ai.quality_threshold', '0.7', 'Minimum quality score threshold')
ON DUPLICATE KEY UPDATE 
    config_value = VALUES(config_value),
    updated_at = CURRENT_TIMESTAMP;

-- 15. Insert default workflows
INSERT INTO megamind_curation_workflows (workflow_id, workflow_name, workflow_type, stages) VALUES
    ('wf_standard', 'Standard Quality Workflow', 'standard_quality', 
     JSON_ARRAY('intake', 'quality_check', 'improvement', 'review', 'approval', 'deployment')),
    ('wf_fast', 'Fast Track Workflow', 'fast_track',
     JSON_ARRAY('intake', 'quality_check', 'approval', 'deployment')),
    ('wf_remediation', 'Quality Remediation Workflow', 'remediation',
     JSON_ARRAY('intake', 'quality_check', 'improvement', 'improvement', 'review', 'approval'))
ON DUPLICATE KEY UPDATE last_used = CURRENT_TIMESTAMP;