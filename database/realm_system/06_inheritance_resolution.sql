-- Phase 2: Inheritance Resolution System
-- Virtual Views and Functions for Realm Inheritance
-- Database: megamind_database (MySQL 8.0+)

-- Enhanced virtual view for realm-aware chunk access with inheritance
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
    c.embedding,
    c.token_count,
    c.complexity_score,
    r.realm_name AS source_realm_name,
    r.realm_type AS source_realm_type,
    CASE 
        WHEN c.realm_id = @current_realm THEN 'direct'
        WHEN c.realm_id = 'GLOBAL' THEN 'inherited_global'
        ELSE 'inherited_project'
    END AS access_type,
    CASE 
        WHEN c.realm_id = @current_realm THEN 1.0
        WHEN c.realm_id = 'GLOBAL' THEN 0.8
        ELSE 0.6
    END AS inheritance_weight,
    -- Selective inheritance filtering
    CASE 
        WHEN c.realm_id = @current_realm THEN TRUE
        WHEN ri.inheritance_type = 'full' THEN TRUE
        WHEN ri.inheritance_type = 'selective' THEN 
            (SELECT _check_selective_inheritance(c.chunk_id, ri.inheritance_config))
        ELSE FALSE
    END AS is_accessible
FROM megamind_chunks c
JOIN megamind_realms r ON c.realm_id = r.realm_id
LEFT JOIN megamind_realm_inheritance ri ON (
    ri.child_realm_id = @current_realm 
    AND ri.parent_realm_id = c.realm_id
    AND ri.is_active = TRUE
)
WHERE c.realm_id = @current_realm
   OR c.realm_id = 'GLOBAL'
   OR c.realm_id IN (
       SELECT ri2.parent_realm_id 
       FROM megamind_realm_inheritance ri2 
       WHERE ri2.child_realm_id = @current_realm 
         AND ri2.inheritance_type IN ('full', 'selective')
         AND ri2.is_active = TRUE
   );

-- View for inheritance chain analysis
CREATE VIEW megamind_inheritance_chains AS
SELECT 
    ri.child_realm_id,
    ri.parent_realm_id,
    ri.inheritance_type,
    ri.priority_order,
    cr.realm_name as child_realm_name,
    pr.realm_name as parent_realm_name,
    cr.realm_type as child_realm_type,
    pr.realm_type as parent_realm_type,
    ri.inheritance_config,
    CASE 
        WHEN ri.inheritance_type = 'full' THEN 'All content inherited'
        WHEN ri.inheritance_type = 'selective' THEN 'Filtered content inherited'
        ELSE 'Read-only access'
    END AS inheritance_description
FROM megamind_realm_inheritance ri
JOIN megamind_realms cr ON ri.child_realm_id = cr.realm_id
JOIN megamind_realms pr ON ri.parent_realm_id = pr.realm_id
WHERE ri.is_active = TRUE
ORDER BY ri.child_realm_id, ri.priority_order;

-- View for realm accessibility matrix
CREATE VIEW megamind_realm_accessibility AS
SELECT 
    child.realm_id as accessing_realm,
    child.realm_name as accessing_realm_name,
    target.realm_id as target_realm,
    target.realm_name as target_realm_name,
    CASE 
        WHEN child.realm_id = target.realm_id THEN 'direct'
        WHEN target.realm_id = 'GLOBAL' THEN 'global_inheritance'
        WHEN EXISTS (
            SELECT 1 FROM megamind_realm_inheritance ri 
            WHERE ri.child_realm_id = child.realm_id 
              AND ri.parent_realm_id = target.realm_id
              AND ri.is_active = TRUE
        ) THEN 'explicit_inheritance'
        ELSE 'no_access'
    END AS access_type,
    COALESCE(ri.inheritance_type, 'none') as inheritance_type,
    COALESCE(ri.priority_order, 999) as priority_order
FROM megamind_realms child
CROSS JOIN megamind_realms target
LEFT JOIN megamind_realm_inheritance ri ON (
    ri.child_realm_id = child.realm_id 
    AND ri.parent_realm_id = target.realm_id
    AND ri.is_active = TRUE
)
WHERE child.is_active = TRUE AND target.is_active = TRUE;

-- Function to check selective inheritance based on configuration
DELIMITER //
CREATE FUNCTION _check_selective_inheritance(p_chunk_id VARCHAR(50), p_config JSON)
RETURNS BOOLEAN
READS SQL DATA
DETERMINISTIC
BEGIN
    DECLARE v_include_tags JSON DEFAULT NULL;
    DECLARE v_exclude_tags JSON DEFAULT NULL;
    DECLARE v_include_types JSON DEFAULT NULL;
    DECLARE v_exclude_types JSON DEFAULT NULL;
    DECLARE v_result BOOLEAN DEFAULT TRUE;
    DECLARE v_tag_count INT DEFAULT 0;
    DECLARE v_type_match INT DEFAULT 0;
    
    -- Handle NULL config (default to TRUE for backward compatibility)
    IF p_config IS NULL THEN
        RETURN TRUE;
    END IF;
    
    -- Extract configuration parameters
    SET v_include_tags = JSON_EXTRACT(p_config, '$.include_tags');
    SET v_exclude_tags = JSON_EXTRACT(p_config, '$.exclude_tags');
    SET v_include_types = JSON_EXTRACT(p_config, '$.include_types');
    SET v_exclude_types = JSON_EXTRACT(p_config, '$.exclude_types');
    
    -- Check exclude tags first (if chunk has any excluded tag, reject)
    IF v_exclude_tags IS NOT NULL THEN
        SELECT COUNT(*) INTO v_tag_count
        FROM megamind_chunk_tags ct
        WHERE ct.chunk_id = p_chunk_id
          AND JSON_CONTAINS(v_exclude_tags, JSON_QUOTE(ct.tag_value));
        
        IF v_tag_count > 0 THEN
            RETURN FALSE;
        END IF;
    END IF;
    
    -- Check exclude types (if chunk has excluded type, reject)
    IF v_exclude_types IS NOT NULL THEN
        SELECT COUNT(*) INTO v_type_match
        FROM megamind_chunks c
        WHERE c.chunk_id = p_chunk_id
          AND JSON_CONTAINS(v_exclude_types, JSON_QUOTE(c.chunk_type));
        
        IF v_type_match > 0 THEN
            RETURN FALSE;
        END IF;
    END IF;
    
    -- Check include tags (if specified, chunk must have at least one)
    IF v_include_tags IS NOT NULL THEN
        SELECT COUNT(*) INTO v_tag_count
        FROM megamind_chunk_tags ct
        WHERE ct.chunk_id = p_chunk_id
          AND JSON_CONTAINS(v_include_tags, JSON_QUOTE(ct.tag_value));
        
        IF v_tag_count = 0 THEN
            RETURN FALSE;
        END IF;
    END IF;
    
    -- Check include types (if specified, chunk type must match)
    IF v_include_types IS NOT NULL THEN
        SELECT COUNT(*) INTO v_type_match
        FROM megamind_chunks c
        WHERE c.chunk_id = p_chunk_id
          AND JSON_CONTAINS(v_include_types, JSON_QUOTE(c.chunk_type));
        
        IF v_type_match = 0 THEN
            RETURN FALSE;
        END IF;
    END IF;
    
    RETURN v_result;
END //
DELIMITER ;

-- Function to get effective chunks for a realm with inheritance
DELIMITER //
CREATE FUNCTION get_realm_chunks(p_realm_id VARCHAR(50))
RETURNS JSON
READS SQL DATA
DETERMINISTIC
BEGIN
    DECLARE result JSON;
    
    -- Set session variable for the view
    SET @current_realm = p_realm_id;
    
    SELECT JSON_ARRAYAGG(
        JSON_OBJECT(
            'chunk_id', chunk_id,
            'content', content,
            'source_realm_id', source_realm_id,
            'access_type', access_type,
            'inheritance_weight', inheritance_weight,
            'source_realm_name', source_realm_name,
            'chunk_type', chunk_type,
            'access_count', access_count
        )
    ) INTO result
    FROM megamind_chunks_with_inheritance
    WHERE is_accessible = TRUE
    ORDER BY inheritance_weight DESC, access_count DESC
    LIMIT 1000;  -- Reasonable limit for JSON function
    
    RETURN result;
END //
DELIMITER ;

-- Function to resolve inheritance conflicts
DELIMITER //
CREATE FUNCTION resolve_inheritance_conflict(
    p_chunk_id VARCHAR(50), 
    p_accessing_realm VARCHAR(50)
) 
RETURNS JSON
READS SQL DATA
DETERMINISTIC
BEGIN
    DECLARE result JSON;
    DECLARE v_direct_access BOOLEAN DEFAULT FALSE;
    DECLARE v_highest_priority INT DEFAULT 999;
    DECLARE v_winning_realm VARCHAR(50) DEFAULT NULL;
    
    -- Check for direct access first
    SELECT COUNT(*) > 0 INTO v_direct_access
    FROM megamind_chunks c
    WHERE c.chunk_id = p_chunk_id AND c.realm_id = p_accessing_realm;
    
    IF v_direct_access THEN
        SET result = JSON_OBJECT(
            'access_granted', TRUE,
            'access_type', 'direct',
            'source_realm', p_accessing_realm,
            'reason', 'Direct access to own realm'
        );
        RETURN result;
    END IF;
    
    -- Find highest priority inheritance path
    SELECT MIN(ri.priority_order), ri.parent_realm_id 
    INTO v_highest_priority, v_winning_realm
    FROM megamind_chunks c
    JOIN megamind_realm_inheritance ri ON ri.parent_realm_id = c.realm_id
    WHERE c.chunk_id = p_chunk_id 
      AND ri.child_realm_id = p_accessing_realm
      AND ri.is_active = TRUE;
    
    IF v_winning_realm IS NOT NULL THEN
        SET result = JSON_OBJECT(
            'access_granted', TRUE,
            'access_type', 'inherited',
            'source_realm', v_winning_realm,
            'priority_order', v_highest_priority,
            'reason', 'Inherited through highest priority path'
        );
    ELSE
        SET result = JSON_OBJECT(
            'access_granted', FALSE,
            'access_type', 'denied',
            'reason', 'No inheritance path found'
        );
    END IF;
    
    RETURN result;
END //
DELIMITER ;

-- Function to get inheritance path for a chunk
DELIMITER //
CREATE FUNCTION get_inheritance_path(p_chunk_id VARCHAR(50), p_accessing_realm VARCHAR(50))
RETURNS JSON
READS SQL DATA
DETERMINISTIC
BEGIN
    DECLARE result JSON;
    
    SELECT JSON_ARRAYAGG(
        JSON_OBJECT(
            'realm_id', r.realm_id,
            'realm_name', r.realm_name,
            'realm_type', r.realm_type,
            'inheritance_type', COALESCE(ri.inheritance_type, 'direct'),
            'priority_order', COALESCE(ri.priority_order, 0),
            'step', ROW_NUMBER() OVER (ORDER BY COALESCE(ri.priority_order, 0))
        )
    ) INTO result
    FROM megamind_chunks c
    JOIN megamind_realms r ON c.realm_id = r.realm_id
    LEFT JOIN megamind_realm_inheritance ri ON (
        ri.parent_realm_id = c.realm_id 
        AND ri.child_realm_id = p_accessing_realm
    )
    WHERE c.chunk_id = p_chunk_id
      AND (c.realm_id = p_accessing_realm 
           OR ri.child_realm_id = p_accessing_realm
           OR c.realm_id = 'GLOBAL');
    
    RETURN result;
END //
DELIMITER ;

-- Procedure to validate inheritance configuration
DELIMITER //
CREATE PROCEDURE validate_inheritance_configuration(
    IN p_child_realm VARCHAR(50),
    IN p_parent_realm VARCHAR(50),
    IN p_inheritance_type VARCHAR(20),
    OUT p_is_valid BOOLEAN,
    OUT p_error_message TEXT
)
BEGIN
    DECLARE v_circular_check INT DEFAULT 0;
    DECLARE v_realm_exists INT DEFAULT 0;
    DECLARE v_already_exists INT DEFAULT 0;
    
    SET p_is_valid = TRUE;
    SET p_error_message = '';
    
    -- Check if realms exist
    SELECT COUNT(*) INTO v_realm_exists
    FROM megamind_realms 
    WHERE realm_id IN (p_child_realm, p_parent_realm) AND is_active = TRUE;
    
    IF v_realm_exists < 2 THEN
        SET p_is_valid = FALSE;
        SET p_error_message = 'One or both realms do not exist or are inactive';
        LEAVE validate_inheritance_configuration;
    END IF;
    
    -- Check for circular inheritance
    SELECT COUNT(*) INTO v_circular_check
    FROM megamind_realm_inheritance
    WHERE (child_realm_id = p_parent_realm AND parent_realm_id = p_child_realm)
       OR (child_realm_id = p_child_realm AND parent_realm_id = p_parent_realm);
    
    IF v_circular_check > 0 THEN
        SET p_is_valid = FALSE;
        SET p_error_message = 'Circular inheritance detected';
        LEAVE validate_inheritance_configuration;
    END IF;
    
    -- Check if inheritance already exists
    SELECT COUNT(*) INTO v_already_exists
    FROM megamind_realm_inheritance
    WHERE child_realm_id = p_child_realm 
      AND parent_realm_id = p_parent_realm
      AND is_active = TRUE;
    
    IF v_already_exists > 0 THEN
        SET p_is_valid = FALSE;
        SET p_error_message = 'Inheritance relationship already exists';
        LEAVE validate_inheritance_configuration;
    END IF;
    
    -- Additional validation for inheritance type
    IF p_inheritance_type NOT IN ('full', 'selective', 'read_only') THEN
        SET p_is_valid = FALSE;
        SET p_error_message = 'Invalid inheritance type';
        LEAVE validate_inheritance_configuration;
    END IF;
    
END //
DELIMITER ;

-- Procedure to create inheritance relationship with validation
DELIMITER //
CREATE PROCEDURE create_inheritance_relationship(
    IN p_child_realm VARCHAR(50),
    IN p_parent_realm VARCHAR(50),
    IN p_inheritance_type VARCHAR(20),
    IN p_inheritance_config JSON,
    IN p_priority_order INT,
    OUT p_success BOOLEAN,
    OUT p_message TEXT
)
BEGIN
    DECLARE v_is_valid BOOLEAN DEFAULT FALSE;
    DECLARE v_error_message TEXT DEFAULT '';
    
    -- Validate the inheritance configuration
    CALL validate_inheritance_configuration(p_child_realm, p_parent_realm, p_inheritance_type, v_is_valid, v_error_message);
    
    IF NOT v_is_valid THEN
        SET p_success = FALSE;
        SET p_message = v_error_message;
        LEAVE create_inheritance_relationship;
    END IF;
    
    -- Create the inheritance relationship
    INSERT INTO megamind_realm_inheritance (
        child_realm_id, 
        parent_realm_id, 
        inheritance_type, 
        priority_order, 
        inheritance_config,
        is_active
    ) VALUES (
        p_child_realm, 
        p_parent_realm, 
        p_inheritance_type, 
        COALESCE(p_priority_order, 1),
        p_inheritance_config,
        TRUE
    );
    
    SET p_success = TRUE;
    SET p_message = 'Inheritance relationship created successfully';
    
END //
DELIMITER ;

-- Add indexes for inheritance performance
CREATE INDEX idx_inheritance_resolution ON megamind_realm_inheritance (child_realm_id, priority_order, is_active);
CREATE INDEX idx_inheritance_parent_lookup ON megamind_realm_inheritance (parent_realm_id, inheritance_type, is_active);
CREATE INDEX idx_chunks_inheritance_lookup ON megamind_chunks (realm_id, chunk_type, access_count DESC);

-- View for inheritance performance monitoring
CREATE VIEW inheritance_performance_stats AS
SELECT 
    ri.child_realm_id,
    ri.parent_realm_id,
    ri.inheritance_type,
    COUNT(c.chunk_id) as inherited_chunks,
    AVG(c.access_count) as avg_access_count,
    SUM(CASE WHEN _check_selective_inheritance(c.chunk_id, ri.inheritance_config) THEN 1 ELSE 0 END) as accessible_chunks,
    (SUM(CASE WHEN _check_selective_inheritance(c.chunk_id, ri.inheritance_config) THEN 1 ELSE 0 END) / COUNT(c.chunk_id)) * 100 as accessibility_percentage
FROM megamind_realm_inheritance ri
JOIN megamind_chunks c ON c.realm_id = ri.parent_realm_id
WHERE ri.is_active = TRUE
GROUP BY ri.child_realm_id, ri.parent_realm_id, ri.inheritance_type;