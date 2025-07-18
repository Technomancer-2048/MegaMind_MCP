-- Migration 026: Add chunk approval status tracking
-- GitHub Issue #26: Decouple chunk approval from session system
-- This migration adds approval status directly to megamind_chunks table

USE megamind_database;

-- Add approval status columns to megamind_chunks table
ALTER TABLE megamind_chunks 
ADD COLUMN approval_status ENUM('pending', 'approved', 'rejected') DEFAULT 'pending' AFTER access_count,
ADD COLUMN approved_at TIMESTAMP NULL AFTER approval_status,
ADD COLUMN approved_by VARCHAR(100) NULL AFTER approved_at,
ADD COLUMN rejection_reason TEXT NULL AFTER approved_by;

-- Add indexes for efficient approval status queries
ALTER TABLE megamind_chunks 
ADD INDEX idx_approval_status (approval_status, created_at DESC),
ADD INDEX idx_approval_status_realm (approval_status, realm_id),
ADD INDEX idx_approved_by (approved_by);

-- Migrate existing chunks to 'approved' status for backwards compatibility
-- This ensures all existing chunks remain accessible after migration
UPDATE megamind_chunks 
SET approval_status = 'approved', 
    approved_at = created_at,
    approved_by = 'system_migration'
WHERE approval_status = 'pending';

-- Verify migration results
SELECT 
    approval_status,
    COUNT(*) as chunk_count,
    MIN(created_at) as oldest_chunk,
    MAX(created_at) as newest_chunk
FROM megamind_chunks 
GROUP BY approval_status
ORDER BY approval_status;

-- Show index creation results
SHOW INDEX FROM megamind_chunks WHERE Key_name LIKE 'idx_approval%';