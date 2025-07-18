-- Rollback Migration 026: Remove chunk approval status tracking
-- GitHub Issue #26: Rollback script to remove approval status columns
-- Use this script to revert changes if needed

USE megamind_database;

-- Remove approval status indexes
ALTER TABLE megamind_chunks 
DROP INDEX IF EXISTS idx_approval_status,
DROP INDEX IF EXISTS idx_approval_status_realm,
DROP INDEX IF EXISTS idx_approved_by;

-- Remove approval status columns
ALTER TABLE megamind_chunks 
DROP COLUMN IF EXISTS rejection_reason,
DROP COLUMN IF EXISTS approved_by,
DROP COLUMN IF EXISTS approved_at,
DROP COLUMN IF EXISTS approval_status;

-- Verify rollback results
DESCRIBE megamind_chunks;

-- Show remaining indexes
SHOW INDEX FROM megamind_chunks;