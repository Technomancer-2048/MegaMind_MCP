#!/bin/bash
# Schema Migration Script for GitHub Issue #5
# Migrates existing MegaMind databases to complete schema

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
DB_HOST="${MEGAMIND_DB_HOST:-10.255.250.22}"
DB_PORT="${MEGAMIND_DB_PORT:-3309}"
DB_NAME="${MEGAMIND_DB_NAME:-megamind_database}"
DB_USER="${MEGAMIND_DB_USER:-megamind_user}"
DB_PASSWORD="${MEGAMIND_DB_PASSWORD}"
BACKUP_DIR="./backups"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if MySQL is accessible
check_mysql_connection() {
    print_status "Checking MySQL connection..."
    if ! mysql -h "$DB_HOST" -P "$DB_PORT" -u "$DB_USER" -p"$DB_PASSWORD" -e "SELECT 1;" > /dev/null 2>&1; then
        print_error "Cannot connect to MySQL database"
        print_error "Host: $DB_HOST:$DB_PORT, User: $DB_USER, Database: $DB_NAME"
        exit 1
    fi
    print_success "MySQL connection established"
}

# Function to create backup
create_backup() {
    print_status "Creating database backup..."
    mkdir -p "$BACKUP_DIR"
    BACKUP_FILE="$BACKUP_DIR/megamind_backup_$(date +%Y%m%d_%H%M%S).sql"
    
    if mysqldump -h "$DB_HOST" -P "$DB_PORT" -u "$DB_USER" -p"$DB_PASSWORD" \
        --single-transaction --routines --triggers "$DB_NAME" > "$BACKUP_FILE" 2>/dev/null; then
        print_success "Backup created: $BACKUP_FILE"
    else
        print_error "Failed to create backup"
        exit 1
    fi
}

# Function to check existing schema
check_existing_schema() {
    print_status "Analyzing existing schema..."
    
    # Get list of existing tables
    EXISTING_TABLES=$(mysql -h "$DB_HOST" -P "$DB_PORT" -u "$DB_USER" -p"$DB_PASSWORD" \
        -D "$DB_NAME" -se "SHOW TABLES LIKE 'megamind_%';" 2>/dev/null | wc -l)
    
    print_status "Found $EXISTING_TABLES MegaMind tables"
    
    # Check for specific missing elements
    MISSING_ELEMENTS=()
    
    # Check for megamind_realms table
    if ! mysql -h "$DB_HOST" -P "$DB_PORT" -u "$DB_USER" -p"$DB_PASSWORD" \
        -D "$DB_NAME" -se "SHOW TABLES LIKE 'megamind_realms';" > /dev/null 2>&1; then
        MISSING_ELEMENTS+=("megamind_realms table")
    fi
    
    # Check for megamind_realm_inheritance table
    if ! mysql -h "$DB_HOST" -P "$DB_PORT" -u "$DB_USER" -p"$DB_PASSWORD" \
        -D "$DB_NAME" -se "SHOW TABLES LIKE 'megamind_realm_inheritance';" > /dev/null 2>&1; then
        MISSING_ELEMENTS+=("megamind_realm_inheritance table")
    fi
    
    # Check for is_cross_realm column
    if ! mysql -h "$DB_HOST" -P "$DB_PORT" -u "$DB_USER" -p"$DB_PASSWORD" \
        -D "$DB_NAME" -se "SHOW COLUMNS FROM megamind_chunk_relationships LIKE 'is_cross_realm';" > /dev/null 2>&1; then
        MISSING_ELEMENTS+=("is_cross_realm column in megamind_chunk_relationships")
    fi
    
    if [ ${#MISSING_ELEMENTS[@]} -eq 0 ]; then
        print_success "Schema appears to be complete"
        return 0
    else
        print_warning "Missing schema elements found:"
        for element in "${MISSING_ELEMENTS[@]}"; do
            echo "  - $element"
        done
        return 1
    fi
}

# Function to apply missing schema elements
apply_schema_migration() {
    print_status "Applying schema migration..."
    
    # Create migration SQL
    MIGRATION_SQL="/tmp/megamind_migration.sql"
    cat > "$MIGRATION_SQL" << 'EOF'
USE megamind_database;

-- Add megamind_realms table if missing
CREATE TABLE IF NOT EXISTS megamind_realms (
    realm_id VARCHAR(50) PRIMARY KEY,
    realm_name VARCHAR(255) NOT NULL,
    realm_type ENUM('global', 'project', 'team', 'personal') NOT NULL,
    parent_realm_id VARCHAR(50) DEFAULT NULL,
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    realm_config JSON DEFAULT NULL,
    created_by VARCHAR(100) DEFAULT 'system',
    access_level ENUM('read_only', 'read_write', 'admin') DEFAULT 'read_write',
    INDEX idx_realms_type (realm_type, is_active),
    INDEX idx_realms_active (is_active, created_at DESC),
    FOREIGN KEY (parent_realm_id) REFERENCES megamind_realms(realm_id) ON DELETE SET NULL
) ENGINE=InnoDB;

-- Add megamind_realm_inheritance table if missing
CREATE TABLE IF NOT EXISTS megamind_realm_inheritance (
    inheritance_id INT PRIMARY KEY AUTO_INCREMENT,
    child_realm_id VARCHAR(50) NOT NULL,
    parent_realm_id VARCHAR(50) NOT NULL,
    inheritance_type ENUM('full', 'selective', 'read_only') NOT NULL DEFAULT 'full',
    priority_order INT DEFAULT 1,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    inheritance_config JSON DEFAULT NULL,
    INDEX idx_inheritance_child (child_realm_id),
    INDEX idx_inheritance_parent (parent_realm_id),
    FOREIGN KEY (child_realm_id) REFERENCES megamind_realms(realm_id) ON DELETE CASCADE,
    FOREIGN KEY (parent_realm_id) REFERENCES megamind_realms(realm_id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- Add default realms
INSERT IGNORE INTO megamind_realms (realm_id, realm_name, realm_type, description, created_by) VALUES
('GLOBAL', 'Global Knowledge Base', 'global', 'Universal knowledge repository - primary source for all queries', 'system'),
('PROJECT', 'Default Project Realm', 'project', 'Project-specific knowledge realm - secondary to GLOBAL', 'system');

-- Add is_cross_realm column if missing
SET @sql = (
    SELECT IF(
        COUNT(*) = 0,
        'ALTER TABLE megamind_chunk_relationships ADD COLUMN is_cross_realm BOOLEAN DEFAULT FALSE;',
        'SELECT "is_cross_realm column already exists" as status;'
    )
    FROM information_schema.columns 
    WHERE table_schema = 'megamind_database' 
    AND table_name = 'megamind_chunk_relationships' 
    AND column_name = 'is_cross_realm'
);

PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- Add missing indexes if needed
ALTER TABLE megamind_chunk_relationships ADD INDEX IF NOT EXISTS idx_cross_realm (is_cross_realm);

-- Add individual FULLTEXT index on content if missing
SET @sql = (
    SELECT IF(
        COUNT(*) = 0,
        'ALTER TABLE megamind_chunks ADD FULLTEXT(content);',
        'SELECT "content FULLTEXT index already exists" as status;'
    )
    FROM information_schema.statistics 
    WHERE table_schema = 'megamind_database' 
    AND table_name = 'megamind_chunks' 
    AND index_name = 'content'
    AND index_type = 'FULLTEXT'
);

PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;
EOF

    # Apply migration
    if mysql -h "$DB_HOST" -P "$DB_PORT" -u "$DB_USER" -p"$DB_PASSWORD" < "$MIGRATION_SQL" 2>/dev/null; then
        print_success "Schema migration applied successfully"
        rm -f "$MIGRATION_SQL"
    else
        print_error "Failed to apply schema migration"
        print_error "Migration SQL saved at: $MIGRATION_SQL"
        exit 1
    fi
}

# Function to validate migration
validate_migration() {
    print_status "Validating migration..."
    
    # Check table count
    TABLE_COUNT=$(mysql -h "$DB_HOST" -P "$DB_PORT" -u "$DB_USER" -p"$DB_PASSWORD" \
        -D "$DB_NAME" -se "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'megamind_database' AND table_name LIKE 'megamind_%';" 2>/dev/null)
    
    print_status "Total MegaMind tables: $TABLE_COUNT"
    
    # Check sample data
    SAMPLE_COUNT=$(mysql -h "$DB_HOST" -P "$DB_PORT" -u "$DB_USER" -p"$DB_PASSWORD" \
        -D "$DB_NAME" -se "SELECT COUNT(*) FROM megamind_chunks WHERE realm_id = 'GLOBAL';" 2>/dev/null)
    
    print_status "Sample chunks in GLOBAL realm: $SAMPLE_COUNT"
    
    # Final schema completeness check
    if check_existing_schema; then
        print_success "Migration validation passed"
        return 0
    else
        print_warning "Migration validation found remaining issues"
        return 1
    fi
}

# Main execution
main() {
    echo "============================================"
    echo "MegaMind Database Schema Migration"
    echo "GitHub Issue #5: Database Schema Completeness"
    echo "============================================"
    echo
    
    # Check if password is provided
    if [ -z "$DB_PASSWORD" ]; then
        print_error "Database password not provided"
        print_error "Set MEGAMIND_DB_PASSWORD environment variable or pass via .env file"
        exit 1
    fi
    
    # Load environment if .env exists
    if [ -f .env ]; then
        print_status "Loading environment from .env file"
        source .env
    fi
    
    print_status "Configuration:"
    echo "  Host: $DB_HOST:$DB_PORT"
    echo "  Database: $DB_NAME"
    echo "  User: $DB_USER"
    echo
    
    # Execute migration steps
    check_mysql_connection
    
    if [ "$1" != "--no-backup" ]; then
        create_backup
    else
        print_warning "Skipping backup (--no-backup flag provided)"
    fi
    
    if check_existing_schema; then
        print_success "Schema is already complete - no migration needed"
        exit 0
    fi
    
    apply_schema_migration
    validate_migration
    
    print_success "Schema migration completed successfully!"
    echo
    print_status "Next steps:"
    echo "  1. Restart MCP server containers to use updated schema"
    echo "  2. Test chunk retrieval functionality"
    echo "  3. Verify GLOBAL realm search prioritization"
}

# Handle command line arguments
case "${1:-}" in
    -h|--help)
        echo "Usage: $0 [OPTIONS]"
        echo
        echo "Options:"
        echo "  --no-backup    Skip database backup"
        echo "  -h, --help     Show this help message"
        echo
        echo "Environment variables:"
        echo "  MEGAMIND_DB_HOST      Database host (default: 10.255.250.22)"
        echo "  MEGAMIND_DB_PORT      Database port (default: 3309)"
        echo "  MEGAMIND_DB_NAME      Database name (default: megamind_database)"
        echo "  MEGAMIND_DB_USER      Database user (default: megamind_user)"
        echo "  MEGAMIND_DB_PASSWORD  Database password (required)"
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac