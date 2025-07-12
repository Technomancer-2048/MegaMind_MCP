#!/usr/bin/env python3
"""
Database initialization script for MegaMind MCP Server
Ensures all required tables exist before starting the server
"""

import os
import time
import logging
import mysql.connector
from mysql.connector import pooling

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def wait_for_database(config, max_retries=30, delay=2):
    """Wait for database to be available"""
    for attempt in range(max_retries):
        try:
            connection = mysql.connector.connect(
                host=config['host'],
                port=config['port'],
                user=config['user'],
                password=config['password']
            )
            connection.close()
            logger.info("Database is available")
            return True
        except mysql.connector.Error as e:
            logger.info(f"Waiting for database... (attempt {attempt + 1}/{max_retries})")
            time.sleep(delay)
    
    logger.error("Database not available after maximum retries")
    return False

def create_database_if_not_exists(config):
    """Create the megamind_database if it doesn't exist"""
    try:
        # Connect without specifying database
        connection = mysql.connector.connect(
            host=config['host'],
            port=config['port'],
            user=config['user'],
            password=config['password']
        )
        cursor = connection.cursor()
        
        # Create database if not exists
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {config['database']}")
        logger.info(f"Database '{config['database']}' ensured to exist")
        
        cursor.close()
        connection.close()
        return True
        
    except Exception as e:
        logger.error(f"Failed to create database: {e}")
        return False

def execute_sql_file(cursor, file_path):
    """Execute SQL statements from a file"""
    try:
        with open(file_path, 'r') as f:
            sql_content = f.read()
        
        # Remove comments and clean up content
        lines = sql_content.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # Skip empty lines and comment lines
            if line and not line.startswith('--') and not line.startswith('#'):
                cleaned_lines.append(line)
        
        sql_content = '\n'.join(cleaned_lines)
        
        # Handle DELIMITER statements for stored procedures/functions
        if 'DELIMITER' in sql_content:
            # Split by DELIMITER statements
            delimiter_sections = sql_content.split('DELIMITER')
            current_delimiter = ';'
            
            for section in delimiter_sections:
                section = section.strip()
                if not section:
                    continue
                
                lines = section.split('\n')
                if lines and lines[0].strip() in ['//', '$$', '|']:
                    # This is a delimiter change
                    current_delimiter = lines[0].strip()
                    section = '\n'.join(lines[1:])
                
                if section.strip():
                    # Split by current delimiter
                    statements = [stmt.strip() for stmt in section.split(current_delimiter) if stmt.strip()]
                    for statement in statements:
                        if statement:
                            logger.debug(f"Executing statement: {statement[:100]}...")
                            cursor.execute(statement)
        else:
            # Regular SQL file without stored procedures
            statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]
            for statement in statements:
                if statement:
                    logger.debug(f"Executing statement: {statement[:100]}...")
                    cursor.execute(statement)
        
        logger.info(f"Successfully executed SQL file: {os.path.basename(file_path)}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to execute SQL file {file_path}: {e}")
        logger.error(f"Error context: {str(e)}")
        return False

def initialize_schema(config):
    """Initialize database schema from SQL files"""
    try:
        # Connect to the specific database
        connection = mysql.connector.connect(**config)
        cursor = connection.cursor()
        
        # Get the database directory path (in container it's /app/database)
        database_dir = os.path.join('/app', 'database')
        
        # Execute schema files in order
        schema_files = [
            # Core tables first
            os.path.join(os.path.dirname(__file__), 'init_schema.sql'),
            
            # Note: Realm system tables will be added later when needed
            # For now, we just have the core tables and a simple stored function
        ]
        
        for schema_file in schema_files:
            if os.path.exists(schema_file):
                logger.info(f"Executing schema file: {os.path.basename(schema_file)}")
                if not execute_sql_file(cursor, schema_file):
                    logger.error(f"Failed to execute {schema_file}")
                    return False
                connection.commit()
            else:
                logger.warning(f"Schema file not found: {schema_file}")
        
        # Add missing realm tables that the server needs
        logger.info("Adding missing realm tables...")
        try:
            # Create basic realm tables
            realm_tables_sql = """
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
                INDEX idx_realms_active (is_active, created_at DESC)
            ) ENGINE=InnoDB;
            
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
                INDEX idx_inheritance_parent (parent_realm_id)
            ) ENGINE=InnoDB;
            
            INSERT IGNORE INTO megamind_realms (realm_id, realm_name, realm_type, description, created_by) VALUES
            ('GLOBAL', 'Global Knowledge Base', 'global', 'Shared knowledge accessible to all realms', 'system'),
            ('PROJ_ECOMMERCE', 'E-Commerce Platform', 'project', 'E-commerce platform specific knowledge', 'system');
            """
            
            statements = [stmt.strip() for stmt in realm_tables_sql.split(';') if stmt.strip()]
            for statement in statements:
                if statement:
                    cursor.execute(statement)
            connection.commit()
            logger.info("Successfully created realm tables")
        except Exception as e:
            logger.warning(f"Failed to create realm tables: {e}")
        
        # Fix FULLTEXT indexes for dual-realm search
        logger.info("Adding missing FULLTEXT indexes...")
        try:
            # Add individual FULLTEXT index on content column for realm-aware search
            fulltext_sql = """
            ALTER TABLE megamind_chunks ADD FULLTEXT(content);
            """
            cursor.execute("ALTER TABLE megamind_chunks ADD FULLTEXT(content)")
            connection.commit()
            logger.info("Successfully created FULLTEXT index on content column")
        except Exception as e:
            if "Duplicate key name" in str(e) or "already exists" in str(e):
                logger.info("FULLTEXT index on content already exists")
            else:
                logger.warning(f"Failed to create FULLTEXT index: {e}")
        
        # Add missing stored function that inheritance resolver needs
        logger.info("Adding missing stored functions...")
        try:
            cursor.execute("DROP FUNCTION IF EXISTS resolve_inheritance_conflict")
            
            # Create a simple version of the resolve_inheritance_conflict function
            function_sql = """
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
                DECLARE v_chunk_realm VARCHAR(50) DEFAULT NULL;
                
                -- Check if chunk exists and get its realm
                SELECT realm_id INTO v_chunk_realm
                FROM megamind_chunks c
                WHERE c.chunk_id = p_chunk_id;
                
                IF v_chunk_realm IS NULL THEN
                    SET result = JSON_OBJECT(
                        'access_granted', FALSE,
                        'access_type', 'denied',
                        'reason', 'Chunk not found'
                    );
                    RETURN result;
                END IF;
                
                -- Check for direct access (same realm)
                IF v_chunk_realm = p_accessing_realm THEN
                    SET result = JSON_OBJECT(
                        'access_granted', TRUE,
                        'access_type', 'direct',
                        'source_realm', p_accessing_realm,
                        'reason', 'Direct access to own realm'
                    );
                    RETURN result;
                END IF;
                
                -- Check for global realm access
                IF v_chunk_realm = 'GLOBAL' THEN
                    SET result = JSON_OBJECT(
                        'access_granted', TRUE,
                        'access_type', 'inherited',
                        'source_realm', 'GLOBAL',
                        'priority_order', 999,
                        'reason', 'Global realm inheritance'
                    );
                    RETURN result;
                END IF;
                
                -- Deny access for now (no complex inheritance yet)
                SET result = JSON_OBJECT(
                    'access_granted', FALSE,
                    'access_type', 'denied',
                    'reason', 'No inheritance path found'
                );
                
                RETURN result;
            END
            """
            cursor.execute(function_sql)
            connection.commit()
            logger.info("Successfully created resolve_inheritance_conflict function")
            
        except Exception as e:
            logger.warning(f"Failed to create stored functions: {e}")
        
        logger.info("Database schema initialized successfully")
        
        cursor.close()
        connection.close()
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize schema: {e}")
        return False

def verify_tables(config):
    """Verify all required tables exist"""
    try:
        connection = mysql.connector.connect(**config)
        cursor = connection.cursor()
        
        cursor.execute("SHOW TABLES")
        tables = [table[0] for table in cursor.fetchall()]
        
        required_tables = [
            'megamind_chunks',
            'megamind_chunk_relationships',
            'megamind_chunk_tags',
            'megamind_embeddings',
            'megamind_session_changes',
            'megamind_session_metadata',
            'megamind_knowledge_contributions',
            'megamind_performance_metrics',
            'megamind_system_health',
            # Realm system tables
            'megamind_realms',
            'megamind_realm_inheritance'
        ]
        
        missing_tables = [table for table in required_tables if table not in tables]
        
        if missing_tables:
            logger.error(f"Missing tables: {missing_tables}")
            return False
        
        logger.info(f"All {len(required_tables)} required tables exist")
        
        cursor.close()
        connection.close()
        return True
        
    except Exception as e:
        logger.error(f"Failed to verify tables: {e}")
        return False

def main():
    """Main initialization function"""
    config = {
        'host': os.getenv('MEGAMIND_DB_HOST', '10.255.250.22'),
        'port': int(os.getenv('MEGAMIND_DB_PORT', '3309')),
        'database': os.getenv('MEGAMIND_DB_NAME', 'megamind_database'),
        'user': os.getenv('MEGAMIND_DB_USER', 'megamind_user'),
        'password': os.getenv('MEGAMIND_DB_PASSWORD', ''),
    }
    
    if not config['password']:
        logger.error("Database password not configured. Set MEGAMIND_DB_PASSWORD environment variable.")
        return False
    
    logger.info("Starting database initialization...")
    
    # Wait for database to be available
    if not wait_for_database(config):
        return False
    
    # Create database if not exists
    if not create_database_if_not_exists(config):
        return False
    
    # Initialize schema
    if not initialize_schema(config):
        return False
    
    # Verify tables
    if not verify_tables(config):
        return False
    
    logger.info("Database initialization completed successfully")
    return True

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)