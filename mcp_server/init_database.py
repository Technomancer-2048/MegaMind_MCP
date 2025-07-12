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

def initialize_schema(config):
    """Initialize database schema from SQL file"""
    try:
        # Connect to the specific database
        connection = mysql.connector.connect(**config)
        cursor = connection.cursor()
        
        # Read and execute schema file
        schema_file = os.path.join(os.path.dirname(__file__), 'init_schema.sql')
        with open(schema_file, 'r') as f:
            sql_content = f.read()
        
        # Split and execute statements
        statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]
        
        for statement in statements:
            if statement:
                cursor.execute(statement)
        
        connection.commit()
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
            'megamind_system_health'
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