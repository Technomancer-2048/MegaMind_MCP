#!/usr/bin/env python3
"""
MegaMind Context Database - Markdown Ingestion Tool
Phase 1: Core Infrastructure

Ingests markdown files into the context database with semantic chunking.
Preserves formatting, extracts metadata, and generates chunk relationships.
"""

import os
import re
import json
import hashlib
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import mysql.connector
from mysql.connector import pooling
import argparse
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ChunkMetadata:
    """Metadata for a content chunk"""
    chunk_id: str
    content: str
    source_document: str
    section_path: str
    chunk_type: str
    line_count: int
    token_count: int
    start_line: int
    end_line: int

class MarkdownIngester:
    """Ingests markdown files into the MegaMind context database"""
    
    def __init__(self, db_config: Dict[str, str]):
        """Initialize the ingester with database configuration"""
        self.db_config = db_config
        self.connection_pool = None
        self._setup_connection_pool()
        
    def _setup_connection_pool(self):
        """Setup MySQL connection pool"""
        try:
            pool_config = {
                'pool_name': 'megamind_pool',
                'pool_size': 5,
                'host': self.db_config['host'],
                'port': int(self.db_config['port']),
                'database': self.db_config['database'],
                'user': self.db_config['user'],
                'password': self.db_config['password'],
                'autocommit': False
            }
            self.connection_pool = pooling.MySQLConnectionPool(**pool_config)
            logger.info("Database connection pool established")
        except Exception as e:
            logger.error(f"Failed to setup database connection pool: {e}")
            raise
    
    def get_connection(self):
        """Get a connection from the pool"""
        return self.connection_pool.get_connection()
    
    def parse_markdown_file(self, file_path: str) -> List[ChunkMetadata]:
        """Parse a markdown file into semantic chunks"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            chunks = []
            current_chunk = []
            current_section_path = ""
            chunk_start_line = 1
            
            for i, line in enumerate(lines, 1):
                # Detect section headers
                if line.startswith('#'):
                    # Save previous chunk if it exists
                    if current_chunk:
                        chunk_content = '\n'.join(current_chunk)
                        if chunk_content.strip():
                            chunk = self._create_chunk_metadata(
                                chunk_content, file_path, current_section_path,
                                chunk_start_line, i - 1
                            )
                            chunks.append(chunk)
                    
                    # Start new chunk
                    current_chunk = [line]
                    current_section_path = self._extract_section_path(line, current_section_path)
                    chunk_start_line = i
                else:
                    current_chunk.append(line)
            
            # Add final chunk
            if current_chunk:
                chunk_content = '\n'.join(current_chunk)
                if chunk_content.strip():
                    chunk = self._create_chunk_metadata(
                        chunk_content, file_path, current_section_path,
                        chunk_start_line, len(lines)
                    )
                    chunks.append(chunk)
            
            logger.info(f"Parsed {len(chunks)} chunks from {file_path}")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {e}")
            return []
    
    def _extract_section_path(self, header_line: str, parent_path: str) -> str:
        """Extract hierarchical section path from header"""
        level = len(header_line) - len(header_line.lstrip('#'))
        title = header_line.strip('#').strip()
        
        # Sanitize title for path
        title = re.sub(r'[^\w\s-]', '', title)
        title = re.sub(r'[-\s]+', '_', title).lower()
        
        if level == 1:
            return f"/{title}"
        else:
            # For now, simple hierarchical path
            return f"{parent_path}/{title}"
    
    def _create_chunk_metadata(self, content: str, file_path: str, section_path: str, 
                              start_line: int, end_line: int) -> ChunkMetadata:
        """Create chunk metadata from content"""
        # Generate chunk ID
        source_doc = os.path.basename(file_path)
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        sanitized_path = re.sub(r'[^\w/]', '_', section_path)
        chunk_id = f"{source_doc}_{sanitized_path}_{content_hash}".replace('/', '_').replace('.', '_')
        
        # Determine chunk type
        chunk_type = self._determine_chunk_type(content)
        
        # Calculate metrics
        line_count = content.count('\n') + 1
        token_count = self._estimate_token_count(content)
        
        return ChunkMetadata(
            chunk_id=chunk_id,
            content=content,
            source_document=source_doc,
            section_path=section_path,
            chunk_type=chunk_type,
            line_count=line_count,
            token_count=token_count,
            start_line=start_line,
            end_line=end_line
        )
    
    def _determine_chunk_type(self, content: str) -> str:
        """Determine the type of content chunk"""
        content_lower = content.lower()
        
        if '```' in content and ('function' in content_lower or 'def ' in content or 'CREATE' in content):
            return 'function'
        elif any(keyword in content_lower for keyword in ['rule', 'must', 'should', 'required', 'mandatory']):
            return 'rule'
        elif '```' in content or 'example' in content_lower:
            return 'example'
        else:
            return 'section'
    
    def _estimate_token_count(self, text: str) -> int:
        """Rough token count estimation (1 token ≈ 4 characters)"""
        return len(text) // 4
    
    def insert_chunks(self, chunks: List[ChunkMetadata]) -> bool:
        """Insert chunks into the database"""
        if not chunks:
            return True
            
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor()
            
            # Prepare insert statement
            insert_query = """
            INSERT INTO megamind_chunks 
            (chunk_id, content, source_document, section_path, chunk_type, 
             line_count, token_count, created_at, last_accessed, access_count)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
            content = VALUES(content),
            section_path = VALUES(section_path),
            chunk_type = VALUES(chunk_type),
            line_count = VALUES(line_count),
            token_count = VALUES(token_count),
            last_accessed = CURRENT_TIMESTAMP
            """
            
            # Prepare data
            now = datetime.now()
            chunk_data = [
                (chunk.chunk_id, chunk.content, chunk.source_document, 
                 chunk.section_path, chunk.chunk_type, chunk.line_count,
                 chunk.token_count, now, now, 1)  # Creation counts as first access
                for chunk in chunks
            ]
            
            # Execute batch insert
            cursor.executemany(insert_query, chunk_data)
            connection.commit()
            
            logger.info(f"Successfully inserted {len(chunks)} chunks")
            return True
            
        except Exception as e:
            if connection:
                connection.rollback()
            logger.error(f"Failed to insert chunks: {e}")
            return False
        finally:
            if connection:
                connection.close()
    
    def ingest_directory(self, directory_path: str, file_pattern: str = "*.md") -> int:
        """Ingest all markdown files from a directory"""
        total_chunks = 0
        directory = Path(directory_path)
        
        if not directory.exists():
            logger.error(f"Directory does not exist: {directory_path}")
            return 0
        
        # Find markdown files
        md_files = list(directory.glob(file_pattern))
        if not md_files:
            logger.warning(f"No markdown files found in {directory_path}")
            return 0
        
        logger.info(f"Found {len(md_files)} markdown files to process")
        
        for md_file in md_files:
            logger.info(f"Processing {md_file}")
            chunks = self.parse_markdown_file(str(md_file))
            
            if chunks and self.insert_chunks(chunks):
                total_chunks += len(chunks)
                logger.info(f"Successfully ingested {len(chunks)} chunks from {md_file}")
            else:
                logger.error(f"Failed to ingest {md_file}")
        
        logger.info(f"Total chunks ingested: {total_chunks}")
        return total_chunks

def main():
    """Main entry point for the markdown ingester"""
    parser = argparse.ArgumentParser(description='Ingest markdown files into MegaMind context database')
    parser.add_argument('directory', help='Directory containing markdown files')
    parser.add_argument('--pattern', default='*.md', help='File pattern to match (default: *.md)')
    parser.add_argument('--host', default='10.255.250.22', help='Database host')
    parser.add_argument('--port', default='3309', help='Database port')
    parser.add_argument('--database', default='megamind_database', help='Database name')
    parser.add_argument('--user', default='megamind_user', help='Database user')
    parser.add_argument('--password', required=True, help='Database password')
    
    args = parser.parse_args()
    
    # Database configuration
    db_config = {
        'host': args.host,
        'port': args.port,
        'database': args.database,
        'user': args.user,
        'password': args.password
    }
    
    try:
        # Initialize ingester
        ingester = MarkdownIngester(db_config)
        
        # Ingest directory
        total_chunks = ingester.ingest_directory(args.directory, args.pattern)
        
        if total_chunks > 0:
            print(f"Successfully ingested {total_chunks} chunks from {args.directory}")
        else:
            print("No chunks were ingested")
            return 1
            
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())