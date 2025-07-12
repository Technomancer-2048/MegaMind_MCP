#!/usr/bin/env python3
"""
MegaMind Context Database - Realm-Aware Markdown Ingestion Tool
Phase 3: Ingestion Integration

Enhanced markdown ingester with realm awareness and semantic embedding generation.
Integrates with RealmAwareMegaMindDatabase for full semantic search capabilities.
"""

import os
import re
import json
import hashlib
import logging
import sys
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
import argparse
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'mcp_server'))

from realm_aware_database import RealmAwareMegaMindDatabase
from services.embedding_service import get_embedding_service

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class RealmAwareChunkMetadata:
    """Enhanced metadata for realm-aware content chunks with embedding support"""
    chunk_id: str
    content: str
    source_document: str
    section_path: str
    chunk_type: str
    line_count: int
    token_count: int
    start_line: int
    end_line: int
    realm_id: str
    embedding: Optional[List[float]] = None
    embedding_hash: Optional[str] = None
    content_hash: Optional[str] = None

class RealmAwareMarkdownIngester:
    """Enhanced markdown ingester with realm awareness and semantic embedding generation"""
    
    def __init__(self, db_config: Dict[str, str], target_realm: str = None, session_id: str = None):
        """
        Initialize the realm-aware ingester
        
        Args:
            db_config: Database configuration
            target_realm: Target realm for chunks (defaults to PROJECT from environment)
            session_id: Session ID for change management
        """
        self.db_config = db_config
        self.target_realm = target_realm or os.getenv('MEGAMIND_DEFAULT_TARGET', 'PROJECT')
        self.project_realm = os.getenv('MEGAMIND_PROJECT_REALM', 'PROJECT_DEFAULT')
        self.session_id = session_id or f"ingestion_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize components
        self.database = RealmAwareMegaMindDatabase(db_config)
        self.embedding_service = get_embedding_service()
        
        # Statistics tracking
        self.stats = {
            'files_processed': 0,
            'chunks_created': 0,
            'embeddings_generated': 0,
            'realm_assignments': {'PROJECT': 0, 'GLOBAL': 0, 'OTHER': 0},
            'processing_time': 0.0,
            'errors': []
        }
        
        logger.info(f"Initialized RealmAwareMarkdownIngester:")
        logger.info(f"  Target realm: {self.target_realm}")
        logger.info(f"  Project realm: {self.project_realm}")
        logger.info(f"  Session ID: {self.session_id}")
        logger.info(f"  Embedding service available: {self.embedding_service.is_available()}")
    
    def ingest_file(self, file_path: str, explicit_realm: str = None) -> Dict[str, any]:
        """
        Ingest a single markdown file with realm-aware processing
        
        Args:
            file_path: Path to the markdown file
            explicit_realm: Override default realm assignment
            
        Returns:
            Dict with ingestion results and statistics
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Ingesting file: {file_path}")
            
            # Parse markdown into chunks
            chunks = self.parse_markdown_file(file_path)
            if not chunks:
                logger.warning(f"No chunks extracted from {file_path}")
                return {'success': False, 'error': 'No chunks extracted'}
            
            # Process chunks with realm assignment and embedding generation
            processed_chunks = []
            for chunk in chunks:
                processed_chunk = self._process_chunk_with_realm(chunk, explicit_realm)
                if processed_chunk:
                    processed_chunks.append(processed_chunk)
            
            # Insert chunks using database's session-based approach
            ingestion_results = self._insert_chunks_with_session(processed_chunks)
            
            # Update statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            self.stats['files_processed'] += 1
            self.stats['chunks_created'] += len(processed_chunks)
            self.stats['processing_time'] += processing_time
            
            logger.info(f"Successfully ingested {file_path}: {len(processed_chunks)} chunks in {processing_time:.2f}s")
            
            return {
                'success': True,
                'file_path': file_path,
                'chunks_processed': len(processed_chunks),
                'processing_time': processing_time,
                'session_id': self.session_id,
                'ingestion_results': ingestion_results
            }
            
        except Exception as e:
            error_msg = f"Failed to ingest {file_path}: {str(e)}"
            logger.error(error_msg)
            self.stats['errors'].append(error_msg)
            return {'success': False, 'error': error_msg}
    
    def ingest_directory(self, directory_path: str, pattern: str = "*.md", 
                        recursive: bool = True) -> Dict[str, any]:
        """
        Ingest all markdown files in a directory
        
        Args:
            directory_path: Path to directory containing markdown files
            pattern: File pattern to match (default: *.md)
            recursive: Search subdirectories recursively
            
        Returns:
            Dict with batch ingestion results
        """
        start_time = datetime.now()
        
        try:
            directory = Path(directory_path)
            if not directory.exists():
                raise FileNotFoundError(f"Directory not found: {directory_path}")
            
            # Find markdown files
            if recursive:
                files = list(directory.rglob(pattern))
            else:
                files = list(directory.glob(pattern))
            
            if not files:
                logger.warning(f"No files matching {pattern} found in {directory_path}")
                return {'success': False, 'error': 'No files found'}
            
            logger.info(f"Found {len(files)} files to process")
            
            # Process files in batches for efficiency
            batch_results = []
            successful_files = 0
            
            for file_path in files:
                result = self.ingest_file(str(file_path))
                batch_results.append(result)
                if result['success']:
                    successful_files += 1
            
            # Calculate batch statistics
            total_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'success': True,
                'directory_path': directory_path,
                'total_files': len(files),
                'successful_files': successful_files,
                'failed_files': len(files) - successful_files,
                'total_processing_time': total_time,
                'session_id': self.session_id,
                'batch_results': batch_results,
                'statistics': self.get_ingestion_statistics()
            }
            
        except Exception as e:
            error_msg = f"Failed to ingest directory {directory_path}: {str(e)}"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}
    
    def parse_markdown_file(self, file_path: str) -> List[RealmAwareChunkMetadata]:
        """Parse a markdown file into semantic chunks with enhanced metadata"""
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
    
    def _process_chunk_with_realm(self, chunk: RealmAwareChunkMetadata, 
                                 explicit_realm: str = None) -> Optional[RealmAwareChunkMetadata]:
        """
        Enhanced chunk processing with embedding generation and realm assignment
        
        Args:
            chunk: Raw chunk metadata
            explicit_realm: Override realm assignment
            
        Returns:
            Processed chunk with embedding and realm assignment
        """
        try:
            # Determine realm assignment
            realm_id = explicit_realm if explicit_realm else self._determine_chunk_realm(chunk)
            chunk.realm_id = realm_id
            
            # Generate content hash for deduplication
            chunk.content_hash = hashlib.sha256(chunk.content.encode()).hexdigest()
            
            # Generate embedding with realm context
            if self.embedding_service.is_available():
                start_time = datetime.now()
                embedding = self.embedding_service.generate_embedding(
                    chunk.content, 
                    realm_context=realm_id
                )
                embedding_time = (datetime.now() - start_time).total_seconds()
                
                if embedding is not None:
                    chunk.embedding = embedding
                    chunk.embedding_hash = hashlib.md5(json.dumps(embedding).encode()).hexdigest()
                    self.stats['embeddings_generated'] += 1
                    logger.debug(f"Generated embedding for chunk {chunk.chunk_id} in {embedding_time:.3f}s")
                else:
                    logger.warning(f"Failed to generate embedding for chunk {chunk.chunk_id}")
            else:
                logger.debug(f"Embedding service not available, skipping embedding for {chunk.chunk_id}")
            
            # Update realm statistics
            if realm_id == self.project_realm:
                self.stats['realm_assignments']['PROJECT'] += 1
            elif realm_id == 'GLOBAL':
                self.stats['realm_assignments']['GLOBAL'] += 1
            else:
                self.stats['realm_assignments']['OTHER'] += 1
            
            return chunk
            
        except Exception as e:
            logger.error(f"Failed to process chunk {chunk.chunk_id}: {e}")
            self.stats['errors'].append(f"Chunk processing error: {e}")
            return None
    
    def _determine_chunk_realm(self, chunk: RealmAwareChunkMetadata) -> str:
        """
        Determine appropriate realm for chunk based on content patterns and configuration
        
        Args:
            chunk: Chunk to analyze
            
        Returns:
            Assigned realm ID
        """
        content_lower = chunk.content.lower()
        
        # Global realm indicators
        global_indicators = [
            'standard', 'policy', 'guideline', 'organization', 'company',
            'template', 'framework', 'architecture', 'best practice',
            'governance', 'compliance', 'security', 'general'
        ]
        
        # Project realm indicators
        project_indicators = [
            'implementation', 'feature', 'requirement', 'specification',
            'design', 'api', 'endpoint', 'component', 'module',
            'task', 'issue', 'bug', 'development'
        ]
        
        # Check for explicit realm hints in content or metadata
        if any(indicator in content_lower for indicator in global_indicators):
            return 'GLOBAL'
        elif any(indicator in content_lower for indicator in project_indicators):
            return self.project_realm
        
        # Check section path for realm hints
        section_lower = chunk.section_path.lower()
        if 'global' in section_lower or 'standard' in section_lower:
            return 'GLOBAL'
        elif 'project' in section_lower or 'implementation' in section_lower:
            return self.project_realm
        
        # Default to configured target realm
        if self.target_realm == 'PROJECT':
            return self.project_realm
        elif self.target_realm == 'GLOBAL':
            return 'GLOBAL'
        else:
            return self.project_realm  # Safe default
    
    def _insert_chunks_with_session(self, chunks: List[RealmAwareChunkMetadata]) -> Dict[str, any]:
        """
        Insert chunks using the database's session-based change management
        
        Args:
            chunks: List of processed chunks to insert
            
        Returns:
            Dict with insertion results
        """
        insertion_results = {
            'successful_chunks': 0,
            'failed_chunks': 0,
            'chunk_ids': [],
            'errors': []
        }
        
        for chunk in chunks:
            try:
                # Use the database's create_chunk_with_target method
                chunk_id = self.database.create_chunk_with_target(
                    content=chunk.content,
                    source_document=chunk.source_document,
                    section_path=chunk.section_path,
                    session_id=self.session_id,
                    target_realm=chunk.realm_id
                )
                
                insertion_results['successful_chunks'] += 1
                insertion_results['chunk_ids'].append(chunk_id)
                
                logger.debug(f"Successfully created chunk {chunk_id} in realm {chunk.realm_id}")
                
            except Exception as e:
                error_msg = f"Failed to insert chunk {chunk.chunk_id}: {str(e)}"
                logger.error(error_msg)
                insertion_results['failed_chunks'] += 1
                insertion_results['errors'].append(error_msg)
        
        return insertion_results
    
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
            # Build hierarchical path
            if parent_path:
                return f"{parent_path}/{title}"
            else:
                return f"/{title}"
    
    def _create_chunk_metadata(self, content: str, file_path: str, section_path: str, 
                              start_line: int, end_line: int) -> RealmAwareChunkMetadata:
        """Create enhanced chunk metadata from content"""
        # Generate chunk ID
        source_doc = os.path.basename(file_path)
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        sanitized_path = re.sub(r'[^\w/]', '_', section_path)
        chunk_id = f"{source_doc}_{sanitized_path}_{content_hash}".replace('/', '_').replace('.', '_')
        
        # Determine chunk type with enhanced logic
        chunk_type = self._determine_chunk_type(content)
        
        # Calculate metrics
        line_count = content.count('\n') + 1
        token_count = self._estimate_token_count(content)
        
        return RealmAwareChunkMetadata(
            chunk_id=chunk_id,
            content=content,
            source_document=source_doc,
            section_path=section_path,
            chunk_type=chunk_type,
            line_count=line_count,
            token_count=token_count,
            start_line=start_line,
            end_line=end_line,
            realm_id=""  # Will be set by _process_chunk_with_realm
        )
    
    def _determine_chunk_type(self, content: str) -> str:
        """Determine the type of content chunk with enhanced logic"""
        content_lower = content.lower()
        
        # Enhanced type detection
        if '```' in content and any(lang in content for lang in ['python', 'javascript', 'sql', 'bash']):
            return 'function'
        elif any(keyword in content_lower for keyword in ['rule', 'must', 'should', 'required', 'mandatory', 'policy']):
            return 'rule'
        elif '```' in content or 'example' in content_lower or 'demo' in content_lower:
            return 'example'
        elif any(keyword in content_lower for keyword in ['api', 'endpoint', 'schema', 'interface']):
            return 'function'
        else:
            return 'section'
    
    def _estimate_token_count(self, text: str) -> int:
        """Enhanced token count estimation"""
        # More accurate token estimation
        words = len(text.split())
        chars = len(text)
        # Average of word-based and character-based estimation
        return int((words + chars // 4) / 2)
    
    def get_ingestion_statistics(self) -> Dict[str, any]:
        """Get comprehensive ingestion statistics"""
        return {
            'files_processed': self.stats['files_processed'],
            'chunks_created': self.stats['chunks_created'],
            'embeddings_generated': self.stats['embeddings_generated'],
            'realm_assignments': self.stats['realm_assignments'].copy(),
            'processing_time': self.stats['processing_time'],
            'embedding_coverage': (
                self.stats['embeddings_generated'] / max(self.stats['chunks_created'], 1) * 100
            ),
            'errors_count': len(self.stats['errors']),
            'errors': self.stats['errors'].copy(),
            'session_id': self.session_id,
            'embedding_service_available': self.embedding_service.is_available()
        }
    
    def get_session_changes(self) -> List[Dict[str, any]]:
        """Get pending changes for the current session"""
        try:
            return self.database.get_pending_changes(self.session_id)
        except Exception as e:
            logger.error(f"Failed to get session changes: {e}")
            return []
    
    def commit_session_changes(self, approved_change_ids: List[str] = None) -> Dict[str, any]:
        """
        Commit session changes to the database
        
        Args:
            approved_change_ids: List of specific change IDs to commit (all if None)
            
        Returns:
            Commit results
        """
        try:
            if approved_change_ids is None:
                # Get all pending changes for this session
                pending_changes = self.get_session_changes()
                approved_change_ids = [change['change_id'] for change in pending_changes]
            
            if not approved_change_ids:
                return {'success': True, 'message': 'No changes to commit'}
            
            result = self.database.commit_session_changes(self.session_id, approved_change_ids)
            logger.info(f"Committed {len(approved_change_ids)} changes for session {self.session_id}")
            return result
            
        except Exception as e:
            error_msg = f"Failed to commit session changes: {str(e)}"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}


def load_db_config() -> Dict[str, str]:
    """Load database configuration from environment variables"""
    return {
        'host': os.getenv('MEGAMIND_DB_HOST', 'localhost'),
        'port': os.getenv('MEGAMIND_DB_PORT', '3306'),
        'database': os.getenv('MEGAMIND_DB_NAME', 'megamind_context'),
        'user': os.getenv('MEGAMIND_DB_USER', 'megamind_user'),
        'password': os.getenv('MEGAMIND_DB_PASSWORD', '')
    }


def main():
    """Command-line interface for realm-aware markdown ingestion"""
    parser = argparse.ArgumentParser(description='Realm-Aware Markdown Ingester for MegaMind Context Database')
    parser.add_argument('path', help='Path to markdown file or directory')
    parser.add_argument('--realm', choices=['PROJECT', 'GLOBAL'], 
                       help='Explicit realm assignment (overrides automatic detection)')
    parser.add_argument('--session-id', help='Custom session ID for change management')
    parser.add_argument('--recursive', '-r', action='store_true', 
                       help='Process directories recursively')
    parser.add_argument('--pattern', default='*.md', 
                       help='File pattern for directory processing (default: *.md)')
    parser.add_argument('--commit', action='store_true', 
                       help='Automatically commit changes after ingestion')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    db_config = load_db_config()
    if not db_config['password']:
        logger.error("Database password not configured. Set MEGAMIND_DB_PASSWORD environment variable.")
        return 1
    
    try:
        # Initialize ingester
        ingester = RealmAwareMarkdownIngester(
            db_config=db_config,
            target_realm=args.realm,
            session_id=args.session_id
        )
        
        # Process input path
        input_path = Path(args.path)
        if input_path.is_file():
            result = ingester.ingest_file(str(input_path), explicit_realm=args.realm)
        elif input_path.is_dir():
            result = ingester.ingest_directory(
                str(input_path), 
                pattern=args.pattern, 
                recursive=args.recursive
            )
        else:
            logger.error(f"Path not found: {args.path}")
            return 1
        
        # Display results
        if result['success']:
            print(f"\n✅ Ingestion completed successfully!")
            if 'chunks_processed' in result:
                print(f"   File: {result['file_path']}")
                print(f"   Chunks processed: {result['chunks_processed']}")
            else:
                print(f"   Directory: {result['directory_path']}")
                print(f"   Files processed: {result['successful_files']}/{result['total_files']}")
            
            print(f"   Session ID: {result['session_id']}")
            
            # Show statistics
            stats = ingester.get_ingestion_statistics()
            print(f"\n📊 Statistics:")
            print(f"   Total chunks: {stats['chunks_created']}")
            print(f"   Embeddings generated: {stats['embeddings_generated']}")
            print(f"   Embedding coverage: {stats['embedding_coverage']:.1f}%")
            print(f"   Realm assignments: {stats['realm_assignments']}")
            print(f"   Processing time: {stats['processing_time']:.2f}s")
            
            if stats['errors_count'] > 0:
                print(f"   ⚠️  Errors: {stats['errors_count']}")
            
            # Auto-commit if requested
            if args.commit:
                print(f"\n🔄 Committing changes...")
                commit_result = ingester.commit_session_changes()
                if commit_result.get('success'):
                    print(f"   ✅ Changes committed successfully")
                else:
                    print(f"   ❌ Commit failed: {commit_result.get('error')}")
            else:
                print(f"\n💡 Use --commit flag to automatically commit changes")
                print(f"   Or commit manually using session ID: {result['session_id']}")
        else:
            print(f"❌ Ingestion failed: {result['error']}")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())