#!/usr/bin/env python3
"""
MegaMind Context Database - Semantic Analysis Engine
Phase 2: Intelligence Layer

Generates embeddings, discovers relationships, and performs semantic analysis
of documentation chunks using sentence transformers.
"""

import os
import json
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import mysql.connector
from mysql.connector import pooling
import argparse
from datetime import datetime
import hashlib
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

# Sentence transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    import torch
except ImportError:
    print("Please install required dependencies:")
    print("pip install sentence-transformers torch")
    exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ChunkEmbedding:
    """Embedding data for a chunk"""
    chunk_id: str
    embedding: np.ndarray
    content: str
    chunk_type: str
    source_document: str
    token_count: int

@dataclass
class RelationshipCandidate:
    """Potential relationship between chunks"""
    chunk_id_1: str
    chunk_id_2: str
    similarity_score: float
    relationship_type: str
    confidence: float

@dataclass
class TagCandidate:
    """Potential tag for a chunk"""
    chunk_id: str
    tag_type: str
    tag_value: str
    confidence: float

class SemanticAnalyzer:
    """Semantic analysis engine for MegaMind context database"""
    
    def __init__(self, db_config: Dict[str, str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the semantic analyzer"""
        self.db_config = db_config
        self.model_name = model_name
        self.model = None
        self.connection_pool = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Similarity thresholds for relationship detection
        self.similarity_thresholds = {
            'references': 0.75,
            'depends_on': 0.70,
            'enhances': 0.65,
            'implements': 0.80,
            'supersedes': 0.85
        }
        
        # Tag classification patterns
        self.tag_patterns = {
            'subsystem': {
                'database': ['sql', 'mysql', 'database', 'table', 'query', 'schema'],
                'mcp': ['mcp', 'server', 'function', 'tool', 'protocol'],
                'security': ['auth', 'security', 'permission', 'access', 'token'],
                'api': ['api', 'endpoint', 'request', 'response', 'http'],
                'analytics': ['analytics', 'metrics', 'dashboard', 'chart', 'data']
            },
            'function_type': {
                'utility': ['helper', 'util', 'common', 'shared'],
                'validation': ['validate', 'check', 'verify', 'ensure'],
                'transformation': ['convert', 'transform', 'parse', 'format'],
                'aggregation': ['sum', 'count', 'average', 'total', 'aggregate'],
                'configuration': ['config', 'setting', 'option', 'parameter']
            },
            'difficulty': {
                'beginner': ['simple', 'basic', 'easy', 'intro'],
                'intermediate': ['medium', 'moderate', 'standard'],
                'advanced': ['complex', 'advanced', 'expert', 'sophisticated']
            }
        }
        
        self._setup_model()
        self._setup_connection_pool()
    
    def _setup_model(self):
        """Initialize the sentence transformer model"""
        try:
            logger.info(f"Loading model: {self.model_name} on device: {self.device}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _setup_connection_pool(self):
        """Setup MySQL connection pool"""
        try:
            pool_config = {
                'pool_name': 'semantic_analyzer_pool',
                'pool_size': 5,
                'host': self.db_config['host'],
                'port': int(self.db_config['port']),
                'database': self.db_config['database'],
                'user': self.db_config['user'],
                'password': self.db_config['password'],
                'autocommit': False,
                'charset': 'utf8mb4',
                'use_unicode': True
            }
            self.connection_pool = pooling.MySQLConnectionPool(**pool_config)
            logger.info("Database connection pool established")
        except Exception as e:
            logger.error(f"Failed to setup database connection pool: {e}")
            raise
    
    def get_connection(self):
        """Get a connection from the pool"""
        return self.connection_pool.get_connection()
    
    def generate_embeddings_batch(self, chunks: List[Dict]) -> List[ChunkEmbedding]:
        """Generate embeddings for a batch of chunks"""
        try:
            # Prepare content for embedding
            contents = [chunk['content'] for chunk in chunks]
            
            logger.info(f"Generating embeddings for {len(contents)} chunks...")
            start_time = time.time()
            
            # Generate embeddings in batch for efficiency
            embeddings = self.model.encode(contents, batch_size=32, show_progress_bar=True)
            
            elapsed = time.time() - start_time
            logger.info(f"Generated {len(embeddings)} embeddings in {elapsed:.2f}s ({len(embeddings)/elapsed:.1f} chunks/sec)")
            
            # Create ChunkEmbedding objects
            chunk_embeddings = []
            for chunk, embedding in zip(chunks, embeddings):
                chunk_embedding = ChunkEmbedding(
                    chunk_id=chunk['chunk_id'],
                    embedding=embedding,
                    content=chunk['content'],
                    chunk_type=chunk['chunk_type'],
                    source_document=chunk['source_document'],
                    token_count=chunk.get('token_count', 0)
                )
                chunk_embeddings.append(chunk_embedding)
            
            return chunk_embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return []
    
    def store_embeddings(self, chunk_embeddings: List[ChunkEmbedding]) -> bool:
        """Store embeddings in the database"""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor()
            
            # Update chunks with embeddings
            update_query = """
            UPDATE megamind_chunks 
            SET embedding = %s 
            WHERE chunk_id = %s
            """
            
            updates = []
            for chunk_emb in chunk_embeddings:
                # Convert numpy array to JSON-serializable list
                embedding_json = json.dumps(chunk_emb.embedding.tolist())
                updates.append((embedding_json, chunk_emb.chunk_id))
            
            cursor.executemany(update_query, updates)
            connection.commit()
            
            logger.info(f"Stored {len(chunk_embeddings)} embeddings in database")
            return True
            
        except Exception as e:
            if connection:
                connection.rollback()
            logger.error(f"Failed to store embeddings: {e}")
            return False
        finally:
            if connection:
                connection.close()
    
    def discover_relationships(self, chunk_embeddings: List[ChunkEmbedding]) -> List[RelationshipCandidate]:
        """Discover semantic relationships between chunks"""
        relationships = []
        
        logger.info(f"Discovering relationships among {len(chunk_embeddings)} chunks...")
        
        # Calculate similarity matrix
        embeddings_matrix = np.array([emb.embedding for emb in chunk_embeddings])
        
        for i, chunk_1 in enumerate(chunk_embeddings):
            for j, chunk_2 in enumerate(chunk_embeddings[i+1:], i+1):
                # Calculate cosine similarity
                similarity = np.dot(chunk_1.embedding, chunk_2.embedding) / (
                    np.linalg.norm(chunk_1.embedding) * np.linalg.norm(chunk_2.embedding)
                )
                
                # Determine relationship type based on content analysis
                relationship_type, confidence = self._classify_relationship(
                    chunk_1, chunk_2, similarity
                )
                
                if relationship_type and similarity >= self.similarity_thresholds.get(relationship_type, 0.6):
                    relationship = RelationshipCandidate(
                        chunk_id_1=chunk_1.chunk_id,
                        chunk_id_2=chunk_2.chunk_id,
                        similarity_score=float(similarity),
                        relationship_type=relationship_type,
                        confidence=confidence
                    )
                    relationships.append(relationship)
        
        logger.info(f"Discovered {len(relationships)} potential relationships")
        return relationships
    
    def _classify_relationship(self, chunk_1: ChunkEmbedding, chunk_2: ChunkEmbedding, 
                              similarity: float) -> Tuple[Optional[str], float]:
        """Classify the type of relationship between two chunks"""
        content_1 = chunk_1.content.lower()
        content_2 = chunk_2.content.lower()
        
        # Rule-based relationship classification
        if chunk_1.chunk_type == 'function' and chunk_2.chunk_type == 'function':
            if any(word in content_1 and word in content_2 for word in ['call', 'invoke', 'use']):
                return 'depends_on', 0.8
        
        if chunk_1.chunk_type == 'example' and chunk_2.chunk_type == 'function':
            return 'implements', 0.9
        
        if 'see also' in content_1 or 'reference' in content_1:
            return 'references', 0.85
        
        if chunk_1.source_document == chunk_2.source_document:
            if abs(len(content_1) - len(content_2)) < 100:  # Similar length
                return 'enhances', 0.7
        
        # Default to enhances for high similarity
        if similarity > 0.8:
            return 'enhances', similarity * 0.8
        
        return None, 0.0
    
    def generate_semantic_tags(self, chunk_embeddings: List[ChunkEmbedding]) -> List[TagCandidate]:
        """Generate semantic tags for chunks"""
        tag_candidates = []
        
        logger.info(f"Generating semantic tags for {len(chunk_embeddings)} chunks...")
        
        for chunk_emb in chunk_embeddings:
            content_lower = chunk_emb.content.lower()
            
            # Generate tags based on content patterns
            for tag_type, categories in self.tag_patterns.items():
                for category, keywords in categories.items():
                    # Calculate keyword match score
                    matches = sum(1 for keyword in keywords if keyword in content_lower)
                    if matches > 0:
                        confidence = min(0.95, matches / len(keywords) + 0.3)
                        
                        tag = TagCandidate(
                            chunk_id=chunk_emb.chunk_id,
                            tag_type=tag_type,
                            tag_value=category,
                            confidence=confidence
                        )
                        tag_candidates.append(tag)
            
            # Language detection
            if '```' in chunk_emb.content:
                # Extract code language from markdown code blocks
                import re
                code_blocks = re.findall(r'```(\w+)', chunk_emb.content)
                for lang in code_blocks:
                    tag = TagCandidate(
                        chunk_id=chunk_emb.chunk_id,
                        tag_type='language',
                        tag_value=lang.lower(),
                        confidence=0.95
                    )
                    tag_candidates.append(tag)
        
        logger.info(f"Generated {len(tag_candidates)} tag candidates")
        return tag_candidates
    
    def store_relationships(self, relationships: List[RelationshipCandidate]) -> bool:
        """Store discovered relationships in the database"""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor()
            
            # Insert relationships
            insert_query = """
            INSERT INTO megamind_chunk_relationships 
            (chunk_id, related_chunk_id, relationship_type, strength, discovered_by)
            VALUES (%s, %s, %s, %s, 'ai_analysis')
            ON DUPLICATE KEY UPDATE
            strength = GREATEST(strength, VALUES(strength))
            """
            
            relationship_data = [
                (rel.chunk_id_1, rel.chunk_id_2, rel.relationship_type, rel.similarity_score)
                for rel in relationships
            ]
            
            cursor.executemany(insert_query, relationship_data)
            connection.commit()
            
            logger.info(f"Stored {len(relationships)} relationships in database")
            return True
            
        except Exception as e:
            if connection:
                connection.rollback()
            logger.error(f"Failed to store relationships: {e}")
            return False
        finally:
            if connection:
                connection.close()
    
    def store_tags(self, tag_candidates: List[TagCandidate]) -> bool:
        """Store semantic tags in the database"""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor()
            
            # Filter tags by confidence threshold
            high_confidence_tags = [tag for tag in tag_candidates if tag.confidence >= 0.7]
            
            # Insert tags
            insert_query = """
            INSERT INTO megamind_chunk_tags 
            (chunk_id, tag_type, tag_value, confidence, created_by)
            VALUES (%s, %s, %s, %s, 'ai_analysis')
            ON DUPLICATE KEY UPDATE
            confidence = GREATEST(confidence, VALUES(confidence))
            """
            
            tag_data = [
                (tag.chunk_id, tag.tag_type, tag.tag_value, tag.confidence)
                for tag in high_confidence_tags
            ]
            
            cursor.executemany(insert_query, tag_data)
            connection.commit()
            
            logger.info(f"Stored {len(high_confidence_tags)} high-confidence tags in database")
            return True
            
        except Exception as e:
            if connection:
                connection.rollback()
            logger.error(f"Failed to store tags: {e}")
            return False
        finally:
            if connection:
                connection.close()
    
    def load_chunks_for_analysis(self, limit: Optional[int] = None) -> List[Dict]:
        """Load chunks from database for analysis"""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            # Load chunks without embeddings
            query = """
            SELECT chunk_id, content, source_document, section_path, 
                   chunk_type, line_count, token_count
            FROM megamind_chunks
            WHERE embedding IS NULL OR JSON_LENGTH(embedding) = 0
            ORDER BY access_count DESC, created_at ASC
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            cursor.execute(query)
            chunks = cursor.fetchall()
            
            logger.info(f"Loaded {len(chunks)} chunks for analysis")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to load chunks: {e}")
            return []
        finally:
            if connection:
                connection.close()
    
    def analyze_full_corpus(self, batch_size: int = 100) -> bool:
        """Perform full semantic analysis on the corpus"""
        try:
            logger.info("Starting full corpus semantic analysis...")
            
            # Load all chunks needing analysis
            chunks = self.load_chunks_for_analysis()
            
            if not chunks:
                logger.info("No chunks found needing analysis")
                return True
            
            all_embeddings = []
            
            # Process in batches
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
                
                # Generate embeddings
                batch_embeddings = self.generate_embeddings_batch(batch)
                if not batch_embeddings:
                    logger.error(f"Failed to generate embeddings for batch {i//batch_size + 1}")
                    continue
                
                # Store embeddings
                if not self.store_embeddings(batch_embeddings):
                    logger.error(f"Failed to store embeddings for batch {i//batch_size + 1}")
                    continue
                
                all_embeddings.extend(batch_embeddings)
            
            if not all_embeddings:
                logger.error("No embeddings generated")
                return False
            
            # Discover relationships
            relationships = self.discover_relationships(all_embeddings)
            if relationships:
                self.store_relationships(relationships)
            
            # Generate tags
            tags = self.generate_semantic_tags(all_embeddings)
            if tags:
                self.store_tags(tags)
            
            logger.info("Full corpus semantic analysis completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Full corpus analysis failed: {e}")
            return False

def main():
    """Main entry point for semantic analyzer"""
    parser = argparse.ArgumentParser(description='MegaMind semantic analysis engine')
    parser.add_argument('--host', default='10.255.250.22', help='Database host')
    parser.add_argument('--port', default='3309', help='Database port')
    parser.add_argument('--database', default='megamind_database', help='Database name')
    parser.add_argument('--user', default='megamind_user', help='Database user')
    parser.add_argument('--password', required=True, help='Database password')
    parser.add_argument('--model', default='sentence-transformers/all-MiniLM-L6-v2', help='Sentence transformer model')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size for processing')
    
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
        # Initialize analyzer
        analyzer = SemanticAnalyzer(db_config, args.model)
        
        # Run full analysis
        success = analyzer.analyze_full_corpus(args.batch_size)
        
        if success:
            print("✅ Semantic analysis completed successfully")
            return 0
        else:
            print("❌ Semantic analysis failed")
            return 1
            
    except Exception as e:
        logger.error(f"Semantic analysis failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())