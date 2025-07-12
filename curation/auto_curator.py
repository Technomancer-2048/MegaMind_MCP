#!/usr/bin/env python3
"""
Automated Curation System for MegaMind Context Database
Phase 4: Advanced Optimization

Intelligent curation system for identifying underutilized chunks, 
consolidating related content, and maintaining knowledge base quality.
"""

import json
import logging
import os
import time
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import mysql.connector
from mysql.connector import pooling

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class CurationRecommendation:
    """Curation recommendation structure"""
    recommendation_id: str
    recommendation_type: str  # remove, consolidate, archive, update
    target_chunks: List[str]
    confidence_score: float
    impact_assessment: Dict[str, Any]
    rationale: str
    potential_savings: Dict[str, int]

@dataclass
class ConsolidationCandidate:
    """Consolidation candidate structure"""
    primary_chunk_id: str
    related_chunk_ids: List[str]
    similarity_score: float
    consolidation_benefit: Dict[str, Any]
    suggested_content: str

class AutoCurator:
    """Automated curation system for knowledge base optimization"""
    
    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config
        self.connection_pool = None
        self.curation_thresholds = {
            'cold_chunk_days': 60,
            'min_access_count': 2,
            'similarity_threshold': 0.85,
            'consolidation_min_chunks': 3,
            'removal_confidence_threshold': 0.8
        }
        self._setup_connection_pool()
    
    def _setup_connection_pool(self):
        """Setup MySQL connection pool"""
        try:
            pool_config = {
                'pool_name': 'curation_pool',
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
            logger.info("Curation system database connection pool established")
        except Exception as e:
            logger.error(f"Failed to setup curation database connection pool: {e}")
            raise
    
    def get_connection(self):
        """Get a connection from the pool"""
        return self.connection_pool.get_connection()
    
    def identify_cold_chunks(self, days_threshold: int = None, access_threshold: int = None) -> List[Dict[str, Any]]:
        """Identify chunks that haven't been accessed recently"""
        if days_threshold is None:
            days_threshold = self.curation_thresholds['cold_chunk_days']
        if access_threshold is None:
            access_threshold = self.curation_thresholds['min_access_count']
        
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            query = """
            SELECT c.chunk_id, c.content, c.source_document, c.section_path, 
                   c.chunk_type, c.line_count, c.token_count, c.access_count, 
                   c.last_accessed, c.created_at,
                   DATEDIFF(NOW(), COALESCE(c.last_accessed, c.created_at)) as days_since_access,
                   COUNT(r.relationship_id) as relationship_count,
                   COUNT(t.tag_id) as tag_count
            FROM megamind_chunks c
            LEFT JOIN megamind_chunk_relationships r ON c.chunk_id = r.chunk_id
            LEFT JOIN megamind_chunk_tags t ON c.chunk_id = t.chunk_id
            WHERE c.access_count <= %s 
              AND (c.last_accessed IS NULL OR c.last_accessed <= DATE_SUB(NOW(), INTERVAL %s DAY))
            GROUP BY c.chunk_id
            ORDER BY c.access_count ASC, days_since_access DESC
            """
            
            cursor.execute(query, (access_threshold, days_threshold))
            results = cursor.fetchall()
            
            cold_chunks = []
            for row in results:
                # Calculate curation score based on multiple factors
                base_score = 1.0
                
                # Factor in access patterns
                if row['access_count'] == 0:
                    base_score *= 0.9  # Never accessed
                elif row['access_count'] <= 1:
                    base_score *= 0.7  # Rarely accessed
                
                # Factor in relationships
                if row['relationship_count'] == 0:
                    base_score *= 0.8  # No relationships
                elif row['relationship_count'] >= 3:
                    base_score *= 1.2  # Well connected
                
                # Factor in content size
                if row['token_count'] and row['token_count'] < 50:
                    base_score *= 0.6  # Very small content
                elif row['token_count'] and row['token_count'] > 500:
                    base_score *= 1.1  # Substantial content
                
                # Factor in age
                days_old = row['days_since_access']
                if days_old > 120:
                    base_score *= 0.5  # Very old
                elif days_old > 90:
                    base_score *= 0.7  # Old
                
                # Determine curation priority
                if base_score <= 0.4:
                    priority = "high"  # Strong candidate for removal
                elif base_score <= 0.6:
                    priority = "medium"  # Candidate for consolidation
                else:
                    priority = "low"  # Monitor or archive
                
                cold_chunks.append({
                    "chunk_id": row['chunk_id'],
                    "source_document": row['source_document'],
                    "section_path": row['section_path'],
                    "chunk_type": row['chunk_type'],
                    "token_count": row['token_count'] or 0,
                    "access_count": row['access_count'],
                    "relationship_count": row['relationship_count'],
                    "tag_count": row['tag_count'],
                    "days_since_access": days_old,
                    "curation_score": round(base_score, 2),
                    "curation_priority": priority,
                    "content_preview": row['content'][:200] + "..." if len(row['content']) > 200 else row['content']
                })
            
            logger.info(f"Identified {len(cold_chunks)} cold chunks for curation")
            return cold_chunks
            
        except Exception as e:
            logger.error(f"Failed to identify cold chunks: {e}")
            return []
        finally:
            if connection:
                connection.close()
    
    def find_consolidation_candidates(self, similarity_threshold: float = None) -> List[ConsolidationCandidate]:
        """Find chunks that could be consolidated based on similarity"""
        if similarity_threshold is None:
            similarity_threshold = self.curation_thresholds['similarity_threshold']
        
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            # Get chunks that are potential consolidation targets
            # Focus on chunks with low-medium access but similar content
            query = """
            SELECT chunk_id, content, source_document, section_path, 
                   chunk_type, token_count, access_count
            FROM megamind_chunks
            WHERE access_count BETWEEN 1 AND 10 
              AND token_count BETWEEN 50 AND 300
            ORDER BY source_document, section_path
            """
            
            cursor.execute(query)
            chunks = cursor.fetchall()
            
            # Group chunks by document and analyze for potential consolidation
            document_groups = {}
            for chunk in chunks:
                doc_key = f"{chunk['source_document']}::{chunk['chunk_type']}"
                if doc_key not in document_groups:
                    document_groups[doc_key] = []
                document_groups[doc_key].append(chunk)
            
            consolidation_candidates = []
            
            for doc_key, chunk_group in document_groups.items():
                if len(chunk_group) < self.curation_thresholds['consolidation_min_chunks']:
                    continue
                
                # Simple content similarity analysis
                for i, primary_chunk in enumerate(chunk_group):
                    similar_chunks = []
                    
                    for j, candidate_chunk in enumerate(chunk_group):
                        if i == j:
                            continue
                        
                        # Calculate basic similarity based on content overlap
                        similarity = self._calculate_content_similarity(
                            primary_chunk['content'], 
                            candidate_chunk['content']
                        )
                        
                        if similarity >= similarity_threshold:
                            similar_chunks.append(candidate_chunk['chunk_id'])
                    
                    if len(similar_chunks) >= 2:  # At least 2 similar chunks
                        total_tokens = sum(chunk['token_count'] for chunk in chunk_group if chunk['chunk_id'] in similar_chunks + [primary_chunk['chunk_id']])
                        
                        consolidation_benefit = {
                            "token_reduction": total_tokens * 0.3,  # Estimated 30% reduction
                            "maintenance_reduction": len(similar_chunks) + 1,
                            "coherence_improvement": 0.8
                        }
                        
                        # Generate suggested consolidated content
                        suggested_content = self._generate_consolidated_content(
                            primary_chunk, 
                            [chunk for chunk in chunk_group if chunk['chunk_id'] in similar_chunks]
                        )
                        
                        candidate = ConsolidationCandidate(
                            primary_chunk_id=primary_chunk['chunk_id'],
                            related_chunk_ids=similar_chunks,
                            similarity_score=similarity,
                            consolidation_benefit=consolidation_benefit,
                            suggested_content=suggested_content
                        )
                        
                        consolidation_candidates.append(candidate)
                        break  # Only one consolidation per primary chunk
            
            logger.info(f"Found {len(consolidation_candidates)} consolidation candidates")
            return consolidation_candidates
            
        except Exception as e:
            logger.error(f"Failed to find consolidation candidates: {e}")
            return []
        finally:
            if connection:
                connection.close()
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate simple content similarity"""
        # Simple word-based similarity
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        jaccard_similarity = len(intersection) / len(union) if union else 0.0
        
        # Boost similarity if sections are in same document path
        if hasattr(self, '_current_path_comparison'):
            if self._current_path_comparison:
                jaccard_similarity *= 1.2
        
        return min(jaccard_similarity, 1.0)
    
    def _generate_consolidated_content(self, primary_chunk: Dict[str, Any], related_chunks: List[Dict[str, Any]]) -> str:
        """Generate consolidated content from multiple chunks"""
        # Start with primary chunk content
        consolidated = primary_chunk['content']
        
        # Add unique information from related chunks
        primary_words = set(primary_chunk['content'].lower().split())
        
        for chunk in related_chunks:
            chunk_words = set(chunk['content'].lower().split())
            unique_words = chunk_words - primary_words
            
            if len(unique_words) > 5:  # If chunk has substantial unique content
                # Extract sentences with unique information
                sentences = chunk['content'].split('.')
                for sentence in sentences:
                    sentence_words = set(sentence.lower().split())
                    if len(sentence_words.intersection(unique_words)) >= 2:
                        consolidated += f"\n\n{sentence.strip()}."
                        break
        
        return consolidated[:2000] + "..." if len(consolidated) > 2000 else consolidated
    
    def generate_curation_recommendations(self) -> List[CurationRecommendation]:
        """Generate comprehensive curation recommendations"""
        recommendations = []
        
        # 1. Identify removal candidates
        cold_chunks = self.identify_cold_chunks()
        high_priority_cold = [chunk for chunk in cold_chunks if chunk['curation_priority'] == 'high']
        
        for chunk in high_priority_cold[:20]:  # Limit to top 20 removal candidates
            rec = CurationRecommendation(
                recommendation_id=f"remove_{chunk['chunk_id']}",
                recommendation_type="remove",
                target_chunks=[chunk['chunk_id']],
                confidence_score=chunk['curation_score'],
                impact_assessment={
                    "token_savings": chunk['token_count'],
                    "maintenance_reduction": 1,
                    "risk_level": "low" if chunk['relationship_count'] == 0 else "medium"
                },
                rationale=f"Chunk has {chunk['access_count']} accesses in {chunk['days_since_access']} days, "
                         f"with {chunk['relationship_count']} relationships",
                potential_savings={
                    "tokens": chunk['token_count'],
                    "chunks": 1
                }
            )
            recommendations.append(rec)
        
        # 2. Identify consolidation candidates
        consolidation_candidates = self.find_consolidation_candidates()
        
        for candidate in consolidation_candidates[:10]:  # Limit to top 10 consolidation candidates
            total_tokens = int(candidate.consolidation_benefit['token_reduction'])
            chunk_count = len(candidate.related_chunk_ids) + 1
            
            rec = CurationRecommendation(
                recommendation_id=f"consolidate_{candidate.primary_chunk_id}",
                recommendation_type="consolidate",
                target_chunks=[candidate.primary_chunk_id] + candidate.related_chunk_ids,
                confidence_score=candidate.similarity_score,
                impact_assessment={
                    "token_savings": total_tokens,
                    "maintenance_reduction": chunk_count - 1,
                    "coherence_improvement": candidate.consolidation_benefit['coherence_improvement']
                },
                rationale=f"Consolidate {chunk_count} similar chunks with {candidate.similarity_score:.2f} similarity",
                potential_savings={
                    "tokens": total_tokens,
                    "chunks": chunk_count - 1
                }
            )
            recommendations.append(rec)
        
        # 3. Identify archival candidates
        medium_priority_cold = [chunk for chunk in cold_chunks if chunk['curation_priority'] == 'medium']
        
        for chunk in medium_priority_cold[:15]:  # Limit to top 15 archive candidates
            rec = CurationRecommendation(
                recommendation_id=f"archive_{chunk['chunk_id']}",
                recommendation_type="archive",
                target_chunks=[chunk['chunk_id']],
                confidence_score=chunk['curation_score'],
                impact_assessment={
                    "token_savings": chunk['token_count'] * 0.5,  # Partial savings through archival
                    "maintenance_reduction": 0.5,
                    "risk_level": "low"
                },
                rationale=f"Archive underutilized chunk with {chunk['relationship_count']} relationships",
                potential_savings={
                    "tokens": int(chunk['token_count'] * 0.5),
                    "chunks": 0  # Not removed, just archived
                }
            )
            recommendations.append(rec)
        
        # Sort recommendations by potential token savings
        recommendations.sort(key=lambda x: x.potential_savings['tokens'], reverse=True)
        
        logger.info(f"Generated {len(recommendations)} curation recommendations")
        return recommendations
    
    def execute_curation_recommendation(self, recommendation: CurationRecommendation, dry_run: bool = True) -> Dict[str, Any]:
        """Execute a curation recommendation"""
        if dry_run:
            return {
                "success": True,
                "dry_run": True,
                "recommendation_id": recommendation.recommendation_id,
                "action": "simulated",
                "impact": recommendation.impact_assessment
            }
        
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            connection.start_transaction()
            
            if recommendation.recommendation_type == "remove":
                # Remove chunk and its relationships
                for chunk_id in recommendation.target_chunks:
                    # Remove relationships
                    cursor.execute("""
                        DELETE FROM megamind_chunk_relationships 
                        WHERE chunk_id = %s OR related_chunk_id = %s
                    """, (chunk_id, chunk_id))
                    
                    # Remove tags
                    cursor.execute("""
                        DELETE FROM megamind_chunk_tags WHERE chunk_id = %s
                    """, (chunk_id,))
                    
                    # Remove chunk
                    cursor.execute("""
                        DELETE FROM megamind_chunks WHERE chunk_id = %s
                    """, (chunk_id,))
            
            elif recommendation.recommendation_type == "consolidate":
                # Implement consolidation logic
                primary_chunk_id = recommendation.target_chunks[0]
                related_chunk_ids = recommendation.target_chunks[1:]
                
                # Get consolidated content from recommendation analysis
                # This would need to be implemented based on the consolidation candidate
                
                # For now, mark related chunks as consolidated
                for chunk_id in related_chunk_ids:
                    cursor.execute("""
                        UPDATE megamind_chunks 
                        SET chunk_type = 'consolidated', 
                            section_path = CONCAT(section_path, '_consolidated')
                        WHERE chunk_id = %s
                    """, (chunk_id,))
            
            elif recommendation.recommendation_type == "archive":
                # Move chunks to archive status
                for chunk_id in recommendation.target_chunks:
                    cursor.execute("""
                        UPDATE megamind_chunks 
                        SET chunk_type = 'archived',
                            section_path = CONCAT('archive/', section_path)
                        WHERE chunk_id = %s
                    """, (chunk_id,))
            
            # Log curation action
            cursor.execute("""
                INSERT INTO megamind_knowledge_contributions 
                (contribution_id, session_id, chunks_modified, chunks_created, 
                 relationships_added, tags_added, rollback_data)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                recommendation.recommendation_id,
                "curation_system",
                len(recommendation.target_chunks) if recommendation.recommendation_type != "remove" else 0,
                0,
                0,
                0,
                json.dumps({
                    "curation_type": recommendation.recommendation_type,
                    "impact": recommendation.impact_assessment,
                    "target_chunks": recommendation.target_chunks
                })
            ))
            
            connection.commit()
            
            return {
                "success": True,
                "dry_run": False,
                "recommendation_id": recommendation.recommendation_id,
                "action": "executed",
                "chunks_affected": len(recommendation.target_chunks),
                "impact": recommendation.impact_assessment
            }
            
        except Exception as e:
            logger.error(f"Failed to execute curation recommendation: {e}")
            if connection:
                connection.rollback()
            return {
                "success": False,
                "error": str(e),
                "recommendation_id": recommendation.recommendation_id
            }
        finally:
            if connection:
                connection.close()
    
    def generate_curation_report(self) -> Dict[str, Any]:
        """Generate comprehensive curation report"""
        recommendations = self.generate_curation_recommendations()
        
        # Calculate potential impact
        total_token_savings = sum(rec.potential_savings['tokens'] for rec in recommendations)
        total_chunk_reduction = sum(rec.potential_savings['chunks'] for rec in recommendations)
        
        # Group recommendations by type
        by_type = {
            "remove": [rec for rec in recommendations if rec.recommendation_type == "remove"],
            "consolidate": [rec for rec in recommendations if rec.recommendation_type == "consolidate"],
            "archive": [rec for rec in recommendations if rec.recommendation_type == "archive"]
        }
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_recommendations": len(recommendations),
            "potential_impact": {
                "token_savings": total_token_savings,
                "chunk_reduction": total_chunk_reduction,
                "estimated_efficiency_gain": round((total_token_savings / 10000) * 100, 2)  # Rough estimate
            },
            "recommendations_by_type": {
                rec_type: {
                    "count": len(recs),
                    "top_recommendations": [
                        {
                            "id": rec.recommendation_id,
                            "confidence": rec.confidence_score,
                            "savings": rec.potential_savings,
                            "rationale": rec.rationale
                        }
                        for rec in recs[:5]  # Top 5 per type
                    ]
                }
                for rec_type, recs in by_type.items()
            },
            "execution_priority": [
                {
                    "recommendation_id": rec.recommendation_id,
                    "type": rec.recommendation_type,
                    "confidence": rec.confidence_score,
                    "impact_score": rec.potential_savings['tokens'],
                    "risk_level": rec.impact_assessment.get('risk_level', 'medium')
                }
                for rec in recommendations[:20]  # Top 20 overall
            ]
        }
        
        return report

def load_config():
    """Load configuration from environment variables"""
    return {
        'host': os.getenv('MEGAMIND_DB_HOST', '10.255.250.22'),
        'port': os.getenv('MEGAMIND_DB_PORT', '3309'),
        'database': os.getenv('MEGAMIND_DB_NAME', 'megamind_database'),
        'user': os.getenv('MEGAMIND_DB_USER', 'megamind_user'),
        'password': os.getenv('MEGAMIND_DB_PASSWORD', ''),
    }

def main():
    """Main entry point for automated curation"""
    try:
        # Load configuration
        db_config = load_config()
        
        if not db_config['password']:
            logger.error("Database password not configured. Set MEGAMIND_DB_PASSWORD environment variable.")
            return 1
        
        # Initialize curation system
        curator = AutoCurator(db_config)
        
        logger.info("Starting automated curation analysis...")
        
        # Generate curation report
        report = curator.generate_curation_report()
        
        # Save report
        report_path = Path("curation_reports")
        report_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = report_path / f"curation_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Curation report saved to {report_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("AUTOMATED CURATION SUMMARY")
        print("="*60)
        print(f"Total recommendations: {report['total_recommendations']}")
        print(f"Potential token savings: {report['potential_impact']['token_savings']:,}")
        print(f"Potential chunk reduction: {report['potential_impact']['chunk_reduction']}")
        print(f"Estimated efficiency gain: {report['potential_impact']['estimated_efficiency_gain']}%")
        print("\nTop recommendations:")
        
        for i, rec in enumerate(report['execution_priority'][:5], 1):
            print(f"{i}. {rec['type'].upper()}: {rec['recommendation_id']}")
            print(f"   Confidence: {rec['confidence']:.2f}, Impact: {rec['impact_score']} tokens")
        
        print(f"\nFull report saved to: {report_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Curation system failed: {e}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())