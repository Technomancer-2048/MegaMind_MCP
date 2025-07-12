#!/usr/bin/env python3
"""
Realm Analytics and Reporting System for MegaMind Context Database
Provides comprehensive analytics, usage tracking, and performance insights
"""

import json
import logging
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Union
from enum import Enum

try:
    import mysql.connector
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False

logger = logging.getLogger(__name__)

class MetricType(Enum):
    CHUNKS_CREATED = "chunks_created"
    CHUNKS_ACCESSED = "chunks_accessed"
    RELATIONSHIPS_FORMED = "relationships_formed"
    SEARCHES_PERFORMED = "searches_performed"
    PROMOTIONS_REQUESTED = "promotions_requested"

class MeasurementPeriod(Enum):
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"

@dataclass
class RealmMetric:
    """Represents a realm performance metric"""
    metric_id: str
    realm_id: str
    metric_type: MetricType
    metric_value: int
    measurement_period: MeasurementPeriod
    period_start: datetime
    period_end: datetime
    created_at: datetime

@dataclass
class UsageAnalytics:
    """Represents usage analytics for a realm"""
    realm_id: str
    realm_name: str
    total_chunks: int
    chunks_accessed_today: int
    chunks_created_this_week: int
    avg_access_count: float
    most_popular_subsystems: List[str]
    search_patterns: Dict[str, int]
    relationship_density: float
    user_engagement_score: float

@dataclass
class PerformanceInsight:
    """Represents a performance insight or recommendation"""
    insight_id: str
    realm_id: str
    insight_type: str
    severity: str
    title: str
    description: str
    recommendation: str
    impact_score: float
    created_at: datetime
    expires_at: Optional[datetime] = None

@dataclass
class RealmHealth:
    """Represents realm health metrics"""
    realm_id: str
    health_score: float
    chunk_distribution: Dict[str, int]
    access_patterns: Dict[str, Any]
    relationship_health: Dict[str, Any]
    inheritance_status: Dict[str, Any]
    performance_metrics: Dict[str, Any]

class RealmAnalytics:
    """Comprehensive analytics and reporting for realm system"""
    
    def __init__(self, db_connection):
        self.db = db_connection
        self.logger = logging.getLogger(__name__)
    
    # ===================================================================
    # Metrics Collection and Tracking
    # ===================================================================
    
    def record_metric(self, realm_id: str, metric_type: MetricType, 
                     value: int = 1, period: MeasurementPeriod = MeasurementPeriod.DAILY) -> str:
        """Record a metric for a realm"""
        try:
            cursor = self.db.cursor()
            
            # Calculate period boundaries
            now = datetime.now()
            period_start = self._get_period_start(now, period)
            period_end = self._get_period_end(period_start, period)
            
            metric_id = f"metric_{uuid.uuid4().hex[:12]}"
            
            query = """
            INSERT INTO megamind_realm_metrics 
            (metric_id, realm_id, metric_type, metric_value, measurement_period, period_start, period_end)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE 
            metric_value = metric_value + VALUES(metric_value)
            """
            
            cursor.execute(query, (
                metric_id, realm_id, metric_type.value, value, 
                period.value, period_start, period_end
            ))
            
            self.db.commit()
            return metric_id
            
        except Exception as e:
            self.logger.error(f"Failed to record metric: {e}")
            self.db.rollback()
            raise
        finally:
            if cursor:
                cursor.close()
    
    def get_realm_metrics(self, realm_id: str, metric_type: Optional[MetricType] = None,
                         period: MeasurementPeriod = MeasurementPeriod.DAILY,
                         days_back: int = 30) -> List[RealmMetric]:
        """Get metrics for a realm"""
        try:
            cursor = self.db.cursor(dictionary=True)
            
            query = """
            SELECT * FROM megamind_realm_metrics 
            WHERE realm_id = %s 
            AND measurement_period = %s
            AND period_start >= DATE_SUB(NOW(), INTERVAL %s DAY)
            """
            params = [realm_id, period.value, days_back]
            
            if metric_type:
                query += " AND metric_type = %s"
                params.append(metric_type.value)
            
            query += " ORDER BY period_start DESC"
            
            cursor.execute(query, params)
            results = cursor.fetchall()
            
            metrics = []
            for row in results:
                metric = RealmMetric(
                    metric_id=row['metric_id'],
                    realm_id=row['realm_id'],
                    metric_type=MetricType(row['metric_type']),
                    metric_value=row['metric_value'],
                    measurement_period=MeasurementPeriod(row['measurement_period']),
                    period_start=row['period_start'],
                    period_end=row['period_end'],
                    created_at=row['created_at']
                )
                metrics.append(metric)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to get realm metrics: {e}")
            return []
        finally:
            if cursor:
                cursor.close()
    
    # ===================================================================
    # Usage Analytics and Insights
    # ===================================================================
    
    def analyze_realm_usage(self, realm_id: str) -> Optional[UsageAnalytics]:
        """Comprehensive usage analysis for a realm"""
        try:
            cursor = self.db.cursor(dictionary=True)
            
            # Get basic realm info and chunk statistics
            realm_query = """
            SELECT 
                r.realm_id,
                r.realm_name,
                COUNT(c.chunk_id) as total_chunks,
                COUNT(CASE WHEN c.last_accessed >= DATE_SUB(NOW(), INTERVAL 1 DAY) THEN 1 END) as chunks_accessed_today,
                COUNT(CASE WHEN c.created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY) THEN 1 END) as chunks_created_this_week,
                AVG(c.access_count) as avg_access_count
            FROM megamind_realms r
            LEFT JOIN megamind_chunks c ON r.realm_id = c.realm_id
            WHERE r.realm_id = %s
            GROUP BY r.realm_id, r.realm_name
            """
            
            cursor.execute(realm_query, (realm_id,))
            realm_data = cursor.fetchone()
            
            if not realm_data:
                return None
            
            # Get most popular subsystems
            subsystem_query = """
            SELECT ct.tag_value, COUNT(*) as chunk_count
            FROM megamind_chunk_tags ct
            JOIN megamind_chunks c ON ct.chunk_id = c.chunk_id
            WHERE c.realm_id = %s AND ct.tag_type = 'subsystem'
            GROUP BY ct.tag_value
            ORDER BY chunk_count DESC
            LIMIT 5
            """
            
            cursor.execute(subsystem_query, (realm_id,))
            subsystem_results = cursor.fetchall()
            most_popular_subsystems = [row['tag_value'] for row in subsystem_results]
            
            # Calculate relationship density
            relationship_query = """
            SELECT COUNT(*) as total_relationships,
                   COUNT(DISTINCT cr.chunk_id) as chunks_with_relationships,
                   (SELECT COUNT(*) FROM megamind_chunks WHERE realm_id = %s) as total_chunks
            FROM megamind_chunk_relationships cr
            JOIN megamind_chunks c ON cr.chunk_id = c.chunk_id
            WHERE c.realm_id = %s
            """
            
            cursor.execute(relationship_query, (realm_id, realm_id))
            rel_data = cursor.fetchone()
            
            relationship_density = 0.0
            if rel_data and rel_data['total_chunks'] > 0:
                relationship_density = rel_data['chunks_with_relationships'] / rel_data['total_chunks']
            
            # Calculate user engagement score (based on access patterns and diversity)
            engagement_query = """
            SELECT 
                COUNT(DISTINCT c.chunk_id) as unique_chunks_accessed,
                SUM(c.access_count) as total_accesses,
                COUNT(DISTINCT ct.tag_value) as subsystem_diversity
            FROM megamind_chunks c
            LEFT JOIN megamind_chunk_tags ct ON c.chunk_id = ct.chunk_id AND ct.tag_type = 'subsystem'
            WHERE c.realm_id = %s AND c.access_count > 1
            """
            
            cursor.execute(engagement_query, (realm_id,))
            engagement_data = cursor.fetchone()
            
            user_engagement_score = 0.0
            if engagement_data and realm_data['total_chunks'] > 0:
                access_diversity = engagement_data['unique_chunks_accessed'] / realm_data['total_chunks']
                subsystem_diversity = min(engagement_data['subsystem_diversity'] / 10, 1.0)  # Normalize to 0-1
                user_engagement_score = (access_diversity + subsystem_diversity) / 2
            
            # Mock search patterns (would be populated by actual search tracking)
            search_patterns = {
                "security": 25,
                "database": 20,
                "api": 15,
                "error": 12,
                "deployment": 8
            }
            
            return UsageAnalytics(
                realm_id=realm_data['realm_id'],
                realm_name=realm_data['realm_name'],
                total_chunks=realm_data['total_chunks'] or 0,
                chunks_accessed_today=realm_data['chunks_accessed_today'] or 0,
                chunks_created_this_week=realm_data['chunks_created_this_week'] or 0,
                avg_access_count=float(realm_data['avg_access_count'] or 0),
                most_popular_subsystems=most_popular_subsystems,
                search_patterns=search_patterns,
                relationship_density=relationship_density,
                user_engagement_score=user_engagement_score
            )
            
        except Exception as e:
            self.logger.error(f"Failed to analyze realm usage: {e}")
            return None
        finally:
            if cursor:
                cursor.close()
    
    def generate_performance_insights(self, realm_id: str) -> List[PerformanceInsight]:
        """Generate performance insights and recommendations"""
        insights = []
        
        try:
            # Get realm analytics
            analytics = self.analyze_realm_usage(realm_id)
            if not analytics:
                return insights
            
            # Low engagement insight
            if analytics.user_engagement_score < 0.3:
                insight = PerformanceInsight(
                    insight_id=f"insight_{uuid.uuid4().hex[:12]}",
                    realm_id=realm_id,
                    insight_type="engagement",
                    severity="medium",
                    title="Low User Engagement",
                    description=f"User engagement score is {analytics.user_engagement_score:.2f}, indicating limited knowledge utilization.",
                    recommendation="Consider organizing knowledge better, adding more detailed examples, or providing training on effective search techniques.",
                    impact_score=0.7,
                    created_at=datetime.now(),
                    expires_at=datetime.now() + timedelta(days=30)
                )
                insights.append(insight)
            
            # Low relationship density insight
            if analytics.relationship_density < 0.2:
                insight = PerformanceInsight(
                    insight_id=f"insight_{uuid.uuid4().hex[:12]}",
                    realm_id=realm_id,
                    insight_type="relationships",
                    severity="medium",
                    title="Sparse Knowledge Relationships",
                    description=f"Only {analytics.relationship_density:.1%} of chunks have relationships, limiting knowledge discovery.",
                    recommendation="Review chunks for missing relationships, implement automated relationship discovery, or train users on relationship creation.",
                    impact_score=0.6,
                    created_at=datetime.now(),
                    expires_at=datetime.now() + timedelta(days=45)
                )
                insights.append(insight)
            
            # Unbalanced subsystem distribution
            if len(analytics.most_popular_subsystems) > 0:
                # Check if top subsystem dominates (>50% of chunks)
                cursor = self.db.cursor()
                cursor.execute("""
                    SELECT COUNT(*) as dominant_count,
                           (SELECT COUNT(*) FROM megamind_chunks WHERE realm_id = %s) as total_count
                    FROM megamind_chunk_tags ct
                    JOIN megamind_chunks c ON ct.chunk_id = c.chunk_id
                    WHERE c.realm_id = %s AND ct.tag_type = 'subsystem' AND ct.tag_value = %s
                """, (realm_id, realm_id, analytics.most_popular_subsystems[0]))
                
                result = cursor.fetchone()
                if result and result[1] > 0:
                    dominance_ratio = result[0] / result[1]
                    if dominance_ratio > 0.5:
                        insight = PerformanceInsight(
                            insight_id=f"insight_{uuid.uuid4().hex[:12]}",
                            realm_id=realm_id,
                            insight_type="distribution",
                            severity="low",
                            title="Unbalanced Knowledge Distribution",
                            description=f"The '{analytics.most_popular_subsystems[0]}' subsystem dominates with {dominance_ratio:.1%} of chunks.",
                            recommendation="Consider diversifying knowledge content or splitting large subsystems into more specific categories.",
                            impact_score=0.4,
                            created_at=datetime.now(),
                            expires_at=datetime.now() + timedelta(days=60)
                        )
                        insights.append(insight)
                cursor.close()
            
            # Stale content insight
            cursor = self.db.cursor()
            cursor.execute("""
                SELECT COUNT(*) as stale_count,
                       (SELECT COUNT(*) FROM megamind_chunks WHERE realm_id = %s) as total_count
                FROM megamind_chunks 
                WHERE realm_id = %s 
                AND last_accessed < DATE_SUB(NOW(), INTERVAL 90 DAY)
                AND access_count <= 2
            """, (realm_id, realm_id))
            
            result = cursor.fetchone()
            if result and result[1] > 0:
                stale_ratio = result[0] / result[1]
                if stale_ratio > 0.3:
                    insight = PerformanceInsight(
                        insight_id=f"insight_{uuid.uuid4().hex[:12]}",
                        realm_id=realm_id,
                        insight_type="content_health",
                        severity="medium",
                        title="Stale Content Detected",
                        description=f"{stale_ratio:.1%} of chunks haven't been accessed in 90+ days and have low usage.",
                        recommendation="Review stale content for relevance, update outdated information, or consider archival for unused chunks.",
                        impact_score=0.5,
                        created_at=datetime.now(),
                        expires_at=datetime.now() + timedelta(days=30)
                    )
                    insights.append(insight)
            cursor.close()
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Failed to generate performance insights: {e}")
            return insights
    
    # ===================================================================
    # Realm Health Monitoring
    # ===================================================================
    
    def assess_realm_health(self, realm_id: str) -> Optional[RealmHealth]:
        """Comprehensive realm health assessment"""
        try:
            cursor = self.db.cursor(dictionary=True)
            
            # Chunk distribution by type and subsystem
            dist_query = """
            SELECT 
                c.chunk_type,
                ct.tag_value as subsystem,
                COUNT(*) as count
            FROM megamind_chunks c
            LEFT JOIN megamind_chunk_tags ct ON c.chunk_id = ct.chunk_id AND ct.tag_type = 'subsystem'
            WHERE c.realm_id = %s
            GROUP BY c.chunk_type, ct.tag_value
            """
            
            cursor.execute(dist_query, (realm_id,))
            dist_results = cursor.fetchall()
            
            chunk_distribution = {}
            for row in dist_results:
                key = f"{row['chunk_type']}_{row['subsystem'] or 'uncategorized'}"
                chunk_distribution[key] = row['count']
            
            # Access patterns analysis
            access_query = """
            SELECT 
                AVG(access_count) as avg_access,
                MAX(access_count) as max_access,
                MIN(access_count) as min_access,
                STDDEV(access_count) as access_stddev,
                COUNT(CASE WHEN access_count = 1 THEN 1 END) as never_accessed_count,
                COUNT(*) as total_chunks
            FROM megamind_chunks
            WHERE realm_id = %s
            """
            
            cursor.execute(access_query, (realm_id,))
            access_data = cursor.fetchone()
            
            access_patterns = {
                "avg_access": float(access_data['avg_access'] or 0),
                "max_access": access_data['max_access'] or 0,
                "min_access": access_data['min_access'] or 0,
                "access_variance": float(access_data['access_stddev'] or 0) ** 2,
                "never_accessed_ratio": (access_data['never_accessed_count'] or 0) / max(access_data['total_chunks'] or 1, 1)
            }
            
            # Relationship health
            rel_query = """
            SELECT 
                COUNT(*) as total_relationships,
                AVG(strength) as avg_strength,
                COUNT(DISTINCT relationship_type) as relationship_types,
                COUNT(DISTINCT chunk_id) as chunks_with_relationships,
                (SELECT COUNT(*) FROM megamind_chunks WHERE realm_id = %s) as total_chunks
            FROM megamind_chunk_relationships cr
            JOIN megamind_chunks c ON cr.chunk_id = c.chunk_id
            WHERE c.realm_id = %s
            """
            
            cursor.execute(rel_query, (realm_id, realm_id))
            rel_data = cursor.fetchone()
            
            relationship_health = {
                "total_relationships": rel_data['total_relationships'] or 0,
                "avg_strength": float(rel_data['avg_strength'] or 0),
                "relationship_types": rel_data['relationship_types'] or 0,
                "connected_chunks_ratio": (rel_data['chunks_with_relationships'] or 0) / max(rel_data['total_chunks'] or 1, 1),
                "relationship_density": (rel_data['total_relationships'] or 0) / max(rel_data['total_chunks'] or 1, 1)
            }
            
            # Inheritance status (for project realms)
            inheritance_query = """
            SELECT 
                ri.inheritance_type,
                COUNT(DISTINCT gc.chunk_id) as inherited_chunks
            FROM megamind_realm_inheritance ri
            LEFT JOIN megamind_chunks gc ON ri.parent_realm_id = gc.realm_id
            WHERE ri.child_realm_id = %s
            GROUP BY ri.inheritance_type
            """
            
            cursor.execute(inheritance_query, (realm_id,))
            inheritance_results = cursor.fetchall()
            
            inheritance_status = {}
            for row in inheritance_results:
                inheritance_status[row['inheritance_type']] = row['inherited_chunks']
            
            # Performance metrics
            perf_query = """
            SELECT 
                metric_type,
                SUM(metric_value) as total_value
            FROM megamind_realm_metrics
            WHERE realm_id = %s 
            AND period_start >= DATE_SUB(NOW(), INTERVAL 7 DAY)
            GROUP BY metric_type
            """
            
            cursor.execute(perf_query, (realm_id,))
            perf_results = cursor.fetchall()
            
            performance_metrics = {}
            for row in perf_results:
                performance_metrics[row['metric_type']] = row['total_value']
            
            # Calculate overall health score
            health_score = self._calculate_health_score(
                access_patterns, relationship_health, chunk_distribution
            )
            
            return RealmHealth(
                realm_id=realm_id,
                health_score=health_score,
                chunk_distribution=chunk_distribution,
                access_patterns=access_patterns,
                relationship_health=relationship_health,
                inheritance_status=inheritance_status,
                performance_metrics=performance_metrics
            )
            
        except Exception as e:
            self.logger.error(f"Failed to assess realm health: {e}")
            return None
        finally:
            if cursor:
                cursor.close()
    
    # ===================================================================
    # Reporting and Dashboards
    # ===================================================================
    
    def generate_realm_report(self, realm_id: str, include_insights: bool = True) -> Dict[str, Any]:
        """Generate comprehensive realm report"""
        try:
            # Get all analytics components
            usage_analytics = self.analyze_realm_usage(realm_id)
            realm_health = self.assess_realm_health(realm_id)
            insights = self.generate_performance_insights(realm_id) if include_insights else []
            recent_metrics = self.get_realm_metrics(realm_id, days_back=7)
            
            report = {
                "realm_id": realm_id,
                "report_generated": datetime.now().isoformat(),
                "summary": {
                    "total_chunks": usage_analytics.total_chunks if usage_analytics else 0,
                    "health_score": realm_health.health_score if realm_health else 0,
                    "engagement_score": usage_analytics.user_engagement_score if usage_analytics else 0,
                    "insights_count": len(insights)
                },
                "usage_analytics": asdict(usage_analytics) if usage_analytics else None,
                "realm_health": asdict(realm_health) if realm_health else None,
                "performance_insights": [asdict(insight) for insight in insights],
                "recent_metrics": [asdict(metric) for metric in recent_metrics]
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate realm report: {e}")
            return {"error": f"Report generation failed: {e}"}
    
    def get_cross_realm_analytics(self) -> Dict[str, Any]:
        """Get analytics across all realms"""
        try:
            cursor = self.db.cursor(dictionary=True)
            
            # Get realm comparison data
            comparison_query = """
            SELECT 
                r.realm_id,
                r.realm_name,
                r.realm_type,
                COUNT(c.chunk_id) as chunk_count,
                AVG(c.access_count) as avg_access,
                COUNT(DISTINCT ct.tag_value) as subsystem_count,
                COUNT(cr.relationship_id) as relationship_count
            FROM megamind_realms r
            LEFT JOIN megamind_chunks c ON r.realm_id = c.realm_id
            LEFT JOIN megamind_chunk_tags ct ON c.chunk_id = ct.chunk_id AND ct.tag_type = 'subsystem'
            LEFT JOIN megamind_chunk_relationships cr ON c.chunk_id = cr.chunk_id
            WHERE r.is_active = TRUE
            GROUP BY r.realm_id, r.realm_name, r.realm_type
            ORDER BY chunk_count DESC
            """
            
            cursor.execute(comparison_query)
            realm_data = cursor.fetchall()
            
            # Get cross-realm relationships
            cross_realm_query = """
            SELECT 
                cr.source_realm_id,
                cr.target_realm_id,
                COUNT(*) as relationship_count,
                AVG(cr.strength) as avg_strength
            FROM megamind_chunk_relationships cr
            WHERE cr.source_realm_id != cr.target_realm_id
            GROUP BY cr.source_realm_id, cr.target_realm_id
            ORDER BY relationship_count DESC
            """
            
            cursor.execute(cross_realm_query)
            cross_realm_data = cursor.fetchall()
            
            # Calculate system-wide metrics
            system_query = """
            SELECT 
                COUNT(DISTINCT r.realm_id) as total_realms,
                COUNT(DISTINCT c.chunk_id) as total_chunks,
                COUNT(DISTINCT cr.relationship_id) as total_relationships,
                AVG(c.access_count) as system_avg_access
            FROM megamind_realms r
            LEFT JOIN megamind_chunks c ON r.realm_id = c.realm_id
            LEFT JOIN megamind_chunk_relationships cr ON c.chunk_id = cr.chunk_id
            WHERE r.is_active = TRUE
            """
            
            cursor.execute(system_query)
            system_data = cursor.fetchone()
            
            return {
                "system_overview": system_data,
                "realm_comparison": realm_data,
                "cross_realm_relationships": cross_realm_data,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get cross-realm analytics: {e}")
            return {"error": f"Cross-realm analytics failed: {e}"}
        finally:
            if cursor:
                cursor.close()
    
    # ===================================================================
    # Helper Methods
    # ===================================================================
    
    def _get_period_start(self, timestamp: datetime, period: MeasurementPeriod) -> datetime:
        """Get the start of a measurement period"""
        if period == MeasurementPeriod.HOURLY:
            return timestamp.replace(minute=0, second=0, microsecond=0)
        elif period == MeasurementPeriod.DAILY:
            return timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == MeasurementPeriod.WEEKLY:
            days_since_monday = timestamp.weekday()
            return (timestamp - timedelta(days=days_since_monday)).replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == MeasurementPeriod.MONTHLY:
            return timestamp.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            return timestamp
    
    def _get_period_end(self, period_start: datetime, period: MeasurementPeriod) -> datetime:
        """Get the end of a measurement period"""
        if period == MeasurementPeriod.HOURLY:
            return period_start + timedelta(hours=1)
        elif period == MeasurementPeriod.DAILY:
            return period_start + timedelta(days=1)
        elif period == MeasurementPeriod.WEEKLY:
            return period_start + timedelta(weeks=1)
        elif period == MeasurementPeriod.MONTHLY:
            # Handle month boundaries
            if period_start.month == 12:
                return period_start.replace(year=period_start.year + 1, month=1)
            else:
                return period_start.replace(month=period_start.month + 1)
        else:
            return period_start + timedelta(days=1)
    
    def _calculate_health_score(self, access_patterns: Dict, relationship_health: Dict, 
                               chunk_distribution: Dict) -> float:
        """Calculate overall realm health score (0-100)"""
        try:
            # Access health (0-40 points)
            access_score = 0
            if access_patterns['avg_access'] > 1:
                access_score += min(access_patterns['avg_access'] * 10, 25)
            if access_patterns['never_accessed_ratio'] < 0.3:
                access_score += 15
            
            # Relationship health (0-35 points)
            relationship_score = 0
            if relationship_health['connected_chunks_ratio'] > 0.2:
                relationship_score += min(relationship_health['connected_chunks_ratio'] * 50, 20)
            if relationship_health['avg_strength'] > 0.5:
                relationship_score += min(relationship_health['avg_strength'] * 30, 15)
            
            # Distribution health (0-25 points)
            distribution_score = 0
            total_chunks = sum(chunk_distribution.values())
            if total_chunks > 0:
                # Diversity bonus (more subsystems = better)
                unique_subsystems = len(set(key.split('_', 1)[1] for key in chunk_distribution.keys()))
                distribution_score += min(unique_subsystems * 3, 15)
                
                # Balance bonus (no single category dominates)
                max_category = max(chunk_distribution.values()) if chunk_distribution else 0
                if max_category / total_chunks < 0.6:
                    distribution_score += 10
            
            total_score = access_score + relationship_score + distribution_score
            return min(max(total_score, 0), 100)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate health score: {e}")
            return 50.0  # Default neutral score