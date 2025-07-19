"""
Rejection reason tracking and analysis module
Provides comprehensive tracking of rejection reasons and patterns
"""

from typing import List, Dict, Any, Optional, Tuple
import mysql.connector
from mysql.connector import Error
import json
import logging
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import re

logger = logging.getLogger(__name__)

class RejectionTracker:
    """
    Tracks and analyzes rejection reasons for chunks
    Provides insights into common rejection patterns and trends
    """
    
    def __init__(self, chunk_service):
        self.chunk_service = chunk_service
        self.rejection_categories = {
            'technical_accuracy': {
                'name': 'Technical Accuracy',
                'keywords': ['incorrect', 'wrong', 'error', 'bug', 'inaccurate', 'outdated'],
                'color': '#f56565'
            },
            'style_compliance': {
                'name': 'Style Compliance',
                'keywords': ['style', 'format', 'formatting', 'guideline', 'standard'],
                'color': '#ed8936'
            },
            'completeness': {
                'name': 'Completeness',
                'keywords': ['incomplete', 'missing', 'partial', 'unfinished', 'empty'],
                'color': '#4299e1'
            },
            'quality': {
                'name': 'Quality Issues',
                'keywords': ['quality', 'poor', 'unclear', 'confusing', 'vague'],
                'color': '#9f7aea'
            },
            'security': {
                'name': 'Security Concerns',
                'keywords': ['security', 'sensitive', 'private', 'confidential', 'credential'],
                'color': '#f093fb'
            },
            'duplicate': {
                'name': 'Duplicate Content',
                'keywords': ['duplicate', 'redundant', 'repeat', 'already exists'],
                'color': '#38b2ac'
            },
            'compliance': {
                'name': 'Regulatory Compliance',
                'keywords': ['compliance', 'regulation', 'legal', 'policy', 'requirement'],
                'color': '#68d391'
            },
            'other': {
                'name': 'Other',
                'keywords': [],
                'color': '#a0aec0'
            }
        }
    
    def categorize_rejection_reason(self, reason: str) -> str:
        """
        Categorize rejection reason based on keywords
        
        Args:
            reason: The rejection reason text
            
        Returns:
            Category name
        """
        if not reason:
            return 'other'
        
        reason_lower = reason.lower()
        
        # Check each category for keyword matches
        for category, config in self.rejection_categories.items():
            if category == 'other':
                continue
                
            for keyword in config['keywords']:
                if keyword in reason_lower:
                    return category
        
        return 'other'
    
    def extract_rejection_severity(self, reason: str) -> str:
        """
        Extract severity level from rejection reason
        
        Args:
            reason: The rejection reason text
            
        Returns:
            Severity level (critical, high, medium, low)
        """
        if not reason:
            return 'medium'
        
        reason_lower = reason.lower()
        
        # Critical severity indicators
        critical_keywords = ['critical', 'severe', 'dangerous', 'security', 'vulnerability']
        if any(keyword in reason_lower for keyword in critical_keywords):
            return 'critical'
        
        # High severity indicators
        high_keywords = ['major', 'serious', 'important', 'urgent', 'broken']
        if any(keyword in reason_lower for keyword in high_keywords):
            return 'high'
        
        # Low severity indicators
        low_keywords = ['minor', 'small', 'cosmetic', 'suggestion', 'improvement']
        if any(keyword in reason_lower for keyword in low_keywords):
            return 'low'
        
        return 'medium'
    
    def analyze_rejection_patterns(self, days: int = 30) -> Dict[str, Any]:
        """
        Analyze rejection patterns over a specified period
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Analysis results including patterns, trends, and recommendations
        """
        try:
            conn = self.chunk_service._get_connection()
            cursor = conn.cursor(dictionary=True)
            
            # Get rejections from the specified period
            start_date = datetime.now() - timedelta(days=days)
            
            query = """
            SELECT chunk_id, rejection_reason, rejected_by, rejected_at,
                   source_document, chunk_type, complexity_score
            FROM megamind_chunks 
            WHERE approval_status = 'rejected' 
              AND rejected_at >= %s
            ORDER BY rejected_at DESC
            """
            
            cursor.execute(query, (start_date,))
            rejections = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            # Analyze patterns
            analysis = self._analyze_rejection_data(rejections)
            analysis['period_days'] = days
            analysis['total_rejections'] = len(rejections)
            
            return analysis
            
        except Error as e:
            logger.error(f"Error analyzing rejection patterns: {e}")
            return {
                'error': str(e),
                'period_days': days,
                'total_rejections': 0
            }
    
    def _analyze_rejection_data(self, rejections: List[Dict]) -> Dict[str, Any]:
        """
        Analyze rejection data to extract patterns and insights
        
        Args:
            rejections: List of rejection records
            
        Returns:
            Analysis results
        """
        if not rejections:
            return {
                'categories': {},
                'severities': {},
                'trends': {},
                'top_reasons': [],
                'recommendations': []
            }
        
        # Category analysis
        categories = Counter()
        severities = Counter()
        by_source = defaultdict(list)
        by_type = defaultdict(list)
        by_reviewer = defaultdict(list)
        reasons_text = []
        
        for rejection in rejections:
            reason = rejection['rejection_reason'] or ''
            
            # Categorize
            category = self.categorize_rejection_reason(reason)
            categories[category] += 1
            
            # Severity
            severity = self.extract_rejection_severity(reason)
            severities[severity] += 1
            
            # Group by various attributes
            by_source[rejection['source_document']].append(rejection)
            by_type[rejection['chunk_type']].append(rejection)
            by_reviewer[rejection['rejected_by']].append(rejection)
            
            reasons_text.append(reason)
        
        # Generate analysis
        analysis = {
            'categories': dict(categories),
            'severities': dict(severities),
            'by_source_document': {k: len(v) for k, v in by_source.items()},
            'by_chunk_type': {k: len(v) for k, v in by_type.items()},
            'by_reviewer': {k: len(v) for k, v in by_reviewer.items()},
            'top_reasons': self._extract_top_reasons(reasons_text),
            'recommendations': self._generate_recommendations(categories, severities, by_source, by_type),
            'trends': self._analyze_trends(rejections)
        }
        
        return analysis
    
    def _extract_top_reasons(self, reasons: List[str], limit: int = 10) -> List[Dict[str, Any]]:
        """
        Extract the most common rejection reasons
        
        Args:
            reasons: List of rejection reason texts
            limit: Maximum number of reasons to return
            
        Returns:
            List of top reasons with counts and categories
        """
        # Clean and normalize reasons
        cleaned_reasons = []
        for reason in reasons:
            if reason:
                # Remove extra whitespace and normalize
                cleaned = ' '.join(reason.split())
                cleaned_reasons.append(cleaned)
        
        # Count occurrences
        reason_counts = Counter(cleaned_reasons)
        
        # Get top reasons
        top_reasons = []
        for reason, count in reason_counts.most_common(limit):
            category = self.categorize_rejection_reason(reason)
            severity = self.extract_rejection_severity(reason)
            
            top_reasons.append({
                'reason': reason,
                'count': count,
                'category': category,
                'severity': severity,
                'percentage': (count / len(reasons)) * 100 if reasons else 0
            })
        
        return top_reasons
    
    def _generate_recommendations(self, categories: Counter, severities: Counter, 
                                by_source: Dict, by_type: Dict) -> List[Dict[str, Any]]:
        """
        Generate recommendations based on rejection patterns
        
        Args:
            categories: Category counts
            severities: Severity counts
            by_source: Rejections by source document
            by_type: Rejections by chunk type
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Most common category
        if categories:
            top_category = categories.most_common(1)[0][0]
            category_config = self.rejection_categories.get(top_category, {})
            
            recommendations.append({
                'type': 'category_focus',
                'title': f'Focus on {category_config.get("name", top_category)}',
                'description': f'Most rejections ({categories[top_category]}) are related to {category_config.get("name", top_category)}',
                'priority': 'high',
                'action': f'Review and improve {category_config.get("name", top_category)} guidelines'
            })
        
        # High severity issues
        if severities.get('critical', 0) > 0:
            recommendations.append({
                'type': 'severity_alert',
                'title': 'Critical Issues Detected',
                'description': f'{severities["critical"]} critical severity rejections found',
                'priority': 'urgent',
                'action': 'Immediately review and address critical issues'
            })
        
        # Source document with many rejections
        if by_source:
            problematic_sources = [(source, len(rejections)) for source, rejections in by_source.items()]
            problematic_sources.sort(key=lambda x: x[1], reverse=True)
            
            if problematic_sources[0][1] > 5:  # More than 5 rejections
                recommendations.append({
                    'type': 'source_review',
                    'title': f'Review {problematic_sources[0][0]}',
                    'description': f'This source has {problematic_sources[0][1]} rejections',
                    'priority': 'medium',
                    'action': f'Comprehensive review of {problematic_sources[0][0]} needed'
                })
        
        # Chunk type with many rejections
        if by_type:
            problematic_types = [(chunk_type, len(rejections)) for chunk_type, rejections in by_type.items()]
            problematic_types.sort(key=lambda x: x[1], reverse=True)
            
            if problematic_types[0][1] > 3:  # More than 3 rejections
                recommendations.append({
                    'type': 'type_improvement',
                    'title': f'Improve {problematic_types[0][0]} Quality',
                    'description': f'{problematic_types[0][0]} chunks have {problematic_types[0][1]} rejections',
                    'priority': 'medium',
                    'action': f'Review chunking strategy for {problematic_types[0][0]} content'
                })
        
        return recommendations
    
    def _analyze_trends(self, rejections: List[Dict]) -> Dict[str, Any]:
        """
        Analyze rejection trends over time
        
        Args:
            rejections: List of rejection records
            
        Returns:
            Trend analysis
        """
        if not rejections:
            return {}
        
        # Group by day
        by_day = defaultdict(list)
        for rejection in rejections:
            if rejection['rejected_at']:
                day = rejection['rejected_at'].date()
                by_day[day].append(rejection)
        
        # Calculate trend
        daily_counts = [(day, len(rejections)) for day, rejections in by_day.items()]
        daily_counts.sort(key=lambda x: x[0])
        
        if len(daily_counts) >= 2:
            # Simple trend calculation
            recent_avg = sum(count for _, count in daily_counts[-7:]) / min(7, len(daily_counts))
            overall_avg = sum(count for _, count in daily_counts) / len(daily_counts)
            
            trend = 'increasing' if recent_avg > overall_avg else 'decreasing'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'daily_counts': daily_counts,
            'recent_average': recent_avg if len(daily_counts) >= 2 else 0,
            'overall_average': sum(count for _, count in daily_counts) / len(daily_counts) if daily_counts else 0
        }
    
    def get_rejection_templates(self) -> List[Dict[str, Any]]:
        """
        Get rejection reason templates organized by category
        
        Returns:
            List of rejection templates
        """
        templates = []
        
        for category, config in self.rejection_categories.items():
            if category == 'other':
                continue
            
            category_templates = self._get_category_templates(category)
            
            for template in category_templates:
                templates.append({
                    'category': category,
                    'category_name': config['name'],
                    'template': template,
                    'color': config['color']
                })
        
        return templates
    
    def _get_category_templates(self, category: str) -> List[str]:
        """
        Get rejection templates for a specific category
        
        Args:
            category: Category name
            
        Returns:
            List of template strings
        """
        templates = {
            'technical_accuracy': [
                'Technical information is incorrect or outdated',
                'Code examples are broken or non-functional',
                'API documentation does not match current implementation',
                'Contains factual errors that need correction',
                'Version information is outdated'
            ],
            'style_compliance': [
                'Does not meet style guide requirements',
                'Formatting is inconsistent with project standards',
                'Code formatting needs improvement',
                'Documentation structure does not follow template',
                'Naming conventions are not followed'
            ],
            'completeness': [
                'Content is incomplete or missing key information',
                'Missing required sections or documentation',
                'Examples are incomplete or insufficient',
                'Important details are not explained',
                'Incomplete implementation or partial content'
            ],
            'quality': [
                'Content quality is below standards',
                'Unclear or confusing explanations',
                'Poor organization and structure',
                'Lacks sufficient detail for understanding',
                'Content is not well-written or professional'
            ],
            'security': [
                'Contains sensitive information that should not be included',
                'Security vulnerabilities or concerns identified',
                'Exposes credentials or private information',
                'Does not follow security best practices',
                'Contains potentially dangerous code examples'
            ],
            'duplicate': [
                'Duplicate content already exists elsewhere',
                'Redundant information that should be consolidated',
                'Similar content found in other sections',
                'Overlaps with existing documentation',
                'Content should be merged with existing material'
            ],
            'compliance': [
                'Does not meet regulatory requirements',
                'Violates company policies or guidelines',
                'Legal compliance issues identified',
                'Does not follow required procedures',
                'Regulatory standards not met'
            ]
        }
        
        return templates.get(category, [])
    
    def track_rejection_resolution(self, chunk_id: str, resolution_action: str, 
                                 resolution_notes: str = '') -> Dict[str, Any]:
        """
        Track how rejections are resolved
        
        Args:
            chunk_id: ID of the chunk
            resolution_action: Action taken (revised, deleted, approved_override)
            resolution_notes: Notes about the resolution
            
        Returns:
            Tracking result
        """
        try:
            conn = self.chunk_service._get_connection()
            cursor = conn.cursor(dictionary=True)
            
            # Get current rejection info
            chunk_query = """
            SELECT rejection_reason, rejected_by, rejected_at, approval_status
            FROM megamind_chunks 
            WHERE chunk_id = %s
            """
            
            cursor.execute(chunk_query, (chunk_id,))
            chunk_info = cursor.fetchone()
            
            if not chunk_info:
                return {'success': False, 'error': 'Chunk not found'}
            
            # Create resolution tracking record
            resolution_record = {
                'chunk_id': chunk_id,
                'original_rejection_reason': chunk_info['rejection_reason'],
                'rejected_by': chunk_info['rejected_by'],
                'rejected_at': chunk_info['rejected_at'].isoformat() if chunk_info['rejected_at'] else None,
                'resolution_action': resolution_action,
                'resolution_notes': resolution_notes,
                'resolution_timestamp': datetime.now().isoformat()
            }
            
            # Store in rejection tracking table (would need to be created)
            # For now, we'll store as JSON in a text field or log it
            
            cursor.close()
            conn.close()
            
            logger.info(f"Tracked rejection resolution for chunk {chunk_id}: {resolution_action}")
            
            return {
                'success': True,
                'resolution_record': resolution_record
            }
            
        except Error as e:
            logger.error(f"Error tracking rejection resolution: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_rejection_statistics(self, days: int = 30) -> Dict[str, Any]:
        """
        Get comprehensive rejection statistics
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Statistics including rates, categories, and trends
        """
        try:
            conn = self.chunk_service._get_connection()
            cursor = conn.cursor(dictionary=True)
            
            start_date = datetime.now() - timedelta(days=days)
            
            # Get overall statistics
            stats_query = """
            SELECT 
                COUNT(*) as total_chunks,
                SUM(CASE WHEN approval_status = 'approved' THEN 1 ELSE 0 END) as approved_count,
                SUM(CASE WHEN approval_status = 'rejected' THEN 1 ELSE 0 END) as rejected_count,
                SUM(CASE WHEN approval_status = 'pending' THEN 1 ELSE 0 END) as pending_count
            FROM megamind_chunks 
            WHERE created_at >= %s
            """
            
            cursor.execute(stats_query, (start_date,))
            stats = cursor.fetchone()
            
            # Calculate rates
            total = stats['total_chunks']
            rejection_rate = (stats['rejected_count'] / total * 100) if total > 0 else 0
            approval_rate = (stats['approved_count'] / total * 100) if total > 0 else 0
            
            # Get detailed rejection analysis
            rejection_analysis = self.analyze_rejection_patterns(days)
            
            cursor.close()
            conn.close()
            
            return {
                'period_days': days,
                'total_chunks': total,
                'approved_count': stats['approved_count'],
                'rejected_count': stats['rejected_count'],
                'pending_count': stats['pending_count'],
                'rejection_rate': rejection_rate,
                'approval_rate': approval_rate,
                'rejection_analysis': rejection_analysis,
                'generated_at': datetime.now().isoformat()
            }
            
        except Error as e:
            logger.error(f"Error getting rejection statistics: {e}")
            return {
                'error': str(e),
                'period_days': days,
                'generated_at': datetime.now().isoformat()
            }
    
    def export_rejection_data(self, days: int = 30, format: str = 'json') -> Dict[str, Any]:
        """
        Export rejection data for analysis or reporting
        
        Args:
            days: Number of days to export
            format: Export format (json, csv)
            
        Returns:
            Export result with data or file path
        """
        try:
            # Get comprehensive data
            statistics = self.get_rejection_statistics(days)
            patterns = self.analyze_rejection_patterns(days)
            templates = self.get_rejection_templates()
            
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'period_days': days,
                'statistics': statistics,
                'patterns': patterns,
                'templates': templates,
                'categories': self.rejection_categories
            }
            
            if format == 'json':
                return {
                    'success': True,
                    'data': export_data,
                    'format': 'json'
                }
            elif format == 'csv':
                # Convert to CSV format (simplified)
                csv_data = self._convert_to_csv(export_data)
                return {
                    'success': True,
                    'data': csv_data,
                    'format': 'csv'
                }
            else:
                return {
                    'success': False,
                    'error': f'Unsupported format: {format}'
                }
                
        except Exception as e:
            logger.error(f"Error exporting rejection data: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _convert_to_csv(self, data: Dict) -> str:
        """
        Convert rejection data to CSV format
        
        Args:
            data: Export data dictionary
            
        Returns:
            CSV formatted string
        """
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write headers
        writer.writerow(['Metric', 'Value'])
        
        # Write statistics
        stats = data.get('statistics', {})
        for key, value in stats.items():
            if key not in ['rejection_analysis', 'generated_at']:
                writer.writerow([key, value])
        
        # Write category data
        patterns = data.get('patterns', {})
        categories = patterns.get('categories', {})
        
        writer.writerow([])  # Empty row
        writer.writerow(['Category', 'Count'])
        
        for category, count in categories.items():
            writer.writerow([category, count])
        
        return output.getvalue()