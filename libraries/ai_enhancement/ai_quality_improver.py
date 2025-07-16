"""
AI Quality Improvement System
Analyzes quality issues and suggests improvements for chunks
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import json
import re
import logging

logger = logging.getLogger(__name__)

@dataclass
class QualityIssue:
    """Represents a specific quality issue in a chunk"""
    dimension: str
    severity: str  # 'low', 'medium', 'high'
    description: str
    impact_score: float
    
@dataclass
class QualityImprovement:
    """Suggested improvement for a quality issue"""
    issue: QualityIssue
    suggestion: str
    confidence: float
    automated: bool
    implementation: Optional[str] = None

class AIQualityImprover:
    """AI-powered system for improving chunk quality"""
    
    def __init__(self):
        self.improvement_patterns = self._initialize_patterns()
        self.quality_thresholds = {
            'readability': 0.7,
            'technical_accuracy': 0.8,
            'completeness': 0.75,
            'relevance': 0.7,
            'freshness': 0.6,
            'coherence': 0.75,
            'uniqueness': 0.5,
            'authority': 0.6
        }
        
    def _initialize_patterns(self) -> Dict[str, List[Dict]]:
        """Initialize improvement patterns for each quality dimension"""
        return {
            'readability': [
                {
                    'pattern': r'\b\w{20,}\b',
                    'issue': 'Very long words reduce readability',
                    'suggestion': 'Break down complex terminology or add explanations',
                    'automated': False
                },
                {
                    'pattern': r'[^.!?]{200,}',
                    'issue': 'Sentences too long',
                    'suggestion': 'Split into shorter, clearer sentences',
                    'automated': True
                },
                {
                    'pattern': r'(\b\w+\b)(?:\s+\1){2,}',
                    'issue': 'Repeated words',
                    'suggestion': 'Remove duplicate words',
                    'automated': True
                }
            ],
            'technical_accuracy': [
                {
                    'pattern': r'\b(?:TODO|FIXME|XXX|HACK)\b',
                    'issue': 'Contains incomplete technical notes',
                    'suggestion': 'Complete or remove technical debt markers',
                    'automated': False
                },
                {
                    'pattern': r'(?:deprecated|obsolete|outdated)',
                    'issue': 'May contain outdated information',
                    'suggestion': 'Update with current best practices',
                    'automated': False
                }
            ],
            'completeness': [
                {
                    'pattern': r'(?:etc\.?|and so on|and more)$',
                    'issue': 'Incomplete list or explanation',
                    'suggestion': 'Provide complete information or clear boundaries',
                    'automated': False
                },
                {
                    'pattern': r'^\s*[-*]\s*$',
                    'issue': 'Empty list item',
                    'suggestion': 'Remove empty items or add content',
                    'automated': True
                }
            ],
            'coherence': [
                {
                    'pattern': r'^(?:However|But|Although)',
                    'issue': 'Starting with contradiction without context',
                    'suggestion': 'Provide context or merge with previous chunk',
                    'automated': False
                },
                {
                    'pattern': r'(?:this|that|these|those)\s+(?:is|are|was|were)',
                    'issue': 'Unclear references',
                    'suggestion': 'Replace pronouns with specific references',
                    'automated': False
                }
            ]
        }
        
    def analyze_quality_issues(self, chunk: Dict[str, Any], quality_scores: Dict[str, float]) -> List[QualityIssue]:
        """Analyze chunk for quality issues based on scores"""
        issues = []
        
        # Check each dimension against threshold
        for dimension, score in quality_scores.items():
            threshold = self.quality_thresholds.get(dimension, 0.7)
            if score < threshold:
                severity = self._calculate_severity(score, threshold)
                issues.append(QualityIssue(
                    dimension=dimension,
                    severity=severity,
                    description=f"{dimension.replace('_', ' ').title()} below threshold ({score:.2f} < {threshold})",
                    impact_score=threshold - score
                ))
                
        # Pattern-based issue detection
        content = chunk.get('content', '')
        for dimension, patterns in self.improvement_patterns.items():
            for pattern_info in patterns:
                if re.search(pattern_info['pattern'], content, re.IGNORECASE):
                    issues.append(QualityIssue(
                        dimension=dimension,
                        severity='medium',
                        description=pattern_info['issue'],
                        impact_score=0.3
                    ))
                    
        return issues
        
    def suggest_improvements(self, chunk: Dict[str, Any], issues: List[QualityIssue]) -> List[QualityImprovement]:
        """Generate improvement suggestions for identified issues"""
        improvements = []
        content = chunk.get('content', '')
        
        for issue in issues:
            # Pattern-based suggestions
            if issue.dimension in self.improvement_patterns:
                for pattern_info in self.improvement_patterns[issue.dimension]:
                    if pattern_info['issue'] == issue.description:
                        improvement = QualityImprovement(
                            issue=issue,
                            suggestion=pattern_info['suggestion'],
                            confidence=0.8 if pattern_info['automated'] else 0.6,
                            automated=pattern_info['automated']
                        )
                        
                        # Generate automated implementation if possible
                        if pattern_info['automated']:
                            implementation = self._generate_implementation(
                                content, pattern_info['pattern'], issue.dimension
                            )
                            if implementation:
                                improvement.implementation = implementation
                                
                        improvements.append(improvement)
                        
            # Dimension-specific suggestions
            else:
                suggestion = self._generate_dimension_suggestion(issue.dimension, issue.impact_score)
                improvements.append(QualityImprovement(
                    issue=issue,
                    suggestion=suggestion,
                    confidence=0.7,
                    automated=False
                ))
                
        return improvements
        
    def apply_automated_improvements(self, chunk: Dict[str, Any], improvements: List[QualityImprovement]) -> Dict[str, Any]:
        """Apply automated improvements to chunk content"""
        content = chunk.get('content', '')
        applied_count = 0
        
        for improvement in improvements:
            if improvement.automated and improvement.implementation:
                try:
                    # Apply the improvement
                    content = improvement.implementation
                    applied_count += 1
                    logger.info(f"Applied automated improvement for {improvement.issue.dimension}")
                except Exception as e:
                    logger.error(f"Failed to apply improvement: {e}")
                    
        if applied_count > 0:
            chunk['content'] = content
            chunk['quality_improvements_applied'] = applied_count
            chunk['last_improved'] = datetime.now().isoformat()
            
        return chunk
        
    def calculate_improvement_priority(self, improvements: List[QualityImprovement]) -> List[QualityImprovement]:
        """Calculate priority order for improvements"""
        # Sort by impact score and confidence
        return sorted(
            improvements,
            key=lambda x: (x.issue.impact_score * x.confidence, x.automated),
            reverse=True
        )
        
    def _calculate_severity(self, score: float, threshold: float) -> str:
        """Calculate severity level based on how far below threshold"""
        gap = threshold - score
        if gap > 0.3:
            return 'high'
        elif gap > 0.15:
            return 'medium'
        else:
            return 'low'
            
    def _generate_implementation(self, content: str, pattern: str, dimension: str) -> Optional[str]:
        """Generate automated implementation for specific patterns"""
        if dimension == 'readability':
            if 'Repeated words' in pattern:
                # Remove duplicate words
                return re.sub(r'(\b\w+\b)(?:\s+\1)+', r'\1', content)
            elif 'Sentences too long' in pattern:
                # This would need more sophisticated NLP for proper implementation
                return None
                
        elif dimension == 'completeness':
            if 'Empty list item' in pattern:
                # Remove empty list items
                lines = content.split('\n')
                filtered = [line for line in lines if not re.match(r'^\s*[-*]\s*$', line)]
                return '\n'.join(filtered)
                
        return None
        
    def _generate_dimension_suggestion(self, dimension: str, impact_score: float) -> str:
        """Generate generic suggestion for dimension issues"""
        suggestions = {
            'readability': "Simplify language, use shorter sentences, and add clear structure",
            'technical_accuracy': "Verify technical details and update with current information",
            'completeness': "Add missing information and ensure all aspects are covered",
            'relevance': "Focus content on the main topic and remove tangential information",
            'freshness': "Update with recent developments and current best practices",
            'coherence': "Improve logical flow and add connecting statements",
            'uniqueness': "Add distinctive insights or remove redundant content",
            'authority': "Add credible sources and expert validation"
        }
        
        base_suggestion = suggestions.get(dimension, "Improve content quality")
        if impact_score > 0.2:
            return f"{base_suggestion} (significant improvement needed)"
        else:
            return base_suggestion
            
    def generate_improvement_report(self, chunk_id: str, issues: List[QualityIssue], 
                                   improvements: List[QualityImprovement]) -> Dict[str, Any]:
        """Generate comprehensive improvement report"""
        return {
            'chunk_id': chunk_id,
            'timestamp': datetime.now().isoformat(),
            'total_issues': len(issues),
            'issues_by_severity': {
                'high': len([i for i in issues if i.severity == 'high']),
                'medium': len([i for i in issues if i.severity == 'medium']),
                'low': len([i for i in issues if i.severity == 'low'])
            },
            'total_improvements': len(improvements),
            'automated_improvements': len([i for i in improvements if i.automated]),
            'manual_improvements': len([i for i in improvements if not i.automated]),
            'priority_improvements': [
                {
                    'dimension': imp.issue.dimension,
                    'suggestion': imp.suggestion,
                    'confidence': imp.confidence,
                    'automated': imp.automated
                }
                for imp in self.calculate_improvement_priority(improvements)[:5]
            ]
        }