"""
Automated Curation System
Implements workflows for automatic content curation and quality enforcement
"""

from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class CurationAction(Enum):
    """Types of curation actions"""
    APPROVE = "approve"
    REJECT = "reject"
    IMPROVE = "improve"
    MERGE = "merge"
    SPLIT = "split"
    ARCHIVE = "archive"
    PROMOTE = "promote"
    
class WorkflowStage(Enum):
    """Stages in curation workflow"""
    INTAKE = "intake"
    QUALITY_CHECK = "quality_check"
    IMPROVEMENT = "improvement"
    REVIEW = "review"
    APPROVAL = "approval"
    DEPLOYMENT = "deployment"

@dataclass
class CurationRule:
    """Rule for automated curation decisions"""
    name: str
    condition: Callable[[Dict[str, Any]], bool]
    action: CurationAction
    parameters: Dict[str, Any]
    priority: int = 0
    
@dataclass
class CurationDecision:
    """Decision made by curator"""
    chunk_id: str
    action: CurationAction
    reason: str
    confidence: float
    parameters: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    
@dataclass
class CurationWorkflow:
    """Complete curation workflow"""
    workflow_id: str
    name: str
    stages: List[WorkflowStage]
    rules: List[CurationRule]
    created: datetime = field(default_factory=datetime.now)
    
@dataclass
class CurationReport:
    """Report of curation activities"""
    period_start: datetime
    period_end: datetime
    total_processed: int
    decisions: Dict[str, int]
    quality_improvements: Dict[str, float]
    automated_rate: float

class AutomatedCurator:
    """System for automated content curation"""
    
    def __init__(self, quality_threshold: float = 0.7):
        self.quality_threshold = quality_threshold
        self.workflows = self._initialize_workflows()
        self.rules = self._initialize_rules()
        self.decision_history: List[CurationDecision] = []
        self.performance_metrics = defaultdict(float)
        
    def _initialize_workflows(self) -> Dict[str, CurationWorkflow]:
        """Initialize standard curation workflows"""
        workflows = {}
        
        # Standard quality workflow
        workflows['standard_quality'] = CurationWorkflow(
            workflow_id='standard_quality',
            name='Standard Quality Curation',
            stages=[
                WorkflowStage.INTAKE,
                WorkflowStage.QUALITY_CHECK,
                WorkflowStage.IMPROVEMENT,
                WorkflowStage.REVIEW,
                WorkflowStage.APPROVAL,
                WorkflowStage.DEPLOYMENT
            ],
            rules=[]
        )
        
        # Fast-track workflow for high-quality content
        workflows['fast_track'] = CurationWorkflow(
            workflow_id='fast_track',
            name='Fast Track High Quality',
            stages=[
                WorkflowStage.INTAKE,
                WorkflowStage.QUALITY_CHECK,
                WorkflowStage.APPROVAL,
                WorkflowStage.DEPLOYMENT
            ],
            rules=[]
        )
        
        # Remediation workflow for low-quality content
        workflows['remediation'] = CurationWorkflow(
            workflow_id='remediation',
            name='Quality Remediation',
            stages=[
                WorkflowStage.INTAKE,
                WorkflowStage.QUALITY_CHECK,
                WorkflowStage.IMPROVEMENT,
                WorkflowStage.IMPROVEMENT,  # Double improvement
                WorkflowStage.REVIEW,
                WorkflowStage.APPROVAL
            ],
            rules=[]
        )
        
        return workflows
        
    def _initialize_rules(self) -> List[CurationRule]:
        """Initialize curation rules"""
        rules = []
        
        # High quality auto-approval
        rules.append(CurationRule(
            name='high_quality_auto_approve',
            condition=lambda chunk: chunk.get('quality_score', 0) >= 0.85,
            action=CurationAction.APPROVE,
            parameters={'reason': 'High quality score'},
            priority=10
        ))
        
        # Low quality rejection
        rules.append(CurationRule(
            name='low_quality_reject',
            condition=lambda chunk: chunk.get('quality_score', 0) < 0.4,
            action=CurationAction.REJECT,
            parameters={'reason': 'Quality below minimum threshold'},
            priority=20
        ))
        
        # Duplicate content merge
        rules.append(CurationRule(
            name='duplicate_merge',
            condition=lambda chunk: chunk.get('similarity_score', 0) > 0.95,
            action=CurationAction.MERGE,
            parameters={'merge_strategy': 'keep_highest_quality'},
            priority=15
        ))
        
        # Large chunk split
        rules.append(CurationRule(
            name='large_chunk_split',
            condition=lambda chunk: chunk.get('token_count', 0) > 1000,
            action=CurationAction.SPLIT,
            parameters={'target_size': 512},
            priority=5
        ))
        
        # Outdated content archive
        rules.append(CurationRule(
            name='outdated_archive',
            condition=lambda chunk: self._is_outdated(chunk),
            action=CurationAction.ARCHIVE,
            parameters={'archive_reason': 'Content outdated'},
            priority=8
        ))
        
        # High-value promotion
        rules.append(CurationRule(
            name='high_value_promote',
            condition=lambda chunk: self._is_high_value(chunk),
            action=CurationAction.PROMOTE,
            parameters={'target_realm': 'GLOBAL'},
            priority=12
        ))
        
        return sorted(rules, key=lambda x: x.priority, reverse=True)
        
    def curate_chunk(self, chunk: Dict[str, Any], workflow_id: str = 'standard_quality') -> CurationDecision:
        """Curate a single chunk through workflow"""
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            workflow = self.workflows['standard_quality']
            
        # Apply rules in priority order
        for rule in self.rules:
            if rule.condition(chunk):
                decision = CurationDecision(
                    chunk_id=chunk['chunk_id'],
                    action=rule.action,
                    reason=rule.parameters.get('reason', rule.name),
                    confidence=0.8,  # Base confidence
                    parameters=rule.parameters
                )
                
                # Adjust confidence based on quality score
                if 'quality_score' in chunk:
                    if chunk['quality_score'] > 0.8:
                        decision.confidence *= 1.2
                    elif chunk['quality_score'] < 0.5:
                        decision.confidence *= 0.8
                        
                decision.confidence = min(decision.confidence, 1.0)
                
                self.decision_history.append(decision)
                return decision
                
        # Default decision if no rules match
        return CurationDecision(
            chunk_id=chunk['chunk_id'],
            action=CurationAction.IMPROVE,
            reason='No specific rules matched, recommend improvement',
            confidence=0.6,
            parameters={}
        )
        
    def curate_batch(self, chunks: List[Dict[str, Any]], workflow_id: str = 'standard_quality') -> List[CurationDecision]:
        """Curate multiple chunks"""
        decisions = []
        
        # Group similar chunks for potential merging
        similarity_groups = self._find_similar_chunks(chunks)
        
        for chunk in chunks:
            # Check if chunk is part of a similarity group
            group_id = next(
                (gid for gid, group in similarity_groups.items() if chunk['chunk_id'] in group),
                None
            )
            
            if group_id and len(similarity_groups[group_id]) > 1:
                # Handle as group
                chunk['similarity_score'] = 0.96  # Mark for merge rule
                
            decision = self.curate_chunk(chunk, workflow_id)
            decisions.append(decision)
            
        return decisions
        
    def apply_decision(self, chunk: Dict[str, Any], decision: CurationDecision) -> Dict[str, Any]:
        """Apply curation decision to chunk"""
        if decision.action == CurationAction.APPROVE:
            chunk['curation_status'] = 'approved'
            chunk['approved_date'] = datetime.now().isoformat()
            
        elif decision.action == CurationAction.REJECT:
            chunk['curation_status'] = 'rejected'
            chunk['rejection_reason'] = decision.reason
            
        elif decision.action == CurationAction.IMPROVE:
            # This would trigger improvement workflow
            chunk['curation_status'] = 'improvement_needed'
            chunk['improvement_priority'] = decision.parameters.get('priority', 'medium')
            
        elif decision.action == CurationAction.ARCHIVE:
            chunk['curation_status'] = 'archived'
            chunk['archive_date'] = datetime.now().isoformat()
            
        elif decision.action == CurationAction.PROMOTE:
            chunk['promotion_requested'] = True
            chunk['target_realm'] = decision.parameters.get('target_realm', 'GLOBAL')
            
        chunk['last_curated'] = datetime.now().isoformat()
        chunk['curation_confidence'] = decision.confidence
        
        return chunk
        
    def execute_workflow(self, chunks: List[Dict[str, Any]], workflow_id: str) -> Dict[str, Any]:
        """Execute complete curation workflow"""
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
            
        results = {
            'workflow_id': workflow_id,
            'total_chunks': len(chunks),
            'stage_results': {},
            'final_decisions': []
        }
        
        current_chunks = chunks.copy()
        
        for stage in workflow.stages:
            stage_results = self._execute_stage(stage, current_chunks)
            results['stage_results'][stage.value] = stage_results
            
            # Update chunks based on stage results
            if stage == WorkflowStage.QUALITY_CHECK:
                # Filter out rejected chunks
                current_chunks = [
                    c for c in current_chunks
                    if c.get('quality_score', 0) >= self.quality_threshold * 0.5
                ]
                
            elif stage == WorkflowStage.IMPROVEMENT:
                # Apply improvements
                current_chunks = [
                    self._apply_improvements(c) for c in current_chunks
                ]
                
        # Final curation decisions
        final_decisions = self.curate_batch(current_chunks, workflow_id)
        results['final_decisions'] = final_decisions
        
        # Apply decisions
        for chunk, decision in zip(current_chunks, final_decisions):
            self.apply_decision(chunk, decision)
            
        return results
        
    def _execute_stage(self, stage: WorkflowStage, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute a single workflow stage"""
        stage_result = {
            'stage': stage.value,
            'chunks_processed': len(chunks),
            'outcomes': defaultdict(int)
        }
        
        if stage == WorkflowStage.INTAKE:
            # Validate and prepare chunks
            for chunk in chunks:
                if self._validate_chunk(chunk):
                    stage_result['outcomes']['valid'] += 1
                else:
                    stage_result['outcomes']['invalid'] += 1
                    
        elif stage == WorkflowStage.QUALITY_CHECK:
            # Check quality scores
            for chunk in chunks:
                quality = chunk.get('quality_score', 0)
                if quality >= self.quality_threshold:
                    stage_result['outcomes']['passed'] += 1
                else:
                    stage_result['outcomes']['failed'] += 1
                    
        elif stage == WorkflowStage.IMPROVEMENT:
            # Count improvements applied
            for chunk in chunks:
                if chunk.get('improvements_applied', 0) > 0:
                    stage_result['outcomes']['improved'] += 1
                else:
                    stage_result['outcomes']['unchanged'] += 1
                    
        return stage_result
        
    def _find_similar_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Find groups of similar chunks"""
        similarity_groups = defaultdict(list)
        
        # Simple grouping by content similarity
        # In practice, this would use embeddings
        for i, chunk1 in enumerate(chunks):
            for j, chunk2 in enumerate(chunks[i+1:], i+1):
                similarity = self._calculate_similarity(chunk1, chunk2)
                if similarity > 0.9:
                    group_id = f"group_{i}"
                    similarity_groups[group_id].extend([
                        chunk1['chunk_id'],
                        chunk2['chunk_id']
                    ])
                    
        return similarity_groups
        
    def _calculate_similarity(self, chunk1: Dict[str, Any], chunk2: Dict[str, Any]) -> float:
        """Calculate similarity between chunks"""
        # Simplified similarity calculation
        content1 = chunk1.get('content', '').lower()
        content2 = chunk2.get('content', '').lower()
        
        if not content1 or not content2:
            return 0.0
            
        # Jaccard similarity of words
        words1 = set(content1.split())
        words2 = set(content2.split())
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
        
    def _is_outdated(self, chunk: Dict[str, Any]) -> bool:
        """Check if chunk is outdated"""
        last_updated = chunk.get('last_updated')
        if not last_updated:
            return False
            
        # Parse date
        if isinstance(last_updated, str):
            last_updated = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
            
        # Consider outdated if older than 180 days
        age = datetime.now() - last_updated
        return age > timedelta(days=180)
        
    def _is_high_value(self, chunk: Dict[str, Any]) -> bool:
        """Check if chunk is high value for promotion"""
        quality = chunk.get('quality_score', 0)
        usage = chunk.get('usage_count', 0)
        importance = chunk.get('importance_score', 0)
        
        # High value if high quality, frequently used, and important
        return quality > 0.8 and usage > 50 and importance > 0.7
        
    def _validate_chunk(self, chunk: Dict[str, Any]) -> bool:
        """Validate chunk has required fields"""
        required_fields = ['chunk_id', 'content']
        return all(field in chunk for field in required_fields)
        
    def _apply_improvements(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Apply automated improvements to chunk"""
        # This would integrate with AIQualityImprover
        chunk['improvements_applied'] = chunk.get('improvements_applied', 0) + 1
        chunk['last_improved'] = datetime.now().isoformat()
        return chunk
        
    def generate_curation_report(self, start_date: datetime, end_date: datetime) -> CurationReport:
        """Generate report of curation activities"""
        period_decisions = [
            d for d in self.decision_history
            if start_date <= d.timestamp <= end_date
        ]
        
        decision_counts = defaultdict(int)
        for decision in period_decisions:
            decision_counts[decision.action.value] += 1
            
        # Calculate quality improvements
        quality_improvements = {
            'average_quality_increase': 0.15,  # Would calculate from actual data
            'chunks_improved': decision_counts.get(CurationAction.IMPROVE.value, 0),
            'chunks_promoted': decision_counts.get(CurationAction.PROMOTE.value, 0)
        }
        
        # Calculate automation rate
        total_decisions = len(period_decisions)
        automated_decisions = sum(
            1 for d in period_decisions if d.confidence > 0.7
        )
        automation_rate = automated_decisions / total_decisions if total_decisions > 0 else 0
        
        return CurationReport(
            period_start=start_date,
            period_end=end_date,
            total_processed=total_decisions,
            decisions=dict(decision_counts),
            quality_improvements=quality_improvements,
            automated_rate=automation_rate
        )