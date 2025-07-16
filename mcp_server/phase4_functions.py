"""
Phase 4 MCP Functions: AI Enhancement
Provides quality improvement, adaptive learning, and performance optimization
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

# Import Phase 4 libraries
try:
    from ..libraries.ai_enhancement import (
        AIQualityImprover, QualityImprovement,
        AdaptiveLearningEngine, UserFeedback, FeedbackType,
        AutomatedCurator, CurationWorkflow,
        PerformanceOptimizer, OptimizationType
    )
except ImportError:
    import sys
    sys.path.append('/Data/MCP_Servers/MegaMind_MCP')
    from libraries.ai_enhancement import (
        AIQualityImprover, QualityImprovement,
        AdaptiveLearningEngine, UserFeedback, FeedbackType,
        AutomatedCurator, CurationWorkflow,
        PerformanceOptimizer, OptimizationType
    )

logger = logging.getLogger(__name__)

class Phase4Functions:
    """MCP function implementations for Phase 4 AI Enhancement"""
    
    def __init__(self, db_manager):
        """Initialize Phase 4 functions with database manager"""
        self.db = db_manager
        self.quality_improver = AIQualityImprover()
        self.learning_engine = AdaptiveLearningEngine()
        self.curator = AutomatedCurator()
        self.optimizer = PerformanceOptimizer()
        
    async def ai_improve_chunk_quality(self, chunk_id: str, session_id: str, 
                                      apply_automated: bool = False) -> Dict[str, Any]:
        """
        Analyze chunk quality and suggest/apply improvements
        
        Args:
            chunk_id: ID of chunk to improve
            session_id: Current session ID
            apply_automated: Whether to automatically apply improvements
            
        Returns:
            Quality improvement report with suggestions and results
        """
        try:
            # Get chunk data
            chunk = await self.db.get_chunk(chunk_id)
            if not chunk:
                return {'error': f'Chunk {chunk_id} not found'}
                
            # Get quality scores
            quality_scores = await self._get_quality_scores(chunk_id)
            
            # Analyze quality issues
            issues = self.quality_improver.analyze_quality_issues(chunk, quality_scores)
            
            # Generate improvement suggestions
            improvements = self.quality_improver.suggest_improvements(chunk, issues)
            
            # Apply automated improvements if requested
            applied_count = 0
            if apply_automated:
                improved_chunk = self.quality_improver.apply_automated_improvements(
                    chunk, improvements
                )
                if improved_chunk.get('quality_improvements_applied', 0) > 0:
                    # Save improved chunk
                    await self._update_chunk_content(chunk_id, improved_chunk['content'])
                    applied_count = improved_chunk['quality_improvements_applied']
                    
            # Store improvement history
            for improvement in improvements:
                await self._store_improvement_history(
                    chunk_id, improvement, session_id,
                    applied=improvement.automated and apply_automated
                )
                
            # Generate report
            report = self.quality_improver.generate_improvement_report(
                chunk_id, issues, improvements
            )
            report['applied_count'] = applied_count
            
            return report
            
        except Exception as e:
            logger.error(f"Error improving chunk quality: {e}")
            return {'error': str(e)}
            
    async def ai_record_user_feedback(self, feedback_type: str, target_id: str,
                                     rating: float, details: Dict[str, Any],
                                     user_id: str, session_id: str) -> Dict[str, Any]:
        """
        Record user feedback and trigger adaptive learning
        
        Args:
            feedback_type: Type of feedback (chunk_quality, boundary_accuracy, etc.)
            target_id: ID of target (chunk, document, etc.)
            rating: Rating from 0.0 to 1.0
            details: Additional feedback details
            user_id: User providing feedback
            session_id: Current session ID
            
        Returns:
            Feedback recording confirmation and learning insights
        """
        try:
            # Create feedback object
            feedback = UserFeedback(
                feedback_type=FeedbackType(feedback_type),
                target_id=target_id,
                rating=rating,
                details=details
            )
            
            # Record in learning engine
            self.learning_engine.record_feedback(feedback)
            
            # Store in database
            feedback_id = await self._store_user_feedback(
                feedback_type, target_id, rating, details, user_id, session_id
            )
            
            # Get learning insights
            insights = self.learning_engine.get_learning_insights()
            
            return {
                'feedback_id': feedback_id,
                'feedback_recorded': True,
                'total_feedback': insights['total_feedback'],
                'learning_triggered': insights['total_feedback'] % 10 == 0,
                'current_strategy': insights['current_strategy'],
                'performance_trend': insights.get('performance_trend', {})
            }
            
        except Exception as e:
            logger.error(f"Error recording user feedback: {e}")
            return {'error': str(e)}
            
    async def ai_get_adaptive_strategy(self, context: Dict[str, Any],
                                      session_id: str) -> Dict[str, Any]:
        """
        Get current adaptive strategy based on learned patterns
        
        Args:
            context: Context information for strategy selection
            session_id: Current session ID
            
        Returns:
            Current adaptive strategy with configuration
        """
        try:
            # Get current strategy from learning engine
            strategy = self.learning_engine.get_current_strategy(context)
            
            # Get performance metrics
            insights = self.learning_engine.get_learning_insights()
            
            # Store strategy usage
            await self._store_strategy_usage(strategy, session_id)
            
            return {
                'strategy_name': strategy.name,
                'preferred_chunk_size': strategy.preferred_size,
                'boundary_patterns': strategy.boundary_patterns,
                'quality_weights': strategy.quality_weights,
                'confidence': strategy.confidence,
                'total_feedback_used': insights['total_feedback'],
                'learned_patterns': insights['learned_patterns']
            }
            
        except Exception as e:
            logger.error(f"Error getting adaptive strategy: {e}")
            return {'error': str(e)}
            
    async def ai_curate_chunks(self, chunk_ids: List[str], workflow_id: str,
                              session_id: str, auto_apply: bool = False) -> Dict[str, Any]:
        """
        Run automated curation workflow on chunks
        
        Args:
            chunk_ids: List of chunk IDs to curate
            workflow_id: Curation workflow to use
            session_id: Current session ID
            auto_apply: Whether to automatically apply decisions
            
        Returns:
            Curation results with decisions and actions
        """
        try:
            # Get chunks data
            chunks = []
            for chunk_id in chunk_ids:
                chunk = await self.db.get_chunk(chunk_id)
                if chunk:
                    # Add quality score
                    quality_scores = await self._get_quality_scores(chunk_id)
                    chunk['quality_score'] = sum(quality_scores.values()) / len(quality_scores)
                    chunks.append(chunk)
                    
            if not chunks:
                return {'error': 'No valid chunks found'}
                
            # Execute curation workflow
            results = self.curator.execute_workflow(chunks, workflow_id)
            
            # Apply decisions if requested
            applied_count = 0
            if auto_apply:
                for chunk, decision in zip(chunks, results['final_decisions']):
                    updated_chunk = self.curator.apply_decision(chunk, decision)
                    await self._update_chunk_metadata(chunk['chunk_id'], updated_chunk)
                    applied_count += 1
                    
            # Store curation decisions
            for decision in results['final_decisions']:
                await self._store_curation_decision(decision, workflow_id, session_id)
                
            return {
                'workflow_id': workflow_id,
                'total_chunks': results['total_chunks'],
                'decisions': [
                    {
                        'chunk_id': d.chunk_id,
                        'action': d.action.value,
                        'reason': d.reason,
                        'confidence': d.confidence
                    }
                    for d in results['final_decisions']
                ],
                'stage_results': results['stage_results'],
                'applied_count': applied_count if auto_apply else 0
            }
            
        except Exception as e:
            logger.error(f"Error curating chunks: {e}")
            return {'error': str(e)}
            
    async def ai_optimize_performance(self, optimization_type: str,
                                     parameters: Dict[str, Any],
                                     session_id: str, apply: bool = False) -> Dict[str, Any]:
        """
        Optimize system performance based on usage patterns
        
        Args:
            optimization_type: Type of optimization (batch_size, cache_strategy, etc.)
            parameters: Parameters for optimization
            session_id: Current session ID
            apply: Whether to apply the optimization
            
        Returns:
            Optimization recommendations and results
        """
        try:
            # Get optimization based on type
            opt_type = OptimizationType(optimization_type)
            
            if opt_type == OptimizationType.BATCH_SIZE:
                result = self.optimizer.optimize_batch_size(parameters)
            elif opt_type == OptimizationType.CACHE_STRATEGY:
                result = self.optimizer.optimize_cache_strategy(parameters)
            elif opt_type == OptimizationType.MODEL_SELECTION:
                result = self.optimizer.optimize_model_selection(parameters)
            elif opt_type == OptimizationType.PREPROCESSING:
                result = self.optimizer.optimize_preprocessing(parameters)
            else:
                return {'error': f'Unknown optimization type: {optimization_type}'}
                
            # Apply optimization if requested
            if apply and result.confidence > 0.6:
                self.optimizer.apply_optimization(result)
                await self._store_optimization_history(result, session_id, applied=True)
            else:
                await self._store_optimization_history(result, session_id, applied=False)
                
            return {
                'optimization_type': result.optimization_type.value,
                'original_config': result.original_config,
                'optimized_config': result.optimized_config,
                'expected_improvement': result.improvement,
                'confidence': result.confidence,
                'applied': apply and result.confidence > 0.6
            }
            
        except Exception as e:
            logger.error(f"Error optimizing performance: {e}")
            return {'error': str(e)}
            
    async def ai_get_performance_insights(self, session_id: str) -> Dict[str, Any]:
        """
        Get current performance insights and recommendations
        
        Args:
            session_id: Current session ID
            
        Returns:
            Performance insights with metrics and recommendations
        """
        try:
            # Get performance insights
            insights = self.optimizer.get_performance_insights()
            
            # Get optimization report
            report = self.optimizer.generate_optimization_report()
            
            # Get recent metrics from database
            recent_metrics = await self._get_recent_performance_metrics()
            
            return {
                'current_config': insights['current_config'],
                'average_metrics': insights['average_metrics'],
                'performance_trends': insights['trends'],
                'bottlenecks': insights['bottlenecks'],
                'optimization_opportunities': insights['optimization_opportunities'],
                'cache_statistics': report['cache_statistics'],
                'recommendations': report['recommendations'],
                'recent_metrics': recent_metrics
            }
            
        except Exception as e:
            logger.error(f"Error getting performance insights: {e}")
            return {'error': str(e)}
            
    async def ai_generate_enhancement_report(self, report_type: str,
                                           start_date: str, end_date: str,
                                           session_id: str) -> Dict[str, Any]:
        """
        Generate comprehensive AI enhancement report
        
        Args:
            report_type: Type of report (quality, learning, curation, optimization)
            start_date: Report period start
            end_date: Report period end
            session_id: Current session ID
            
        Returns:
            Comprehensive report with metrics and insights
        """
        try:
            from datetime import datetime
            start = datetime.fromisoformat(start_date)
            end = datetime.fromisoformat(end_date)
            
            report_data = {}
            
            if report_type == 'quality':
                # Quality improvement report
                improvements = await self._get_quality_improvements(start, end)
                report_data = {
                    'total_improvements': len(improvements),
                    'automated_improvements': sum(1 for i in improvements if i['automated']),
                    'average_score_increase': sum(i['score_increase'] for i in improvements) / len(improvements) if improvements else 0,
                    'improvements_by_type': self._group_by_type(improvements)
                }
                
            elif report_type == 'learning':
                # Learning insights report
                insights = self.learning_engine.get_learning_insights()
                patterns = await self._get_learning_patterns()
                report_data = {
                    'total_feedback': insights['total_feedback'],
                    'feedback_distribution': dict(insights['feedback_types']),
                    'learned_patterns': patterns,
                    'performance_trend': insights['performance_trend'],
                    'current_strategy': insights['current_strategy']
                }
                
            elif report_type == 'curation':
                # Curation summary report
                report_data = self.curator.generate_curation_report(start, end).__dict__
                
            elif report_type == 'optimization':
                # Optimization report
                report_data = self.optimizer.generate_optimization_report()
                
            else:
                return {'error': f'Unknown report type: {report_type}'}
                
            # Store report
            report_id = await self._store_ai_report(
                report_type, start, end, report_data, session_id
            )
            
            return {
                'report_id': report_id,
                'report_type': report_type,
                'period': {'start': start_date, 'end': end_date},
                'data': report_data,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating enhancement report: {e}")
            return {'error': str(e)}
            
    # Helper methods for database operations
    
    async def _get_quality_scores(self, chunk_id: str) -> Dict[str, float]:
        """Get quality scores for a chunk"""
        # In practice, this would query the quality assessment table
        # For now, return mock scores
        return {
            'readability': 0.75,
            'technical_accuracy': 0.82,
            'completeness': 0.68,
            'relevance': 0.79,
            'freshness': 0.85,
            'coherence': 0.71,
            'uniqueness': 0.66,
            'authority': 0.73
        }
        
    async def _update_chunk_content(self, chunk_id: str, content: str):
        """Update chunk content in database"""
        query = """
        UPDATE megamind_knowledge_chunks 
        SET content = %s, last_improved = NOW()
        WHERE chunk_id = %s
        """
        await self.db.execute_query(query, (content, chunk_id))
        
    async def _update_chunk_metadata(self, chunk_id: str, metadata: Dict[str, Any]):
        """Update chunk metadata"""
        query = """
        UPDATE megamind_knowledge_chunks
        SET metadata = %s, quality_score = %s
        WHERE chunk_id = %s
        """
        await self.db.execute_query(
            query,
            (json.dumps(metadata), metadata.get('quality_score', 0), chunk_id)
        )
        
    async def _store_improvement_history(self, chunk_id: str, improvement: QualityImprovement,
                                       session_id: str, applied: bool = False):
        """Store quality improvement in history"""
        improvement_id = f"imp_{uuid.uuid4().hex[:12]}"
        query = """
        INSERT INTO megamind_quality_improvements 
        (improvement_id, chunk_id, improvement_type, original_score, improved_score,
         improvement_status, suggestion, implementation, automated, confidence, session_id)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        status = 'applied' if applied else 'suggested'
        await self.db.execute_query(query, (
            improvement_id, chunk_id, improvement.issue.dimension,
            0.5, 0.7,  # Mock scores
            status, improvement.suggestion, improvement.implementation,
            improvement.automated, improvement.confidence, session_id
        ))
        
    async def _store_user_feedback(self, feedback_type: str, target_id: str,
                                  rating: float, details: Dict[str, Any],
                                  user_id: str, session_id: str) -> str:
        """Store user feedback in database"""
        feedback_id = f"fb_{uuid.uuid4().hex[:12]}"
        query = """
        INSERT INTO megamind_user_feedback
        (feedback_id, feedback_type, target_id, rating, details, user_id, session_id)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        await self.db.execute_query(query, (
            feedback_id, feedback_type, target_id, rating,
            json.dumps(details), user_id, session_id
        ))
        return feedback_id
        
    async def _store_strategy_usage(self, strategy, session_id: str):
        """Store adaptive strategy usage"""
        strategy_id = f"strat_{uuid.uuid4().hex[:12]}"
        query = """
        INSERT INTO megamind_adaptive_strategies
        (strategy_id, strategy_name, strategy_type, preferred_chunk_size,
         boundary_patterns, quality_weights, confidence, is_active)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
        confidence = %s, last_updated = NOW()
        """
        await self.db.execute_query(query, (
            strategy_id, strategy.name, 'chunking', strategy.preferred_size,
            json.dumps(strategy.boundary_patterns),
            json.dumps(strategy.quality_weights),
            strategy.confidence, True, strategy.confidence
        ))
        
    async def _store_curation_decision(self, decision, workflow_id: str, session_id: str):
        """Store curation decision"""
        decision_id = f"dec_{uuid.uuid4().hex[:12]}"
        query = """
        INSERT INTO megamind_curation_decisions
        (decision_id, chunk_id, action, reason, confidence, parameters,
         workflow_id, automated, session_id)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        await self.db.execute_query(query, (
            decision_id, decision.chunk_id, decision.action.value,
            decision.reason, decision.confidence,
            json.dumps(decision.parameters),
            workflow_id, decision.confidence > 0.7, session_id
        ))
        
    async def _store_optimization_history(self, result, session_id: str, applied: bool):
        """Store optimization in history"""
        opt_id = f"opt_{uuid.uuid4().hex[:12]}"
        query = """
        INSERT INTO megamind_optimization_history
        (optimization_id, optimization_type, original_config, optimized_config,
         improvement_metrics, confidence, applied, session_id)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        await self.db.execute_query(query, (
            opt_id, result.optimization_type.value,
            json.dumps(result.original_config),
            json.dumps(result.optimized_config),
            json.dumps(result.improvement),
            result.confidence, applied, session_id
        ))
        
    async def _get_recent_performance_metrics(self) -> List[Dict[str, Any]]:
        """Get recent performance metrics from database"""
        query = """
        SELECT metric_type, operation, AVG(value) as avg_value
        FROM megamind_performance_metrics
        WHERE timestamp > DATE_SUB(NOW(), INTERVAL 1 HOUR)
        GROUP BY metric_type, operation
        ORDER BY metric_type
        """
        results = await self.db.execute_query(query)
        return [
            {
                'metric': row['metric_type'],
                'operation': row['operation'],
                'average': row['avg_value']
            }
            for row in results
        ]
        
    async def _get_quality_improvements(self, start: datetime, end: datetime) -> List[Dict[str, Any]]:
        """Get quality improvements in date range"""
        query = """
        SELECT improvement_type, automated,
               (improved_score - original_score) as score_increase
        FROM megamind_quality_improvements
        WHERE created_date BETWEEN %s AND %s
        AND improvement_status = 'applied'
        """
        results = await self.db.execute_query(query, (start, end))
        return results
        
    async def _get_learning_patterns(self) -> Dict[str, int]:
        """Get current learning patterns"""
        query = """
        SELECT pattern_type, COUNT(*) as count
        FROM megamind_learning_patterns
        WHERE confidence > 0.6
        GROUP BY pattern_type
        """
        results = await self.db.execute_query(query)
        return {row['pattern_type']: row['count'] for row in results}
        
    async def _store_ai_report(self, report_type: str, start: datetime, end: datetime,
                             report_data: Dict[str, Any], session_id: str) -> str:
        """Store AI enhancement report"""
        report_id = f"report_{uuid.uuid4().hex[:12]}"
        query = """
        INSERT INTO megamind_ai_reports
        (report_id, report_type, period_start, period_end, report_data, created_by)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        await self.db.execute_query(query, (
            report_id, report_type, start, end,
            json.dumps(report_data), session_id
        ))
        return report_id
        
    def _group_by_type(self, items: List[Dict[str, Any]]) -> Dict[str, int]:
        """Group items by type"""
        groups = {}
        for item in items:
            type_key = item.get('improvement_type', 'unknown')
            groups[type_key] = groups.get(type_key, 0) + 1
        return groups