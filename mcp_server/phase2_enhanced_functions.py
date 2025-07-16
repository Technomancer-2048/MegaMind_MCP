"""
Phase 2 Function Consolidation - Enhanced Master Functions
GitHub Issue #19: Function Name Standardization - Phase 2

This module implements Phase 2 enhancements to the consolidated MCP functions,
adding intelligent features like parameter inference, batch operations, adaptive
routing, performance optimization, and function composition.

Building upon Phase 1's 19 master functions with advanced capabilities.
"""

import asyncio
import logging
import json
import time
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class RoutingStrategy(Enum):
    """Routing strategies for adaptive routing system."""
    PERFORMANCE = "performance"
    ACCURACY = "accuracy"
    BALANCED = "balanced"
    LEARNED = "learned"

@dataclass
class RoutingDecision:
    """Represents a routing decision with metadata."""
    function_name: str
    parameters: Dict[str, Any]
    routed_to: str
    execution_time: float
    success: bool
    timestamp: datetime
    user_context: Optional[str] = None

@dataclass
class BatchOperation:
    """Represents a batch operation request."""
    operation_type: str
    items: List[Dict[str, Any]]
    batch_id: str
    created_at: datetime = field(default_factory=datetime.now)
    
@dataclass
class WorkflowStep:
    """Represents a step in a workflow composition."""
    function_name: str
    parameters: Dict[str, Any]
    depends_on: List[str] = field(default_factory=list)
    step_id: str = ""

class Phase2EnhancedFunctions:
    """
    Phase 2 Enhanced MCP Functions with advanced capabilities:
    - Smart Parameter Inference
    - Batch Operations
    - Adaptive Routing
    - Performance Optimization
    - Function Composition
    - Workflow Integration
    """
    
    def __init__(self, base_functions, db_manager, session_manager=None):
        """
        Initialize Phase 2 enhanced functions.
        
        Args:
            base_functions: Phase 1 ConsolidatedMCPFunctions instance
            db_manager: RealmAwareMegaMindDatabase instance
            session_manager: Optional session manager
        """
        self.base_functions = base_functions
        self.db = db_manager
        self.session_manager = session_manager
        
        # Adaptive routing system
        self.routing_history = deque(maxlen=10000)  # Keep last 10k decisions
        self.routing_cache = {}  # Cache for routing decisions
        self.usage_patterns = defaultdict(list)  # Track usage patterns
        self.performance_metrics = defaultdict(list)  # Track performance
        
        # Batch processing
        self.batch_queue = {}  # Active batch operations
        self.batch_results = {}  # Completed batch results
        
        # Workflow composition
        self.active_workflows = {}  # Active workflow compositions
        self.workflow_templates = {}  # Reusable workflow templates
        
        # Parameter inference
        self.parameter_patterns = defaultdict(dict)  # Learn parameter patterns
        self.context_history = deque(maxlen=1000)  # Context for inference
        
        logger.info("Phase 2 Enhanced Functions initialized")
    
    # ========================================
    # 🧠 Smart Parameter Inference
    # ========================================
    
    def infer_search_parameters(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Intelligently infer search parameters based on query content and context.
        
        Args:
            query: Search query text
            context: Optional context from previous operations
            
        Returns:
            Dict with inferred parameters
        """
        inferred = {
            "search_type": "hybrid",  # Default
            "limit": 10,
            "threshold": 0.7
        }
        
        # Analyze query characteristics
        query_lower = query.lower()
        
        # Infer search type based on query patterns
        if any(word in query_lower for word in ["similar", "like", "related to"]):
            inferred["search_type"] = "similarity"
            
        elif any(word in query_lower for word in ["semantic", "meaning", "concept"]):
            inferred["search_type"] = "semantic"
            inferred["threshold"] = 0.8  # Higher threshold for semantic
            
        elif any(word in query_lower for word in ["exact", "keyword", "find"]):
            inferred["search_type"] = "keyword"
            
        # Infer limit based on query urgency
        if any(word in query_lower for word in ["quick", "brief", "summary"]):
            inferred["limit"] = 5
        elif any(word in query_lower for word in ["comprehensive", "detailed", "thorough"]):
            inferred["limit"] = 20
            
        # Use context from previous operations
        if context:
            if context.get("last_search_type") == "semantic" and "more" in query_lower:
                inferred["search_type"] = "semantic"
            if context.get("last_limit", 0) > 0:
                inferred["limit"] = min(inferred["limit"], context["last_limit"] * 2)
        
        # Learn from usage patterns
        if query in self.parameter_patterns:
            learned_params = self.parameter_patterns[query]
            for key, value in learned_params.items():
                if key in inferred:
                    inferred[key] = value
                    
        return inferred
    
    def learn_from_usage(self, function_name: str, parameters: Dict[str, Any], 
                        success: bool, execution_time: float):
        """
        Learn from function usage to improve future parameter inference.
        
        Args:
            function_name: Name of the function used
            parameters: Parameters that were used
            success: Whether the operation was successful
            execution_time: How long the operation took
        """
        # Store usage pattern
        if success and execution_time < 1.0:  # Only learn from fast, successful operations
            key = parameters.get("query", function_name)
            self.parameter_patterns[key].update(parameters)
            
        # Track performance metrics
        self.performance_metrics[function_name].append({
            "parameters": parameters,
            "success": success,
            "execution_time": execution_time,
            "timestamp": datetime.now()
        })
        
        # Keep only recent metrics
        if len(self.performance_metrics[function_name]) > 100:
            self.performance_metrics[function_name] = self.performance_metrics[function_name][-100:]
    
    # ========================================
    # 🚀 Batch Operations
    # ========================================
    
    async def create_batch_operation(self, operation_type: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a batch operation for processing multiple items efficiently.
        
        Args:
            operation_type: Type of operation (search, content_create, etc.)
            items: List of items to process
            
        Returns:
            Dict with batch operation ID and status
        """
        batch_id = f"batch_{int(time.time() * 1000)}"
        
        batch_op = BatchOperation(
            operation_type=operation_type,
            items=items,
            batch_id=batch_id
        )
        
        self.batch_queue[batch_id] = batch_op
        
        logger.info(f"Created batch operation {batch_id} with {len(items)} items")
        
        return {
            "success": True,
            "batch_id": batch_id,
            "operation_type": operation_type,
            "item_count": len(items),
            "status": "queued"
        }
    
    async def process_batch_operation(self, batch_id: str) -> Dict[str, Any]:
        """
        Process a queued batch operation.
        
        Args:
            batch_id: ID of the batch operation to process
            
        Returns:
            Dict with batch processing results
        """
        if batch_id not in self.batch_queue:
            return {
                "success": False,
                "error": f"Batch operation {batch_id} not found"
            }
        
        batch_op = self.batch_queue[batch_id]
        start_time = time.time()
        
        results = []
        errors = []
        
        try:
            # Process items based on operation type
            if batch_op.operation_type == "search":
                for item in batch_op.items:
                    try:
                        result = await self.base_functions.search_query(**item)
                        results.append(result)
                    except Exception as e:
                        errors.append({"item": item, "error": str(e)})
                        
            elif batch_op.operation_type == "content_create":
                for item in batch_op.items:
                    try:
                        result = await self.base_functions.content_create(**item)
                        results.append(result)
                    except Exception as e:
                        errors.append({"item": item, "error": str(e)})
                        
            elif batch_op.operation_type == "content_update":
                for item in batch_op.items:
                    try:
                        result = await self.base_functions.content_update(**item)
                        results.append(result)
                    except Exception as e:
                        errors.append({"item": item, "error": str(e)})
                        
            else:
                return {
                    "success": False,
                    "error": f"Unsupported batch operation type: {batch_op.operation_type}"
                }
            
            # Store results
            execution_time = time.time() - start_time
            
            batch_result = {
                "success": True,
                "batch_id": batch_id,
                "operation_type": batch_op.operation_type,
                "processed_count": len(results),
                "error_count": len(errors),
                "execution_time": execution_time,
                "results": results,
                "errors": errors,
                "completed_at": datetime.now()
            }
            
            self.batch_results[batch_id] = batch_result
            
            # Clean up queue
            del self.batch_queue[batch_id]
            
            logger.info(f"Completed batch operation {batch_id} in {execution_time:.2f}s")
            
            return batch_result
            
        except Exception as e:
            logger.error(f"Error processing batch operation {batch_id}: {str(e)}")
            return {
                "success": False,
                "batch_id": batch_id,
                "error": str(e),
                "execution_time": time.time() - start_time
            }
    
    async def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """
        Get the status of a batch operation.
        
        Args:
            batch_id: ID of the batch operation
            
        Returns:
            Dict with batch operation status
        """
        if batch_id in self.batch_queue:
            batch_op = self.batch_queue[batch_id]
            return {
                "success": True,
                "batch_id": batch_id,
                "status": "queued",
                "operation_type": batch_op.operation_type,
                "item_count": len(batch_op.items),
                "created_at": batch_op.created_at.isoformat()
            }
        elif batch_id in self.batch_results:
            return self.batch_results[batch_id]
        else:
            return {
                "success": False,
                "error": f"Batch operation {batch_id} not found"
            }
    
    # ========================================
    # 🎯 Adaptive Routing with Learning
    # ========================================
    
    def get_optimal_routing(self, function_name: str, parameters: Dict[str, Any], 
                           strategy: RoutingStrategy = RoutingStrategy.BALANCED) -> Tuple[str, Dict[str, Any]]:
        """
        Get optimal routing decision based on learned patterns and strategy.
        
        Args:
            function_name: Name of the function to route
            parameters: Parameters for the function
            strategy: Routing strategy to use
            
        Returns:
            Tuple of (routed_function_name, optimized_parameters)
        """
        # Create cache key
        cache_key = f"{function_name}_{hash(str(sorted(parameters.items())))}"
        
        # Check cache for recent decisions
        if cache_key in self.routing_cache:
            cached_decision = self.routing_cache[cache_key]
            if datetime.now() - cached_decision["timestamp"] < timedelta(minutes=5):
                return cached_decision["routed_to"], cached_decision["parameters"]
        
        # Analyze historical performance for this function
        if function_name in self.performance_metrics:
            metrics = self.performance_metrics[function_name]
            
            if strategy == RoutingStrategy.PERFORMANCE:
                # Route to fastest successful combination
                successful_metrics = [m for m in metrics if m["success"]]
                if successful_metrics:
                    best_metric = min(successful_metrics, key=lambda x: x["execution_time"])
                    optimized_params = best_metric["parameters"].copy()
                    optimized_params.update(parameters)  # Override with current params
                    
                    # Cache the decision
                    self.routing_cache[cache_key] = {
                        "routed_to": function_name,
                        "parameters": optimized_params,
                        "timestamp": datetime.now()
                    }
                    
                    return function_name, optimized_params
            
            elif strategy == RoutingStrategy.ACCURACY:
                # Route to most accurate combination
                successful_metrics = [m for m in metrics if m["success"]]
                if successful_metrics:
                    # Use most recent successful parameters
                    best_metric = max(successful_metrics, key=lambda x: x["timestamp"])
                    optimized_params = best_metric["parameters"].copy()
                    optimized_params.update(parameters)
                    
                    self.routing_cache[cache_key] = {
                        "routed_to": function_name,
                        "parameters": optimized_params,
                        "timestamp": datetime.now()
                    }
                    
                    return function_name, optimized_params
        
        # Default routing with parameter inference
        if function_name.startswith("search_"):
            inferred_params = self.infer_search_parameters(
                parameters.get("query", ""),
                {"last_search_type": parameters.get("search_type")}
            )
            optimized_params = inferred_params.copy()
            optimized_params.update(parameters)  # Override with explicit params
            
            self.routing_cache[cache_key] = {
                "routed_to": function_name,
                "parameters": optimized_params,
                "timestamp": datetime.now()
            }
            
            return function_name, optimized_params
        
        # Default: return original parameters
        return function_name, parameters
    
    def record_routing_decision(self, function_name: str, parameters: Dict[str, Any], 
                              routed_to: str, execution_time: float, success: bool):
        """
        Record a routing decision for learning purposes.
        
        Args:
            function_name: Original function name
            parameters: Parameters used
            routed_to: Where the function was routed
            execution_time: How long it took
            success: Whether it was successful
        """
        decision = RoutingDecision(
            function_name=function_name,
            parameters=parameters,
            routed_to=routed_to,
            execution_time=execution_time,
            success=success,
            timestamp=datetime.now()
        )
        
        self.routing_history.append(decision)
        self.learn_from_usage(function_name, parameters, success, execution_time)
    
    # ========================================
    # 🔄 Function Composition & Workflows
    # ========================================
    
    async def create_workflow(self, workflow_name: str, steps: List[WorkflowStep]) -> Dict[str, Any]:
        """
        Create a workflow composition with multiple function steps.
        
        Args:
            workflow_name: Name for the workflow
            steps: List of workflow steps
            
        Returns:
            Dict with workflow creation result
        """
        workflow_id = f"workflow_{int(time.time() * 1000)}"
        
        # Validate workflow steps
        for i, step in enumerate(steps):
            if not step.step_id:
                step.step_id = f"step_{i}"
        
        workflow = {
            "workflow_id": workflow_id,
            "name": workflow_name,
            "steps": steps,
            "created_at": datetime.now(),
            "status": "created"
        }
        
        self.active_workflows[workflow_id] = workflow
        
        logger.info(f"Created workflow {workflow_id} with {len(steps)} steps")
        
        return {
            "success": True,
            "workflow_id": workflow_id,
            "workflow_name": workflow_name,
            "step_count": len(steps),
            "status": "created"
        }
    
    async def execute_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """
        Execute a workflow composition.
        
        Args:
            workflow_id: ID of the workflow to execute
            
        Returns:
            Dict with workflow execution results
        """
        if workflow_id not in self.active_workflows:
            return {
                "success": False,
                "error": f"Workflow {workflow_id} not found"
            }
        
        workflow = self.active_workflows[workflow_id]
        workflow["status"] = "running"
        
        start_time = time.time()
        step_results = {}
        
        try:
            # Execute steps in dependency order
            for step in workflow["steps"]:
                # Check dependencies
                if step.depends_on:
                    for dep_id in step.depends_on:
                        if dep_id not in step_results:
                            return {
                                "success": False,
                                "error": f"Dependency {dep_id} not found for step {step.step_id}"
                            }
                
                # Execute step
                try:
                    if step.function_name.startswith("search_"):
                        result = await self.enhanced_search_query(**step.parameters)
                    elif step.function_name.startswith("content_"):
                        result = await self.enhanced_content_create(**step.parameters)
                    else:
                        # Fall back to base functions
                        base_func = getattr(self.base_functions, step.function_name.split("_", 2)[-1])
                        result = await base_func(**step.parameters)
                    
                    step_results[step.step_id] = result
                    
                except Exception as e:
                    step_results[step.step_id] = {
                        "success": False,
                        "error": str(e)
                    }
            
            execution_time = time.time() - start_time
            
            workflow_result = {
                "success": True,
                "workflow_id": workflow_id,
                "workflow_name": workflow["name"],
                "step_results": step_results,
                "execution_time": execution_time,
                "completed_at": datetime.now()
            }
            
            workflow["status"] = "completed"
            workflow["result"] = workflow_result
            
            logger.info(f"Completed workflow {workflow_id} in {execution_time:.2f}s")
            
            return workflow_result
            
        except Exception as e:
            workflow["status"] = "failed"
            logger.error(f"Error executing workflow {workflow_id}: {str(e)}")
            return {
                "success": False,
                "workflow_id": workflow_id,
                "error": str(e),
                "execution_time": time.time() - start_time
            }
    
    # ========================================
    # 🎯 Enhanced Master Functions
    # ========================================
    
    async def enhanced_search_query(self, query: str, search_type: str = "auto", 
                                   limit: int = 0, threshold: float = 0.0,
                                   reference_chunk_id: str = None, 
                                   enable_inference: bool = True) -> Dict[str, Any]:
        """
        Enhanced search query with smart parameter inference and adaptive routing.
        
        Args:
            query: Search query text
            search_type: Search type ("auto" enables inference)
            limit: Maximum results (0 enables inference)
            threshold: Similarity threshold (0.0 enables inference)
            reference_chunk_id: Reference chunk for similarity search
            enable_inference: Whether to enable parameter inference
            
        Returns:
            Dict with search results and enhancement metadata
        """
        start_time = time.time()
        
        # Smart parameter inference if enabled
        if enable_inference and (search_type == "auto" or limit == 0 or threshold == 0.0):
            inferred_params = self.infer_search_parameters(query)
            
            if search_type == "auto":
                search_type = inferred_params["search_type"]
            if limit == 0:
                limit = inferred_params["limit"]
            if threshold == 0.0:
                threshold = inferred_params["threshold"]
        
        # Adaptive routing
        optimal_func, optimal_params = self.get_optimal_routing(
            "search_query", 
            {"query": query, "search_type": search_type, "limit": limit, "threshold": threshold}
        )
        
        try:
            # Execute with base function
            result = await self.base_functions.search_query(
                query=query,
                search_type=search_type,
                limit=limit,
                threshold=threshold,
                reference_chunk_id=reference_chunk_id
            )
            
            execution_time = time.time() - start_time
            
            # Record routing decision
            self.record_routing_decision(
                "search_query", 
                optimal_params, 
                optimal_func, 
                execution_time, 
                result.get("success", False)
            )
            
            # Enhance result with metadata
            result["enhancement_metadata"] = {
                "parameter_inference_used": enable_inference,
                "inferred_search_type": search_type if enable_inference else None,
                "adaptive_routing_used": True,
                "execution_time": execution_time,
                "optimization_applied": True
            }
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.record_routing_decision("search_query", optimal_params, optimal_func, execution_time, False)
            
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "enhancement_metadata": {
                    "parameter_inference_used": enable_inference,
                    "execution_time": execution_time,
                    "optimization_applied": False
                }
            }
    
    async def enhanced_content_create(self, content: str, source_document: str,
                                     section_path: str = "", session_id: str = "",
                                     target_realm: str = "PROJECT", 
                                     enable_inference: bool = True,
                                     auto_relationships: bool = True) -> Dict[str, Any]:
        """
        Enhanced content creation with smart relationship inference.
        
        Args:
            content: Content to create
            source_document: Source document name
            section_path: Section path within document
            session_id: Session ID for tracking
            target_realm: Target realm for creation
            enable_inference: Whether to enable smart inference
            auto_relationships: Whether to auto-create relationships
            
        Returns:
            Dict with creation result and enhancement metadata
        """
        start_time = time.time()
        
        # Smart relationship inference if enabled
        relationship_targets = []
        if enable_inference and auto_relationships:
            try:
                # Search for related content
                related_search = await self.enhanced_search_query(
                    query=content[:200],  # Use first 200 chars for search
                    search_type="semantic",
                    limit=5,
                    enable_inference=False
                )
                
                if related_search.get("success") and related_search.get("results"):
                    relationship_targets = [
                        chunk.get("chunk_id") for chunk in related_search["results"]
                        if chunk.get("chunk_id")
                    ]
            except Exception as e:
                logger.warning(f"Failed to infer relationships: {str(e)}")
        
        try:
            # Execute with base function
            result = await self.base_functions.content_create(
                content=content,
                source_document=source_document,
                section_path=section_path,
                session_id=session_id,
                target_realm=target_realm,
                create_relationships=auto_relationships,
                relationship_targets=relationship_targets,
                relationship_type="semantic_similarity"
            )
            
            execution_time = time.time() - start_time
            
            # Enhance result with metadata
            result["enhancement_metadata"] = {
                "relationship_inference_used": enable_inference and auto_relationships,
                "relationships_created": len(relationship_targets),
                "execution_time": execution_time,
                "optimization_applied": True
            }
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            return {
                "success": False,
                "error": str(e),
                "content": content[:100] + "..." if len(content) > 100 else content,
                "enhancement_metadata": {
                    "relationship_inference_used": enable_inference and auto_relationships,
                    "execution_time": execution_time,
                    "optimization_applied": False
                }
            }
    
    # ========================================
    # 📊 Performance Analytics
    # ========================================
    
    def get_performance_analytics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance analytics for Phase 2 enhancements.
        
        Returns:
            Dict with performance analytics
        """
        analytics = {
            "routing_decisions": len(self.routing_history),
            "cache_size": len(self.routing_cache),
            "active_batches": len(self.batch_queue),
            "completed_batches": len(self.batch_results),
            "active_workflows": len(self.active_workflows),
            "learned_patterns": len(self.parameter_patterns)
        }
        
        # Function performance metrics
        function_metrics = {}
        for func_name, metrics in self.performance_metrics.items():
            if metrics:
                avg_time = sum(m["execution_time"] for m in metrics) / len(metrics)
                success_rate = sum(1 for m in metrics if m["success"]) / len(metrics)
                
                function_metrics[func_name] = {
                    "average_execution_time": avg_time,
                    "success_rate": success_rate,
                    "total_calls": len(metrics)
                }
        
        analytics["function_metrics"] = function_metrics
        
        # Recent routing decisions
        recent_decisions = []
        for decision in list(self.routing_history)[-10:]:
            recent_decisions.append({
                "function_name": decision.function_name,
                "routed_to": decision.routed_to,
                "execution_time": decision.execution_time,
                "success": decision.success,
                "timestamp": decision.timestamp.isoformat()
            })
        
        analytics["recent_routing_decisions"] = recent_decisions
        
        return analytics
    
    def cleanup_caches(self, max_age_minutes: int = 30):
        """
        Clean up old cache entries and optimize memory usage.
        
        Args:
            max_age_minutes: Maximum age for cache entries in minutes
        """
        cutoff_time = datetime.now() - timedelta(minutes=max_age_minutes)
        
        # Clean routing cache
        expired_keys = [
            key for key, value in self.routing_cache.items()
            if value["timestamp"] < cutoff_time
        ]
        
        for key in expired_keys:
            del self.routing_cache[key]
        
        # Clean completed batch results
        expired_batches = [
            batch_id for batch_id, result in self.batch_results.items()
            if result.get("completed_at", datetime.now()) < cutoff_time
        ]
        
        for batch_id in expired_batches:
            del self.batch_results[batch_id]
        
        logger.info(f"Cleaned {len(expired_keys)} cache entries and {len(expired_batches)} batch results")