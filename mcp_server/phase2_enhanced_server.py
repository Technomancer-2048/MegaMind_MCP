"""
Phase 2 Enhanced MCP Server - Advanced Function Consolidation
GitHub Issue #19: Function Name Standardization - Phase 2

This module implements the Phase 2 enhanced MCP server that extends the Phase 1
consolidated functions with advanced capabilities including smart parameter inference,
batch operations, adaptive routing, and workflow composition.

Extends the existing 19 master functions with enhanced versions and new capabilities.
"""

import json
import logging
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from decimal import Decimal

try:
    from .consolidated_mcp_server import ConsolidatedMCPServer
    from .phase2_enhanced_functions import Phase2EnhancedFunctions, RoutingStrategy
    from .megamind_database_server import MegaMindJSONEncoder, clean_decimal_objects
    from .realm_aware_database import RealmAwareMegaMindDatabase
except ImportError:
    from consolidated_mcp_server import ConsolidatedMCPServer
    from phase2_enhanced_functions import Phase2EnhancedFunctions, RoutingStrategy
    from megamind_database_server import MegaMindJSONEncoder, clean_decimal_objects
    from realm_aware_database import RealmAwareMegaMindDatabase

logger = logging.getLogger(__name__)

class Phase2EnhancedMCPServer(ConsolidatedMCPServer):
    """
    Phase 2 Enhanced MCP Server with advanced consolidation capabilities.
    
    Extends the Phase 1 consolidated server with:
    - Smart parameter inference
    - Batch operations
    - Adaptive routing
    - Workflow composition
    - Performance optimization
    - Auto-documentation
    """
    
    def __init__(self, db_manager):
        """Initialize Phase 2 enhanced MCP server."""
        super().__init__(db_manager)
        
        # Initialize Phase 2 enhanced functions
        self.enhanced_functions = Phase2EnhancedFunctions(
            self.consolidated_functions,
            self.db_manager,
            self.session_manager
        )
        
        logger.info("Phase 2 Enhanced MCP Server initialized with advanced capabilities")
    
    def get_tools_list(self) -> List[Dict[str, Any]]:
        """Get the enhanced list of MCP tools with Phase 2 capabilities."""
        # Start with Phase 1 consolidated tools
        tools = super().get_tools_list()
        
        # Add Phase 2 enhanced tools
        enhanced_tools = [
            # Enhanced Search Functions
            {
                "name": "mcp__megamind__search_enhanced",
                "description": "Enhanced search with smart parameter inference and adaptive routing",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query text"},
                        "search_type": {"type": "string", "default": "auto", "enum": ["auto", "hybrid", "semantic", "similarity", "keyword"], "description": "Search type (auto enables inference)"},
                        "limit": {"type": "integer", "default": 0, "description": "Maximum results (0 enables inference)"},
                        "threshold": {"type": "number", "default": 0.0, "description": "Similarity threshold (0.0 enables inference)"},
                        "reference_chunk_id": {"type": "string", "description": "Reference chunk ID for similarity search"},
                        "enable_inference": {"type": "boolean", "default": True, "description": "Enable smart parameter inference"}
                    },
                    "required": ["query"]
                }
            },
            
            # Enhanced Content Functions
            {
                "name": "mcp__megamind__content_enhanced",
                "description": "Enhanced content creation with smart relationship inference",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Content to create"},
                        "source_document": {"type": "string", "description": "Source document name"},
                        "section_path": {"type": "string", "description": "Section path within document"},
                        "session_id": {"type": "string", "description": "Session ID for tracking"},
                        "target_realm": {"type": "string", "default": "PROJECT", "description": "Target realm for creation"},
                        "enable_inference": {"type": "boolean", "default": True, "description": "Enable smart inference"},
                        "auto_relationships": {"type": "boolean", "default": True, "description": "Auto-create relationships"}
                    },
                    "required": ["content", "source_document"]
                }
            },
            
            # Batch Operations
            {
                "name": "mcp__megamind__batch_create",
                "description": "Create a batch operation for processing multiple items",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "operation_type": {"type": "string", "enum": ["search", "content_create", "content_update"], "description": "Type of batch operation"},
                        "items": {"type": "array", "items": {"type": "object"}, "description": "Items to process in batch"}
                    },
                    "required": ["operation_type", "items"]
                }
            },
            {
                "name": "mcp__megamind__batch_process",
                "description": "Process a queued batch operation",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "batch_id": {"type": "string", "description": "ID of batch operation to process"}
                    },
                    "required": ["batch_id"]
                }
            },
            {
                "name": "mcp__megamind__batch_status",
                "description": "Get status of a batch operation",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "batch_id": {"type": "string", "description": "ID of batch operation"}
                    },
                    "required": ["batch_id"]
                }
            },
            
            # Workflow Composition
            {
                "name": "mcp__megamind__workflow_create",
                "description": "Create a workflow composition with multiple function steps",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "workflow_name": {"type": "string", "description": "Name for the workflow"},
                        "steps": {"type": "array", "items": {"type": "object"}, "description": "Workflow steps"}
                    },
                    "required": ["workflow_name", "steps"]
                }
            },
            {
                "name": "mcp__megamind__workflow_execute",
                "description": "Execute a workflow composition",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "workflow_id": {"type": "string", "description": "ID of workflow to execute"}
                    },
                    "required": ["workflow_id"]
                }
            },
            {
                "name": "mcp__megamind__workflow_status",
                "description": "Get status of a workflow",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "workflow_id": {"type": "string", "description": "ID of workflow"}
                    },
                    "required": ["workflow_id"]
                }
            },
            
            # Performance Analytics
            {
                "name": "mcp__megamind__analytics_performance",
                "description": "Get comprehensive performance analytics for Phase 2 enhancements",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "include_metrics": {"type": "boolean", "default": True, "description": "Include function metrics"},
                        "include_routing": {"type": "boolean", "default": True, "description": "Include routing decisions"},
                        "include_cache": {"type": "boolean", "default": False, "description": "Include cache statistics"}
                    },
                    "required": []
                }
            },
            
            # System Maintenance
            {
                "name": "mcp__megamind__system_cleanup",
                "description": "Clean up caches and optimize system performance",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "max_age_minutes": {"type": "integer", "default": 30, "description": "Maximum age for cache entries"},
                        "cleanup_type": {"type": "string", "default": "all", "enum": ["all", "cache", "batches", "workflows"], "description": "Type of cleanup to perform"}
                    },
                    "required": []
                }
            }
        ]
        
        # Combine Phase 1 and Phase 2 tools
        return tools + enhanced_tools
    
    async def handle_tool_call(self, params: Dict[str, Any], request_id: int) -> Dict[str, Any]:
        """Handle tool calls including Phase 2 enhanced functions."""
        try:
            tool_name = params.get('name', '')
            tool_args = params.get('arguments', {})
            
            # Handle Phase 2 enhanced functions
            if tool_name == 'mcp__megamind__search_enhanced':
                result = await self.enhanced_functions.enhanced_search_query(**tool_args)
            elif tool_name == 'mcp__megamind__content_enhanced':
                result = await self.enhanced_functions.enhanced_content_create(**tool_args)
            elif tool_name == 'mcp__megamind__batch_create':
                result = await self.enhanced_functions.create_batch_operation(**tool_args)
            elif tool_name == 'mcp__megamind__batch_process':
                result = await self.enhanced_functions.process_batch_operation(**tool_args)
            elif tool_name == 'mcp__megamind__batch_status':
                result = await self.enhanced_functions.get_batch_status(**tool_args)
            elif tool_name == 'mcp__megamind__workflow_create':
                result = await self.enhanced_functions.create_workflow(**tool_args)
            elif tool_name == 'mcp__megamind__workflow_execute':
                result = await self.enhanced_functions.execute_workflow(**tool_args)
            elif tool_name == 'mcp__megamind__workflow_status':
                workflow_id = tool_args.get('workflow_id')
                if workflow_id in self.enhanced_functions.active_workflows:
                    workflow = self.enhanced_functions.active_workflows[workflow_id]
                    result = {
                        "success": True,
                        "workflow_id": workflow_id,
                        "status": workflow.get("status", "unknown"),
                        "step_count": len(workflow.get("steps", [])),
                        "created_at": workflow.get("created_at", datetime.now()).isoformat()
                    }
                else:
                    result = {
                        "success": False,
                        "error": f"Workflow {workflow_id} not found"
                    }
            elif tool_name == 'mcp__megamind__analytics_performance':
                result = self.enhanced_functions.get_performance_analytics()
                result["success"] = True
            elif tool_name == 'mcp__megamind__system_cleanup':
                max_age = tool_args.get('max_age_minutes', 30)
                self.enhanced_functions.cleanup_caches(max_age)
                result = {
                    "success": True,
                    "cleanup_performed": True,
                    "max_age_minutes": max_age,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                # Fall back to Phase 1 consolidated functions
                return await super().handle_tool_call(params, request_id)
            
            # Return standardized response
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(clean_decimal_objects(result), indent=2, cls=MegaMindJSONEncoder)
                        }
                    ]
                }
            }
            
        except Exception as e:
            logger.error(f"Error handling Phase 2 tool call {tool_name}: {str(e)}")
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": f"Phase 2 tool execution error: {str(e)}"
                }
            }
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP requests with Phase 2 enhanced capabilities."""
        try:
            method = request.get('method', '')
            params = request.get('params', {})
            request_id = request.get('id', 0)
            
            # Handle standard MCP protocol methods
            if method == 'initialize':
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {
                            "tools": {},
                            "resources": {},
                            "prompts": {}
                        },
                        "serverInfo": {
                            "name": "megamind-phase2-enhanced-mcp-server",
                            "version": "2.0.0",
                            "description": "Phase 2 Enhanced MCP Server with advanced consolidation capabilities"
                        }
                    }
                }
            
            elif method == 'tools/list':
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "tools": self.get_tools_list()
                    }
                }
            
            elif method == 'tools/call':
                return await self.handle_tool_call(params, request_id)
            
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}"
                    }
                }
                
        except Exception as e:
            logger.error(f"Error handling Phase 2 request: {str(e)}")
            return {
                "jsonrpc": "2.0",
                "id": request.get('id', 0),
                "error": {
                    "code": -32603,
                    "message": f"Phase 2 internal error: {str(e)}"
                }
            }