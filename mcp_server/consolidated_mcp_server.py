"""
Consolidated MCP Server - Phase 1 Implementation
GitHub Issue #19: Function Name Standardization

This module implements the new consolidated MCP server with master functions
that replace the previous 44 functions with 19 standardized functions.
"""

import json
import logging
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from decimal import Decimal

try:
    from .megamind_database_server import MCPServer, MegaMindJSONEncoder, clean_decimal_objects
    from .consolidated_functions import ConsolidatedMCPFunctions
    from .realm_aware_database import RealmAwareMegaMindDatabase
except ImportError:
    from megamind_database_server import MCPServer, MegaMindJSONEncoder, clean_decimal_objects
    from consolidated_functions import ConsolidatedMCPFunctions
    from realm_aware_database import RealmAwareMegaMindDatabase

logger = logging.getLogger(__name__)

class ConsolidatedMCPServer(MCPServer):
    """
    Consolidated MCP Server implementing standardized function architecture.
    
    Provides 19 master functions that intelligently route to existing subfunctions
    while maintaining backward compatibility with all existing functionality.
    """
    
    def __init__(self, db_manager):
        """Initialize consolidated MCP server with master functions."""
        super().__init__(db_manager)
        
        # Initialize consolidated functions
        self.consolidated_functions = ConsolidatedMCPFunctions(
            self.db_manager, 
            self.session_manager
        )
        
        logger.info("Consolidated MCP Server initialized with 19 master functions")
    
    def get_tools_list(self) -> List[Dict[str, Any]]:
        """Get the consolidated list of 19 MCP tools."""
        return [
            # 🔍 SEARCH CLASS - 3 Master Functions
            {
                "name": "mcp__megamind__search_query",
                "description": "Master search function with intelligent routing (hybrid, semantic, similarity, keyword)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query text"},
                        "search_type": {"type": "string", "default": "hybrid", "enum": ["hybrid", "semantic", "similarity", "keyword"], "description": "Search type"},
                        "limit": {"type": "integer", "default": 10, "description": "Maximum results"},
                        "threshold": {"type": "number", "default": 0.7, "description": "Similarity threshold"},
                        "reference_chunk_id": {"type": "string", "description": "Reference chunk ID (required for similarity search)"}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "mcp__megamind__search_related",
                "description": "Master function for finding related chunks and contexts",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "chunk_id": {"type": "string", "description": "Reference chunk ID"},
                        "max_depth": {"type": "integer", "default": 2, "description": "Relationship traversal depth"},
                        "include_hot_contexts": {"type": "boolean", "default": False, "description": "Include hot contexts"},
                        "model_type": {"type": "string", "default": "sonnet", "description": "Model type for hot contexts"}
                    },
                    "required": ["chunk_id"]
                }
            },
            {
                "name": "mcp__megamind__search_retrieve",
                "description": "Master function for retrieving specific chunks by ID",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "chunk_id": {"type": "string", "description": "Unique chunk identifier"},
                        "include_relationships": {"type": "boolean", "default": True, "description": "Include related chunks"},
                        "track_access": {"type": "boolean", "default": True, "description": "Track access for analytics"},
                        "query_context": {"type": "string", "description": "Context for access tracking"}
                    },
                    "required": ["chunk_id"]
                }
            },
            
            # 📝 CONTENT CLASS - 4 Master Functions
            {
                "name": "mcp__megamind__content_create",
                "description": "Master function for creating new chunks and relationships",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Chunk content"},
                        "source_document": {"type": "string", "description": "Source document name"},
                        "section_path": {"type": "string", "description": "Section path within document"},
                        "session_id": {"type": "string", "description": "Session ID for tracking"},
                        "target_realm": {"type": "string", "default": "PROJECT", "description": "Target realm for creation"},
                        "create_relationships": {"type": "boolean", "default": True, "description": "Create relationships with existing chunks"},
                        "relationship_targets": {"type": "array", "items": {"type": "string"}, "description": "Specific chunk IDs to relate to"},
                        "relationship_type": {"type": "string", "default": "related", "description": "Type of relationship"}
                    },
                    "required": ["content", "source_document"]
                }
            },
            {
                "name": "mcp__megamind__content_update",
                "description": "Master function for modifying existing chunks",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "chunk_id": {"type": "string", "description": "Chunk ID to update"},
                        "new_content": {"type": "string", "description": "Updated content"},
                        "session_id": {"type": "string", "description": "Session ID for tracking"},
                        "update_embeddings": {"type": "boolean", "default": True, "description": "Regenerate embeddings"},
                        "update_relationships": {"type": "boolean", "default": False, "description": "Update relationships"},
                        "new_relationships": {"type": "array", "items": {"type": "object"}, "description": "New relationships to add"}
                    },
                    "required": ["chunk_id", "new_content"]
                }
            },
            {
                "name": "mcp__megamind__content_process",
                "description": "Master document processing function (analyze, chunk, optimize)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Document content to process"},
                        "document_name": {"type": "string", "description": "Name of the document"},
                        "session_id": {"type": "string", "description": "Session ID for tracking"},
                        "strategy": {"type": "string", "default": "auto", "enum": ["auto", "semantic", "fixed"], "description": "Chunking strategy"},
                        "max_tokens": {"type": "integer", "default": 150, "description": "Maximum tokens per chunk"},
                        "target_realm": {"type": "string", "default": "PROJECT", "description": "Target realm for chunks"},
                        "analyze_first": {"type": "boolean", "default": True, "description": "Analyze document structure first"},
                        "optimize_after": {"type": "boolean", "default": True, "description": "Optimize embeddings after creation"},
                        "batch_size": {"type": "integer", "default": 10, "description": "Batch size for processing"}
                    },
                    "required": ["content", "document_name"]
                }
            },
            {
                "name": "mcp__megamind__content_manage",
                "description": "Master content management function (ingest, discover, optimize, get_related)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["ingest", "discover", "optimize", "get_related"], "description": "Action to perform"},
                        "document_path": {"type": "string", "description": "Document path (for ingest)"},
                        "chunk_ids": {"type": "array", "items": {"type": "string"}, "description": "Chunk IDs (for discover)"},
                        "chunk_id": {"type": "string", "description": "Chunk ID (for get_related)"},
                        "target_queries": {"type": "array", "items": {"type": "string"}, "description": "Target queries (for optimize)"},
                        "session_id": {"type": "string", "description": "Session ID for tracking"},
                        "processing_options": {"type": "object", "description": "Processing options"},
                        "discovery_method": {"type": "string", "default": "semantic", "description": "Discovery method"},
                        "optimization_strategy": {"type": "string", "default": "performance", "description": "Optimization strategy"},
                        "relation_types": {"type": "array", "items": {"type": "string"}, "description": "Relation types"}
                    },
                    "required": ["action"]
                }
            },
            
            # 🚀 PROMOTION CLASS - 3 Master Functions
            {
                "name": "mcp__megamind__promotion_request",
                "description": "Master function for creating and managing promotion requests",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "chunk_id": {"type": "string", "description": "Chunk ID to promote"},
                        "target_realm": {"type": "string", "description": "Target realm for promotion"},
                        "justification": {"type": "string", "description": "Justification for promotion"},
                        "session_id": {"type": "string", "description": "Session ID for tracking"},
                        "auto_analyze": {"type": "boolean", "default": True, "description": "Automatically analyze impact"}
                    },
                    "required": ["chunk_id", "target_realm", "justification"]
                }
            },
            {
                "name": "mcp__megamind__promotion_review",
                "description": "Master function for reviewing promotions (approve/reject)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "promotion_id": {"type": "string", "description": "Promotion ID to review"},
                        "action": {"type": "string", "enum": ["approve", "reject"], "description": "Action to perform"},
                        "reason": {"type": "string", "description": "Reason for action"},
                        "session_id": {"type": "string", "description": "Session ID for tracking"},
                        "analyze_before": {"type": "boolean", "default": True, "description": "Analyze impact before action"}
                    },
                    "required": ["promotion_id", "action", "reason"]
                }
            },
            {
                "name": "mcp__megamind__promotion_monitor",
                "description": "Master function for monitoring promotion queue",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "filter_status": {"type": "string", "enum": ["", "pending", "approved", "rejected"], "description": "Filter by status"},
                        "filter_realm": {"type": "string", "description": "Filter by target realm"},
                        "limit": {"type": "integer", "default": 20, "description": "Maximum number of requests"},
                        "include_summary": {"type": "boolean", "default": True, "description": "Include queue summary"}
                    },
                    "required": []
                }
            },
            
            # 🔄 SESSION CLASS - 4 Master Functions
            {
                "name": "mcp__megamind__session_create",
                "description": "Master function for creating sessions",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_type": {"type": "string", "enum": ["processing", "operational", "general"], "description": "Type of session"},
                        "created_by": {"type": "string", "description": "Creator identifier"},
                        "description": {"type": "string", "description": "Session description"},
                        "metadata": {"type": "object", "description": "Session metadata"},
                        "auto_prime": {"type": "boolean", "default": True, "description": "Automatically prime context"}
                    },
                    "required": ["session_type", "created_by"]
                }
            },
            {
                "name": "mcp__megamind__session_manage",
                "description": "Master function for session management (get_state, track_action, prime_context)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "description": "Session ID to manage"},
                        "action": {"type": "string", "enum": ["get_state", "track_action", "prime_context"], "description": "Action to perform"},
                        "action_type": {"type": "string", "description": "Action type (for track_action)"},
                        "action_details": {"type": "object", "description": "Action details (for track_action)"},
                        "context_type": {"type": "string", "default": "auto", "description": "Context type (for prime_context)"},
                        "metadata": {"type": "object", "description": "Additional metadata"}
                    },
                    "required": ["session_id", "action"]
                }
            },
            {
                "name": "mcp__megamind__session_review",
                "description": "Master function for session review (recap, pending, recent)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "description": "Session ID to review"},
                        "include_recap": {"type": "boolean", "default": True, "description": "Include session recap"},
                        "include_pending": {"type": "boolean", "default": True, "description": "Include pending changes"},
                        "include_recent": {"type": "boolean", "default": False, "description": "Include recent sessions"}
                    },
                    "required": ["session_id"]
                }
            },
            {
                "name": "mcp__megamind__session_commit",
                "description": "Master function for session commitment and closure",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "description": "Session ID to commit"},
                        "approved_changes": {"type": "array", "items": {"type": "string"}, "description": "Approved change IDs"},
                        "close_session": {"type": "boolean", "default": True, "description": "Close session after commit"},
                        "completion_status": {"type": "string", "default": "completed", "description": "Completion status"}
                    },
                    "required": ["session_id"]
                }
            },
            
            # 🤖 AI CLASS - 3 Master Functions
            {
                "name": "mcp__megamind__ai_enhance",
                "description": "Master AI enhancement function (quality, curation, optimization)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "chunk_ids": {"type": "array", "items": {"type": "string"}, "description": "Chunk IDs to enhance"},
                        "session_id": {"type": "string", "description": "Session ID for tracking"},
                        "enhancement_type": {"type": "string", "default": "comprehensive", "enum": ["quality", "curation", "optimization", "comprehensive"], "description": "Enhancement type"},
                        "include_curation": {"type": "boolean", "default": True, "description": "Include curation workflow"},
                        "include_optimization": {"type": "boolean", "default": True, "description": "Include performance optimization"}
                    },
                    "required": ["chunk_ids"]
                }
            },
            {
                "name": "mcp__megamind__ai_learn",
                "description": "Master AI learning function (feedback recording and strategy updates)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "feedback_data": {"type": "object", "description": "User feedback data"},
                        "session_id": {"type": "string", "description": "Session ID for tracking"},
                        "update_strategy": {"type": "boolean", "default": True, "description": "Update adaptive strategy"}
                    },
                    "required": ["feedback_data"]
                }
            },
            {
                "name": "mcp__megamind__ai_analyze",
                "description": "Master AI analysis function (performance insights and enhancement reports)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "analysis_type": {"type": "string", "enum": ["performance", "enhancement", "comprehensive"], "description": "Analysis type"},
                        "session_id": {"type": "string", "description": "Session ID for tracking"},
                        "target_chunks": {"type": "array", "items": {"type": "string"}, "description": "Specific chunks to analyze"},
                        "include_insights": {"type": "boolean", "default": True, "description": "Include performance insights"},
                        "include_report": {"type": "boolean", "default": True, "description": "Include enhancement report"}
                    },
                    "required": ["analysis_type"]
                }
            },
            
            # 📊 ANALYTICS CLASS - 2 Master Functions
            {
                "name": "mcp__megamind__analytics_track",
                "description": "Master analytics tracking function",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "chunk_id": {"type": "string", "description": "Chunk ID to track"},
                        "query_context": {"type": "string", "description": "Context for the access"},
                        "track_type": {"type": "string", "default": "access", "enum": ["access", "usage", "performance"], "description": "Type of tracking"},
                        "metadata": {"type": "object", "description": "Additional tracking metadata"}
                    },
                    "required": ["chunk_id"]
                }
            },
            {
                "name": "mcp__megamind__analytics_insights",
                "description": "Master analytics insights function",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "insight_type": {"type": "string", "default": "hot_contexts", "enum": ["hot_contexts", "usage_patterns", "performance"], "description": "Type of insights"},
                        "model_type": {"type": "string", "default": "sonnet", "description": "Model type for insights"},
                        "limit": {"type": "integer", "default": 20, "description": "Maximum insights"},
                        "include_metrics": {"type": "boolean", "default": True, "description": "Include detailed metrics"}
                    },
                    "required": []
                }
            }
        ]
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP requests with consolidated function routing."""
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
                            "tools": {}
                        },
                        "serverInfo": {
                            "name": "megamind-consolidated-mcp-server",
                            "version": "1.0.0"
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
            logger.error(f"Error handling request: {str(e)}")
            return {
                "jsonrpc": "2.0",
                "id": request.get('id', 0),
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            }
    
    async def handle_tool_call(self, params: Dict[str, Any], request_id: int) -> Dict[str, Any]:
        """Handle tool calls with consolidated master functions."""
        try:
            tool_name = params.get('name', '')
            tool_args = params.get('arguments', {})
            
            # Route to appropriate master function
            if tool_name == 'mcp__megamind__search_query':
                result = await self.consolidated_functions.search_query(**tool_args)
            elif tool_name == 'mcp__megamind__search_related':
                result = await self.consolidated_functions.search_related(**tool_args)
            elif tool_name == 'mcp__megamind__search_retrieve':
                result = await self.consolidated_functions.search_retrieve(**tool_args)
            elif tool_name == 'mcp__megamind__content_create':
                result = await self.consolidated_functions.content_create(**tool_args)
            elif tool_name == 'mcp__megamind__content_update':
                result = await self.consolidated_functions.content_update(**tool_args)
            elif tool_name == 'mcp__megamind__content_process':
                result = await self.consolidated_functions.content_process(**tool_args)
            elif tool_name == 'mcp__megamind__content_manage':
                result = await self.consolidated_functions.content_manage(**tool_args)
            elif tool_name == 'mcp__megamind__promotion_request':
                result = await self.consolidated_functions.promotion_request(**tool_args)
            elif tool_name == 'mcp__megamind__promotion_review':
                result = await self.consolidated_functions.promotion_review(**tool_args)
            elif tool_name == 'mcp__megamind__promotion_monitor':
                result = await self.consolidated_functions.promotion_monitor(**tool_args)
            elif tool_name == 'mcp__megamind__session_create':
                result = await self.consolidated_functions.session_create(**tool_args)
            elif tool_name == 'mcp__megamind__session_manage':
                result = await self.consolidated_functions.session_manage(**tool_args)
            elif tool_name == 'mcp__megamind__session_review':
                result = await self.consolidated_functions.session_review(**tool_args)
            elif tool_name == 'mcp__megamind__session_commit':
                result = await self.consolidated_functions.session_commit(**tool_args)
            elif tool_name == 'mcp__megamind__ai_enhance':
                result = await self.consolidated_functions.ai_enhance(**tool_args)
            elif tool_name == 'mcp__megamind__ai_learn':
                result = await self.consolidated_functions.ai_learn(**tool_args)
            elif tool_name == 'mcp__megamind__ai_analyze':
                result = await self.consolidated_functions.ai_analyze(**tool_args)
            elif tool_name == 'mcp__megamind__analytics_track':
                result = await self.consolidated_functions.analytics_track(**tool_args)
            elif tool_name == 'mcp__megamind__analytics_insights':
                result = await self.consolidated_functions.analytics_insights(**tool_args)
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Unknown tool: {tool_name}"
                    }
                }
            
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
            logger.error(f"Error handling tool call {tool_name}: {str(e)}")
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": f"Tool execution error: {str(e)}"
                }
            }