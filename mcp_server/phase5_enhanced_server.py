#!/usr/bin/env python3
"""
Phase 5 Enhanced MCP Server with Advanced Session Functions
Integrates all Phase 5 advanced session capabilities (6 core + 4 semantic functions)
"""

import json
import logging
import os
import sys
from typing import Dict, List, Optional, Any

# Import existing components
from enhanced_megamind_server import EnhancedMCPServer
from advanced_session_functions import AdvancedSessionMCPFunctions
from advanced_session_functions_part2 import AdvancedSessionFunctionsPart2

logger = logging.getLogger(__name__)

class Phase5EnhancedMCPServer(EnhancedMCPServer):
    """
    Phase 5 Enhanced MCP Server with Advanced Session Functions
    Extends EnhancedMCPServer with 6 core + 4 semantic advanced session functions
    """
    
    def __init__(self, db_manager):
        # Initialize parent enhanced server
        super().__init__(db_manager)
        
        # Initialize advanced session functions
        try:
            if self.session_manager and self.session_extension:
                self.advanced_functions = AdvancedSessionMCPFunctions(
                    session_manager=self.session_manager,
                    session_extension=self.session_extension,
                    db_manager=db_manager
                )
                
                self.advanced_functions_part2 = AdvancedSessionFunctionsPart2(
                    session_manager=self.session_manager,
                    session_extension=self.session_extension,
                    db_manager=db_manager
                )
                
                logger.info("Phase 5 Advanced Session Functions initialized successfully")
            else:
                self.advanced_functions = None
                self.advanced_functions_part2 = None
                logger.warning("Advanced session functions not initialized (dependencies unavailable)")
                
        except Exception as e:
            logger.error(f"Failed to initialize advanced session functions: {e}")
            self.advanced_functions = None
            self.advanced_functions_part2 = None
    
    def get_tools_list(self) -> List[Dict[str, Any]]:
        """Get enhanced tools list including Phase 5 advanced session functions"""
        # Get base tools from parent (includes Phase 4 session tools)
        base_tools = super().get_tools_list()
        
        # Add Phase 5 advanced session tools if available
        advanced_tools = []
        if self.advanced_functions:
            try:
                advanced_tools = self.advanced_functions.get_advanced_tools_list()
                logger.debug(f"Added {len(advanced_tools)} Phase 5 advanced session tools")
            except Exception as e:
                logger.warning(f"Failed to get advanced session tools: {e}")
        
        # Combine tools
        all_tools = base_tools + advanced_tools
        logger.info(f"Phase 5 tools list: {len(base_tools)} base tools + {len(advanced_tools)} advanced tools = {len(all_tools)} total")
        
        return all_tools
    
    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced request handler with Phase 5 advanced session functionality"""
        try:
            method = request.get('method', '')
            params = request.get('params', {})
            request_id = request.get('id')
            
            # Handle initialization and tools/list with enhanced tools
            if method == 'initialize':
                tools_list = self.get_tools_list()
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {
                            "tools": {tool["name"]: tool for tool in tools_list}
                        },
                        "serverInfo": {
                            "name": "megamind-database-phase5",
                            "version": "3.0.0"
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
                tool_name = params.get('name', '')
                tool_args = params.get('arguments', {})
                
                # Handle Phase 5 advanced session tools
                if self._is_phase5_tool(tool_name):
                    return self._handle_phase5_tool_call(tool_name, tool_args, request_id)
                
                # Delegate to parent for other tools (including Phase 4 session tools)
                else:
                    return super().handle_request(request)
            
            # Delegate other methods to parent
            else:
                return super().handle_request(request)
                
        except Exception as e:
            logger.error(f"Phase 5 request handling failed: {e}")
            return {
                "jsonrpc": "2.0",
                "id": request.get('id'),
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            }
    
    def _is_phase5_tool(self, tool_name: str) -> bool:
        """Check if tool is a Phase 5 advanced session tool"""
        phase5_tools = [
            'mcp__megamind__session_list_user_sessions',
            'mcp__megamind__session_bulk_archive',
            'mcp__megamind__session_get_entries_filtered',
            'mcp__megamind__session_analytics_dashboard',
            'mcp__megamind__session_export',
            'mcp__megamind__session_relationship_tracking',
            'mcp__megamind__session_semantic_similarity',
            'mcp__megamind__session_semantic_clustering',
            'mcp__megamind__session_semantic_insights',
            'mcp__megamind__session_semantic_recommendations'
        ]
        return tool_name in phase5_tools
    
    def _handle_phase5_tool_call(self, tool_name: str, tool_args: Dict[str, Any], request_id: Any) -> Dict[str, Any]:
        """Handle Phase 5 advanced session tool calls"""
        try:
            if not self.advanced_functions or not self.advanced_functions_part2:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": "Advanced session functions not available"
                    }
                }
            
            # Route to appropriate advanced function handler
            if tool_name == 'mcp__megamind__session_list_user_sessions':
                result = self.advanced_functions.handle_session_list_user_sessions(tool_args)
            elif tool_name == 'mcp__megamind__session_bulk_archive':
                result = self.advanced_functions.handle_session_bulk_archive(tool_args)
            elif tool_name == 'mcp__megamind__session_get_entries_filtered':
                result = self.advanced_functions.handle_session_get_entries_filtered(tool_args)
            elif tool_name == 'mcp__megamind__session_analytics_dashboard':
                result = self.advanced_functions_part2.handle_session_analytics_dashboard(tool_args)
            elif tool_name == 'mcp__megamind__session_export':
                result = self.advanced_functions_part2.handle_session_export(tool_args)
            elif tool_name == 'mcp__megamind__session_relationship_tracking':
                result = self.advanced_functions_part2.handle_session_relationship_tracking(tool_args)
            elif tool_name == 'mcp__megamind__session_semantic_similarity':
                result = self.advanced_functions_part2.handle_session_semantic_similarity(tool_args)
            elif tool_name == 'mcp__megamind__session_semantic_clustering':
                result = self.advanced_functions_part2.handle_session_semantic_clustering(tool_args)
            elif tool_name == 'mcp__megamind__session_semantic_insights':
                result = self.advanced_functions_part2.handle_session_semantic_insights(tool_args)
            elif tool_name == 'mcp__megamind__session_semantic_recommendations':
                result = self.advanced_functions_part2.handle_session_semantic_recommendations(tool_args)
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Unknown Phase 5 tool: {tool_name}"
                    }
                }
            
            # Clean result and return
            from megamind_database_server import clean_decimal_objects
            clean_result = clean_decimal_objects(result)
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(clean_result, indent=2)
                        }
                    ]
                }
            }
            
        except Exception as e:
            logger.error(f"Phase 5 tool call failed for {tool_name}: {e}")
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": f"Phase 5 tool error: {str(e)}"
                }
            }
    
    def get_phase5_status(self) -> Dict[str, Any]:
        """Get comprehensive status of Phase 5 system"""
        try:
            # Get base status from parent
            base_status = super().get_session_status()
            
            # Add Phase 5 specific status
            phase5_status = {
                "phase5_advanced_functions_available": self.advanced_functions is not None,
                "phase5_advanced_functions_part2_available": self.advanced_functions_part2 is not None,
                "total_mcp_functions": len(self.get_tools_list()),
                "phase5_function_count": 10  # 6 core + 4 semantic
            }
            
            # Combine statuses
            combined_status = {**base_status, **phase5_status}
            
            # Add tool categorization
            if self.advanced_functions:
                tools = self.get_tools_list()
                phase4_tools = [t for t in tools if t['name'].startswith('mcp__megamind__session_') and not self._is_phase5_tool(t['name'])]
                phase5_tools = [t for t in tools if self._is_phase5_tool(t['name'])]
                base_tools = [t for t in tools if not t['name'].startswith('mcp__megamind__session_')]
                
                combined_status["tool_categorization"] = {
                    "base_tools": len(base_tools),
                    "phase4_session_tools": len(phase4_tools),
                    "phase5_advanced_tools": len(phase5_tools),
                    "total_tools": len(tools)
                }
            
            return combined_status
            
        except Exception as e:
            logger.error(f"Failed to get Phase 5 status: {e}")
            return {"error": str(e)}

def main():
    """Main entry point for Phase 5 enhanced MCP server"""
    try:
        # Get database configuration
        config = {
            'host': os.getenv('DB_HOST', 'megamind-mysql'),
            'port': int(os.getenv('DB_PORT', 3306)),
            'database': os.getenv('DB_DATABASE', 'megamind_database'),
            'user': os.getenv('DB_USER', 'megamind_user'),
            'password': os.getenv('DB_PASSWORD', 'test_db_password_456')
        }
        
        logger.info("Initializing Phase 5 Enhanced MegaMind MCP Server...")
        
        # Initialize realm-aware database
        from realm_aware_database import RealmAwareMegaMindDatabase
        db_manager = RealmAwareMegaMindDatabase(config=config)
        logger.info("Realm-aware database manager initialized")
        
        # Initialize Phase 5 enhanced MCP server
        server = Phase5EnhancedMCPServer(db_manager)
        logger.info("Phase 5 Enhanced MCP server initialized")
        
        # Get and log Phase 5 status
        phase5_status = server.get_phase5_status()
        logger.info(f"Phase 5 system status: {json.dumps(phase5_status, indent=2)}")
        
        # Start server (placeholder for actual server implementation)
        logger.info("Phase 5 Enhanced MCP server ready for requests")
        
        # In a real implementation, this would start the actual server loop
        # For now, return the server instance for testing
        return server
        
    except Exception as e:
        logger.error(f"Failed to initialize Phase 5 enhanced MCP server: {e}")
        return None

if __name__ == "__main__":
    main()