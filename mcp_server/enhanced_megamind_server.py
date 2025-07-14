#!/usr/bin/env python3
"""
Enhanced MegaMind MCP Server with Session System Integration
Extends the existing MCP server with Phase 4 session-aware functionality
"""

import json
import logging
import os
import sys
from typing import Dict, List, Optional, Any

# Import existing components
from megamind_database_server import MCPServer, MegaMindDatabase, clean_decimal_objects
from realm_aware_database import RealmAwareMegaMindDatabase

# Import session system components
from session_manager import SessionManager
from session_mcp_integration import SessionAwareMCPExtension

logger = logging.getLogger(__name__)

class EnhancedMCPServer(MCPServer):
    """
    Enhanced MCP Server with Session System Integration
    Extends the existing MCPServer with session-aware functionality
    """
    
    def __init__(self, db_manager: MegaMindDatabase):
        # Initialize parent MCP server
        super().__init__(db_manager)
        
        # Initialize session management system
        try:
            self.session_manager = SessionManager(db_manager)
            logger.info("SessionManager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize SessionManager: {e}")
            self.session_manager = None
        
        # Initialize session-aware MCP extension
        try:
            embedding_service = getattr(db_manager, 'embedding_service', None)
            if self.session_manager:
                self.session_extension = SessionAwareMCPExtension(
                    db_manager=db_manager,
                    session_manager=self.session_manager,
                    embedding_service=embedding_service
                )
                logger.info("SessionAwareMCPExtension initialized successfully")
            else:
                self.session_extension = None
                logger.warning("SessionAwareMCPExtension not initialized (SessionManager unavailable)")
        except Exception as e:
            logger.error(f"Failed to initialize SessionAwareMCPExtension: {e}")
            self.session_extension = None
    
    def get_tools_list(self) -> List[Dict[str, Any]]:
        """Get enhanced tools list including session management functions"""
        # Get base tools from parent
        base_tools = super().get_tools_list()
        
        # Add session management tools if available
        session_tools = []
        if self.session_extension:
            try:
                session_tools = self.session_extension.get_session_tools_list()
                logger.debug(f"Added {len(session_tools)} session management tools")
            except Exception as e:
                logger.warning(f"Failed to get session tools: {e}")
        
        # Combine tools
        all_tools = base_tools + session_tools
        logger.info(f"Enhanced tools list: {len(base_tools)} base tools + {len(session_tools)} session tools = {len(all_tools)} total")
        
        return all_tools
    
    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced request handler with session-aware functionality"""
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
                            "name": "megamind-database-enhanced",
                            "version": "2.0.0"
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
                
                # Handle session management tools
                if self.session_extension and tool_name.startswith('mcp__megamind__session_'):
                    return self._handle_session_tool_call(tool_name, tool_args, request_id)
                
                # Handle enhanced chunk operations with session awareness
                elif tool_name in ['mcp__megamind__search_chunks', 'mcp__megamind__create_chunk']:
                    return self._handle_enhanced_chunk_operation(tool_name, tool_args, request_id)
                
                # Delegate to parent for other tools
                else:
                    return super().handle_request(request)
            
            # Delegate other methods to parent
            else:
                return super().handle_request(request)
                
        except Exception as e:
            logger.error(f"Enhanced request handling failed: {e}")
            return {
                "jsonrpc": "2.0",
                "id": request.get('id'),
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            }
    
    def _handle_session_tool_call(self, tool_name: str, tool_args: Dict[str, Any], request_id: Any) -> Dict[str, Any]:
        """Handle session management tool calls"""
        try:
            if not self.session_extension:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": "Session management not available"
                    }
                }
            
            # Route to appropriate session handler
            if tool_name == 'mcp__megamind__session_create':
                result = self.session_extension.handle_session_create(tool_args)
            elif tool_name == 'mcp__megamind__session_activate':
                result = self.session_extension.handle_session_activate(tool_args)
            elif tool_name == 'mcp__megamind__session_archive':
                result = self.session_extension.handle_session_archive(tool_args)
            elif tool_name == 'mcp__megamind__session_get_active':
                result = self.session_extension.handle_session_get_active(tool_args)
            elif tool_name == 'mcp__megamind__session_add_entry':
                result = self.session_extension.handle_session_add_entry(tool_args)
            elif tool_name == 'mcp__megamind__session_search_semantic':
                result = self.session_extension.handle_session_search_semantic(tool_args)
            elif tool_name == 'mcp__megamind__session_get_summary':
                result = self.session_extension.handle_session_get_summary(tool_args)
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Unknown session tool: {tool_name}"
                    }
                }
            
            # Clean result and return
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
            logger.error(f"Session tool call failed for {tool_name}: {e}")
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": f"Session tool error: {str(e)}"
                }
            }
    
    def _handle_enhanced_chunk_operation(self, tool_name: str, tool_args: Dict[str, Any], request_id: Any) -> Dict[str, Any]:
        """Handle chunk operations with session awareness"""
        try:
            if tool_name == 'mcp__megamind__search_chunks' and self.session_extension:
                # Use session-aware search if session context provided
                result = self.session_extension.handle_session_aware_search_chunks(tool_args)
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
            
            elif tool_name == 'mcp__megamind__create_chunk' and self.session_extension:
                # Use session-aware chunk creation
                result = self.session_extension.handle_session_aware_create_chunk(tool_args)
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
            
            # Fallback to parent implementation
            else:
                return super().handle_request({
                    "method": "tools/call",
                    "params": {"name": tool_name, "arguments": tool_args},
                    "id": request_id
                })
                
        except Exception as e:
            logger.error(f"Enhanced chunk operation failed for {tool_name}: {e}")
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": f"Enhanced chunk operation error: {str(e)}"
                }
            }
    
    def get_session_status(self) -> Dict[str, Any]:
        """Get comprehensive status of session system"""
        try:
            status = {
                "session_manager_available": self.session_manager is not None,
                "session_extension_available": self.session_extension is not None,
                "embedding_service_available": hasattr(self.db_manager, 'embedding_service') and self.db_manager.embedding_service is not None
            }
            
            if self.session_extension:
                try:
                    service_stats = self.session_extension.session_embedding_service.get_service_stats()
                    status["session_service_stats"] = service_stats
                except Exception as e:
                    status["session_service_error"] = str(e)
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get session status: {e}")
            return {"error": str(e)}

def main():
    """Main entry point for enhanced MCP server"""
    try:
        # Get database configuration
        config = {
            'host': os.getenv('DB_HOST', 'megamind-mysql'),
            'port': int(os.getenv('DB_PORT', 3306)),
            'database': os.getenv('DB_DATABASE', 'megamind_database'),
            'user': os.getenv('DB_USER', 'megamind_user'),
            'password': os.getenv('DB_PASSWORD', 'test_db_password_456')
        }
        
        logger.info("Initializing Enhanced MegaMind MCP Server...")
        
        # Initialize realm-aware database
        db_manager = RealmAwareMegaMindDatabase(
            config=config,
            default_realm=os.getenv('MEGAMIND_PROJECT_REALM', 'PROJECT')
        )
        logger.info("Realm-aware database manager initialized")
        
        # Initialize enhanced MCP server
        server = EnhancedMCPServer(db_manager)
        logger.info("Enhanced MCP server initialized")
        
        # Get and log session status
        session_status = server.get_session_status()
        logger.info(f"Session system status: {json.dumps(session_status, indent=2)}")
        
        # Start server (placeholder for actual server implementation)
        logger.info("Enhanced MCP server ready for requests")
        
        # In a real implementation, this would start the actual server loop
        # For now, return the server instance for testing
        return server
        
    except Exception as e:
        logger.error(f"Failed to initialize enhanced MCP server: {e}")
        return None

if __name__ == "__main__":
    main()