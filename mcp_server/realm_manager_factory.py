#!/usr/bin/env python3
"""
Realm Manager Factory for MCP JSON-RPC Refactoring
Factory for creating and caching realm-specific database managers with shared resources
"""

import logging
import asyncio
from typing import Dict, Any, Optional
from dataclasses import dataclass

try:
    from .realm_aware_database import RealmAwareMegaMindDatabase
    from .realm_config import RealmConfigurationManager, RealmAccessController, RealmConfig
    from .services.embedding_service import get_embedding_service
except ImportError:
    from realm_aware_database import RealmAwareMegaMindDatabase
    from realm_config import RealmConfigurationManager, RealmAccessController, RealmConfig
    from services.embedding_service import get_embedding_service

logger = logging.getLogger(__name__)

@dataclass
class RealmContext:
    """Enhanced realm context supporting dynamic configuration"""
    realm_id: str
    project_name: str
    default_target: str = "PROJECT"
    global_realm: str = 'GLOBAL'
    cross_realm_search_enabled: bool = True
    project_priority_weight: float = 1.2
    global_priority_weight: float = 1.0
    
    @classmethod
    def from_dynamic_config(cls, config: Dict[str, Any]) -> 'RealmContext':
        """Create RealmContext from dynamic configuration"""
        return cls(
            realm_id=config['project_realm'],
            project_name=config['project_name'],
            default_target=config['default_target'],
            global_realm=config.get('global_realm', 'GLOBAL'),
            cross_realm_search_enabled=config.get('cross_realm_search_enabled', True),
            project_priority_weight=config.get('project_priority_weight', 1.2),
            global_priority_weight=config.get('global_priority_weight', 1.0)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'realm_id': self.realm_id,
            'project_name': self.project_name,
            'default_target': self.default_target,
            'global_realm': self.global_realm,
            'cross_realm_search_enabled': self.cross_realm_search_enabled,
            'project_priority_weight': self.project_priority_weight,
            'global_priority_weight': self.global_priority_weight
        }
    
    def create_realm_config(self) -> 'RealmConfig':
        """Create RealmConfig instance for access control"""
        return RealmConfig(
            project_realm=self.realm_id,
            project_name=self.project_name,
            default_target=self.default_target,
            global_realm=self.global_realm,
            cross_realm_search_enabled=self.cross_realm_search_enabled,
            project_priority_weight=self.project_priority_weight,
            global_priority_weight=self.global_priority_weight
        )

class RealmManagerFactory:
    """Factory for managing multiple realm contexts with shared resources"""
    
    def __init__(self, base_config: Dict[str, Any]):
        self.base_config = base_config
        self.shared_embedding_service = None
        self.realm_managers: Dict[str, RealmAwareMegaMindDatabase] = {}
        self.initialization_lock = asyncio.Lock()
        self.initialized = False
        
        logger.info("RealmManagerFactory initialized with base config")
    
    async def initialize_shared_services(self):
        """Initialize services shared across all realms"""
        async with self.initialization_lock:
            if self.initialized:
                logger.info("Shared services already initialized, skipping")
                return
            
            logger.info("=== Initializing Shared Services ===")
            try:
                logger.info("Creating embedding service instance...")
                self.shared_embedding_service = get_embedding_service()
                logger.info("✓ Embedding service instance created")
                
                # Wait for embedding service to be ready
                if hasattr(self.shared_embedding_service, 'initialize'):
                    logger.info("Initializing embedding service...")
                    await self.shared_embedding_service.initialize()
                    logger.info("✓ Embedding service initialization completed")
                
                # Test readiness
                if hasattr(self.shared_embedding_service, 'test_readiness'):
                    logger.info("Testing embedding service readiness...")
                    readiness = self.shared_embedding_service.test_readiness()
                    if not readiness.get('ready', False):
                        logger.error(f"Embedding service readiness test failed: {readiness}")
                        raise RuntimeError(f"Embedding service not ready: {readiness}")
                    logger.info("✓ Embedding service readiness test passed")
                
                self.initialized = True
                logger.info("🎉 Shared embedding service initialized successfully")
                
            except Exception as e:
                logger.error(f"✗ Failed to initialize shared embedding service: {e}")
                logger.error(f"Error type: {type(e).__name__}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                raise
    
    def create_realm_config(self, realm_id: str) -> RealmConfig:
        """Create realm-specific configuration"""
        try:
            # Create a realm configuration for the specified realm
            realm_config = RealmConfig(
                project_realm=realm_id,
                global_realm="GLOBAL",
                project_name=f"Dynamic Realm {realm_id}",
                default_target="PROJECT"
            )
            
            logger.debug(f"Created realm configuration for {realm_id}")
            return realm_config
            
        except Exception as e:
            logger.error(f"Failed to create realm config for {realm_id}: {e}")
            # Return a default configuration as fallback
            return RealmConfig(
                project_realm=realm_id,
                global_realm="GLOBAL", 
                project_name=f"Fallback Realm {realm_id}",
                default_target="PROJECT"
            )
    
    async def get_realm_manager(self, realm_id: str) -> RealmAwareMegaMindDatabase:
        """Get or create realm-specific manager with shared resources"""
        
        # Ensure shared services are initialized
        if not self.initialized:
            await self.initialize_shared_services()
        
        # Return existing manager if available
        if realm_id in self.realm_managers:
            logger.debug(f"Returning existing manager for realm {realm_id}")
            return self.realm_managers[realm_id]
        
        try:
            logger.info(f"Creating new realm manager for {realm_id}")
            
            # Create realm-specific configuration
            realm_config = self.create_realm_config(realm_id)
            
            # Create manager with shared embedding service
            manager = RealmAwareMegaMindDatabase(
                config=self.base_config,
                realm_config=realm_config,
                shared_embedding_service=self.shared_embedding_service
            )
            
            # Cache the manager
            self.realm_managers[realm_id] = manager
            
            logger.info(f"Successfully created realm manager for {realm_id}")
            return manager
            
        except Exception as e:
            logger.error(f"Failed to create realm manager for {realm_id}: {e}")
            raise
    
    async def get_default_realm_manager(self) -> RealmAwareMegaMindDatabase:
        """Get the default realm manager (PROJECT realm)"""
        default_realm = self.base_config.get('default_realm', 'PROJECT')
        return await self.get_realm_manager(default_realm)
    
    def list_realms(self) -> Dict[str, Dict[str, Any]]:
        """List all available realms and their status"""
        realm_info = {}
        
        for realm_id, manager in self.realm_managers.items():
            try:
                # Get basic realm information
                realm_info[realm_id] = {
                    'realm_id': realm_id,
                    'active': True,
                    'manager_type': type(manager).__name__,
                    'has_embedding_service': hasattr(manager, 'embedding_service') and manager.embedding_service is not None
                }
                
                # Add realm configuration if available
                if hasattr(manager, 'realm_config'):
                    realm_info[realm_id]['config'] = {
                        'project_realm': manager.realm_config.project_realm,
                        'global_realm': manager.realm_config.global_realm,
                        'project_name': manager.realm_config.project_name
                    }
                    
            except Exception as e:
                realm_info[realm_id] = {
                    'realm_id': realm_id,
                    'active': False,
                    'error': str(e)
                }
        
        return realm_info
    
    async def check_realm_health(self, realm_id: str) -> Dict[str, Any]:
        """Check health of specific realm"""
        try:
            if realm_id not in self.realm_managers:
                return {
                    'realm_id': realm_id,
                    'healthy': False,
                    'error': 'Realm not initialized'
                }
            
            manager = self.realm_managers[realm_id]
            
            # Perform basic health checks
            health_status = {
                'realm_id': realm_id,
                'healthy': True,
                'checks': {}
            }
            
            # Check database connection
            try:
                if hasattr(manager, 'search_chunks_dual_realm'):
                    test_results = manager.search_chunks_dual_realm("health_check", limit=1)
                    health_status['checks']['database'] = {
                        'status': 'healthy',
                        'test_query_results': len(test_results)
                    }
                else:
                    health_status['checks']['database'] = {
                        'status': 'unknown',
                        'message': 'No dual realm search method available'
                    }
            except Exception as e:
                health_status['checks']['database'] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
                health_status['healthy'] = False
            
            # Check embedding service
            try:
                if hasattr(manager, 'embedding_service') and manager.embedding_service:
                    embedding_available = manager.embedding_service.is_available()
                    health_status['checks']['embedding'] = {
                        'status': 'healthy' if embedding_available else 'unhealthy',
                        'available': embedding_available
                    }
                    
                    if not embedding_available:
                        health_status['healthy'] = False
                else:
                    health_status['checks']['embedding'] = {
                        'status': 'unavailable',
                        'message': 'No embedding service configured'
                    }
            except Exception as e:
                health_status['checks']['embedding'] = {
                    'status': 'error',
                    'error': str(e)
                }
                health_status['healthy'] = False
            
            return health_status
            
        except Exception as e:
            return {
                'realm_id': realm_id,
                'healthy': False,
                'error': f"Health check failed: {str(e)}"
            }
    
    async def cleanup_realm(self, realm_id: str) -> bool:
        """Clean up resources for a specific realm"""
        try:
            if realm_id in self.realm_managers:
                manager = self.realm_managers[realm_id]
                
                # Perform cleanup if manager has cleanup method
                if hasattr(manager, 'cleanup'):
                    await manager.cleanup()
                
                # Remove from cache
                del self.realm_managers[realm_id]
                
                logger.info(f"Successfully cleaned up realm {realm_id}")
                return True
            else:
                logger.warning(f"Attempted to cleanup non-existent realm {realm_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to cleanup realm {realm_id}: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown all realm managers and shared services"""
        try:
            logger.info("Shutting down RealmManagerFactory...")
            
            # Cleanup all realm managers
            for realm_id in list(self.realm_managers.keys()):
                await self.cleanup_realm(realm_id)
            
            # Cleanup shared embedding service
            if self.shared_embedding_service and hasattr(self.shared_embedding_service, 'cleanup'):
                await self.shared_embedding_service.cleanup()
            
            self.initialized = False
            logger.info("RealmManagerFactory shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during RealmManagerFactory shutdown: {e}")
    
    async def create_dynamic_realm_manager(self, realm_context: RealmContext) -> RealmAwareMegaMindDatabase:
        """Create realm manager with dynamic configuration"""
        try:
            # Ensure shared services are initialized
            if not self.initialized:
                await self.initialize_shared_services()
            
            # Create dynamic realm configuration from context
            realm_config = realm_context.create_realm_config()
            
            # Create temporary configuration manager with dynamic config
            config_manager = RealmConfigurationManager()
            config_manager.config = realm_config
            
            # Create access controller with dynamic configuration
            access_controller = RealmAccessController(config_manager)
            
            # Create realm-aware database with dynamic configuration
            realm_manager = RealmAwareMegaMindDatabase(
                config=self.base_config,
                realm_config=realm_config,
                shared_embedding_service=self.shared_embedding_service,
                access_controller=access_controller
            )
            
            logger.info(f"✅ Created dynamic realm manager for {realm_context.realm_id} with full configuration")
            return realm_manager
            
        except Exception as e:
            logger.error(f"Failed to create dynamic realm manager: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise

class DynamicRealmManagerFactory(RealmManagerFactory):
    """Enhanced factory with dynamic realm creation capabilities"""
    
    def __init__(self, base_config: Dict[str, Any]):
        super().__init__(base_config)
        self.realm_creation_lock = asyncio.Lock()
        
    async def create_realm(self, realm_id: str, realm_config: Dict[str, Any]) -> bool:
        """Create a new realm with custom configuration"""
        async with self.realm_creation_lock:
            try:
                if realm_id in self.realm_managers:
                    logger.warning(f"Realm {realm_id} already exists, skipping creation")
                    return True
                
                logger.info(f"Creating new dynamic realm: {realm_id}")
                
                # Create custom realm configuration
                custom_config = RealmConfig(
                    project_realm=realm_id,
                    global_realm=realm_config.get('global_realm', 'GLOBAL'),
                    project_name=realm_config.get('project_name', f"Dynamic Realm {realm_id}"),
                    default_target=realm_config.get('default_target', 'PROJECT')
                )
                
                # Ensure shared services are initialized
                if not self.initialized:
                    await self.initialize_shared_services()
                
                # Create manager with custom configuration
                manager = RealmAwareMegaMindDatabase(
                    config=self.base_config,
                    realm_config=custom_config,
                    shared_embedding_service=self.shared_embedding_service
                )
                
                # Cache the manager
                self.realm_managers[realm_id] = manager
                
                logger.info(f"Successfully created dynamic realm {realm_id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to create dynamic realm {realm_id}: {e}")
                return False
    
    def extract_realm_context(self, request_data: Dict[str, Any], default_realm: str = "PROJECT") -> RealmContext:
        """Extract realm context from request data"""
        try:
            # Check for realm_id in tool arguments
            realm_id = default_realm
            
            if 'params' in request_data and 'arguments' in request_data['params']:
                args = request_data['params']['arguments']
                realm_id = args.get('realm_id', default_realm)
            
            # Create realm context
            context = RealmContext(
                realm_id=realm_id,
                project_name=f"Project {realm_id}",
                default_target="PROJECT"
            )
            
            logger.debug(f"Extracted realm context: {context.to_dict()}")
            return context
            
        except Exception as e:
            logger.error(f"Failed to extract realm context: {e}")
            # Return default context
            return RealmContext(
                realm_id=default_realm,
                project_name=f"Default {default_realm}",
                default_target="PROJECT"
            )