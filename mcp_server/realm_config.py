#!/usr/bin/env python3
"""
Realm Configuration Management System
Handles environment-based realm configuration and dual-realm access patterns
"""

import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class RealmConfig:
    """Configuration for realm-aware operations"""
    project_realm: str
    project_name: str
    default_target: str  # 'PROJECT' or 'GLOBAL'
    global_realm: str = 'GLOBAL'
    cross_realm_search_enabled: bool = True
    project_priority_weight: float = 1.2
    global_priority_weight: float = 1.0
    
    def get_search_realms(self) -> List[str]:
        """Get list of realms to search (dual-realm access)"""
        if self.cross_realm_search_enabled:
            return [self.global_realm, self.project_realm]
        else:
            return [self.project_realm]
    
    def calculate_realm_priority_score(self, realm_id: str, base_score: float) -> float:
        """Calculate priority-weighted score for realm-aware ranking"""
        if realm_id == self.project_realm:
            return base_score * self.project_priority_weight
        elif realm_id == self.global_realm:
            return base_score * self.global_priority_weight
        else:
            return base_score * 0.8  # Lower priority for other realms

class RealmConfigurationManager:
    """Manages realm configuration from environment variables"""
    
    def __init__(self):
        self.config = self._load_from_environment()
        self._validate_configuration()
    
    def _load_from_environment(self) -> RealmConfig:
        """Load realm configuration from environment variables"""
        project_realm = os.getenv('MEGAMIND_PROJECT_REALM', 'PROJ_DEFAULT')
        project_name = os.getenv('MEGAMIND_PROJECT_NAME', 'Default Project')
        default_target = os.getenv('MEGAMIND_DEFAULT_TARGET', 'PROJECT').upper()
        
        # Semantic search configuration
        cross_realm_enabled = os.getenv('CROSS_REALM_SEARCH_ENABLED', 'true').lower() == 'true'
        project_weight = float(os.getenv('REALM_PRIORITY_PROJECT', '1.2'))
        global_weight = float(os.getenv('REALM_PRIORITY_GLOBAL', '1.0'))
        
        return RealmConfig(
            project_realm=project_realm,
            project_name=project_name,
            default_target=default_target,
            cross_realm_search_enabled=cross_realm_enabled,
            project_priority_weight=project_weight,
            global_priority_weight=global_weight
        )
    
    def _validate_configuration(self):
        """Validate realm configuration"""
        if self.config.default_target not in ['PROJECT', 'GLOBAL']:
            logger.warning(f"Invalid MEGAMIND_DEFAULT_TARGET: {self.config.default_target}, defaulting to PROJECT")
            self.config.default_target = 'PROJECT'
        
        if not self.config.project_realm:
            raise ValueError("MEGAMIND_PROJECT_REALM must be configured")
        
        logger.info(f"Realm configuration loaded - Project: {self.config.project_realm}, Default target: {self.config.default_target}")
    
    def get_target_realm(self, explicit_target: Optional[str] = None) -> str:
        """Determine target realm for operations"""
        if explicit_target:
            if explicit_target.upper() == 'GLOBAL':
                return self.config.global_realm
            elif explicit_target.upper() == 'PROJECT':
                return self.config.project_realm
            else:
                logger.warning(f"Unknown target realm: {explicit_target}, using default")
        
        # Use configured default
        if self.config.default_target == 'GLOBAL':
            return self.config.global_realm
        else:
            return self.config.project_realm
    
    def get_search_realms(self) -> List[str]:
        """Get list of realms to search (dual-realm access)"""
        if self.config.cross_realm_search_enabled:
            return [self.config.global_realm, self.config.project_realm]
        else:
            return [self.config.project_realm]
    
    def calculate_realm_priority_score(self, realm_id: str, base_score: float) -> float:
        """Calculate priority-weighted score for realm-aware ranking"""
        if realm_id == self.config.project_realm:
            return base_score * self.config.project_priority_weight
        elif realm_id == self.config.global_realm:
            return base_score * self.config.global_priority_weight
        else:
            return base_score * 0.8  # Lower priority for other realms
    
    def is_cross_realm_operation(self, source_realm: str, target_realm: str) -> bool:
        """Check if operation crosses realm boundaries"""
        return source_realm != target_realm
    
    def get_realm_info(self) -> Dict[str, Any]:
        """Get comprehensive realm configuration information"""
        return {
            "project_realm_id": self.config.project_realm,
            "project_name": self.config.project_name,
            "global_realm_id": self.config.global_realm,
            "default_target": self.config.default_target,
            "cross_realm_search_enabled": self.config.cross_realm_search_enabled,
            "search_realms": self.get_search_realms(),
            "priority_weights": {
                "project": self.config.project_priority_weight,
                "global": self.config.global_priority_weight
            },
            "environment_variables": {
                "MEGAMIND_PROJECT_REALM": os.getenv('MEGAMIND_PROJECT_REALM'),
                "MEGAMIND_PROJECT_NAME": os.getenv('MEGAMIND_PROJECT_NAME'),
                "MEGAMIND_DEFAULT_TARGET": os.getenv('MEGAMIND_DEFAULT_TARGET'),
                "CROSS_REALM_SEARCH_ENABLED": os.getenv('CROSS_REALM_SEARCH_ENABLED'),
                "REALM_PRIORITY_PROJECT": os.getenv('REALM_PRIORITY_PROJECT'),
                "REALM_PRIORITY_GLOBAL": os.getenv('REALM_PRIORITY_GLOBAL')
            }
        }

class RealmAccessController:
    """Controls access patterns and permissions for realm operations"""
    
    def __init__(self, config_manager: RealmConfigurationManager):
        self.config_manager = config_manager
    
    def can_read_realm(self, realm_id: str) -> bool:
        """Check if current configuration can read from specified realm"""
        # Always can read from own project realm and global realm
        allowed_realms = [self.config_manager.config.project_realm, self.config_manager.config.global_realm]
        return realm_id in allowed_realms
    
    def can_write_realm(self, realm_id: str) -> bool:
        """Check if current configuration can write to specified realm"""
        # Can write to project realm, global realm requires explicit targeting
        if realm_id == self.config_manager.config.project_realm:
            return True
        elif realm_id == self.config_manager.config.global_realm:
            # Global realm write requires explicit targeting
            return self.config_manager.config.default_target == 'GLOBAL'
        else:
            return False
    
    def validate_realm_operation(self, operation: str, source_realm: str, target_realm: str = None) -> Tuple[bool, str]:
        """Validate realm operation permissions"""
        if operation in ['read', 'search']:
            if self.can_read_realm(source_realm):
                return True, "Read access granted"
            else:
                return False, f"Read access denied for realm {source_realm}"
        
        elif operation in ['create', 'update', 'delete']:
            if self.can_write_realm(source_realm):
                return True, "Write access granted"
            else:
                return False, f"Write access denied for realm {source_realm}"
        
        elif operation == 'promote':
            # Promotion typically from project to global
            if (self.can_read_realm(source_realm) and 
                target_realm == self.config_manager.config.global_realm):
                return True, "Promotion access granted"
            else:
                return False, f"Promotion access denied from {source_realm} to {target_realm}"
        
        else:
            return False, f"Unknown operation: {operation}"
    
    def get_effective_realms_for_search(self) -> List[Dict[str, Any]]:
        """Get realms that should be included in search operations"""
        search_realms = self.config_manager.get_search_realms()
        realm_info = []
        
        for realm_id in search_realms:
            priority_weight = self.config_manager.calculate_realm_priority_score(realm_id, 1.0)
            realm_info.append({
                "realm_id": realm_id,
                "priority_weight": priority_weight,
                "access_type": "direct" if realm_id == self.config_manager.config.project_realm else "inherited"
            })
        
        # Sort by priority weight (highest first)
        realm_info.sort(key=lambda x: x["priority_weight"], reverse=True)
        return realm_info

# Global configuration instance (singleton pattern)
_realm_config_manager = None
_realm_access_controller = None

def get_realm_config() -> RealmConfigurationManager:
    """Get global realm configuration manager instance"""
    global _realm_config_manager
    if _realm_config_manager is None:
        _realm_config_manager = RealmConfigurationManager()
    return _realm_config_manager

def get_realm_access_controller() -> RealmAccessController:
    """Get global realm access controller instance"""
    global _realm_access_controller
    if _realm_access_controller is None:
        _realm_access_controller = RealmAccessController(get_realm_config())
    return _realm_access_controller

def reset_realm_config():
    """Reset configuration (useful for testing)"""
    global _realm_config_manager, _realm_access_controller
    _realm_config_manager = None
    _realm_access_controller = None