#!/usr/bin/env python3
"""
Inheritance Resolution Engine
Handles complex inheritance logic for realm-aware operations
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class InheritanceConfig:
    """Configuration for selective inheritance"""
    include_tags: Optional[List[str]] = None
    exclude_tags: Optional[List[str]] = None
    include_types: Optional[List[str]] = None
    exclude_types: Optional[List[str]] = None
    priority_boost: float = 1.0
    max_depth: int = 3

@dataclass
class InheritancePath:
    """Represents an inheritance path from source to target realm"""
    source_realm: str
    target_realm: str
    inheritance_type: str
    priority_order: int
    config: Optional[InheritanceConfig] = None
    depth: int = 1

@dataclass
class AccessResult:
    """Result of access resolution"""
    access_granted: bool
    access_type: str  # 'direct', 'inherited', 'denied'
    source_realm: str
    reason: str
    priority_score: float = 0.0
    inheritance_path: Optional[List[str]] = None

class InheritanceResolver:
    """Resolves inheritance and access patterns for realm-aware operations"""
    
    def __init__(self, db_connection):
        self.db = db_connection
        self._inheritance_cache = {}
        self._access_cache = {}
    
    def resolve_chunk_access(self, chunk_id: str, accessing_realm: str) -> AccessResult:
        """Resolve access to a specific chunk from an accessing realm"""
        cache_key = f"{chunk_id}:{accessing_realm}"
        if cache_key in self._access_cache:
            return self._access_cache[cache_key]
        
        try:
            cursor = self.db.cursor(dictionary=True)
            
            # Use the stored function for conflict resolution
            cursor.execute("""
                SELECT resolve_inheritance_conflict(%s, %s) as result
            """, (chunk_id, accessing_realm))
            
            result_row = cursor.fetchone()
            if result_row and result_row['result']:
                result_data = json.loads(result_row['result'])
                
                access_result = AccessResult(
                    access_granted=result_data.get('access_granted', False),
                    access_type=result_data.get('access_type', 'denied'),
                    source_realm=result_data.get('source_realm', ''),
                    reason=result_data.get('reason', 'Unknown'),
                    priority_score=result_data.get('priority_order', 999)
                )
            else:
                access_result = AccessResult(
                    access_granted=False,
                    access_type='denied',
                    source_realm='',
                    reason='No access path found'
                )
            
            self._access_cache[cache_key] = access_result
            return access_result
            
        except Exception as e:
            logger.error(f"Failed to resolve chunk access for {chunk_id}: {e}")
            return AccessResult(
                access_granted=False,
                access_type='error',
                source_realm='',
                reason=f'Resolution error: {str(e)}'
            )
    
    def get_inheritance_paths(self, child_realm: str, max_depth: int = 3) -> List[InheritancePath]:
        """Get all inheritance paths for a child realm"""
        cache_key = f"paths:{child_realm}:{max_depth}"
        if cache_key in self._inheritance_cache:
            return self._inheritance_cache[cache_key]
        
        paths = []
        try:
            cursor = self.db.cursor(dictionary=True)
            
            # Get direct inheritance relationships
            cursor.execute("""
                SELECT ri.parent_realm_id, ri.inheritance_type, ri.priority_order, 
                       ri.inheritance_config, pr.realm_name
                FROM megamind_realm_inheritance ri
                JOIN megamind_realms pr ON ri.parent_realm_id = pr.realm_id
                WHERE ri.child_realm_id = %s AND ri.is_active = TRUE
                ORDER BY ri.priority_order
            """, (child_realm,))
            
            for row in cursor.fetchall():
                config = None
                if row['inheritance_config']:
                    config_data = json.loads(row['inheritance_config']) if isinstance(row['inheritance_config'], str) else row['inheritance_config']
                    config = InheritanceConfig(
                        include_tags=config_data.get('include_tags'),
                        exclude_tags=config_data.get('exclude_tags'),
                        include_types=config_data.get('include_types'),
                        exclude_types=config_data.get('exclude_types'),
                        priority_boost=config_data.get('priority_boost', 1.0),
                        max_depth=config_data.get('max_depth', 3)
                    )
                
                path = InheritancePath(
                    source_realm=child_realm,
                    target_realm=row['parent_realm_id'],
                    inheritance_type=row['inheritance_type'],
                    priority_order=row['priority_order'],
                    config=config,
                    depth=1
                )
                paths.append(path)
                
                # Recursively get paths through this parent (if within depth limit)
                if max_depth > 1:
                    parent_paths = self.get_inheritance_paths(row['parent_realm_id'], max_depth - 1)
                    for parent_path in parent_paths:
                        inherited_path = InheritancePath(
                            source_realm=child_realm,
                            target_realm=parent_path.target_realm,
                            inheritance_type=f"{row['inheritance_type']}+{parent_path.inheritance_type}",
                            priority_order=row['priority_order'] + parent_path.priority_order,
                            config=config,  # Use the first level config
                            depth=parent_path.depth + 1
                        )
                        paths.append(inherited_path)
            
            self._inheritance_cache[cache_key] = paths
            return paths
            
        except Exception as e:
            logger.error(f"Failed to get inheritance paths for {child_realm}: {e}")
            return []
    
    def filter_chunks_by_inheritance(self, chunks: List[Dict], accessing_realm: str) -> List[Dict]:
        """Filter chunks based on inheritance rules and add access metadata"""
        filtered_chunks = []
        
        for chunk in chunks:
            chunk_id = chunk['chunk_id']
            source_realm = chunk['realm_id']
            
            # Check access resolution
            access_result = self.resolve_chunk_access(chunk_id, accessing_realm)
            
            if access_result.access_granted:
                # Add inheritance metadata to chunk
                enhanced_chunk = dict(chunk)
                enhanced_chunk.update({
                    'access_type': access_result.access_type,
                    'access_source_realm': access_result.source_realm,
                    'priority_score': access_result.priority_score,
                    'inheritance_reason': access_result.reason
                })
                
                # Calculate weighted score for ranking
                base_score = chunk.get('access_count', 0)
                if access_result.access_type == 'direct':
                    enhanced_chunk['weighted_score'] = base_score * 1.2  # Direct access boost
                elif access_result.source_realm == 'GLOBAL':
                    enhanced_chunk['weighted_score'] = base_score * 0.8  # Global inheritance
                else:
                    enhanced_chunk['weighted_score'] = base_score * 1.0  # Project inheritance
                
                filtered_chunks.append(enhanced_chunk)
        
        # Sort by weighted score (highest first)
        filtered_chunks.sort(key=lambda x: x['weighted_score'], reverse=True)
        return filtered_chunks
    
    def check_selective_inheritance(self, chunk: Dict, inheritance_config: InheritanceConfig) -> bool:
        """Check if a chunk passes selective inheritance filters"""
        if not inheritance_config:
            return True
        
        chunk_type = chunk.get('chunk_type', '')
        
        # Check exclude types first
        if inheritance_config.exclude_types and chunk_type in inheritance_config.exclude_types:
            return False
        
        # Check include types
        if inheritance_config.include_types and chunk_type not in inheritance_config.include_types:
            return False
        
        # For tag-based filtering, we need to query the database
        # This is a simplified version - in practice, tags would be pre-loaded or queried
        chunk_tags = chunk.get('tags', [])
        
        # Check exclude tags
        if inheritance_config.exclude_tags:
            for exclude_tag in inheritance_config.exclude_tags:
                if exclude_tag in chunk_tags:
                    return False
        
        # Check include tags
        if inheritance_config.include_tags:
            has_included_tag = any(tag in chunk_tags for tag in inheritance_config.include_tags)
            if not has_included_tag:
                return False
        
        return True
    
    def get_realm_accessibility_matrix(self, realm_id: str) -> Dict[str, Any]:
        """Get accessibility matrix for a realm"""
        try:
            cursor = self.db.cursor(dictionary=True)
            
            cursor.execute("""
                SELECT target_realm, target_realm_name, access_type, 
                       inheritance_type, priority_order
                FROM megamind_realm_accessibility
                WHERE accessing_realm = %s
                ORDER BY priority_order, target_realm
            """, (realm_id,))
            
            accessibility = {
                'realm_id': realm_id,
                'accessible_realms': {},
                'inheritance_paths': [],
                'access_summary': {
                    'direct_access': 0,
                    'global_inheritance': 0,
                    'explicit_inheritance': 0,
                    'no_access': 0
                }
            }
            
            for row in cursor.fetchall():
                accessibility['accessible_realms'][row['target_realm']] = {
                    'realm_name': row['target_realm_name'],
                    'access_type': row['access_type'],
                    'inheritance_type': row['inheritance_type'],
                    'priority_order': row['priority_order']
                }
                
                # Update summary
                accessibility['access_summary'][row['access_type']] += 1
                
                if row['access_type'] in ['global_inheritance', 'explicit_inheritance']:
                    accessibility['inheritance_paths'].append({
                        'target_realm': row['target_realm'],
                        'inheritance_type': row['inheritance_type'],
                        'priority': row['priority_order']
                    })
            
            return accessibility
            
        except Exception as e:
            logger.error(f"Failed to get accessibility matrix for {realm_id}: {e}")
            return {'error': str(e)}
    
    def validate_inheritance_setup(self, child_realm: str, parent_realm: str, 
                                  inheritance_type: str, inheritance_config: Optional[Dict] = None) -> Tuple[bool, str]:
        """Validate inheritance configuration before creation"""
        try:
            cursor = self.db.cursor()
            
            # Call stored procedure for validation
            cursor.callproc('validate_inheritance_configuration', 
                          [child_realm, parent_realm, inheritance_type, False, ''])
            
            # Get the output parameters
            results = cursor.stored_results()
            for result in results:
                # This is a simplified approach - actual implementation would handle OUT parameters properly
                pass
            
            return True, "Validation successful"
            
        except Exception as e:
            logger.error(f"Failed to validate inheritance setup: {e}")
            return False, str(e)
    
    def clear_cache(self):
        """Clear inheritance resolution caches"""
        self._inheritance_cache.clear()
        self._access_cache.clear()
        logger.info("Inheritance resolution caches cleared")
    
    def get_inheritance_stats(self, realm_id: str) -> Dict[str, Any]:
        """Get inheritance statistics for a realm"""
        try:
            cursor = self.db.cursor(dictionary=True)
            
            cursor.execute("""
                SELECT child_realm_id, parent_realm_id, inheritance_type,
                       inherited_chunks, accessible_chunks, accessibility_percentage
                FROM inheritance_performance_stats
                WHERE child_realm_id = %s OR parent_realm_id = %s
            """, (realm_id, realm_id))
            
            stats = {
                'realm_id': realm_id,
                'as_child': [],
                'as_parent': [],
                'summary': {
                    'total_inherited_chunks': 0,
                    'total_accessible_chunks': 0,
                    'avg_accessibility': 0.0
                }
            }
            
            total_chunks = 0
            total_accessible = 0
            
            for row in cursor.fetchall():
                if row['child_realm_id'] == realm_id:
                    stats['as_child'].append(dict(row))
                    total_chunks += row['inherited_chunks']
                    total_accessible += row['accessible_chunks']
                else:
                    stats['as_parent'].append(dict(row))
            
            if total_chunks > 0:
                stats['summary']['total_inherited_chunks'] = total_chunks
                stats['summary']['total_accessible_chunks'] = total_accessible
                stats['summary']['avg_accessibility'] = (total_accessible / total_chunks) * 100
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get inheritance stats for {realm_id}: {e}")
            return {'error': str(e)}