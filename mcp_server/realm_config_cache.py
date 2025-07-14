#!/usr/bin/env python3
"""
Realm Configuration Cache
High-performance caching system for dynamic realm configurations with TTL, LRU eviction, and monitoring
"""

import json
import logging
import threading
import time
import hashlib
from typing import Dict, Any, Optional, Tuple, List, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import OrderedDict
from enum import Enum
import weakref

logger = logging.getLogger(__name__)

class CacheEventType(Enum):
    """Cache operation event types"""
    HIT = "hit"
    MISS = "miss"
    SET = "set"
    EVICT = "evict"
    EXPIRE = "expire"
    CLEAR = "clear"

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    ttl_seconds: Optional[int]
    tags: Set[str]
    size_bytes: int
    
    def is_expired(self) -> bool:
        """Check if entry has expired"""
        if self.ttl_seconds is None:
            return False
        return datetime.now() > self.created_at + timedelta(seconds=self.ttl_seconds)
    
    def touch(self):
        """Update last accessed time"""
        self.last_accessed = datetime.now()
        self.access_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        result['created_at'] = self.created_at.isoformat()
        result['last_accessed'] = self.last_accessed.isoformat()
        result['tags'] = list(self.tags)
        return result

@dataclass
class CacheStats:
    """Cache performance statistics"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    evictions: int = 0
    expirations: int = 0
    total_entries: int = 0
    total_size_bytes: int = 0
    average_access_time_ms: float = 0.0
    hit_rate: float = 0.0
    
    def update_hit_rate(self):
        """Update hit rate calculation"""
        total_requests = self.hits + self.misses
        self.hit_rate = (self.hits / total_requests) if total_requests > 0 else 0.0

class RealmConfigCache:
    """High-performance cache for realm configurations with advanced features"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Cache configuration
        self.max_entries = self.config.get('max_entries', 1000)
        self.default_ttl = self.config.get('default_ttl_seconds', 3600)  # 1 hour
        self.max_size_bytes = self.config.get('max_size_bytes', 50 * 1024 * 1024)  # 50MB
        self.cleanup_interval = self.config.get('cleanup_interval_seconds', 300)  # 5 minutes
        self.enable_stats = self.config.get('enable_stats', True)
        self.enable_monitoring = self.config.get('enable_monitoring', True)
        
        # Cache storage and metadata
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._size_bytes = 0
        self._lock = threading.RLock()
        self._stats = CacheStats()
        
        # Tag-based indexing for efficient invalidation
        self._tag_index: Dict[str, Set[str]] = {}
        
        # Background cleanup
        self._cleanup_timer: Optional[threading.Timer] = None
        self._shutdown_flag = threading.Event()
        
        # Performance monitoring
        self._access_times: List[float] = []
        self._max_access_time_samples = 1000
        
        # Start background cleanup
        self._start_cleanup_timer()
        
        logger.info(f"RealmConfigCache initialized: max_entries={self.max_entries}, "
                   f"default_ttl={self.default_ttl}s, max_size={self.max_size_bytes} bytes")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        start_time = time.perf_counter()
        
        try:
            with self._lock:
                entry = self._cache.get(key)
                
                if entry is None:
                    self._record_stats(CacheEventType.MISS)
                    return None
                
                # Check expiration
                if entry.is_expired():
                    self._remove_entry(key, CacheEventType.EXPIRE)
                    self._record_stats(CacheEventType.MISS)
                    return None
                
                # Update access metadata
                entry.touch()
                
                # Move to end (LRU)
                self._cache.move_to_end(key)
                
                self._record_stats(CacheEventType.HIT)
                return entry.value
                
        finally:
            # Record access time
            access_time = (time.perf_counter() - start_time) * 1000
            self._record_access_time(access_time)
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None, 
            tags: Optional[Set[str]] = None) -> bool:
        """Set value in cache"""
        try:
            ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
            tags = tags or set()
            
            # Calculate size
            value_size = self._calculate_size(value)
            
            with self._lock:
                # Check if we need to evict entries
                self._ensure_space(value_size)
                
                # Remove existing entry if present
                if key in self._cache:
                    self._remove_entry(key, CacheEventType.SET)
                
                # Create new entry
                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=datetime.now(),
                    last_accessed=datetime.now(),
                    access_count=1,
                    ttl_seconds=ttl,
                    tags=tags,
                    size_bytes=value_size
                )
                
                # Add to cache
                self._cache[key] = entry
                self._size_bytes += value_size
                
                # Update tag index
                for tag in tags:
                    if tag not in self._tag_index:
                        self._tag_index[tag] = set()
                    self._tag_index[tag].add(key)
                
                self._record_stats(CacheEventType.SET)
                return True
                
        except Exception as e:
            logger.error(f"Failed to set cache entry {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete entry from cache"""
        with self._lock:
            if key in self._cache:
                self._remove_entry(key, CacheEventType.EVICT)
                return True
            return False
    
    def invalidate_by_tag(self, tag: str) -> int:
        """Invalidate all entries with specified tag"""
        with self._lock:
            if tag not in self._tag_index:
                return 0
            
            keys_to_remove = list(self._tag_index[tag])
            for key in keys_to_remove:
                self._remove_entry(key, CacheEventType.EVICT)
            
            return len(keys_to_remove)
    
    def invalidate_by_pattern(self, pattern: str) -> int:
        """Invalidate entries with keys matching pattern"""
        import re
        
        with self._lock:
            regex = re.compile(pattern)
            keys_to_remove = [key for key in self._cache.keys() if regex.match(key)]
            
            for key in keys_to_remove:
                self._remove_entry(key, CacheEventType.EVICT)
            
            return len(keys_to_remove)
    
    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            self._tag_index.clear()
            self._size_bytes = 0
            self._record_stats(CacheEventType.CLEAR)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            stats = asdict(self._stats)
            stats.update({
                'current_entries': len(self._cache),
                'current_size_bytes': self._size_bytes,
                'size_utilization_percent': (self._size_bytes / self.max_size_bytes) * 100,
                'entry_utilization_percent': (len(self._cache) / self.max_entries) * 100,
                'average_entry_size_bytes': self._size_bytes / len(self._cache) if self._cache else 0,
                'tag_count': len(self._tag_index),
                'recent_access_times_ms': self._access_times[-10:] if self._access_times else []
            })
            
            # Update derived stats
            stats['hit_rate'] = self._stats.hit_rate
            stats['average_access_time_ms'] = sum(self._access_times) / len(self._access_times) if self._access_times else 0
            
            return stats
    
    def get_entry_info(self, key: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about specific cache entry"""
        with self._lock:
            entry = self._cache.get(key)
            if entry:
                return entry.to_dict()
            return None
    
    def get_top_entries(self, limit: int = 10, sort_by: str = 'access_count') -> List[Dict[str, Any]]:
        """Get top cache entries by specified metric"""
        with self._lock:
            entries = list(self._cache.values())
            
            if sort_by == 'access_count':
                entries.sort(key=lambda e: e.access_count, reverse=True)
            elif sort_by == 'size':
                entries.sort(key=lambda e: e.size_bytes, reverse=True)
            elif sort_by == 'age':
                entries.sort(key=lambda e: e.created_at)
            
            return [entry.to_dict() for entry in entries[:limit]]
    
    def _ensure_space(self, required_bytes: int):
        """Ensure sufficient space for new entry"""
        # Check size limit
        while (self._size_bytes + required_bytes > self.max_size_bytes or 
               len(self._cache) >= self.max_entries):
            
            if not self._cache:
                break
                
            # Remove least recently used entry
            oldest_key = next(iter(self._cache))
            self._remove_entry(oldest_key, CacheEventType.EVICT)
    
    def _remove_entry(self, key: str, event_type: CacheEventType):
        """Remove entry and update metadata"""
        if key not in self._cache:
            return
        
        entry = self._cache[key]
        
        # Update size
        self._size_bytes -= entry.size_bytes
        
        # Remove from tag index
        for tag in entry.tags:
            if tag in self._tag_index:
                self._tag_index[tag].discard(key)
                if not self._tag_index[tag]:
                    del self._tag_index[tag]
        
        # Remove from cache
        del self._cache[key]
        
        # Record stats
        if event_type == CacheEventType.EVICT:
            self._stats.evictions += 1
        elif event_type == CacheEventType.EXPIRE:
            self._stats.expirations += 1
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value in bytes"""
        try:
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (dict, list)):
                return len(json.dumps(value).encode('utf-8'))
            elif isinstance(value, (int, float)):
                return 8  # Approximate
            elif isinstance(value, bool):
                return 1
            else:
                return len(str(value).encode('utf-8'))
        except Exception:
            return 1024  # Default fallback
    
    def _record_stats(self, event_type: CacheEventType):
        """Record cache operation statistics"""
        if not self.enable_stats:
            return
        
        if event_type == CacheEventType.HIT:
            self._stats.hits += 1
        elif event_type == CacheEventType.MISS:
            self._stats.misses += 1
        elif event_type == CacheEventType.SET:
            self._stats.sets += 1
        
        # Update derived stats
        self._stats.total_entries = len(self._cache)
        self._stats.total_size_bytes = self._size_bytes
        self._stats.update_hit_rate()
    
    def _record_access_time(self, access_time_ms: float):
        """Record access time for performance monitoring"""
        if not self.enable_monitoring:
            return
        
        self._access_times.append(access_time_ms)
        
        # Keep only recent samples
        if len(self._access_times) > self._max_access_time_samples:
            self._access_times = self._access_times[-self._max_access_time_samples:]
    
    def _cleanup_expired_entries(self):
        """Remove expired entries from cache"""
        try:
            with self._lock:
                current_time = datetime.now()
                expired_keys = []
                
                for key, entry in self._cache.items():
                    if entry.is_expired():
                        expired_keys.append(key)
                
                for key in expired_keys:
                    self._remove_entry(key, CacheEventType.EXPIRE)
                
                if expired_keys:
                    logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                    
        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")
    
    def _start_cleanup_timer(self):
        """Start background cleanup timer"""
        if self._shutdown_flag.is_set():
            return
        
        self._cleanup_expired_entries()
        
        # Schedule next cleanup
        self._cleanup_timer = threading.Timer(
            self.cleanup_interval, 
            self._start_cleanup_timer
        )
        self._cleanup_timer.daemon = True
        self._cleanup_timer.start()
    
    def shutdown(self):
        """Shutdown cache and cleanup resources"""
        logger.info("Shutting down RealmConfigCache")
        
        self._shutdown_flag.set()
        
        if self._cleanup_timer:
            self._cleanup_timer.cancel()
        
        with self._lock:
            self.clear()

class RealmConfigurationManager:
    """High-level manager for realm configuration caching"""
    
    def __init__(self, cache_config: Optional[Dict[str, Any]] = None):
        self.cache = RealmConfigCache(cache_config)
        self._realm_config_ttl = cache_config.get('realm_config_ttl', 1800) if cache_config else 1800  # 30 minutes
        self._validation_cache_ttl = cache_config.get('validation_cache_ttl', 300) if cache_config else 300  # 5 minutes
        
        logger.info("RealmConfigurationManager initialized with caching support")
    
    def get_realm_config(self, realm_id: str, config_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached realm configuration"""
        cache_key = f"realm_config:{realm_id}:{config_hash}"
        return self.cache.get(cache_key)
    
    def set_realm_config(self, realm_id: str, config_hash: str, config: Dict[str, Any]) -> bool:
        """Cache realm configuration"""
        cache_key = f"realm_config:{realm_id}:{config_hash}"
        tags = {f"realm:{realm_id}", "realm_config"}
        return self.cache.set(cache_key, config, self._realm_config_ttl, tags)
    
    def get_validation_result(self, config_hash: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached validation result"""
        cache_key = f"validation:{config_hash}"
        return self.cache.get(cache_key)
    
    def set_validation_result(self, config_hash: str, validation_results: List[Dict[str, Any]]) -> bool:
        """Cache validation result"""
        cache_key = f"validation:{config_hash}"
        tags = {"validation_result"}
        return self.cache.set(cache_key, validation_results, self._validation_cache_ttl, tags)
    
    def invalidate_realm(self, realm_id: str) -> int:
        """Invalidate all cache entries for a specific realm"""
        return self.cache.invalidate_by_tag(f"realm:{realm_id}")
    
    def invalidate_all_validations(self) -> int:
        """Invalidate all validation results"""
        return self.cache.invalidate_by_tag("validation_result")
    
    def get_cache_metrics(self) -> Dict[str, Any]:
        """Get comprehensive cache metrics"""
        stats = self.cache.get_stats()
        
        # Add manager-specific metrics
        stats['realm_configs_cached'] = len([k for k in self.cache._cache.keys() if k.startswith('realm_config:')])
        stats['validations_cached'] = len([k for k in self.cache._cache.keys() if k.startswith('validation:')])
        
        return stats
    
    def warm_cache(self, common_realm_configs: List[Tuple[str, Dict[str, Any]]]):
        """Pre-populate cache with common realm configurations"""
        for realm_id, config in common_realm_configs:
            config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
            self.set_realm_config(realm_id, config_hash, config)
        
        logger.info(f"Cache warmed with {len(common_realm_configs)} common configurations")
    
    def shutdown(self):
        """Shutdown configuration manager"""
        self.cache.shutdown()