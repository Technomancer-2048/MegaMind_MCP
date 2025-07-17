"""
Gradual Removal Scheduler for MegaMind MCP Functions
Phase 3: Function Consolidation Cleanup Plan

This module implements the gradual removal system for deprecated functions
with progressive warning levels and automated removal scheduling.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)

class RemovalPhase(Enum):
    """Phases of the gradual removal process"""
    ACTIVE = "active"              # Functions work normally with warnings
    WARNING = "warning"            # Increased warning frequency
    FINAL_WARNING = "final_warning" # Final warning before removal
    REMOVED = "removed"            # Functions removed

class WarningLevel(Enum):
    """Warning levels for deprecated functions"""
    STANDARD = "standard"          # Normal deprecation warning
    ELEVATED = "elevated"          # More frequent warnings
    CRITICAL = "critical"          # Critical warnings before removal
    FINAL = "final"               # Final warnings

class RemovalScheduler:
    """
    Manages the gradual removal schedule for deprecated functions
    with progressive warning levels and automated timeline management.
    """
    
    def __init__(self, config_file: str = "removal_schedule.json"):
        self.config_file = config_file
        self.schedule = self._load_schedule()
        self.start_date = datetime.now()
        
    def _load_schedule(self) -> Dict[str, Any]:
        """Load removal schedule from configuration file"""
        default_schedule = {
            "removal_timeline": {
                "phase_1_duration": 14,  # 2 weeks: Active with standard warnings
                "phase_2_duration": 14,  # 2 weeks: Elevated warnings
                "phase_3_duration": 14,  # 2 weeks: Critical warnings
                "phase_4_duration": 0    # Removal
            },
            "warning_frequencies": {
                "standard": 1,     # Every usage
                "elevated": 1,     # Every usage
                "critical": 1,     # Every usage
                "final": 1         # Every usage
            },
            "functions": {},
            "deployment_date": None
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    loaded_schedule = json.load(f)
                    # Merge with defaults
                    default_schedule.update(loaded_schedule)
                    return default_schedule
            except Exception as e:
                logger.error(f"Error loading schedule: {e}")
                return default_schedule
        
        return default_schedule
    
    def _save_schedule(self):
        """Save current schedule to configuration file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.schedule, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving schedule: {e}")
    
    def initialize_removal_schedule(self, deployment_date: Optional[datetime] = None):
        """Initialize the removal schedule for all deprecated functions"""
        if deployment_date is None:
            deployment_date = datetime.now()
        
        self.schedule["deployment_date"] = deployment_date.isoformat()
        
        # Define the 20 deprecated functions
        deprecated_functions = [
            "search_chunks", "get_chunk", "get_related_chunks",
            "search_chunks_semantic", "search_chunks_by_similarity",
            "create_chunk", "update_chunk", "add_relationship", "batch_generate_embeddings",
            "create_promotion_request", "get_promotion_requests", "approve_promotion_request",
            "reject_promotion_request", "get_promotion_impact", "get_promotion_queue_summary",
            "get_session_primer", "get_pending_changes", "commit_session_changes",
            "track_access", "get_hot_contexts"
        ]
        
        timeline = self.schedule["removal_timeline"]
        
        for func_name in deprecated_functions:
            # Calculate phase dates
            phase_1_end = deployment_date + timedelta(days=timeline["phase_1_duration"])
            phase_2_end = phase_1_end + timedelta(days=timeline["phase_2_duration"])
            phase_3_end = phase_2_end + timedelta(days=timeline["phase_3_duration"])
            removal_date = phase_3_end + timedelta(days=timeline["phase_4_duration"])
            
            self.schedule["functions"][func_name] = {
                "deployment_date": deployment_date.isoformat(),
                "phase_1_end": phase_1_end.isoformat(),
                "phase_2_end": phase_2_end.isoformat(),
                "phase_3_end": phase_3_end.isoformat(),
                "removal_date": removal_date.isoformat(),
                "current_phase": RemovalPhase.ACTIVE.value,
                "usage_count": 0,
                "last_warning": None,
                "warnings_sent": 0
            }
        
        self._save_schedule()
        logger.info(f"Removal schedule initialized for {len(deprecated_functions)} functions")
    
    def get_current_phase(self, function_name: str) -> RemovalPhase:
        """Get current removal phase for a function"""
        if function_name not in self.schedule["functions"]:
            return RemovalPhase.ACTIVE
        
        func_config = self.schedule["functions"][function_name]
        now = datetime.now()
        
        phase_1_end = datetime.fromisoformat(func_config["phase_1_end"])
        phase_2_end = datetime.fromisoformat(func_config["phase_2_end"])
        phase_3_end = datetime.fromisoformat(func_config["phase_3_end"])
        removal_date = datetime.fromisoformat(func_config["removal_date"])
        
        if now < phase_1_end:
            return RemovalPhase.ACTIVE
        elif now < phase_2_end:
            return RemovalPhase.WARNING
        elif now < phase_3_end:
            return RemovalPhase.FINAL_WARNING
        else:
            return RemovalPhase.REMOVED
    
    def get_warning_level(self, function_name: str) -> WarningLevel:
        """Get appropriate warning level for a function"""
        phase = self.get_current_phase(function_name)
        
        if phase == RemovalPhase.ACTIVE:
            return WarningLevel.STANDARD
        elif phase == RemovalPhase.WARNING:
            return WarningLevel.ELEVATED
        elif phase == RemovalPhase.FINAL_WARNING:
            return WarningLevel.CRITICAL
        else:
            return WarningLevel.FINAL
    
    def should_show_warning(self, function_name: str) -> bool:
        """Determine if a warning should be shown for this function usage"""
        if function_name not in self.schedule["functions"]:
            return True  # Show warning for untracked functions
        
        func_config = self.schedule["functions"][function_name]
        warning_level = self.get_warning_level(function_name)
        frequency = self.schedule["warning_frequencies"][warning_level.value]
        
        # For now, show warning on every usage (frequency = 1)
        # In production, this could be adjusted based on usage patterns
        return True
    
    def record_usage(self, function_name: str):
        """Record usage of a deprecated function"""
        if function_name not in self.schedule["functions"]:
            return
        
        func_config = self.schedule["functions"][function_name]
        func_config["usage_count"] += 1
        func_config["last_warning"] = datetime.now().isoformat()
        func_config["warnings_sent"] += 1
        
        self._save_schedule()
    
    def get_removal_timeline(self, function_name: str) -> Dict[str, Any]:
        """Get detailed removal timeline for a function"""
        if function_name not in self.schedule["functions"]:
            return {"error": "Function not found in schedule"}
        
        func_config = self.schedule["functions"][function_name]
        current_phase = self.get_current_phase(function_name)
        
        removal_date = datetime.fromisoformat(func_config["removal_date"])
        days_until_removal = (removal_date - datetime.now()).days
        
        return {
            "function_name": function_name,
            "current_phase": current_phase.value,
            "warning_level": self.get_warning_level(function_name).value,
            "days_until_removal": max(0, days_until_removal),
            "removal_date": func_config["removal_date"],
            "usage_count": func_config["usage_count"],
            "warnings_sent": func_config["warnings_sent"],
            "timeline": {
                "phase_1_end": func_config["phase_1_end"],
                "phase_2_end": func_config["phase_2_end"],
                "phase_3_end": func_config["phase_3_end"],
                "removal_date": func_config["removal_date"]
            }
        }
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get comprehensive usage statistics for all deprecated functions"""
        stats = {
            "total_functions": len(self.schedule["functions"]),
            "deployment_date": self.schedule.get("deployment_date"),
            "phases": {
                "active": 0,
                "warning": 0,
                "final_warning": 0,
                "removed": 0
            },
            "total_usage": 0,
            "total_warnings": 0,
            "functions": {}
        }
        
        for func_name, func_config in self.schedule["functions"].items():
            current_phase = self.get_current_phase(func_name)
            stats["phases"][current_phase.value] += 1
            stats["total_usage"] += func_config["usage_count"]
            stats["total_warnings"] += func_config["warnings_sent"]
            
            stats["functions"][func_name] = {
                "phase": current_phase.value,
                "usage_count": func_config["usage_count"],
                "warnings_sent": func_config["warnings_sent"],
                "last_warning": func_config["last_warning"]
            }
        
        return stats
    
    def get_functions_by_phase(self, phase: RemovalPhase) -> List[str]:
        """Get list of functions in a specific removal phase"""
        functions = []
        for func_name in self.schedule["functions"]:
            if self.get_current_phase(func_name) == phase:
                functions.append(func_name)
        return functions
    
    def is_function_removed(self, function_name: str) -> bool:
        """Check if a function has been removed"""
        return self.get_current_phase(function_name) == RemovalPhase.REMOVED
    
    def get_removal_alert_message(self, function_name: str) -> str:
        """Get appropriate alert message for a function"""
        phase = self.get_current_phase(function_name)
        timeline = self.get_removal_timeline(function_name)
        days_until_removal = timeline["days_until_removal"]
        
        if phase == RemovalPhase.ACTIVE:
            return f"Function '{function_name}' is deprecated and will be removed in {days_until_removal} days. Please migrate to the consolidated equivalent."
        elif phase == RemovalPhase.WARNING:
            return f"⚠️ ELEVATED WARNING: Function '{function_name}' will be removed in {days_until_removal} days. Migration required soon."
        elif phase == RemovalPhase.FINAL_WARNING:
            return f"🚨 FINAL WARNING: Function '{function_name}' will be removed in {days_until_removal} days. Immediate migration required!"
        else:
            return f"❌ Function '{function_name}' has been removed. Please use the consolidated equivalent."
    
    def force_removal(self, function_name: str):
        """Force immediate removal of a function (for testing/admin purposes)"""
        if function_name in self.schedule["functions"]:
            # Update all phase dates to reflect immediate removal
            now = datetime.now()
            func_config = self.schedule["functions"][function_name]
            func_config["current_phase"] = RemovalPhase.REMOVED.value
            func_config["removal_date"] = now.isoformat()
            func_config["phase_1_end"] = (now - timedelta(days=30)).isoformat()
            func_config["phase_2_end"] = (now - timedelta(days=15)).isoformat()
            func_config["phase_3_end"] = (now - timedelta(days=1)).isoformat()
            self._save_schedule()
            logger.info(f"Function '{function_name}' forcibly removed")
    
    def extend_removal_timeline(self, function_name: str, additional_days: int):
        """Extend the removal timeline for a function"""
        if function_name not in self.schedule["functions"]:
            return
        
        func_config = self.schedule["functions"][function_name]
        current_removal = datetime.fromisoformat(func_config["removal_date"])
        new_removal = current_removal + timedelta(days=additional_days)
        
        func_config["removal_date"] = new_removal.isoformat()
        func_config["phase_3_end"] = (new_removal - timedelta(days=self.schedule["removal_timeline"]["phase_4_duration"])).isoformat()
        
        self._save_schedule()
        logger.info(f"Extended removal timeline for '{function_name}' by {additional_days} days")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data for monitoring"""
        stats = self.get_usage_statistics()
        
        # Add timeline information
        timeline_data = []
        for func_name in self.schedule["functions"]:
            timeline_data.append(self.get_removal_timeline(func_name))
        
        # Sort by days until removal
        timeline_data.sort(key=lambda x: x["days_until_removal"])
        
        return {
            "overview": stats,
            "timeline": timeline_data,
            "alerts": {
                "functions_in_final_warning": self.get_functions_by_phase(RemovalPhase.FINAL_WARNING),
                "functions_removed": self.get_functions_by_phase(RemovalPhase.REMOVED),
                "high_usage_functions": [
                    func for func, data in stats["functions"].items() 
                    if data["usage_count"] > 100
                ]
            }
        }