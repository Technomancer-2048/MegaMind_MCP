"""
Usage Statistics Dashboard for MegaMind MCP Functions
Phase 3: Function Consolidation Cleanup Plan

This module provides a comprehensive dashboard for monitoring deprecated
function usage and removal progress.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class FunctionStats:
    """Statistics for a single deprecated function"""
    name: str
    usage_count: int
    warnings_sent: int
    last_warning: Optional[datetime]
    current_phase: str
    days_until_removal: int
    removal_date: datetime

class UsageDashboard:
    """
    Dashboard for monitoring deprecated function usage and removal progress
    """
    
    def __init__(self, removal_scheduler, deprecation_router):
        self.removal_scheduler = removal_scheduler
        self.deprecation_router = deprecation_router
    
    def get_overview_stats(self) -> Dict[str, Any]:
        """Get high-level overview statistics"""
        scheduler_stats = self.removal_scheduler.get_usage_statistics()
        router_stats = self.deprecation_router.get_usage_stats()
        
        return {
            "total_deprecated_functions": 20,
            "functions_tracked": len(scheduler_stats["functions"]),
            "total_usage_events": scheduler_stats["total_usage"],
            "total_warnings_sent": scheduler_stats["total_warnings"],
            "deployment_date": scheduler_stats["deployment_date"],
            "phase_distribution": scheduler_stats["phases"],
            "router_usage": {
                "functions_used": len(router_stats),
                "total_calls": sum(stat["count"] for stat in router_stats.values())
            }
        }
    
    def get_function_details(self, function_name: str) -> Dict[str, Any]:
        """Get detailed information for a specific function"""
        timeline = self.removal_scheduler.get_removal_timeline(function_name)
        router_stats = self.deprecation_router.get_usage_stats().get(function_name, {})
        
        return {
            "function_name": function_name,
            "timeline": timeline,
            "router_usage": router_stats,
            "alert_message": self.removal_scheduler.get_removal_alert_message(function_name),
            "is_removed": self.removal_scheduler.is_function_removed(function_name)
        }
    
    def get_critical_alerts(self) -> List[Dict[str, Any]]:
        """Get functions requiring immediate attention"""
        alerts = []
        
        # Functions in final warning phase
        from .removal_scheduler import RemovalPhase
        final_warning_functions = self.removal_scheduler.get_functions_by_phase(
            RemovalPhase.FINAL_WARNING
        )
        
        for func_name in final_warning_functions:
            timeline = self.removal_scheduler.get_removal_timeline(func_name)
            alerts.append({
                "type": "final_warning",
                "function": func_name,
                "days_until_removal": timeline["days_until_removal"],
                "usage_count": timeline["usage_count"],
                "message": f"🚨 {func_name} will be removed in {timeline['days_until_removal']} days"
            })
        
        # High usage functions still being used
        router_stats = self.deprecation_router.get_usage_stats()
        for func_name, stats in router_stats.items():
            if stats["count"] > 50:  # High usage threshold
                timeline = self.removal_scheduler.get_removal_timeline(func_name)
                alerts.append({
                    "type": "high_usage",
                    "function": func_name,
                    "usage_count": stats["count"],
                    "days_until_removal": timeline["days_until_removal"],
                    "message": f"⚠️ {func_name} has high usage ({stats['count']} calls) and needs migration"
                })
        
        # Recently removed functions still being called
        removed_functions = self.removal_scheduler.get_functions_by_phase(
            RemovalPhase.REMOVED
        )
        
        for func_name in removed_functions:
            if func_name in router_stats and router_stats[func_name]["count"] > 0:
                alerts.append({
                    "type": "removed_but_used",
                    "function": func_name,
                    "usage_count": router_stats[func_name]["count"],
                    "message": f"❌ {func_name} was removed but still being called"
                })
        
        return sorted(alerts, key=lambda x: x.get("days_until_removal", 0))
    
    def get_migration_progress(self) -> Dict[str, Any]:
        """Get migration progress analytics"""
        scheduler_stats = self.removal_scheduler.get_usage_statistics()
        router_stats = self.deprecation_router.get_usage_stats()
        
        # Calculate migration progress
        total_functions = 20
        functions_with_zero_usage = sum(1 for stats in router_stats.values() if stats["count"] == 0)
        migration_percentage = (functions_with_zero_usage / total_functions) * 100
        
        # Functions by migration status
        migration_status = {
            "fully_migrated": [],      # 0 usage
            "low_usage": [],           # 1-10 usage
            "medium_usage": [],        # 11-50 usage
            "high_usage": []           # 51+ usage
        }
        
        for func_name in [
            "search_chunks", "get_chunk", "get_related_chunks",
            "search_chunks_semantic", "search_chunks_by_similarity",
            "create_chunk", "update_chunk", "add_relationship", "batch_generate_embeddings",
            "create_promotion_request", "get_promotion_requests", "approve_promotion_request",
            "reject_promotion_request", "get_promotion_impact", "get_promotion_queue_summary",
            "get_session_primer", "get_pending_changes", "commit_session_changes",
            "track_access", "get_hot_contexts"
        ]:
            usage_count = router_stats.get(func_name, {}).get("count", 0)
            
            if usage_count == 0:
                migration_status["fully_migrated"].append(func_name)
            elif usage_count <= 10:
                migration_status["low_usage"].append(func_name)
            elif usage_count <= 50:
                migration_status["medium_usage"].append(func_name)
            else:
                migration_status["high_usage"].append(func_name)
        
        return {
            "migration_percentage": round(migration_percentage, 1),
            "functions_migrated": len(migration_status["fully_migrated"]),
            "functions_remaining": total_functions - len(migration_status["fully_migrated"]),
            "migration_status": migration_status,
            "priority_migrations": migration_status["high_usage"] + migration_status["medium_usage"]
        }
    
    def get_timeline_view(self) -> List[Dict[str, Any]]:
        """Get timeline view of all functions"""
        timeline_data = []
        
        for func_name in [
            "search_chunks", "get_chunk", "get_related_chunks",
            "search_chunks_semantic", "search_chunks_by_similarity",
            "create_chunk", "update_chunk", "add_relationship", "batch_generate_embeddings",
            "create_promotion_request", "get_promotion_requests", "approve_promotion_request",
            "reject_promotion_request", "get_promotion_impact", "get_promotion_queue_summary",
            "get_session_primer", "get_pending_changes", "commit_session_changes",
            "track_access", "get_hot_contexts"
        ]:
            timeline = self.removal_scheduler.get_removal_timeline(func_name)
            router_stats = self.deprecation_router.get_usage_stats().get(func_name, {})
            
            timeline_data.append({
                "function": func_name,
                "current_phase": timeline["current_phase"],
                "days_until_removal": timeline["days_until_removal"],
                "usage_count": router_stats.get("count", 0),
                "warnings_sent": timeline["warnings_sent"],
                "removal_date": timeline["removal_date"],
                "status_indicator": self._get_status_indicator(timeline, router_stats)
            })
        
        return sorted(timeline_data, key=lambda x: x["days_until_removal"])
    
    def _get_status_indicator(self, timeline: Dict, router_stats: Dict) -> str:
        """Get visual status indicator for a function"""
        usage_count = router_stats.get("count", 0)
        days_until_removal = timeline["days_until_removal"]
        phase = timeline["current_phase"]
        
        if phase == "removed":
            return "🔴 REMOVED" if usage_count > 0 else "✅ REMOVED"
        elif phase == "final_warning":
            return "🚨 FINAL WARNING"
        elif phase == "warning":
            return "⚠️ WARNING"
        elif usage_count == 0:
            return "✅ READY FOR REMOVAL"
        elif usage_count > 50:
            return "🔥 HIGH USAGE"
        else:
            return "📊 ACTIVE"
    
    def get_usage_trends(self) -> Dict[str, Any]:
        """Get usage trends over time"""
        router_stats = self.deprecation_router.get_usage_stats()
        
        # Group functions by usage level
        usage_groups = {
            "high_usage": [],
            "medium_usage": [],
            "low_usage": [],
            "no_usage": []
        }
        
        for func_name, stats in router_stats.items():
            count = stats["count"]
            if count == 0:
                usage_groups["no_usage"].append(func_name)
            elif count <= 10:
                usage_groups["low_usage"].append(func_name)
            elif count <= 50:
                usage_groups["medium_usage"].append(func_name)
            else:
                usage_groups["high_usage"].append(func_name)
        
        return {
            "usage_distribution": {
                "high_usage": len(usage_groups["high_usage"]),
                "medium_usage": len(usage_groups["medium_usage"]),
                "low_usage": len(usage_groups["low_usage"]),
                "no_usage": len(usage_groups["no_usage"])
            },
            "usage_groups": usage_groups,
            "total_calls": sum(stats["count"] for stats in router_stats.values()),
            "most_used_functions": sorted(
                router_stats.items(), 
                key=lambda x: x[1]["count"], 
                reverse=True
            )[:5]
        }
    
    def generate_dashboard_report(self) -> Dict[str, Any]:
        """Generate comprehensive dashboard report"""
        return {
            "generated_at": datetime.now().isoformat(),
            "overview": self.get_overview_stats(),
            "critical_alerts": self.get_critical_alerts(),
            "migration_progress": self.get_migration_progress(),
            "timeline_view": self.get_timeline_view(),
            "usage_trends": self.get_usage_trends()
        }
    
    def get_function_category_stats(self) -> Dict[str, Any]:
        """Get statistics grouped by function category"""
        categories = {
            "search": ["search_chunks", "get_chunk", "get_related_chunks", 
                      "search_chunks_semantic", "search_chunks_by_similarity"],
            "content": ["create_chunk", "update_chunk", "add_relationship", "batch_generate_embeddings"],
            "promotion": ["create_promotion_request", "get_promotion_requests", 
                         "approve_promotion_request", "reject_promotion_request",
                         "get_promotion_impact", "get_promotion_queue_summary"],
            "session": ["get_session_primer", "get_pending_changes", "commit_session_changes"],
            "analytics": ["track_access", "get_hot_contexts"]
        }
        
        router_stats = self.deprecation_router.get_usage_stats()
        category_stats = {}
        
        for category, functions in categories.items():
            total_usage = sum(router_stats.get(func, {}).get("count", 0) for func in functions)
            migrated_count = sum(1 for func in functions if router_stats.get(func, {}).get("count", 0) == 0)
            
            category_stats[category] = {
                "total_functions": len(functions),
                "migrated_functions": migrated_count,
                "total_usage": total_usage,
                "migration_percentage": (migrated_count / len(functions)) * 100,
                "functions": functions
            }
        
        return category_stats
    
    def export_dashboard_data(self, filename: str = "dashboard_export.json"):
        """Export dashboard data to JSON file"""
        dashboard_data = self.generate_dashboard_report()
        
        try:
            with open(filename, 'w') as f:
                json.dump(dashboard_data, f, indent=2, default=str)
            logger.info(f"Dashboard data exported to {filename}")
            return {"success": True, "filename": filename}
        except Exception as e:
            logger.error(f"Error exporting dashboard data: {e}")
            return {"success": False, "error": str(e)}