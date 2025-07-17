"""
Removal Alerts and Notification System for MegaMind MCP Functions
Phase 3: Function Consolidation Cleanup Plan

This module provides automated alerts and notifications for function removal
milestones and critical events.
"""

import json
import logging
import smtplib
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class Alert:
    """Alert data structure"""
    alert_type: str
    severity: str
    function_name: str
    message: str
    timestamp: datetime
    days_until_removal: int
    usage_count: int

class RemovalAlerts:
    """
    Alert system for function removal milestones and critical events
    """
    
    def __init__(self, removal_scheduler, usage_dashboard, alert_config: Dict[str, Any] = None):
        self.removal_scheduler = removal_scheduler
        self.usage_dashboard = usage_dashboard
        self.alert_config = alert_config or self._get_default_config()
        self.alert_history = []
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default alert configuration"""
        return {
            "email_alerts": {
                "enabled": False,
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "sender_email": "",
                "sender_password": "",
                "recipient_emails": []
            },
            "alert_thresholds": {
                "high_usage": 50,
                "critical_days": 7,
                "final_warning_days": 3
            },
            "alert_intervals": {
                "daily": 1,
                "weekly": 7,
                "final_warning": 1
            }
        }
    
    def check_removal_milestones(self) -> List[Alert]:
        """Check for removal milestone alerts"""
        alerts = []
        
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
            days_until_removal = timeline["days_until_removal"]
            usage_count = timeline["usage_count"]
            
            # Critical milestone: 7 days before removal
            if days_until_removal <= self.alert_config["alert_thresholds"]["critical_days"] and days_until_removal > 0:
                alerts.append(Alert(
                    alert_type="critical_milestone",
                    severity="high",
                    function_name=func_name,
                    message=f"🚨 CRITICAL: Function '{func_name}' will be removed in {days_until_removal} days",
                    timestamp=datetime.now(),
                    days_until_removal=days_until_removal,
                    usage_count=usage_count
                ))
            
            # Final warning: 3 days before removal
            if days_until_removal <= self.alert_config["alert_thresholds"]["final_warning_days"] and days_until_removal > 0:
                alerts.append(Alert(
                    alert_type="final_warning",
                    severity="critical",
                    function_name=func_name,
                    message=f"🚨 FINAL WARNING: Function '{func_name}' will be removed in {days_until_removal} days",
                    timestamp=datetime.now(),
                    days_until_removal=days_until_removal,
                    usage_count=usage_count
                ))
            
            # Function removed but still being used
            if days_until_removal <= 0 and usage_count > 0:
                alerts.append(Alert(
                    alert_type="removed_but_used",
                    severity="critical",
                    function_name=func_name,
                    message=f"❌ ERROR: Function '{func_name}' was removed but still has {usage_count} usage calls",
                    timestamp=datetime.now(),
                    days_until_removal=days_until_removal,
                    usage_count=usage_count
                ))
        
        return alerts
    
    def check_usage_alerts(self) -> List[Alert]:
        """Check for high usage alerts"""
        alerts = []
        router_stats = self.usage_dashboard.deprecation_router.get_usage_stats()
        
        for func_name, stats in router_stats.items():
            usage_count = stats["count"]
            timeline = self.removal_scheduler.get_removal_timeline(func_name)
            days_until_removal = timeline["days_until_removal"]
            
            # High usage alert
            if usage_count >= self.alert_config["alert_thresholds"]["high_usage"]:
                alerts.append(Alert(
                    alert_type="high_usage",
                    severity="medium",
                    function_name=func_name,
                    message=f"⚠️ HIGH USAGE: Function '{func_name}' has {usage_count} calls and {days_until_removal} days until removal",
                    timestamp=datetime.now(),
                    days_until_removal=days_until_removal,
                    usage_count=usage_count
                ))
        
        return alerts
    
    def check_phase_transitions(self) -> List[Alert]:
        """Check for phase transition alerts"""
        alerts = []
        
        # Get functions entering warning phase
        from .removal_scheduler import RemovalPhase
        warning_functions = self.removal_scheduler.get_functions_by_phase(
            RemovalPhase.WARNING
        )
        
        for func_name in warning_functions:
            timeline = self.removal_scheduler.get_removal_timeline(func_name)
            alerts.append(Alert(
                alert_type="phase_transition",
                severity="medium",
                function_name=func_name,
                message=f"⚠️ PHASE CHANGE: Function '{func_name}' entered WARNING phase",
                timestamp=datetime.now(),
                days_until_removal=timeline["days_until_removal"],
                usage_count=timeline["usage_count"]
            ))
        
        # Get functions entering final warning phase
        final_warning_functions = self.removal_scheduler.get_functions_by_phase(
            RemovalPhase.FINAL_WARNING
        )
        
        for func_name in final_warning_functions:
            timeline = self.removal_scheduler.get_removal_timeline(func_name)
            alerts.append(Alert(
                alert_type="phase_transition",
                severity="high",
                function_name=func_name,
                message=f"🚨 PHASE CHANGE: Function '{func_name}' entered FINAL WARNING phase",
                timestamp=datetime.now(),
                days_until_removal=timeline["days_until_removal"],
                usage_count=timeline["usage_count"]
            ))
        
        return alerts
    
    def generate_all_alerts(self) -> List[Alert]:
        """Generate all types of alerts"""
        all_alerts = []
        
        all_alerts.extend(self.check_removal_milestones())
        all_alerts.extend(self.check_usage_alerts())
        all_alerts.extend(self.check_phase_transitions())
        
        # Sort by severity and days until removal
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        all_alerts.sort(key=lambda x: (severity_order.get(x.severity, 3), x.days_until_removal))
        
        return all_alerts
    
    def generate_alert_summary(self) -> Dict[str, Any]:
        """Generate alert summary for dashboard"""
        alerts = self.generate_all_alerts()
        
        summary = {
            "total_alerts": len(alerts),
            "critical_alerts": len([a for a in alerts if a.severity == "critical"]),
            "high_alerts": len([a for a in alerts if a.severity == "high"]),
            "medium_alerts": len([a for a in alerts if a.severity == "medium"]),
            "alert_types": {},
            "most_critical": []
        }
        
        # Group by alert type
        for alert in alerts:
            if alert.alert_type not in summary["alert_types"]:
                summary["alert_types"][alert.alert_type] = 0
            summary["alert_types"][alert.alert_type] += 1
        
        # Get most critical alerts
        summary["most_critical"] = [
            {
                "function": alert.function_name,
                "message": alert.message,
                "severity": alert.severity,
                "days_until_removal": alert.days_until_removal
            }
            for alert in alerts[:5]  # Top 5 most critical
        ]
        
        return summary
    
    def send_email_alert(self, alerts: List[Alert]):
        """Send email alert for critical functions"""
        if not self.alert_config["email_alerts"]["enabled"]:
            return
        
        if not alerts:
            return
        
        try:
            # Create email message
            msg = MIMEMultipart()
            msg['From'] = self.alert_config["email_alerts"]["sender_email"]
            msg['To'] = ", ".join(self.alert_config["email_alerts"]["recipient_emails"])
            msg['Subject'] = f"MegaMind MCP Function Removal Alert - {len(alerts)} Critical Issues"
            
            # Create email body
            body = self._create_email_body(alerts)
            msg.attach(MIMEText(body, 'html'))
            
            # Send email
            server = smtplib.SMTP(
                self.alert_config["email_alerts"]["smtp_server"],
                self.alert_config["email_alerts"]["smtp_port"]
            )
            server.starttls()
            server.login(
                self.alert_config["email_alerts"]["sender_email"],
                self.alert_config["email_alerts"]["sender_password"]
            )
            
            for recipient in self.alert_config["email_alerts"]["recipient_emails"]:
                server.sendmail(
                    self.alert_config["email_alerts"]["sender_email"],
                    recipient,
                    msg.as_string()
                )
            
            server.quit()
            logger.info(f"Email alert sent for {len(alerts)} critical functions")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    def _create_email_body(self, alerts: List[Alert]) -> str:
        """Create HTML email body for alerts"""
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .critical {{ color: #dc3545; }}
                .high {{ color: #fd7e14; }}
                .medium {{ color: #ffc107; }}
                .alert-item {{ margin: 10px 0; padding: 10px; border-left: 4px solid; }}
                .critical-item {{ border-left-color: #dc3545; }}
                .high-item {{ border-left-color: #fd7e14; }}
                .medium-item {{ border-left-color: #ffc107; }}
            </style>
        </head>
        <body>
            <h2>MegaMind MCP Function Removal Alert</h2>
            <p><strong>Alert Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Total Alerts:</strong> {len(alerts)}</p>
            
            <h3>Critical Alerts</h3>
        """
        
        for alert in alerts:
            severity_class = f"{alert.severity}-item"
            html += f"""
            <div class="alert-item {severity_class}">
                <strong>{alert.function_name}</strong> - {alert.message}<br>
                <small>Days until removal: {alert.days_until_removal} | Usage count: {alert.usage_count}</small>
            </div>
            """
        
        html += """
            <hr>
            <p><strong>Action Required:</strong> Please migrate from deprecated functions to their consolidated equivalents.</p>
            <p><strong>Migration Guide:</strong> See /docs/Migration_Guide.md for detailed migration instructions.</p>
        </body>
        </html>
        """
        
        return html
    
    def create_alert_report(self) -> Dict[str, Any]:
        """Create comprehensive alert report"""
        alerts = self.generate_all_alerts()
        
        return {
            "report_generated": datetime.now().isoformat(),
            "alert_summary": self.generate_alert_summary(),
            "detailed_alerts": [
                {
                    "alert_type": alert.alert_type,
                    "severity": alert.severity,
                    "function_name": alert.function_name,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat(),
                    "days_until_removal": alert.days_until_removal,
                    "usage_count": alert.usage_count
                }
                for alert in alerts
            ],
            "recommendations": self._generate_recommendations(alerts)
        }
    
    def _generate_recommendations(self, alerts: List[Alert]) -> List[str]:
        """Generate recommendations based on alerts"""
        recommendations = []
        
        critical_alerts = [a for a in alerts if a.severity == "critical"]
        high_usage_alerts = [a for a in alerts if a.alert_type == "high_usage"]
        
        if critical_alerts:
            recommendations.append(
                f"URGENT: {len(critical_alerts)} functions require immediate attention. "
                "Review and migrate these functions as soon as possible."
            )
        
        if high_usage_alerts:
            recommendations.append(
                f"HIGH PRIORITY: {len(high_usage_alerts)} functions have high usage counts. "
                "Plan migration strategy for these heavily used functions."
            )
        
        recommendations.append(
            "Review the Migration Guide (/docs/Migration_Guide.md) for step-by-step migration instructions."
        )
        
        recommendations.append(
            "Monitor the usage dashboard regularly to track migration progress."
        )
        
        return recommendations
    
    def save_alert_history(self, alerts: List[Alert]):
        """Save alert history to file"""
        try:
            history_entry = {
                "timestamp": datetime.now().isoformat(),
                "alerts": [
                    {
                        "alert_type": alert.alert_type,
                        "severity": alert.severity,
                        "function_name": alert.function_name,
                        "message": alert.message,
                        "days_until_removal": alert.days_until_removal,
                        "usage_count": alert.usage_count
                    }
                    for alert in alerts
                ]
            }
            
            self.alert_history.append(history_entry)
            
            # Save to file
            with open("alert_history.json", "w") as f:
                json.dump(self.alert_history, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving alert history: {e}")