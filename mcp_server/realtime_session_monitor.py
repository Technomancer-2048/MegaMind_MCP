#!/usr/bin/env python3
"""
Phase 7: Real-time Session Monitor
WebSocket-based real-time session monitoring with live updates
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from collections import defaultdict, deque
import threading
from dataclasses import dataclass, asdict
import uuid

# WebSocket imports for real-time communication
try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    logging.warning("WebSocket libraries not available - using fallback monitoring")

logger = logging.getLogger(__name__)

@dataclass
class SessionUpdate:
    """Real-time session update data structure"""
    update_id: str
    session_id: str
    user_id: str
    update_type: str  # 'status_change', 'operation', 'performance', 'error', 'completion'
    timestamp: datetime
    data: Dict[str, Any]
    ml_insights: Optional[Dict[str, Any]] = None

@dataclass
class MonitoringClient:
    """WebSocket client connection data"""
    client_id: str
    websocket: Any
    connected_at: datetime
    subscriptions: Set[str]  # Session IDs being monitored
    user_id: Optional[str] = None
    client_type: str = 'dashboard'  # 'dashboard', 'admin', 'developer'

class RealTimeSessionMonitor:
    """
    Real-time WebSocket-based session monitoring system
    Provides live updates on session status, performance, and events
    """
    
    def __init__(self, session_manager, ml_engine=None, analytics_engine=None):
        self.session_manager = session_manager
        self.ml_engine = ml_engine
        self.analytics_engine = analytics_engine
        
        # WebSocket server and client management
        self.websocket_server = None
        self.connected_clients = {}  # client_id -> MonitoringClient
        self.session_subscribers = defaultdict(set)  # session_id -> set of client_ids
        
        # Update tracking
        self.update_queue = asyncio.Queue(maxsize=10000)
        self.update_history = deque(maxlen=1000)
        
        # Session state tracking
        self.session_states = {}  # session_id -> current state data
        self.performance_tracker = {}  # session_id -> performance metrics
        
        # Server configuration
        self.server_host = "0.0.0.0"
        self.server_port = 8766
        self.is_running = False
        
        logger.info("✅ Real-time Session Monitor initialized")
    
    async def start_server(self):
        """Start the WebSocket monitoring server"""
        if not WEBSOCKET_AVAILABLE:
            logger.error("❌ WebSocket not available - cannot start real-time monitoring")
            return False
        
        if self.is_running:
            logger.warning("⚠️ WebSocket server already running")
            return True
        
        try:
            # Start WebSocket server
            self.websocket_server = await websockets.serve(
                self._handle_client_connection,
                self.server_host,
                self.server_port,
                ping_interval=30,
                ping_timeout=10,
                close_timeout=10
            )
            
            # Start update processing
            asyncio.create_task(self._process_updates_loop())
            
            self.is_running = True
            logger.info(f"🌐 Real-time Session Monitor started on {self.server_host}:{self.server_port}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start WebSocket server: {e}")
            return False
    
    async def stop_server(self):
        """Stop the WebSocket monitoring server"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Close all client connections
        for client in list(self.connected_clients.values()):
            try:
                await client.websocket.close()
            except Exception:
                pass
        
        # Close server
        if self.websocket_server:
            self.websocket_server.close()
            await self.websocket_server.wait_closed()
        
        logger.info("⏹️ Real-time Session Monitor stopped")
    
    async def _handle_client_connection(self, websocket: WebSocketServerProtocol, path: str):
        """Handle new WebSocket client connection"""
        client_id = str(uuid.uuid4())
        client = MonitoringClient(
            client_id=client_id,
            websocket=websocket,
            connected_at=datetime.now(),
            subscriptions=set()
        )
        
        self.connected_clients[client_id] = client
        
        try:
            logger.info(f"🔗 New monitoring client connected: {client_id}")
            
            # Send welcome message
            await self._send_to_client(client, {
                'type': 'connection_established',
                'client_id': client_id,
                'server_time': datetime.now().isoformat(),
                'capabilities': {
                    'real_time_updates': True,
                    'session_monitoring': True,
                    'performance_tracking': True,
                    'ml_insights': self.ml_engine is not None,
                    'anomaly_detection': self.analytics_engine is not None
                }
            })
            
            # Handle client messages
            async for message in websocket:
                await self._handle_client_message(client, message)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"🔌 Client disconnected: {client_id}")
        except Exception as e:
            logger.error(f"❌ Error handling client {client_id}: {e}")
        finally:
            # Cleanup client
            await self._cleanup_client(client)
    
    async def _handle_client_message(self, client: MonitoringClient, message: str):
        """Handle incoming message from WebSocket client"""
        try:
            data = json.loads(message)
            message_type = data.get('type')
            
            if message_type == 'subscribe_session':
                await self._handle_session_subscription(client, data)
            elif message_type == 'unsubscribe_session':
                await self._handle_session_unsubscription(client, data)
            elif message_type == 'get_session_status':
                await self._handle_session_status_request(client, data)
            elif message_type == 'get_performance_metrics':
                await self._handle_performance_request(client, data)
            elif message_type == 'client_info':
                await self._handle_client_info_update(client, data)
            else:
                await self._send_error(client, f"Unknown message type: {message_type}")
                
        except json.JSONDecodeError:
            await self._send_error(client, "Invalid JSON message")
        except Exception as e:
            logger.error(f"Error handling message from {client.client_id}: {e}")
            await self._send_error(client, f"Message processing error: {str(e)}")
    
    async def _handle_session_subscription(self, client: MonitoringClient, data: Dict[str, Any]):
        """Handle session subscription request"""
        session_id = data.get('session_id')
        if not session_id:
            await self._send_error(client, "session_id required for subscription")
            return
        
        # Add subscription
        client.subscriptions.add(session_id)
        self.session_subscribers[session_id].add(client.client_id)
        
        # Send current session state
        current_state = await self._get_session_current_state(session_id)
        await self._send_to_client(client, {
            'type': 'subscription_confirmed',
            'session_id': session_id,
            'current_state': current_state
        })
        
        logger.info(f"📊 Client {client.client_id} subscribed to session {session_id}")
    
    async def _handle_session_unsubscription(self, client: MonitoringClient, data: Dict[str, Any]):
        """Handle session unsubscription request"""
        session_id = data.get('session_id')
        if not session_id:
            await self._send_error(client, "session_id required for unsubscription")
            return
        
        # Remove subscription
        client.subscriptions.discard(session_id)
        self.session_subscribers[session_id].discard(client.client_id)
        
        await self._send_to_client(client, {
            'type': 'unsubscription_confirmed',
            'session_id': session_id
        })
        
        logger.info(f"📊 Client {client.client_id} unsubscribed from session {session_id}")
    
    async def _handle_session_status_request(self, client: MonitoringClient, data: Dict[str, Any]):
        """Handle session status request"""
        session_id = data.get('session_id')
        if not session_id:
            await self._send_error(client, "session_id required")
            return
        
        try:
            # Get session details
            session = self.session_manager.get_session_details(session_id)
            if not session:
                await self._send_error(client, f"Session {session_id} not found")
                return
            
            # Get current state and performance
            current_state = await self._get_session_current_state(session_id)
            performance_data = self._get_session_performance(session_id)
            
            # Get ML insights if available
            ml_insights = None
            if self.ml_engine:
                ml_insights = await self._get_session_ml_insights(session_id)
            
            await self._send_to_client(client, {
                'type': 'session_status',
                'session_id': session_id,
                'session_details': {
                    'session_name': session.session_name,
                    'user_id': session.user_id,
                    'status': session.status,
                    'created_at': session.created_at.isoformat(),
                    'priority': session.priority
                },
                'current_state': current_state,
                'performance': performance_data,
                'ml_insights': ml_insights
            })
            
        except Exception as e:
            logger.error(f"Error getting session status: {e}")
            await self._send_error(client, f"Failed to get session status: {str(e)}")
    
    async def _handle_performance_request(self, client: MonitoringClient, data: Dict[str, Any]):
        """Handle performance metrics request"""
        session_id = data.get('session_id')
        time_range = data.get('time_range', '1h')  # '5m', '1h', '24h'
        
        try:
            performance_data = await self._get_performance_history(session_id, time_range)
            
            await self._send_to_client(client, {
                'type': 'performance_metrics',
                'session_id': session_id,
                'time_range': time_range,
                'metrics': performance_data
            })
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            await self._send_error(client, f"Failed to get performance metrics: {str(e)}")
    
    async def _handle_client_info_update(self, client: MonitoringClient, data: Dict[str, Any]):
        """Handle client information update"""
        client.user_id = data.get('user_id')
        client.client_type = data.get('client_type', 'dashboard')
        
        await self._send_to_client(client, {
            'type': 'client_info_updated',
            'client_id': client.client_id
        })
    
    async def _cleanup_client(self, client: MonitoringClient):
        """Clean up disconnected client"""
        # Remove from connected clients
        self.connected_clients.pop(client.client_id, None)
        
        # Remove from all session subscriptions
        for session_id in client.subscriptions:
            self.session_subscribers[session_id].discard(client.client_id)
        
        logger.info(f"🧹 Cleaned up client {client.client_id}")
    
    # ================================================================
    # SESSION UPDATE BROADCASTING
    # ================================================================
    
    async def broadcast_session_update(self, session_id: str, update_type: str, 
                                     data: Dict[str, Any], user_id: str = None):
        """Broadcast session update to all subscribers"""
        if not self.is_running:
            return
        
        update = SessionUpdate(
            update_id=str(uuid.uuid4()),
            session_id=session_id,
            user_id=user_id or "system",
            update_type=update_type,
            timestamp=datetime.now(),
            data=data
        )
        
        # Add ML insights if available
        if self.ml_engine:
            update.ml_insights = await self._get_session_ml_insights(session_id)
        
        # Queue update for processing
        try:
            await self.update_queue.put(update)
        except asyncio.QueueFull:
            logger.warning("Update queue full - dropping update")
    
    async def _process_updates_loop(self):
        """Process queued updates and broadcast to clients"""
        while self.is_running:
            try:
                # Get update from queue
                update = await asyncio.wait_for(self.update_queue.get(), timeout=1.0)
                
                # Process and broadcast update
                await self._process_and_broadcast_update(update)
                
                # Add to history
                self.update_history.append(update)
                
                # Mark task done
                self.update_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing update: {e}")
    
    async def _process_and_broadcast_update(self, update: SessionUpdate):
        """Process update and broadcast to relevant clients"""
        # Update session state tracking
        self._update_session_state(update)
        
        # Get subscribers for this session
        subscribers = self.session_subscribers.get(update.session_id, set())
        if not subscribers:
            return
        
        # Prepare broadcast message
        message = {
            'type': 'session_update',
            'update': {
                'update_id': update.update_id,
                'session_id': update.session_id,
                'update_type': update.update_type,
                'timestamp': update.timestamp.isoformat(),
                'data': update.data,
                'ml_insights': update.ml_insights
            }
        }
        
        # Broadcast to all subscribers
        disconnected_clients = set()
        for client_id in subscribers:
            client = self.connected_clients.get(client_id)
            if client:
                try:
                    await self._send_to_client(client, message)
                except Exception as e:
                    logger.warning(f"Failed to send update to client {client_id}: {e}")
                    disconnected_clients.add(client_id)
        
        # Remove disconnected clients
        for client_id in disconnected_clients:
            self.session_subscribers[update.session_id].discard(client_id)
    
    def _update_session_state(self, update: SessionUpdate):
        """Update internal session state tracking"""
        session_id = update.session_id
        
        if session_id not in self.session_states:
            self.session_states[session_id] = {
                'last_update': update.timestamp,
                'status': 'active',
                'operation_count': 0,
                'error_count': 0,
                'performance_score': 0.0
            }
        
        state = self.session_states[session_id]
        state['last_update'] = update.timestamp
        
        # Update based on update type
        if update.update_type == 'operation':
            state['operation_count'] += 1
        elif update.update_type == 'error':
            state['error_count'] += 1
        elif update.update_type == 'status_change':
            state['status'] = update.data.get('new_status', state['status'])
        elif update.update_type == 'performance':
            state['performance_score'] = update.data.get('score', state['performance_score'])
        
        # Update performance tracking
        if session_id not in self.performance_tracker:
            self.performance_tracker[session_id] = deque(maxlen=1000)
        
        self.performance_tracker[session_id].append({
            'timestamp': update.timestamp,
            'performance_score': state['performance_score'],
            'operation_count': state['operation_count'],
            'error_count': state['error_count']
        })
    
    # ================================================================
    # HELPER METHODS
    # ================================================================
    
    async def _send_to_client(self, client: MonitoringClient, message: Dict[str, Any]):
        """Send message to specific client"""
        try:
            await client.websocket.send(json.dumps(message))
        except Exception as e:
            logger.warning(f"Failed to send message to client {client.client_id}: {e}")
            raise
    
    async def _send_error(self, client: MonitoringClient, error_message: str):
        """Send error message to client"""
        await self._send_to_client(client, {
            'type': 'error',
            'message': error_message,
            'timestamp': datetime.now().isoformat()
        })
    
    async def _get_session_current_state(self, session_id: str) -> Dict[str, Any]:
        """Get current session state"""
        state = self.session_states.get(session_id, {})
        
        # Get live session data
        try:
            session = self.session_manager.get_session_details(session_id)
            if session:
                state.update({
                    'session_name': session.session_name,
                    'status': session.status,
                    'created_at': session.created_at.isoformat(),
                    'priority': session.priority
                })
        except Exception as e:
            logger.warning(f"Failed to get session details: {e}")
        
        return state
    
    def _get_session_performance(self, session_id: str) -> Dict[str, Any]:
        """Get session performance data"""
        performance_data = self.performance_tracker.get(session_id, [])
        
        if not performance_data:
            return {'current_score': 0.0, 'trend': 'stable', 'data_points': 0}
        
        recent_data = list(performance_data)[-50:]  # Last 50 data points
        
        current_score = recent_data[-1]['performance_score'] if recent_data else 0.0
        
        # Calculate trend
        if len(recent_data) >= 10:
            early_avg = sum(d['performance_score'] for d in recent_data[:5]) / 5
            late_avg = sum(d['performance_score'] for d in recent_data[-5:]) / 5
            
            if late_avg > early_avg + 0.1:
                trend = 'improving'
            elif late_avg < early_avg - 0.1:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_data'
        
        return {
            'current_score': current_score,
            'trend': trend,
            'data_points': len(recent_data),
            'operation_count': recent_data[-1]['operation_count'] if recent_data else 0,
            'error_count': recent_data[-1]['error_count'] if recent_data else 0
        }
    
    async def _get_session_ml_insights(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get ML insights for session"""
        if not self.ml_engine:
            return None
        
        try:
            # This would integrate with the ML engine from Phase 6
            insights = {
                'success_probability': 0.8,  # Placeholder
                'performance_prediction': 0.75,
                'recommended_actions': [],
                'risk_factors': []
            }
            
            # Add real ML predictions if methods exist
            if hasattr(self.ml_engine, 'predict_session_success'):
                session_data = self.session_manager.get_session_details(session_id)
                if session_data:
                    prediction = self.ml_engine.predict_session_success(
                        session_data.__dict__, []
                    )
                    insights.update(prediction)
            
            return insights
            
        except Exception as e:
            logger.warning(f"Failed to get ML insights: {e}")
            return None
    
    async def _get_performance_history(self, session_id: str, time_range: str) -> Dict[str, Any]:
        """Get performance history for specified time range"""
        performance_data = self.performance_tracker.get(session_id, [])
        
        # Calculate time cutoff
        now = datetime.now()
        if time_range == '5m':
            cutoff = now - timedelta(minutes=5)
        elif time_range == '1h':
            cutoff = now - timedelta(hours=1)
        elif time_range == '24h':
            cutoff = now - timedelta(hours=24)
        else:
            cutoff = now - timedelta(hours=1)  # Default to 1 hour
        
        # Filter data
        filtered_data = [
            d for d in performance_data 
            if d['timestamp'] >= cutoff
        ]
        
        if not filtered_data:
            return {'data_points': [], 'summary': {'avg_performance': 0.0}}
        
        # Prepare time series data
        data_points = [
            {
                'timestamp': d['timestamp'].isoformat(),
                'performance_score': d['performance_score'],
                'operation_count': d['operation_count'],
                'error_count': d['error_count']
            }
            for d in filtered_data
        ]
        
        # Calculate summary statistics
        avg_performance = sum(d['performance_score'] for d in filtered_data) / len(filtered_data)
        max_performance = max(d['performance_score'] for d in filtered_data)
        min_performance = min(d['performance_score'] for d in filtered_data)
        
        return {
            'data_points': data_points,
            'summary': {
                'avg_performance': avg_performance,
                'max_performance': max_performance,
                'min_performance': min_performance,
                'total_operations': sum(d['operation_count'] for d in filtered_data),
                'total_errors': sum(d['error_count'] for d in filtered_data)
            }
        }
    
    # ================================================================
    # PUBLIC API METHODS
    # ================================================================
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status"""
        return {
            'server_running': self.is_running,
            'server_address': f"{self.server_host}:{self.server_port}",
            'connected_clients': len(self.connected_clients),
            'active_subscriptions': sum(len(subs) for subs in self.session_subscribers.values()),
            'tracked_sessions': len(self.session_states),
            'update_queue_size': self.update_queue.qsize() if self.update_queue else 0,
            'ml_insights_available': self.ml_engine is not None,
            'analytics_integration': self.analytics_engine is not None
        }
    
    def get_connected_clients(self) -> List[Dict[str, Any]]:
        """Get list of connected clients"""
        return [
            {
                'client_id': client.client_id,
                'connected_at': client.connected_at.isoformat(),
                'subscriptions': list(client.subscriptions),
                'user_id': client.user_id,
                'client_type': client.client_type
            }
            for client in self.connected_clients.values()
        ]