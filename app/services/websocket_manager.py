"""
WebSocket manager for real-time communication (for future use)
"""
from typing import List, Dict
from fastapi import WebSocket, WebSocketDisconnect
import json
import asyncio
import logging

logger = logging.getLogger(__name__)


class WebSocketManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_count = 0
    
    async def connect(self, websocket: WebSocket, client_id: str = None):
        """Accept a websocket connection"""
        await websocket.accept()
        
        if not client_id:
            client_id = f"client_{self.connection_count}"
            self.connection_count += 1
        
        self.active_connections[client_id] = websocket
        logger.info(f"✅ WebSocket client connected: {client_id}")
        return client_id
    
    def disconnect(self, client_id: str):
        """Remove a websocket connection"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"❌ WebSocket client disconnected: {client_id}")
    
    async def send_personal_message(self, message: str, client_id: str):
        """Send message to specific client"""
        if client_id in self.active_connections:
            try:
                websocket = self.active_connections[client_id]
                await websocket.send_text(message)
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)
    
    async def broadcast_message(self, message: str):
        """Send message to all connected clients"""
        disconnected_clients = []
        
        for client_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting to {client_id}: {e}")
                disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)
    
    async def broadcast_json(self, data: dict):
        """Broadcast JSON data to all clients"""
        message = json.dumps(data)
        await self.broadcast_message(message)
    
    def get_connection_count(self) -> int:
        """Get number of active connections"""
        return len(self.active_connections)


# Global websocket manager instance
manager = WebSocketManager()
