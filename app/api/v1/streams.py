"""
FastAPI application and streaming endpoints
"""
import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse

from app.config.settings import settings
from app.services.redis_manager import redis_manager
from app.web.router import web_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown"""
    # Startup
    logger.info("🚀 Starting Hailo AI Video Streaming Server")
    logger.info(f"📺 Redis: {settings.REDIS_HOST}:{settings.REDIS_PORT}")
    logger.info(f"📡 Channel: {settings.ANNOTATED_FRAME_OUTPUT_CHANNEL}")
    logger.info(f"🌐 Server: {settings.SERVER_HOST}:{settings.SERVER_PORT}")
    
    # Connect to Redis
    await redis_manager.connect()
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Hailo AI Video Streaming Server")
    await redis_manager.disconnect()


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    app = FastAPI(
        title="Hailo AI Video Stream",
        description="Real-time AI-processed video streaming with technical overlays",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Include web router
    app.include_router(web_router)
    
    @app.get("/video_feed")
    async def video_feed(request: Request):
        """Stream video frames from Redis channel as MJPEG"""
        try:
            logger.info(f"📺 New video feed request from {request.client.host}")
            
            return StreamingResponse(
                redis_manager.frame_generator(settings.ANNOTATED_FRAME_OUTPUT_CHANNEL),
                media_type="multipart/x-mixed-replace; boundary=frame",
                headers={
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0",
                    "Connection": "keep-alive"
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Error creating video feed: {e}")
            return HTMLResponse(
                content="<h1>Video Feed Error</h1><p>Unable to connect to video stream. Please check if the AI inference service is running.</p>",
                status_code=503
            )
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint for monitoring"""
        try:
            await redis_manager.client.ping()
            return {
                "status": "healthy",
                "redis": "connected",
                "channel": settings.ANNOTATED_FRAME_OUTPUT_CHANNEL,
                "timestamp": time.time()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "redis": "disconnected",
                "error": str(e),
                "timestamp": time.time()
            }
    
    @app.get("/stats")
    async def stream_stats():
        """Get streaming statistics"""
        try:
            redis_info = await redis_manager.client.info()
            
            return {
                "redis_host": settings.REDIS_HOST,
                "redis_port": settings.REDIS_PORT,
                "channel": settings.ANNOTATED_FRAME_OUTPUT_CHANNEL,
                "redis_connected_clients": redis_info.get('connected_clients', 0),
                "redis_used_memory": redis_info.get('used_memory_human', 'unknown'),
                "server_host": settings.SERVER_HOST,
                "server_port": settings.SERVER_PORT
            }
        except Exception as e:
            return {"error": str(e)}
    
    return app
