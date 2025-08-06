import asyncio
import time
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
import redis.asyncio as redis
from pydantic_settings import BaseSettings, SettingsConfigDict
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore', case_sensitive=False)
    
    # Redis Configuration
    REDIS_HOST: str = 'localhost'
    REDIS_PORT: int = 6379
    ANNOTATED_FRAME_OUTPUT_CHANNEL: str = 'ai_stream:frames:rpi'
    
    # Server Configuration
    SERVER_HOST: str = '0.0.0.0'
    SERVER_PORT: int = 8001
    
    # Stream Configuration
    STREAM_TIMEOUT: float = 5.0
    RECONNECT_DELAY: float = 1.0
    MAX_RECONNECT_ATTEMPTS: int = 10

settings = Settings()

# Global Redis client
redis_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown."""
    global redis_client
    
    # Startup
    logger.info("🚀 Starting Hailo AI Video Streaming Server")
    logger.info(f"📺 Redis: {settings.REDIS_HOST}:{settings.REDIS_PORT}")
    logger.info(f"📡 Channel: {settings.ANNOTATED_FRAME_OUTPUT_CHANNEL}")
    logger.info(f"🌐 Server: {settings.SERVER_HOST}:{settings.SERVER_PORT}")
    
    try:
        redis_client = redis.from_url(
            f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}",
            decode_responses=False,
            socket_keepalive=True,
            socket_keepalive_options={},
            retry_on_timeout=True
        )
        
        # Test Redis connection
        await redis_client.ping()
        logger.info("✅ Redis connection established")
        
    except Exception as e:
        logger.error(f"❌ Failed to connect to Redis: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Hailo AI Video Streaming Server")
    if redis_client:
        await redis_client.close()
        logger.info("✅ Redis connection closed")

# Initialize FastAPI app with lifespan management
app = FastAPI(
    title="Hailo AI Video Stream",
    description="Real-time AI-processed video streaming with technical overlays",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the main dashboard with live video feed."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Hailo AI Live Video Stream</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                margin: 0;
                padding: 20px;
                color: white;
                min-height: 100vh;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                text-align: center;
            }
            h1 {
                font-size: 2.5em;
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
            }
            .subtitle {
                font-size: 1.2em;
                margin-bottom: 30px;
                opacity: 0.9;
            }
            .video-container {
                background: rgba(0,0,0,0.3);
                border-radius: 15px;
                padding: 20px;
                margin: 20px auto;
                box-shadow: 0 8px 32px rgba(0,0,0,0.3);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255,255,255,0.1);
            }
            .video-stream {
                max-width: 100%;
                height: auto;
                border-radius: 10px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            }
            .status {
                display: flex;
                justify-content: center;
                gap: 20px;
                margin: 20px 0;
                flex-wrap: wrap;
            }
            .status-item {
                background: rgba(255,255,255,0.1);
                padding: 10px 20px;
                border-radius: 25px;
                border: 1px solid rgba(255,255,255,0.2);
            }
            .controls {
                margin: 20px 0;
            }
            .btn {
                background: rgba(255,255,255,0.2);
                border: 1px solid rgba(255,255,255,0.3);
                color: white;
                padding: 10px 20px;
                border-radius: 25px;
                cursor: pointer;
                margin: 0 10px;
                transition: all 0.3s ease;
            }
            .btn:hover {
                background: rgba(255,255,255,0.3);
                transform: translateY(-2px);
            }
            .technical-info {
                text-align: left;
                background: rgba(0,0,0,0.2);
                padding: 15px;
                border-radius: 10px;
                margin: 20px 0;
                font-family: 'Courier New', monospace;
                font-size: 0.9em;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 Hailo AI Live Video Stream</h1>
            <p class="subtitle">Real-time object detection with comprehensive technical overlays</p>
            
            <div class="status">
                <div class="status-item">🎯 YOLOv8 Detection</div>
                <div class="status-item">⚡ Hailo-8L Acceleration</div>
                <div class="status-item">📊 Technical Metrics</div>
                <div class="status-item">🔴 Live Stream</div>
            </div>
            
            <div class="video-container">
                <img id="videoStream" 
                     src="/video_feed" 
                     alt="Live AI Video Feed"
                     class="video-stream"
                     onerror="handleStreamError()"
                     onload="handleStreamLoad()">
            </div>
            
            <div class="controls">
                <button class="btn" onclick="refreshStream()">🔄 Refresh Stream</button>
                <button class="btn" onclick="toggleFullscreen()">📺 Fullscreen</button>
                <button class="btn" onclick="showInfo()">ℹ️ Info</button>
            </div>
            
            <div class="technical-info">
                <h3>📊 Stream Information</h3>
                <p><strong>Endpoint:</strong> /video_feed</p>
                <p><strong>Format:</strong> MJPEG Stream (multipart/x-mixed-replace)</p>
                <p><strong>Source:</strong> Redis Channel: ai_stream:frames:rpi</p>
                <p><strong>Processing:</strong> Hailo AI Inference → Detection Overlay → Stream</p>
                <p><strong>Technical Overlay:</strong> Frame details, processing metrics, detection info, system stats</p>
            </div>
        </div>

        <script>
            let streamErrors = 0;
            const maxErrors = 5;
            
            function handleStreamError() {
                streamErrors++;
                console.error('Stream error #' + streamErrors);
                
                if (streamErrors < maxErrors) {
                    setTimeout(() => {
                        document.getElementById('videoStream').src = '/video_feed?' + new Date().getTime();
                    }, 2000);
                } else {
                    document.getElementById('videoStream').alt = 'Stream unavailable - check AI inference service';
                }
            }
            
            function handleStreamLoad() {
                streamErrors = 0;
                console.log('Stream loaded successfully');
            }
            
            function refreshStream() {
                streamErrors = 0;
                document.getElementById('videoStream').src = '/video_feed?' + new Date().getTime();
            }
            
            function toggleFullscreen() {
                const video = document.getElementById('videoStream');
                if (video.requestFullscreen) {
                    video.requestFullscreen();
                }
            }
            
            function showInfo() {
                alert('Hailo AI Video Stream\\n\\nThis stream shows real-time object detection with comprehensive technical overlays including:\\n\\n• Frame processing metrics\\n• Detection details\\n• System performance\\n• Model information\\n\\nProcessed by Hailo-8L AI accelerator');
            }
            
            // Auto-refresh on connection issues
            setInterval(() => {
                if (streamErrors >= 3) {
                    refreshStream();
                }
            }, 10000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

async def frame_generator(redis_client: redis.Redis, channel_name: str) -> AsyncGenerator[bytes, None]:
    """
    Generate video frames from Redis channel with robust error handling and reconnection.
    """
    reconnect_attempts = 0
    
    while reconnect_attempts < settings.MAX_RECONNECT_ATTEMPTS:
        try:
            logger.info(f"🔗 Subscribing to Redis channel: {channel_name}")
            
            pubsub = redis_client.pubsub()
            await pubsub.subscribe(channel_name)
            
            logger.info(f"✅ Successfully subscribed to {channel_name}")
            reconnect_attempts = 0  # Reset on successful connection
            
            while True:
                try:
                    # Get message with timeout to allow for cancellation
                    message = await pubsub.get_message(
                        ignore_subscribe_messages=True, 
                        timeout=settings.STREAM_TIMEOUT
                    )
                    
                    if message and message['type'] == 'message':
                        frame_data = message['data']
                        
                        # Yield the frame in MJPEG format
                        yield (
                            b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n'
                            b'Content-Length: ' + str(len(frame_data)).encode() + b'\r\n'
                            b'\r\n' + frame_data + b'\r\n'
                        )
                    else:
                        # No message received within timeout, continue loop
                        await asyncio.sleep(0.01)
                        
                except asyncio.CancelledError:
                    logger.info("🛑 Frame generator cancelled")
                    break
                except Exception as e:
                    logger.error(f"❌ Error in frame generation: {e}")
                    break
                    
        except asyncio.CancelledError:
            logger.info("🛑 Frame generator cancelled during setup")
            break
        except Exception as e:
            reconnect_attempts += 1
            logger.error(f"❌ Redis connection error (attempt {reconnect_attempts}): {e}")
            
            if reconnect_attempts < settings.MAX_RECONNECT_ATTEMPTS:
                logger.info(f"🔄 Retrying in {settings.RECONNECT_DELAY} seconds...")
                await asyncio.sleep(settings.RECONNECT_DELAY)
            else:
                logger.error("💥 Max reconnection attempts reached")
                break
        finally:
            try:
                await pubsub.unsubscribe(channel_name)
                await pubsub.close()
            except:
                pass

@app.get("/video_feed")
async def video_feed(request: Request):
    """
    Stream video frames from Redis channel as MJPEG.
    
    This endpoint provides a continuous stream of JPEG frames published by the 
    Hailo AI inference service, formatted as an MJPEG stream for browser display.
    """
    try:
        logger.info(f"📺 New video feed request from {request.client.host}")
        
        return StreamingResponse(
            frame_generator(redis_client, settings.ANNOTATED_FRAME_OUTPUT_CHANNEL),
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
    """Health check endpoint for monitoring."""
    try:
        await redis_client.ping()
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
    """Get streaming statistics."""
    try:
        # Get Redis info
        redis_info = await redis_client.info()
        
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

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Hailo AI Video Streaming Server")
    print(f"📺 Dashboard: http://{settings.SERVER_HOST}:{settings.SERVER_PORT}")
    print(f"📡 Video Feed: http://{settings.SERVER_HOST}:{settings.SERVER_PORT}/video_feed")
    print(f"🔗 Redis Channel: {settings.ANNOTATED_FRAME_OUTPUT_CHANNEL}")
    
    uvicorn.run(
        "main:app",
        host=settings.SERVER_HOST,
        port=settings.SERVER_PORT,
        reload=False,
        access_log=True
    )
