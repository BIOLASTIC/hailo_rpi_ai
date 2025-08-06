"""
Web interface router for HTML pages and static content
"""
from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

web_router = APIRouter()

# Setup static files
BASE_DIR = Path(__file__).resolve().parent

# Mount static files
try:
    web_router.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
except RuntimeError:
    # Handle case where static directory doesn't exist yet
    pass


@web_router.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the main dashboard with live video feed"""
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
            .connection-status {
                position: fixed;
                top: 20px;
                right: 20px;
                padding: 10px;
                border-radius: 5px;
                background: rgba(0,0,0,0.8);
                font-size: 0.9em;
            }
            .status-healthy { color: #00ff00; }
            .status-unhealthy { color: #ff0000; }
        </style>
    </head>
    <body>
        <div class="connection-status" id="connectionStatus">
            🔍 Checking connection...
        </div>
        
        <div class="container">
            <h1>🤖 Hailo AI Live Video Stream</h1>
            <p class="subtitle">Real-time object detection with comprehensive technical overlays</p>
            
            <div class="status">
                <div class="status-item">🎯 YOLOv8 Detection</div>
                <div class="status-item">⚡ Hailo-8L Acceleration</div>
                <div class="status-item">📊 Technical Metrics</div>
                <div class="status-item" id="liveStatus">🔴 Live Stream</div>
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
                <button class="btn" onclick="showStats()">📊 Stats</button>
            </div>
            
            <div class="technical-info">
                <h3>📊 Stream Information</h3>
                <p><strong>Endpoint:</strong> /video_feed</p>
                <p><strong>Format:</strong> MJPEG Stream (multipart/x-mixed-replace)</p>
                <p><strong>Source:</strong> Redis Channel: ai_stream:frames:rpi</p>
                <p><strong>Processing:</strong> Hailo AI Inference → Detection Overlay → Stream</p>
                <p><strong>Technical Overlay:</strong> Frame details, processing metrics, detection info, system stats</p>
                <div id="statsDisplay" style="margin-top: 15px; display: none;">
                    <h4>📈 Live Statistics</h4>
                    <div id="statsContent">Loading...</div>
                </div>
            </div>
        </div>

        <script>
            let streamErrors = 0;
            const maxErrors = 5;
            let healthCheckInterval;
            
            function handleStreamError() {
                streamErrors++;
                console.error('Stream error #' + streamErrors);
                updateLiveStatus('🔴 Stream Error', 'unhealthy');
                
                if (streamErrors < maxErrors) {
                    setTimeout(() => {
                        document.getElementById('videoStream').src = '/video_feed?' + new Date().getTime();
                    }, 2000);
                } else {
                    document.getElementById('videoStream').alt = 'Stream unavailable - check AI inference service';
                    updateLiveStatus('💀 Stream Failed', 'unhealthy');
                }
            }
            
            function handleStreamLoad() {
                streamErrors = 0;
                console.log('Stream loaded successfully');
                updateLiveStatus('🟢 Live Stream', 'healthy');
            }
            
            function refreshStream() {
                streamErrors = 0;
                updateLiveStatus('🔄 Refreshing...', 'neutral');
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
            
            function updateLiveStatus(text, status) {
                const element = document.getElementById('liveStatus');
                element.textContent = text;
                element.className = 'status-item';
                if (status === 'healthy') element.style.color = '#00ff00';
                else if (status === 'unhealthy') element.style.color = '#ff0000';
                else element.style.color = '#ffff00';
            }
            
            function updateConnectionStatus(status, details) {
                const element = document.getElementById('connectionStatus');
                if (status === 'healthy') {
                    element.innerHTML = '✅ Connected';
                    element.className = 'connection-status status-healthy';
                } else {
                    element.innerHTML = '❌ Disconnected';
                    element.className = 'connection-status status-unhealthy';
                }
            }
            
            // Auto-refresh on connection issues
            setInterval(() => {
                if (streamErrors >= 3) {
                    refreshStream();
                }
            }, 10000);

            // Health check functionality
            async function checkHealth() {
                try {
                    const response = await fetch('/health');
                    const data = await response.json();
                    
                    if (data.status === 'healthy') {
                        updateConnectionStatus('healthy');
                    } else {
                        updateConnectionStatus('unhealthy');
                    }
                } catch (error) {
                    console.error('Health check failed:', error);
                    updateConnectionStatus('unhealthy');
                }
            }
            
            async function showStats() {
                try {
                    const response = await fetch('/stats');
                    const data = await response.json();
                    
                    const statsDiv = document.getElementById('statsDisplay');
                    const statsContent = document.getElementById('statsContent');
                    
                    if (data.error) {
                        statsContent.innerHTML = '<p style="color: #ff0000;">❌ Error: ' + data.error + '</p>';
                    } else {
                        statsContent.innerHTML = `
                            <p><strong>Redis Host:</strong> ${data.redis_host}:${data.redis_port}</p>
                            <p><strong>Channel:</strong> ${data.channel}</p>
                            <p><strong>Connected Clients:</strong> ${data.redis_connected_clients}</p>
                            <p><strong>Redis Memory:</strong> ${data.redis_used_memory}</p>
                            <p><strong>Server:</strong> ${data.server_host}:${data.server_port}</p>
                        `;
                    }
                    
                    // Toggle display
                    statsDiv.style.display = statsDiv.style.display === 'none' ? 'block' : 'none';
                    
                } catch (error) {
                    console.error('Stats fetch failed:', error);
                    alert('Failed to fetch statistics. Check console for details.');
                }
            }
            
            // Initialize health check
            checkHealth();
            healthCheckInterval = setInterval(checkHealth, 30000);
            
            // Cleanup on page unload
            window.addEventListener('beforeunload', () => {
                if (healthCheckInterval) {
                    clearInterval(healthCheckInterval);
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
