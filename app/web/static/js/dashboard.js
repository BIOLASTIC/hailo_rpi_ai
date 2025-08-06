// Dashboard JavaScript functionality
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
    alert('Hailo AI Video Stream\n\nThis stream shows real-time object detection with comprehensive technical overlays including:\n\n• Frame processing metrics\n• Detection details\n• System performance\n• Model information\n\nProcessed by Hailo-8L AI accelerator');
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
        console.log('Health check:', data);
        
        // Update UI based on health status
        const statusElement = document.querySelector('.status');
        if (data.status === 'healthy') {
            statusElement.style.borderColor = '#00ff00';
        } else {
            statusElement.style.borderColor = '#ff0000';
        }
    } catch (error) {
        console.error('Health check failed:', error);
    }
}

// Run health check every 30 seconds
setInterval(checkHealth, 30000);
