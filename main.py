#!/usr/bin/env python3
"""
Main entry point for Hailo AI Video Streaming Server
"""
import uvicorn
from app.config.settings import settings
from app.api.v1.streams import create_app


def main():
    """Main entry point"""
    print("🚀 Starting Hailo AI Video Streaming Server")
    print(f"📺 Dashboard: http://{settings.SERVER_HOST}:{settings.SERVER_PORT}")
    print(f"📡 Video Feed: http://{settings.SERVER_HOST}:{settings.SERVER_PORT}/video_feed")
    print(f"🔗 Redis Channel: {settings.ANNOTATED_FRAME_OUTPUT_CHANNEL}")
    
    app = create_app()
    
    uvicorn.run(
        app,
        host=settings.SERVER_HOST,
        port=settings.SERVER_PORT,
        reload=False,
        access_log=True
    )


if __name__ == "__main__":
    main()
