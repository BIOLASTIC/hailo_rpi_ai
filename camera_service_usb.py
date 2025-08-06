#!/usr/bin/env python3
"""
USB Camera Service for capturing frames and publishing to Redis with configurable resolution
"""
import cv2
import redis
import time
import threading
from app.config.settings import settings


class USBCameraService:
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.cap = None
        self.redis_client = None
        self.running = False
        self.thread = None
        # Use configurable camera settings
        self.width = settings.CAMERA_RPI_RESOLUTION_WIDTH
        self.height = settings.CAMERA_RPI_RESOLUTION_HEIGHT
        self.fps = settings.CAMERA_RPI_FPS
        self.jpeg_quality = settings.CAMERA_RPI_JPEG_QUALITY
        
    def initialize(self):
        """Initialize camera and Redis connection with configurable settings"""
        try:
            # Initialize camera
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                raise Exception(f"Cannot open camera {self.camera_id}")
                
            # Set camera properties using configurable values
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Additional camera settings if supported
            if settings.CAMERA_RPI_ISO > 0:
                try:
                    self.cap.set(cv2.CAP_PROP_ISO_SPEED, settings.CAMERA_RPI_ISO)
                except:
                    pass  # Not all cameras support ISO
            
            if settings.CAMERA_RPI_SHUTTER_SPEED > 0:
                try:
                    self.cap.set(cv2.CAP_PROP_EXPOSURE, settings.CAMERA_RPI_SHUTTER_SPEED)
                except:
                    pass  # Not all cameras support manual exposure
            
            # Initialize Redis
            self.redis_client = redis.Redis(
                host=settings.REDIS_HOST, 
                port=settings.REDIS_PORT
            )
            self.redis_client.ping()
            
            print(f"✅ USB Camera {self.camera_id} initialized")
            print(f"📷 Resolution: {self.width}x{self.height} @ {self.fps}fps")
            print(f"🖼️ JPEG Quality: {self.jpeg_quality}%")
            print(f"✅ Redis connected for camera frames")
            return True
            
        except Exception as e:
            print(f"❌ Camera initialization error: {e}")
            return False
    
    def start_capture(self):
        """Start capturing frames in a separate thread"""
        if not self.initialize():
            return False
            
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        print(f"🎥 Camera capture started on channel: {settings.RAW_FRAME_INPUT_CHANNEL}")
        return True
    
    def _capture_loop(self):
        """Main capture loop with configurable settings"""
        frame_count = 0
        
        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Failed to read frame from camera")
                    time.sleep(0.1)
                    continue
                
                # Encode frame as JPEG with configurable quality
                ret, buffer = cv2.imencode('.jpg', frame, 
                    [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
                
                if ret:
                    # Publish to Redis
                    self.redis_client.publish(
                        settings.RAW_FRAME_INPUT_CHANNEL, 
                        buffer.tobytes()
                    )
                    frame_count += 1
                    
                    if frame_count % 60 == 0:
                        print(f"📸 Published frame #{frame_count} ({self.width}x{self.height})")
                
                # Control frame rate
                time.sleep(1.0 / self.fps)
                
            except Exception as e:
                print(f"❌ Capture error: {e}")
                time.sleep(1)
    
    def stop_capture(self):
        """Stop capturing frames"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.cap:
            self.cap.release()
        if self.redis_client:
            self.redis_client.close()
        print("🛑 Camera capture stopped")


def main():
    camera_service = USBCameraService(camera_id=0)
    
    try:
        if camera_service.start_capture():
            print("Press Ctrl+C to stop camera service...")
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping camera service...")
    finally:
        camera_service.stop_capture()


if __name__ == "__main__":
    main()
