"""
Application settings and configuration with frame processing optimization
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore', case_sensitive=False)
    
    # Redis Configuration
    REDIS_HOST: str = 'localhost'
    REDIS_PORT: int = 6379
    RAW_FRAME_INPUT_CHANNEL: str = 'camera:frames:rpi'
    ANNOTATED_FRAME_OUTPUT_CHANNEL: str = 'ai_stream:frames:rpi'
    
    # Server Configuration
    SERVER_HOST: str = '0.0.0.0'
    SERVER_PORT: int = 8001
    
    # AI Model Configuration
    AI_MODEL_PATH: str = 'yolov8m.hef'
    CONFIDENCE_THRESHOLD: float = 0.25
    JPEG_QUALITY: int = 60
    
    # Model Input Size Configuration
    MODEL_INPUT_WIDTH: int = 640
    MODEL_INPUT_HEIGHT: int = 640
    MAINTAIN_ASPECT_RATIO: bool = True
    
    # Frame Processing Optimization (NEW)
    ENABLE_FRAME_SKIPPING: bool = True
    PROCESS_EVERY_N_FRAMES: int = 3
    MAX_PROCESSING_FPS: float = 10.0
    SKIP_FRAMES_ON_HIGH_CPU: bool = True
    CPU_THRESHOLD_FOR_SKIPPING: float = 80.0
    MEMORY_THRESHOLD_FOR_SKIPPING: float = 85.0
    ADAPTIVE_FRAME_SKIPPING: bool = True
    
    # Performance Optimization (NEW)
    ENABLE_PERFORMANCE_MODE: bool = True
    LOW_LATENCY_MODE: bool = False
    BATCH_PROCESSING: bool = False
    OPTIMIZE_FOR_POWER_SAVING: bool = True
    REDUCE_OVERLAY_FREQUENCY: bool = True
    OVERLAY_UPDATE_EVERY_N_FRAMES: int = 5
    
    # Stream Configuration
    STREAM_TIMEOUT: float = 5.0
    RECONNECT_DELAY: float = 1.0
    MAX_RECONNECT_ATTEMPTS: int = 10
    
    # Verbose Overlay Settings
    ENABLE_VERBOSE_OVERLAY: bool = True
    OVERLAY_POSITION: str = 'TOP_LEFT'
    OVERLAY_FONT_SIZE: float = 0.4
    OVERLAY_BACKGROUND: bool = True
    SHOW_FRAME_DETAILS: bool = True
    SHOW_MODEL_DETAILS: bool = True
    SHOW_PROCESSING_METRICS: bool = True
    SHOW_DETECTION_DETAILS: bool = True
    SHOW_SYSTEM_METRICS: bool = True
    SHOW_FRAME_SKIP_STATS: bool = True
    MAX_DETECTION_LINES: int = 8
    OVERLAY_UPDATE_INTERVAL: int = 1
    
    # Camera RPI Settings
    CAMERA_RPI_RESOLUTION_WIDTH: int = 1280
    CAMERA_RPI_RESOLUTION_HEIGHT: int = 720
    CAMERA_RPI_FPS: int = 60
    CAMERA_RPI_JPEG_QUALITY: int = 100
    CAMERA_RPI_SHUTTER_SPEED: int = 0
    CAMERA_RPI_ISO: int = 0
    CAMERA_RPI_MANUAL_FOCUS: float = 0.0


settings = Settings()
