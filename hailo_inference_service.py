#!/usr/bin/env python3
"""
Enhanced Hailo AI Inference Service with Frame Skipping and Robust Redis Connection Handling
"""
import time
import cv2
import redis
import numpy as np
import traceback
from pathlib import Path

# Correct Redis retry imports
try:
    from redis.retry import Retry
    from redis.backoff import ExponentialBackoff
    REDIS_RETRY_AVAILABLE = True
except ImportError:
    REDIS_RETRY_AVAILABLE = False
    print("⚠️ Redis retry functionality not available, using basic connection")

# Import from the new modular structure
from app.config.settings import settings
from app.services.system_monitor import SystemMonitor
from app.services.technical_logger import TechnicalLogger
from app.services.frame_processor import FrameProcessor
from app.core.detection_processor import DetectionProcessor

try:
    from hailo_platform import VDevice, HEF, InferVStreams, InputVStreamParams, OutputVStreamParams
except ImportError as e:
    print(f"[FATAL] HailoRT import error: {e}")
    exit(1)


class OptimizedHailoInferenceService:
    def __init__(self):
        self.system_monitor = SystemMonitor()
        self.technical_logger = TechnicalLogger()
        self.frame_processor = FrameProcessor()
        self.detection_processor = DetectionProcessor()
        self.redis_client = None
        self.pubsub = None
        self.model_info = {}
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        
        # Frame skipping and optimization variables
        self.frame_skip_counter = 0
        self.last_process_time = 0
        self.last_detections = []  # Store last detections for skipped frames
        self.last_processed_frame = None
        self.adaptive_skip_factor = 1
        self.overlay_cache = None
        self.overlay_update_counter = 0
        
    def initialize_redis(self):
        """Initialize Redis with progressive fallback to find working configuration"""
        print("🔗 Attempting Redis connection with progressive fallback...")
        
        # Method 1: Most basic connection (no socket options)
        try:
            print("   Trying basic connection...")
            self.redis_client = redis.Redis(
                host='localhost',
                port=6379,
                decode_responses=False
            )
            self.redis_client.ping()
            print("✅ Redis connected (Method 1: Basic)")
            return True
        except Exception as e:
            print(f"   Method 1 failed: {e}")
        
        # Method 2: With minimal timeouts only
        try:
            print("   Trying with minimal timeouts...")
            self.redis_client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                socket_connect_timeout=5,
                socket_timeout=5,
                decode_responses=False
            )
            self.redis_client.ping()
            print("✅ Redis connected (Method 2: With timeouts)")
            return True
        except Exception as e:
            print(f"   Method 2 failed: {e}")
        
        # Method 3: URL-based connection
        try:
            print("   Trying URL-based connection...")
            self.redis_client = redis.from_url(
                'redis://localhost:6379/0',
                socket_connect_timeout=5,
                socket_timeout=5
            )
            self.redis_client.ping()
            print("✅ Redis connected (Method 3: URL-based)")
            return True
        except Exception as e:
            print(f"   Method 3 failed: {e}")
        
        # Method 4: Connection pool approach
        try:
            print("   Trying connection pool...")
            pool = redis.ConnectionPool(
                host='localhost',
                port=6379,
                max_connections=10
            )
            self.redis_client = redis.Redis(connection_pool=pool)
            self.redis_client.ping()
            print("✅ Redis connected (Method 4: Connection pool)")
            return True
        except Exception as e:
            print(f"   Method 4 failed: {e}")
        
        print("❌ All Redis connection methods failed")
        return False
    
    def setup_redis_pubsub(self):
        """Setup Redis pubsub with enhanced reconnection handling"""
        try:
            if self.pubsub:
                try:
                    self.pubsub.close()
                except:
                    pass
            
            # Ensure Redis client is healthy
            self.redis_client.ping()
            
            self.pubsub = self.redis_client.pubsub()
            self.pubsub.subscribe(settings.RAW_FRAME_INPUT_CHANNEL)
            print(f"✅ Redis pubsub subscribed to {settings.RAW_FRAME_INPUT_CHANNEL}")
            self.reconnect_attempts = 0
            return True
            
        except Exception as e:
            print(f"❌ Redis pubsub setup error: {e}")
            return False
    
    def reconnect_redis(self):
        """Handle Redis reconnection with enhanced exponential backoff"""
        self.reconnect_attempts += 1
        
        if self.reconnect_attempts > self.max_reconnect_attempts:
            print(f"💥 Max Redis reconnection attempts ({self.max_reconnect_attempts}) reached")
            return False
        
        # Enhanced exponential backoff with jitter
        base_wait = 2 ** self.reconnect_attempts
        jitter = np.random.uniform(0.1, 0.5)
        wait_time = min(base_wait + jitter, 30)
        
        print(f"🔄 Redis reconnection attempt {self.reconnect_attempts}/{self.max_reconnect_attempts}")
        print(f"⏳ Waiting {wait_time:.1f} seconds before reconnecting...")
        
        time.sleep(wait_time)
        
        # Close existing connections gracefully
        try:
            if self.pubsub:
                self.pubsub.close()
            if self.redis_client:
                self.redis_client.close()
        except:
            pass
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Reinitialize Redis connection
        if self.initialize_redis() and self.setup_redis_pubsub():
            print("✅ Redis reconnection successful")
            return True
        else:
            print("❌ Redis reconnection failed")
            return False
    
    def get_redis_message_with_retry(self):
        """Get Redis message with enhanced automatic reconnection on failure"""
        max_retries = 5
        base_timeout = 0.5
        
        for attempt in range(max_retries):
            try:
                # Health check before getting message
                if attempt > 0:
                    self.redis_client.ping()
                
                # Get message with adaptive timeout
                timeout = base_timeout * (attempt + 1)
                message = self.pubsub.get_message(
                    ignore_subscribe_messages=True, 
                    timeout=min(timeout, 2.0)
                )
                return message
                
            except (redis.ConnectionError, redis.TimeoutError) as e:
                print(f"❌ Redis connection error (attempt {attempt + 1}): {e}")
                
                if attempt < max_retries - 1:
                    if self.reconnect_redis():
                        continue
                    else:
                        break
                else:
                    raise e
                    
            except Exception as e:
                print(f"❌ Unexpected Redis error: {e}")
                raise e
        
        return None
    
    def setup_hailo_model(self, vdevice):
        """Setup Hailo model and auto-detect input size from HEF file"""
        hef = HEF(settings.AI_MODEL_PATH)
        network_groups = vdevice.configure(hef)
        network_group = network_groups[0]
        
        # Get ACTUAL model dimensions from HEF file
        input_vstream_info = hef.get_input_vstream_infos()[0]
        output_vstream_info = hef.get_output_vstream_infos()[0]
        
        model_height_hef = input_vstream_info.shape[0]
        model_width_hef = input_vstream_info.shape[1]
        
        size_override = (model_width_hef != settings.MODEL_INPUT_WIDTH or 
                        model_height_hef != settings.MODEL_INPUT_HEIGHT)
        
        self.model_info = {
            'name': Path(settings.AI_MODEL_PATH).stem,
            'input_shape': f"{model_width_hef}x{model_height_hef}x{input_vstream_info.shape[2]}",
            'output_shape': str(output_vstream_info.shape),
            'num_classes': 80,
            'actual_height': model_height_hef,
            'actual_width': model_width_hef,
            'configured_height': settings.MODEL_INPUT_HEIGHT,
            'configured_width': settings.MODEL_INPUT_WIDTH,
            'size_override': size_override,
            'maintain_aspect_ratio': settings.MAINTAIN_ASPECT_RATIO
        }
        
        print(f"📐 Model: {self.model_info['name']}")
        print(f"📏 HEF Model Input (ACTUAL): {model_width_hef}x{model_height_hef}")
        if size_override:
            print(f"⚠️ Using HEF's native size ({model_width_hef}x{model_height_hef}) instead of configured size")
        
        return network_group, hef
    
    def should_skip_frame(self, frame_count, current_time):
        """Determine if current frame should be skipped based on multiple criteria"""
        if not settings.ENABLE_FRAME_SKIPPING:
            return False, 'disabled'
        
        # Manual frame skipping (process every N frames)
        if frame_count % settings.PROCESS_EVERY_N_FRAMES != 0:
            return True, 'manual'
        
        # FPS limiting
        if settings.MAX_PROCESSING_FPS > 0:
            min_interval = 1.0 / settings.MAX_PROCESSING_FPS
            if current_time - self.last_process_time < min_interval:
                return True, 'fps_limit'
        
        # System resource-based skipping
        if settings.SKIP_FRAMES_ON_HIGH_CPU:
            cpu_percent = self.system_monitor.cpu_percent
            memory_percent = self.system_monitor.memory_percent
            
            if cpu_percent > settings.CPU_THRESHOLD_FOR_SKIPPING:
                return True, 'high_cpu'
            
            if memory_percent > settings.MEMORY_THRESHOLD_FOR_SKIPPING:
                return True, 'high_memory'
        
        # Adaptive frame skipping
        if settings.ADAPTIVE_FRAME_SKIPPING:
            stats = self.technical_logger.get_stats()
            avg_processing_ms = stats.get('avg_processing_ms', 0)
            
            if avg_processing_ms > 100:
                if frame_count % (settings.PROCESS_EVERY_N_FRAMES * self.adaptive_skip_factor) != 0:
                    return True, 'adaptive'
        
        return False, 'none'
    
    def update_adaptive_skip_factor(self):
        """Update adaptive skip factor based on system performance"""
        if not settings.ADAPTIVE_FRAME_SKIPPING:
            return
        
        cpu_percent = self.system_monitor.cpu_percent
        
        if cpu_percent > settings.CPU_THRESHOLD_FOR_SKIPPING * 0.8:
            self.adaptive_skip_factor = min(self.adaptive_skip_factor + 1, 5)
        elif cpu_percent < settings.CPU_THRESHOLD_FOR_SKIPPING * 0.5:
            self.adaptive_skip_factor = max(self.adaptive_skip_factor - 1, 1)
    
    def process_frame(self, frame, infer_pipeline, input_name, output_name, frame_count):
        """Process a single frame using ACTUAL model input size from HEF"""
        frame_start_time = time.time()
        original_height, original_width = frame.shape[:2]
        
        actual_height = self.model_info['actual_height']
        actual_width = self.model_info['actual_width']
        
        # Preprocessing
        preprocessed_frame, preprocess_time = self.frame_processor.preprocess_frame_with_timing(
            frame, actual_height, actual_width
        )
        
        # Inference
        inference_start = time.time()
        input_data = {input_name: np.expand_dims(preprocessed_frame, axis=0)}
        
        try:
            results = infer_pipeline.infer(input_data)
            raw_detections = results[output_name]
            inference_time = time.time() - inference_start
        except Exception as e:
            print(f"[ERROR] Inference error: {e}")
            return None
        
        # Post-processing
        detections, postprocess_time = self.detection_processor.postprocess_hailo_detections_with_timing(
            raw_detections, original_width, original_height, 
            actual_height, actual_width
        )
        
        self.last_detections = detections
        self.last_processed_frame = frame.copy()
        
        # Draw detections
        annotated_frame = self.frame_processor.draw_detections_with_overlay(frame.copy(), detections)
        
        total_processing_time = time.time() - frame_start_time
        
        frame_metrics = {
            'frame_number': frame_count,
            'timestamp': frame_start_time,
            'width': original_width,
            'height': original_height,
            'model_input_width': actual_width,
            'model_input_height': actual_height,
            'total_time': total_processing_time,
            'preprocess_time': preprocess_time,
            'inference_time': inference_time,
            'postprocess_time': postprocess_time
        }
        
        self.technical_logger.log_frame_metrics(total_processing_time, inference_time)
        self.technical_logger.log_detections(len(detections))
        self.last_process_time = time.time()
        
        return annotated_frame, frame_metrics, detections
    
    def create_skipped_frame_output(self, frame, frame_count):
        """Create output for skipped frame using last detections"""
        if self.last_detections and len(self.last_detections) > 0:
            annotated_frame = self.frame_processor.draw_detections_with_overlay(frame.copy(), self.last_detections)
        else:
            annotated_frame = frame.copy()
        
        frame_metrics = {
            'frame_number': frame_count,
            'timestamp': time.time(),
            'width': frame.shape[1],
            'height': frame.shape[0],
            'model_input_width': self.model_info['actual_width'],
            'model_input_height': self.model_info['actual_height'],
            'total_time': 0,
            'preprocess_time': 0,
            'inference_time': 0,
            'postprocess_time': 0
        }
        
        return annotated_frame, frame_metrics, self.last_detections or []
    
    def publish_frame_with_retry(self, frame_data):
        """Enhanced frame publishing with better error handling"""
        max_retries = 5
        base_delay = 0.1
        
        for attempt in range(max_retries):
            try:
                self.redis_client.ping()
                self.redis_client.publish(settings.ANNOTATED_FRAME_OUTPUT_CHANNEL, frame_data)
                return True
                
            except (redis.ConnectionError, redis.TimeoutError) as e:
                print(f"❌ Redis publish error (attempt {attempt + 1}): {e}")
                
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    jitter = np.random.uniform(0.01, 0.1)
                    time.sleep(min(delay + jitter, 2.0))
                    
                    if self.reconnect_redis():
                        continue
                    else:
                        break
                else:
                    print("❌ Failed to publish frame after all retries")
                    return False
                    
            except Exception as e:
                print(f"❌ Unexpected publish error: {e}")
                return False
        
        return False
    
    def run(self):
        """Main inference loop with frame skipping and performance optimization"""
        print("🚀 === OPTIMIZED Hailo Inference Service ===")
        print(f"📊 Verbose Overlay: {settings.ENABLE_VERBOSE_OVERLAY}")
        print(f"📍 Overlay Position: {settings.OVERLAY_POSITION}")
        print(f"🔍 Max Detection Lines: {settings.MAX_DETECTION_LINES}")
        print(f"🖼️ Maintain Aspect Ratio: {settings.MAINTAIN_ASPECT_RATIO}")
        print(f"⏭️ Frame Skipping: {'Enabled' if settings.ENABLE_FRAME_SKIPPING else 'Disabled'}")
        
        if settings.ENABLE_FRAME_SKIPPING:
            print(f"🔢 Process Every N Frames: {settings.PROCESS_EVERY_N_FRAMES}")
            print(f"⚡ Max Processing FPS: {settings.MAX_PROCESSING_FPS}")
            print(f"🧠 Adaptive Skipping: {settings.ADAPTIVE_FRAME_SKIPPING}")
            print(f"💾 CPU Threshold: {settings.CPU_THRESHOLD_FOR_SKIPPING}%")
            print(f"💾 Memory Threshold: {settings.MEMORY_THRESHOLD_FOR_SKIPPING}%")
        
        print(f"🔄 Max Reconnect Attempts: {self.max_reconnect_attempts}")
        print(f"⚡ Performance Mode: {settings.ENABLE_PERFORMANCE_MODE}")
        print(f"🔋 Power Saving: {settings.OPTIMIZE_FOR_POWER_SAVING}")
        print(f"🔗 Enhanced Redis Connection: {'With Retry' if REDIS_RETRY_AVAILABLE else 'Basic'}")
        
        self.system_monitor.start_monitoring()
        
        try:
            print("\n🔗 Initializing Redis...")
            if not self.initialize_redis():
                print("💥 Failed to initialize Redis. Exiting...")
                return
            
            if not self.setup_redis_pubsub():
                print("💥 Failed to setup Redis pubsub. Exiting...")
                return
            
            print("\n⚡ Initializing Hailo AI model...")
            with VDevice() as vdevice:
                network_group, hef = self.setup_hailo_model(vdevice)
                
                input_vstreams_params = InputVStreamParams.make_from_network_group(network_group, quantized=False)
                output_vstreams_params = OutputVStreamParams.make_from_network_group(network_group, quantized=False)
                network_group_params = network_group.create_params()
                
                with network_group.activate(network_group_params):
                    with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
                        input_name = network_group.get_input_vstream_infos()[0].name
                        output_name = network_group.get_output_vstream_infos()[0].name
                        
                        actual_width = self.model_info['actual_width']
                        actual_height = self.model_info['actual_height']
                        
                        print("✅ Hailo AI model loaded successfully")
                        print("🎯 Starting optimized inference loop...")
                        print(f"📝 Processing: Selective frames → {actual_width}x{actual_height} model input → Original size output")
                        print("=" * 80)
                        
                        frame_count = 0
                        consecutive_failures = 0
                        max_consecutive_failures = 10
                        last_stats_time = time.time()
                        
                        while True:
                            try:
                                message = self.get_redis_message_with_retry()
                                
                                if not message:
                                    consecutive_failures += 1
                                    if consecutive_failures > max_consecutive_failures:
                                        print(f"💥 Too many consecutive failures ({consecutive_failures}). Exiting...")
                                        break
                                    continue
                                
                                consecutive_failures = 0
                                frame_count += 1
                                current_time = time.time()
                                
                                # Decode frame
                                try:
                                    np_array = np.frombuffer(message['data'], np.uint8)
                                    frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
                                    
                                    if frame is None:
                                        continue
                                except Exception as e:
                                    print(f"❌ Frame decode error: {e}")
                                    continue
                                
                                # Determine if frame should be skipped
                                should_skip, skip_reason = self.should_skip_frame(frame_count, current_time)
                                
                                if should_skip:
                                    self.technical_logger.log_skipped_frame(skip_reason)
                                    annotated_frame, frame_metrics, detections = self.create_skipped_frame_output(frame, frame_count)
                                else:
                                    result = self.process_frame(frame, infer_pipeline, input_name, output_name, frame_count)
                                    if result is None:
                                        continue
                                    annotated_frame, frame_metrics, detections = result
                                
                                # Update adaptive skip factor periodically
                                if frame_count % 30 == 0:
                                    self.update_adaptive_skip_factor()
                                
                                # Create overlay
                                if self.frame_processor.should_update_overlay(frame_count):
                                    logger_stats = self.technical_logger.get_stats()
                                    system_stats = self.system_monitor.get_stats()
                                    
                                    overlay_lines = self.frame_processor.create_verbose_overlay_lines(
                                        frame_metrics, detections, self.model_info, system_stats, logger_stats
                                    )
                                    
                                    if settings.REDUCE_OVERLAY_FREQUENCY:
                                        self.overlay_cache = overlay_lines
                                else:
                                    overlay_lines = self.overlay_cache or []
                                
                                # Draw overlay and publish
                                final_frame = self.frame_processor.draw_verbose_technical_overlay(annotated_frame, overlay_lines)
                                
                                if final_frame is not None:
                                    ret, buffer = cv2.imencode('.jpg', final_frame, 
                                        [int(cv2.IMWRITE_JPEG_QUALITY), settings.JPEG_QUALITY])
                                    
                                    if ret:
                                        self.publish_frame_with_retry(buffer.tobytes())
                                
                                # Console output every 60 frames
                                if frame_count % 60 == 0:
                                    current_stats_time = time.time()
                                    time_diff = current_stats_time - last_stats_time
                                    
                                    logger_stats = self.technical_logger.get_stats()
                                    fps = logger_stats.get('fps', 0)
                                    avg_ms = logger_stats.get('avg_processing_ms', 0)
                                    total_dets = logger_stats.get('total_detections', 0)
                                    processed = logger_stats.get('processed_frames', 0)
                                    skipped = logger_stats.get('skipped_frames', 0)
                                    efficiency = logger_stats.get('processing_efficiency', 100)
                                    skip_reasons = logger_stats.get('skip_reasons', {})
                                    
                                    # System metrics
                                    cpu_pct = self.system_monitor.cpu_percent
                                    mem_pct = self.system_monitor.memory_percent
                                    temp = self.system_monitor.temperature
                                    
                                    print(f"\n📊 === FRAME {frame_count} STATISTICS ===")
                                    print(f"⚡ Processing FPS: {fps:.1f} | Avg Time: {avg_ms:.1f}ms | Total Detections: {total_dets}")
                                    print(f"📈 Efficiency: {efficiency:.1f}% | Processed: {processed} | Skipped: {skipped}")
                                    print(f"🖥️ CPU: {cpu_pct:.1f}% | Memory: {mem_pct:.1f}% | Temp: {temp:.1f}°C")
                                    print(f"🔄 Reconnects: {self.reconnect_attempts} | Adaptive Factor: {self.adaptive_skip_factor}")
                                    
                                    if skip_reasons:
                                        skip_summary = ", ".join([f"{k}:{v}" for k, v in skip_reasons.items() if v > 0])
                                        print(f"⏭️ Skip Reasons: {skip_summary}")
                                    
                                    print("=" * 80)
                                    last_stats_time = current_stats_time
                                
                            except KeyboardInterrupt:
                                print("\n🛑 Keyboard interrupt received. Shutting down gracefully...")
                                break
                                
                            except Exception as e:
                                print(f"❌ Unexpected error in main loop: {e}")
                                consecutive_failures += 1
                                if consecutive_failures > max_consecutive_failures:
                                    print(f"💥 Too many consecutive failures ({consecutive_failures}). Exiting...")
                                    break
                                time.sleep(1)
        
        except Exception as e:
            print(f"💥 Fatal error: {e}")
            traceback.print_exc()
            
        finally:
            print("\n🧹 Cleaning up resources...")
            
            # Stop monitoring
            self.system_monitor.stop_monitoring()
            
            # Print final performance summary
            try:
                performance_summary = self.technical_logger.get_performance_summary()
                print("\n📊 === FINAL PERFORMANCE SUMMARY ===")
                for key, value in performance_summary.items():
                    print(f"{key.replace('_', ' ').title()}: {value}")
                
                # Additional cleanup stats
                final_stats = self.technical_logger.get_stats()
                print(f"\nTotal Runtime: {final_stats.get('uptime_seconds', 0):.1f} seconds")
                print(f"Total Frames Processed: {final_stats.get('total_frames', 0)}")
                print(f"Average FPS: {final_stats.get('fps', 0):.2f}")
                print(f"Connection Stability: {((60 - self.reconnect_attempts) / 60 * 100):.1f}%")
                
            except Exception as cleanup_error:
                print(f"❌ Error generating performance summary: {cleanup_error}")
            
            # Clean up Redis connections
            try:
                if self.pubsub:
                    print("🧹 Closing Redis pubsub...")
                    self.pubsub.close()
                if self.redis_client:
                    print("🧹 Closing Redis client...")
                    self.redis_client.close()
                print("✅ Redis connections closed cleanly")
            except Exception as redis_cleanup_error:
                print(f"❌ Error during Redis cleanup: {redis_cleanup_error}")
            
            print("✅ Cleanup completed. Service stopped.")


def main():
    """Main entry point with enhanced error handling"""
    print("🚀 Starting Optimized Hailo AI Inference Service")
    print("=" * 60)
    
    try:
        service = OptimizedHailoInferenceService()
        service.run()
    except KeyboardInterrupt:
        print("\n🛑 Service interrupted by user")
    except Exception as e:
        print(f"💥 Service failed to start: {e}")
        traceback.print_exc()
    finally:
        print("👋 Hailo AI Inference Service terminated")


if __name__ == "__main__":
    main()
