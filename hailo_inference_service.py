#!/usr/bin/env python3
"""
Hailo AI Inference Service Optimized for Maximum Hailo HAT Utilization
"""
import time
import cv2
import redis
import numpy as np
import traceback
import subprocess
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
    from hailo_platform import VDevice, HEF, InferVStreams, InputVStreamParams, OutputVStreamParams, ConfigureParams, HailoStreamInterface
except ImportError as e:
    print(f"[FATAL] HailoRT import error: {e}")
    exit(1)


class MaxHailoUtilizationInferenceService:
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
        
        # Hailo optimization variables
        self.frame_skip_counter = 0
        self.last_process_time = 0
        self.last_detections = []
        self.last_processed_frame = None
        self.adaptive_skip_factor = 1
        self.overlay_cache = None
        self.overlay_update_counter = 0
        
        # Hailo utilization tracking
        self.hailo_utilization = 0.0
        self.last_hailo_check = time.time()
        self.batch_frames = []
        self.batch_size = 4
        self.enable_batching = True
        self.use_hailo_nms = True
        
    def initialize_redis(self):
        """Initialize Redis with progressive fallback to find working configuration"""
        print("🔗 Attempting Redis connection with progressive fallback...")
        
        # Method 1: Enhanced persistent connection with aggressive keepalive
        try:
            print("   Trying enhanced persistent connection...")
            self.redis_client = redis.Redis(
                host='localhost',
                port=6379,
                decode_responses=False,
                socket_keepalive=True,
                socket_keepalive_options={
                    1: 1,    # TCP_KEEPIDLE: 1 second
                    2: 1,    # TCP_KEEPINTVL: 1 second interval
                    3: 3     # TCP_KEEPCNT: 3 probes
                },
                socket_connect_timeout=10,
                socket_timeout=30,
                health_check_interval=15,
                retry_on_timeout=True
            )
            self.redis_client.ping()
            print("✅ Redis connected with enhanced persistence")
            return True
        except Exception as e:
            print(f"   Enhanced method failed: {e}")
        
        # Method 2: Basic connection fallback
        try:
            print("   Trying basic connection...")
            self.redis_client = redis.Redis(
                host='localhost',
                port=6379,
                decode_responses=False
            )
            self.redis_client.ping()
            print("✅ Redis connected (Method 2: Basic)")
            return True
        except Exception as e:
            print(f"   Basic method failed: {e}")
        
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
        
        base_wait = 2 ** self.reconnect_attempts
        jitter = np.random.uniform(0.1, 0.5)
        wait_time = min(base_wait + jitter, 30)
        
        print(f"🔄 Redis reconnection attempt {self.reconnect_attempts}/{self.max_reconnect_attempts}")
        print(f"⏳ Waiting {wait_time:.1f} seconds before reconnecting...")
        
        time.sleep(wait_time)
        
        try:
            if self.pubsub:
                self.pubsub.close()
            if self.redis_client:
                self.redis_client.close()
        except:
            pass
        
        import gc
        gc.collect()
        
        if self.initialize_redis() and self.setup_redis_pubsub():
            print("✅ Redis reconnection successful")
            return True
        else:
            print("❌ Redis reconnection failed")
            return False
    
    def get_redis_message_with_retry(self):
        """Get Redis message with enhanced automatic reconnection on failure"""
        max_retries = 5
        base_timeout = 0.1  # Reduced timeout for faster throughput
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    self.redis_client.ping()
                
                timeout = base_timeout * (attempt + 1)
                message = self.pubsub.get_message(
                    ignore_subscribe_messages=True, 
                    timeout=min(timeout, 1.0)
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
    
    def setup_hailo_model_with_nms(self, vdevice):
        """Setup Hailo model with built-in post-processing for maximum utilization"""
        hef = HEF(settings.AI_MODEL_PATH)
        
        try:
            # Try to enable Hailo's built-in NMS post-processing
            configure_params = ConfigureParams.create_from_hef(
                hef, 
                interface=HailoStreamInterface.PCIe
            )
            
            # Configure NMS to run on Hailo chip instead of CPU
            nms_enabled = False
            for network_group in configure_params.network_groups:
                for network in network_group.networks:
                    for output_stream in network.output_streams:
                        if hasattr(output_stream, 'nms_config'):
                            output_stream.nms_config.engine = 'hailo'
                            nms_enabled = True
                            print("✅ Enabled Hailo NMS post-processing")
            
            if nms_enabled:
                network_groups = vdevice.configure(configure_params)
                self.use_hailo_nms = True
            else:
                network_groups = vdevice.configure(hef)
                print("⚠️ Hailo NMS not available, using CPU post-processing")
                
        except Exception as e:
            print(f"⚠️ Advanced Hailo configuration failed: {e}")
            print("📝 Using standard configuration")
            network_groups = vdevice.configure(hef)
        
        network_group = network_groups[0]
        
        # Get model dimensions
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
            'maintain_aspect_ratio': settings.MAINTAIN_ASPECT_RATIO,
            'hailo_nms_enabled': self.use_hailo_nms
        }
        
        print(f"📐 Model: {self.model_info['name']}")
        print(f"📏 HEF Model Input (ACTUAL): {model_width_hef}x{model_height_hef}")
        print(f"🔥 Hailo NMS Enabled: {self.use_hailo_nms}")
        if size_override:
            print(f"⚠️ Using HEF's native size ({model_width_hef}x{model_height_hef}) instead of configured size")
        
        return network_group, hef
    
    def get_hailo_utilization(self):
        """Get current Hailo device utilization"""
        current_time = time.time()
        
        # Only check every 2 seconds to avoid overhead
        if current_time - self.last_hailo_check < 2:
            return self.hailo_utilization
        
        try:
            result = subprocess.run(['hailortcli', 'monitor'], 
                                  capture_output=True, text=True, timeout=1)
            
            for line in result.stdout.split('\n'):
                if 'Utilization' in line and '%' in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if '%' in part:
                            util_str = part.replace('%', '')
                            try:
                                self.hailo_utilization = float(util_str)
                                self.last_hailo_check = current_time
                                return self.hailo_utilization
                            except:
                                continue
        except:
            pass
        
        return self.hailo_utilization
    
    def should_skip_frame(self, frame_count, current_time):
        """Determine if current frame should be skipped - OPTIMIZED FOR MAX HAILO USAGE"""
        # Disable most frame skipping to maximize Hailo utilization
        if not settings.ENABLE_FRAME_SKIPPING:
            return False, 'disabled'
        
        # Only skip if absolutely necessary (emergency CPU protection)
        if settings.SKIP_FRAMES_ON_HIGH_CPU:
            cpu_percent = self.system_monitor.cpu_percent
            
            # Only skip at very high CPU usage (95%+)
            if cpu_percent > 95.0:
                return True, 'emergency_cpu'
        
        # Process every frame by default for maximum Hailo utilization
        return False, 'none'
    
    def process_frame_batch(self, frames_batch, infer_pipeline, input_name, output_name, frame_counts):
        """Process multiple frames together for maximum Hailo utilization"""
        if len(frames_batch) == 0:
            return []
        
        batch_start_time = time.time()
        results = []
        
        # Prepare batch input
        batch_input = []
        for frame in frames_batch:
            preprocessed, _ = self.frame_processor.preprocess_frame_with_timing(
                frame, self.model_info['actual_height'], self.model_info['actual_width']
            )
            batch_input.append(preprocessed)
        
        # Single batched inference - more efficient for Hailo
        try:
            if len(batch_input) == 1:
                # Single frame inference
                input_data = {input_name: np.expand_dims(batch_input[0], axis=0)}
                inference_results = infer_pipeline.infer(input_data)
                raw_detections = [inference_results[output_name]]
            else:
                # Batch inference
                batch_array = np.array(batch_input)
                input_data = {input_name: batch_array}
                inference_results = infer_pipeline.infer(input_data)
                raw_detections = inference_results[output_name]
            
            inference_time = time.time() - batch_start_time
            
            # Process each frame's results
            for i, (frame, frame_count) in enumerate(zip(frames_batch, frame_counts)):
                try:
                    # Extract detections for this frame
                    if len(raw_detections) > i:
                        frame_detections = raw_detections[i:i+1]
                    else:
                        frame_detections = raw_detections[0:1]
                    
                    # Post-processing
                    detections, postprocess_time = self.detection_processor.postprocess_hailo_detections_with_timing(
                        frame_detections, frame.shape[1], frame.shape[0], 
                        self.model_info['actual_height'], self.model_info['actual_width']
                    )
                    
                    # Store last detections for potential skipped frames
                    if detections:
                        self.last_detections = detections
                    
                    # Draw detections
                    annotated_frame = self.frame_processor.draw_detections_with_overlay(frame.copy(), detections)
                    
                    # Calculate frame metrics
                    total_time = (time.time() - batch_start_time) / len(frames_batch)
                    frame_metrics = {
                        'frame_number': frame_count,
                        'timestamp': batch_start_time,
                        'width': frame.shape[1],
                        'height': frame.shape[0],
                        'model_input_width': self.model_info['actual_width'],
                        'model_input_height': self.model_info['actual_height'],
                        'total_time': total_time,
                        'preprocess_time': 0,
                        'inference_time': inference_time / len(frames_batch),
                        'postprocess_time': postprocess_time
                    }
                    
                    # Update loggers
                    self.technical_logger.log_frame_metrics(total_time, inference_time / len(frames_batch))
                    self.technical_logger.log_detections(len(detections))
                    
                    results.append((annotated_frame, frame_metrics, detections))
                    
                except Exception as e:
                    print(f"[ERROR] Frame {i} processing error: {e}")
                    continue
                    
        except Exception as e:
            print(f"[ERROR] Batch inference error: {e}")
            return []
        
        self.last_process_time = time.time()
        return results
    
    def process_single_frame(self, frame, infer_pipeline, input_name, output_name, frame_count):
        """Process a single frame - fallback method"""
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
        max_retries = 3
        base_delay = 0.05  # Reduced delay for faster throughput
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    self.redis_client.ping()
                
                self.redis_client.publish(settings.ANNOTATED_FRAME_OUTPUT_CHANNEL, frame_data)
                return True
                
            except (redis.ConnectionError, redis.TimeoutError) as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    time.sleep(delay)
                    if self.reconnect_redis():
                        continue
                else:
                    return False
            except Exception as e:
                return False
        
        return False
    
    def run(self):
        """Main inference loop optimized for MAXIMUM HAILO UTILIZATION"""
        print("🚀 === MAXIMUM HAILO UTILIZATION Inference Service ===")
        print(f"🔥 Target: Move CPU load to Hailo HAT")
        print(f"📊 Frame Skipping: {'Minimized' if settings.ENABLE_FRAME_SKIPPING else 'Disabled'}")
        print(f"⚡ Batch Processing: {'Enabled' if self.enable_batching else 'Disabled'}")
        print(f"🔥 Hailo NMS: {'Enabled' if self.use_hailo_nms else 'Disabled'}")
        print(f"🎯 Goal: 50-80% Hailo utilization, 25-35% CPU usage")
        
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
                network_group, hef = self.setup_hailo_model_with_nms(vdevice)
                
                input_vstreams_params = InputVStreamParams.make_from_network_group(network_group, quantized=False)
                output_vstreams_params = OutputVStreamParams.make_from_network_group(network_group, quantized=False)
                network_group_params = network_group.create_params()
                
                with network_group.activate(network_group_params):
                    with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
                        input_name = network_group.get_input_vstream_infos()[0].name
                        output_name = network_group.get_output_vstream_infos()[0].name
                        
                        actual_width = self.model_info['actual_width']
                        actual_height = self.model_info['actual_height']
                        
                        print("🎯 MAXIMUM HAILO UTILIZATION inference loop starting...")
                        print(f"📝 Processing: ALL frames → {actual_width}x{actual_height} → Hailo acceleration")
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
                                
                                # Check if frame should be skipped (minimal skipping for max Hailo usage)
                                should_skip, skip_reason = self.should_skip_frame(frame_count, current_time)
                                
                                if should_skip:
                                    self.technical_logger.log_skipped_frame(skip_reason)
                                    annotated_frame, frame_metrics, detections = self.create_skipped_frame_output(frame, frame_count)
                                else:
                                    # Process frame for maximum Hailo utilization
                                    if self.enable_batching and len(self.batch_frames) < self.batch_size:
                                        # Collect frames for batch processing
                                        self.batch_frames.append((frame, frame_count))
                                        
                                        if len(self.batch_frames) < self.batch_size:
                                            continue  # Wait for full batch
                                        
                                        # Process batch
                                        frames = [f[0] for f in self.batch_frames]
                                        counts = [f[1] for f in self.batch_frames]
                                        batch_results = self.process_frame_batch(frames, infer_pipeline, input_name, output_name, counts)
                                        
                                        # Process batch results
                                        for result in batch_results:
                                            if result:
                                                annotated_frame, frame_metrics, detections = result
                                                
                                                # Create overlay
                                                if frame_count % 5 == 0:  # Reduce overlay frequency
                                                    logger_stats = self.technical_logger.get_stats()
                                                    system_stats = self.system_monitor.get_stats()
                                                    overlay_lines = self.frame_processor.create_verbose_overlay_lines(
                                                        frame_metrics, detections, self.model_info, system_stats, logger_stats
                                                    )
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
                                        
                                        self.batch_frames = []  # Clear batch
                                        continue
                                    else:
                                        # Single frame processing
                                        result = self.process_single_frame(frame, infer_pipeline, input_name, output_name, frame_count)
                                        if result is None:
                                            continue
                                        annotated_frame, frame_metrics, detections = result
                                
                                # Create overlay (reduced frequency)
                                if frame_count % 5 == 0:
                                    logger_stats = self.technical_logger.get_stats()
                                    system_stats = self.system_monitor.get_stats()
                                    overlay_lines = self.frame_processor.create_verbose_overlay_lines(
                                        frame_metrics, detections, self.model_info, system_stats, logger_stats
                                    )
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
                                
                                # Enhanced console output every 60 frames
                                if frame_count % 60 == 0:
                                    current_stats_time = time.time()
                                    
                                    logger_stats = self.technical_logger.get_stats()
                                    fps = logger_stats.get('fps', 0)
                                    avg_ms = logger_stats.get('avg_processing_ms', 0)
                                    total_dets = logger_stats.get('total_detections', 0)
                                    processed = logger_stats.get('processed_frames', 0)
                                    skipped = logger_stats.get('skipped_frames', 0)
                                    efficiency = logger_stats.get('processing_efficiency', 100)
                                    
                                    # System metrics
                                    cpu_pct = self.system_monitor.cpu_percent
                                    mem_pct = self.system_monitor.memory_percent
                                    temp = self.system_monitor.temperature
                                    
                                    # Get Hailo utilization
                                    hailo_util = self.get_hailo_utilization()
                                    
                                    # Calculate load balance
                                    if hailo_util > cpu_pct:
                                        balance_status = "🎯 OPTIMAL (Hailo > CPU)"
                                    elif hailo_util > 30:
                                        balance_status = "⚡ GOOD (Hailo Active)"
                                    else:
                                        balance_status = "⚠️ CPU-HEAVY (Low Hailo)"
                                    
                                    print(f"\n🔥 === FRAME {frame_count} HAILO UTILIZATION STATS ===")
                                    print(f"🔥 Hailo: {hailo_util:.1f}% | CPU: {cpu_pct:.1f}% | {balance_status}")
                                    print(f"⚡ Processing FPS: {fps:.1f} | Avg Time: {avg_ms:.1f}ms | Detections: {total_dets}")
                                    print(f"📈 Efficiency: {efficiency:.1f}% | Processed: {processed} | Skipped: {skipped}")
                                    print(f"🖥️ Memory: {mem_pct:.1f}% | Temp: {temp:.1f}°C | Reconnects: {self.reconnect_attempts}")
                                    print(f"🔥 Hailo NMS: {'ON' if self.use_hailo_nms else 'OFF'} | Batching: {'ON' if self.enable_batching else 'OFF'}")
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
                                time.sleep(0.1)
        
        except Exception as e:
            print(f"💥 Fatal error: {e}")
            traceback.print_exc()
            
        finally:
            print("\n🧹 Cleaning up resources...")
            self.system_monitor.stop_monitoring()
            
            # Print final performance summary
            try:
                performance_summary = self.technical_logger.get_performance_summary()
                hailo_final = self.get_hailo_utilization()
                
                print(f"\n🔥 === FINAL HAILO UTILIZATION SUMMARY ===")
                print(f"🔥 Final Hailo Usage: {hailo_final:.1f}%")
                for key, value in performance_summary.items():
                    print(f"{key.replace('_', ' ').title()}: {value}")
                
                final_stats = self.technical_logger.get_stats()
                cpu_final = self.system_monitor.cpu_percent
                
                print(f"\n🎯 OPTIMIZATION RESULTS:")
                print(f"🔥 Hailo Utilization: {hailo_final:.1f}%")
                print(f"💻 CPU Usage: {cpu_final:.1f}%")
                print(f"⚖️ Load Balance: {'OPTIMAL' if hailo_final > cpu_final else 'NEEDS IMPROVEMENT'}")
                print(f"📊 Total Frames: {final_stats.get('total_frames', 0)}")
                print(f"⚡ Average FPS: {final_stats.get('fps', 0):.2f}")
                
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
            
            print("✅ Maximum Hailo Utilization Service stopped.")


def main():
    """Main entry point optimized for Hailo utilization"""
    print("🔥 Starting MAXIMUM HAILO UTILIZATION Inference Service")
    print("🎯 Goal: Move CPU load to Hailo HAT for optimal performance")
    print("=" * 60)
    
    try:
        service = MaxHailoUtilizationInferenceService()
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
