import time
import cv2
import redis
import numpy as np
import psutil
import threading
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
import traceback
from collections import deque

try:
    from hailo_platform import VDevice, HEF, InferVStreams, InputVStreamParams, OutputVStreamParams
except ImportError as e:
    print(f"[FATAL] HailoRT import error: {e}")
    exit(1)

class VerboseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore', case_sensitive=False)
    
    # Existing settings
    REDIS_HOST: str = 'localhost'
    REDIS_PORT: int = 6379
    RAW_FRAME_INPUT_CHANNEL: str = 'camera:frames:rpi'
    ANNOTATED_FRAME_OUTPUT_CHANNEL: str = 'ai_stream:frames:rpi'
    AI_MODEL_PATH: str = 'yolov8s.hef'
    CONFIDENCE_THRESHOLD: float = 0.25
    JPEG_QUALITY: int = 60
    
    # Verbose logging settings
    ENABLE_VERBOSE_OVERLAY: bool = True
    OVERLAY_POSITION: str = 'TOP_LEFT'
    OVERLAY_FONT_SIZE: float = 0.4
    OVERLAY_BACKGROUND: bool = True
    SHOW_FRAME_DETAILS: bool = True
    SHOW_MODEL_DETAILS: bool = True
    SHOW_PROCESSING_METRICS: bool = True
    SHOW_DETECTION_DETAILS: bool = True
    SHOW_SYSTEM_METRICS: bool = True
    MAX_DETECTION_LINES: int = 8
    OVERLAY_UPDATE_INTERVAL: int = 1

settings = VerboseSettings()

COCO_LABELS = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}

class SystemMonitor:
    def __init__(self):
        self.cpu_percent = 0
        self.memory_percent = 0
        self.temperature = 0
        self.running = False
        self.thread = None
    
    def start_monitoring(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
    
    def _monitor_loop(self):
        while self.running:
            try:
                self.cpu_percent = psutil.cpu_percent(interval=1)
                self.memory_percent = psutil.virtual_memory().percent
                
                # Try to get temperature (Raspberry Pi)
                try:
                    with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                        self.temperature = int(f.read()) / 1000.0
                except:
                    self.temperature = 0
                    
            except Exception as e:
                print(f"[MONITOR] Error: {e}")
            
            time.sleep(1)
    
    def stop_monitoring(self):
        self.running = False

class TechnicalLogger:
    def __init__(self):
        self.frame_times = deque(maxlen=30)
        self.processing_times = deque(maxlen=30)
        self.inference_times = deque(maxlen=30)
        self.encoding_times = deque(maxlen=30)
        self.total_frames = 0
        self.total_detections = 0
        self.start_time = time.time()
        
    def log_frame_metrics(self, processing_time, inference_time=None, encoding_time=None):
        current_time = time.time()
        self.frame_times.append(current_time)
        self.processing_times.append(processing_time)
        
        if inference_time:
            self.inference_times.append(inference_time)
        if encoding_time:
            self.encoding_times.append(encoding_time)
            
        self.total_frames += 1
    
    def log_detections(self, count):
        self.total_detections += count
    
    def get_stats(self):
        if len(self.frame_times) < 2:
            return {}
        
        fps = len(self.frame_times) / (self.frame_times[-1] - self.frame_times[0])
        avg_processing = np.mean(self.processing_times) * 1000
        avg_inference = np.mean(self.inference_times) * 1000 if self.inference_times else 0
        avg_encoding = np.mean(self.encoding_times) * 1000 if self.encoding_times else 0
        
        return {
            'fps': fps,
            'avg_processing_ms': avg_processing,
            'avg_inference_ms': avg_inference,
            'avg_encoding_ms': avg_encoding,
            'total_frames': self.total_frames,
            'total_detections': self.total_detections,
            'uptime_seconds': time.time() - self.start_time
        }

def create_verbose_overlay_lines(frame_metrics, detections, model_info, system_stats, logger_stats):
    """Create comprehensive technical overlay lines for video frame."""
    lines = []
    
    if settings.SHOW_FRAME_DETAILS:
        lines.extend([
            f"=== FRAME DETAILS ===",
            f"Frame: #{frame_metrics.get('frame_number', 0)}",
            f"Timestamp: {frame_metrics.get('timestamp', time.time()):.3f}",
            f"Resolution: {frame_metrics.get('width', 0)}x{frame_metrics.get('height', 0)}",
            f"FPS: {logger_stats.get('fps', 0):.1f}",
            ""
        ])
    
    if settings.SHOW_PROCESSING_METRICS:
        lines.extend([
            f"=== PROCESSING METRICS ===",
            f"Frame Process: {frame_metrics.get('total_time', 0)*1000:.1f}ms",
            f"Preprocessing: {frame_metrics.get('preprocess_time', 0)*1000:.1f}ms", 
            f"Inference: {frame_metrics.get('inference_time', 0)*1000:.1f}ms",
            f"Postprocess: {frame_metrics.get('postprocess_time', 0)*1000:.1f}ms",
            f"Encoding: {frame_metrics.get('encoding_time', 0)*1000:.1f}ms",
            f"Avg Process: {logger_stats.get('avg_processing_ms', 0):.1f}ms",
            ""
        ])
    
    if settings.SHOW_MODEL_DETAILS:
        lines.extend([
            f"=== MODEL INFO ===",
            f"Model: {model_info.get('name', 'Unknown')}",
            f"Input: {model_info.get('input_shape', 'N/A')}",
            f"Output: {model_info.get('output_shape', 'N/A')}",
            f"Confidence: {settings.CONFIDENCE_THRESHOLD}",
            f"Classes: {model_info.get('num_classes', 80)}",
            ""
        ])
    
    if settings.SHOW_DETECTION_DETAILS:
        lines.extend([
            f"=== DETECTIONS ({len(detections)}) ===",
        ])
        
        detection_count = min(len(detections), settings.MAX_DETECTION_LINES)
        for i in range(detection_count):
            det = detections[i]
            class_name = COCO_LABELS.get(det['class_id'], f"cls_{det['class_id']}")
            bbox = det['bbox']
            conf = det['confidence']
            box_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            
            lines.append(f"{i+1:2d}: {class_name} ({conf:.3f})")
            lines.append(f"    Box: [{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}]")
            lines.append(f"    Area: {box_area}px²")
        
        if len(detections) > settings.MAX_DETECTION_LINES:
            lines.append(f"    ... +{len(detections) - settings.MAX_DETECTION_LINES} more")
        lines.append("")
    
    if settings.SHOW_SYSTEM_METRICS:
        lines.extend([
            f"=== SYSTEM METRICS ===",
            f"CPU: {system_stats.get('cpu_percent', 0):.1f}%",
            f"Memory: {system_stats.get('memory_percent', 0):.1f}%",
            f"Temp: {system_stats.get('temperature', 0):.1f}°C",
            f"Uptime: {logger_stats.get('uptime_seconds', 0):.0f}s",
            f"Total Frames: {logger_stats.get('total_frames', 0)}",
            f"Total Detections: {logger_stats.get('total_detections', 0)}",
        ])
    
    return lines

def draw_verbose_technical_overlay(frame, overlay_lines):
    """Draw comprehensive technical overlay on video frame."""
    if not settings.ENABLE_VERBOSE_OVERLAY or not overlay_lines:
        return frame
    
    # Configuration
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = settings.OVERLAY_FONT_SIZE
    thickness = 1
    line_spacing = int(15 * font_scale) + 8
    
    # Position settings
    if settings.OVERLAY_POSITION == 'TOP_LEFT':
        x_start, y_start = 10, 20
    elif settings.OVERLAY_POSITION == 'TOP_RIGHT':
        x_start, y_start = frame.shape[1] - 400, 20
    else:
        x_start, y_start = 10, 20
    
    # Draw each line
    for i, line in enumerate(overlay_lines):
        y = y_start + i * line_spacing
        
        # Skip if line would be off-screen
        if y > frame.shape[0] - 10:
            break
        
        # Get text size for background
        (text_width, text_height), baseline = cv2.getTextSize(line, font, font_scale, thickness)
        
        # Draw background rectangle for readability
        if settings.OVERLAY_BACKGROUND:
            padding = 3
            cv2.rectangle(
                frame, 
                (x_start - padding, y - text_height - padding),
                (x_start + text_width + padding, y + baseline + padding),
                (0, 0, 0, 180),  # Semi-transparent black
                -1
            )
        
        # Choose text color based on line type
        if line.startswith('==='):
            color = (0, 255, 255)  # Yellow for headers
        elif 'ERROR' in line or 'FATAL' in line:
            color = (0, 0, 255)    # Red for errors
        elif any(keyword in line for keyword in ['SUCCESS', 'DETECTION', 'Found']):
            color = (0, 255, 0)    # Green for success
        else:
            color = (255, 255, 255)  # White for regular text
        
        # Draw the text
        cv2.putText(frame, line, (x_start, y), font, font_scale, color, thickness, cv2.LINE_AA)
    
    return frame

def preprocess_frame_with_timing(frame, target_height=640, target_width=640):
    """Preprocessing with detailed timing."""
    start_time = time.time()
    resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    end_time = time.time()
    return resized, end_time - start_time

def postprocess_hailo_detections_with_timing(raw_detections, original_height, original_width, model_height, model_width, confidence_threshold=0.25):
    """Post-processing with detailed timing and logging."""
    start_time = time.time()
    detections = []
    
    if not raw_detections or len(raw_detections) == 0:
        return [], time.time() - start_time
    
    try:
        detection_batch = raw_detections[0]
        
        if isinstance(detection_batch, list) and len(detection_batch) == 80:
            for class_id, class_detections in enumerate(detection_batch):
                if hasattr(class_detections, 'shape') and class_detections.shape[0] > 0:
                    for detection in class_detections:
                        if len(detection) >= 5:
                            x_center, y_center, width, height, confidence = detection[:5]
                            
                            if confidence < confidence_threshold:
                                continue
                            
                            # Convert to pixel coordinates
                            x_center_px = x_center * original_width
                            y_center_px = y_center * original_height
                            width_px = width * original_width
                            height_px = height * original_height
                            
                            # Convert to corner format
                            x_min = int(x_center_px - width_px / 2)
                            y_min = int(y_center_px - height_px / 2)
                            x_max = int(x_center_px + width_px / 2)
                            y_max = int(y_center_px + height_px / 2)
                            
                            # Clamp to image bounds
                            x_min = max(0, min(x_min, original_width - 1))
                            y_min = max(0, min(y_min, original_height - 1))
                            x_max = max(x_min + 1, min(x_max, original_width))
                            y_max = max(y_min + 1, min(y_max, original_height))
                            
                            # Validate box size
                            if (x_max - x_min) >= 10 and (y_max - y_min) >= 10:
                                detections.append({
                                    'bbox': [x_min, y_min, x_max, y_max],
                                    'confidence': float(confidence),
                                    'class_id': int(class_id)
                                })
    
    except Exception as e:
        print(f"[ERROR] Postprocessing error: {e}")
    
    return detections, time.time() - start_time

def draw_detections_with_overlay(frame, detections):
    """Draw detections with bounding boxes."""
    for detection in detections:
        x_min, y_min, x_max, y_max = detection['bbox']
        class_id = detection['class_id']
        confidence = detection['confidence']
        
        # Color based on confidence
        if confidence > 0.7:
            color = (0, 255, 0)    # Green
        elif confidence > 0.5:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 165, 255)  # Orange
        
        # Draw bounding box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
        
        # Draw label
        label = COCO_LABELS.get(class_id, f"class_{class_id}")
        label_text = f"{label} {confidence:.2f}"
        
        # Position label
        (label_width, label_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        # Draw label background
        cv2.rectangle(
            frame,
            (x_min, y_min - label_height - 10),
            (x_min + label_width, y_min),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            frame, label_text, (x_min, y_min - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
    
    return frame

def main():
    print("🚀 === ULTRA-VERBOSE Hailo Inference Service ===")
    print(f"📊 Verbose Overlay: {settings.ENABLE_VERBOSE_OVERLAY}")
    print(f"📍 Overlay Position: {settings.OVERLAY_POSITION}")
    print(f"🔍 Max Detection Lines: {settings.MAX_DETECTION_LINES}")
    
    # Initialize monitoring systems
    system_monitor = SystemMonitor()
    system_monitor.start_monitoring()
    
    technical_logger = TechnicalLogger()
    
    try:
        # Setup Redis
        redis_client = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT)
        redis_client.ping()
        print("✅ Redis connected with verbose logging enabled")
        
        # Setup Hailo
        with VDevice() as vdevice:
            hef = HEF(settings.AI_MODEL_PATH)
            network_groups = vdevice.configure(hef)
            network_group = network_groups[0]
            
            # Get model info
            input_vstream_info = hef.get_input_vstream_infos()[0]
            output_vstream_info = hef.get_output_vstream_infos()[0]
            model_height, model_width = input_vstream_info.shape[0], input_vstream_info.shape[1]
            
            model_info = {
                'name': Path(settings.AI_MODEL_PATH).stem,
                'input_shape': f"{model_width}x{model_height}x{input_vstream_info.shape[2]}",
                'output_shape': str(output_vstream_info.shape),
                'num_classes': 80
            }
            
            print(f"📐 Model: {model_info['name']} | Input: {model_info['input_shape']}")
            
            # Setup inference pipeline
            pubsub = redis_client.pubsub()
            pubsub.subscribe(settings.RAW_FRAME_INPUT_CHANNEL)
            
            input_vstreams_params = InputVStreamParams.make_from_network_group(network_group, quantized=False)
            output_vstreams_params = OutputVStreamParams.make_from_network_group(network_group, quantized=False)
            network_group_params = network_group.create_params()
            
            with network_group.activate(network_group_params):
                with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
                    input_name = network_group.get_input_vstream_infos()[0].name
                    output_name = network_group.get_output_vstream_infos()[0].name
                    
                    print("🎯 Ultra-verbose inference loop starting...")
                    print(f"📝 Overlay shows: Frame details, Processing metrics, Model info, Detection details, System stats")
                    
                    frame_count = 0
                    
                    while True:
                        message = pubsub.get_message(ignore_subscribe_messages=True, timeout=0.1)
                        if not message:
                            continue
                        
                        frame_start_time = time.time()
                        frame_count += 1
                        
                        # Decode frame
                        try:
                            decode_start = time.time()
                            np_array = np.frombuffer(message['data'], np.uint8)
                            frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
                            decode_time = time.time() - decode_start
                            
                            if frame is None:
                                continue
                        except Exception:
                            continue
                        
                        original_height, original_width = frame.shape[:2]
                        
                        # Preprocessing with timing
                        preprocessed_frame, preprocess_time = preprocess_frame_with_timing(
                            frame, model_height, model_width
                        )
                        
                        # Inference with timing
                        inference_start = time.time()
                        input_data = {input_name: np.expand_dims(preprocessed_frame, axis=0)}
                        
                        try:
                            results = infer_pipeline.infer(input_data)
                            raw_detections = results[output_name]
                            inference_time = time.time() - inference_start
                        except Exception:
                            continue
                        
                        # Post-processing with timing
                        detections, postprocess_time = postprocess_hailo_detections_with_timing(
                            raw_detections, original_width, original_height, model_height, model_width
                        )
                        
                        # Draw detections
                        annotated_frame = draw_detections_with_overlay(frame.copy(), detections)
                        
                        # Encoding with timing
                        encoding_start = time.time()
                        ret, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), settings.JPEG_QUALITY])
                        encoding_time = time.time() - encoding_start
                        
                        # Calculate total processing time
                        total_processing_time = time.time() - frame_start_time
                        
                        # Prepare frame metrics
                        frame_metrics = {
                            'frame_number': frame_count,
                            'timestamp': frame_start_time,
                            'width': original_width,
                            'height': original_height,
                            'total_time': total_processing_time,
                            'decode_time': decode_time,
                            'preprocess_time': preprocess_time,
                            'inference_time': inference_time,
                            'postprocess_time': postprocess_time,
                            'encoding_time': encoding_time
                        }
                        
                        # Update loggers
                        technical_logger.log_frame_metrics(total_processing_time, inference_time, encoding_time)
                        technical_logger.log_detections(len(detections))
                        
                        # Get current stats
                        logger_stats = technical_logger.get_stats()
                        system_stats = {
                            'cpu_percent': system_monitor.cpu_percent,
                            'memory_percent': system_monitor.memory_percent,
                            'temperature': system_monitor.temperature
                        }
                        
                        # Create verbose overlay
                        overlay_lines = create_verbose_overlay_lines(
                            frame_metrics, detections, model_info, system_stats, logger_stats
                        )
                        
                        # Draw technical overlay
                        final_frame = draw_verbose_technical_overlay(annotated_frame, overlay_lines)
                        
                        # Re-encode with overlay
                        ret, buffer = cv2.imencode('.jpg', final_frame, [int(cv2.IMWRITE_JPEG_QUALITY), settings.JPEG_QUALITY])
                        
                        if ret:
                            redis_client.publish(settings.ANNOTATED_FRAME_OUTPUT_CHANNEL, buffer.tobytes())
                        
                        # Console output every 60 frames
                        if frame_count % 60 == 0:
                            fps = logger_stats.get('fps', 0)
                            avg_ms = logger_stats.get('avg_processing_ms', 0)
                            total_dets = logger_stats.get('total_detections', 0)
                            print(f"📊 Frame {frame_count}: {fps:.1f} FPS | {avg_ms:.1f}ms avg | {total_dets} total detections")
    
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        traceback.print_exc()
    finally:
        system_monitor.stop_monitoring()

if __name__ == "__main__":
    main()
