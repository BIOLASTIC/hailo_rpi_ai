"""
Frame processing utilities for preprocessing and visualization with frame skipping support
"""
import cv2
import time
import numpy as np
from app.config.settings import settings


COCO_LABELS = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 
    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 
    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 
    27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 
    32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 
    36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 
    51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 
    57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 
    62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 
    67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 
    72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 
    77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}


class FrameProcessor:
    @staticmethod
    def preprocess_frame_with_timing(frame, target_height, target_width):
        """
        Preprocessing with detailed timing using ACTUAL model input size
        Note: target_height and target_width must match the HEF model's compiled input size
        """
        start_time = time.time()
        
        original_height, original_width = frame.shape[:2]
        
        if settings.MAINTAIN_ASPECT_RATIO:
            # Calculate aspect ratio preserving resize
            aspect_ratio = original_width / original_height
            target_aspect_ratio = target_width / target_height
            
            if aspect_ratio > target_aspect_ratio:
                # Image is wider than target, fit by width
                new_width = target_width
                new_height = int(target_width / aspect_ratio)
            else:
                # Image is taller than target, fit by height
                new_height = target_height
                new_width = int(target_height * aspect_ratio)
            
            # Resize maintaining aspect ratio
            resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            
            # Create target size image with padding (black background)
            processed_frame = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            
            # Calculate padding offsets (center the image)
            y_offset = (target_height - new_height) // 2
            x_offset = (target_width - new_width) // 2
            
            # Place resized image in center
            processed_frame[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
            
        else:
            # Direct resize without maintaining aspect ratio
            processed_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        
        end_time = time.time()
        
        return processed_frame, end_time - start_time
    
    @staticmethod
    def get_model_input_info(actual_width, actual_height):
        """Get current model input configuration info with actual model dimensions"""
        return {
            'actual_width': actual_width,
            'actual_height': actual_height,
            'configured_width': settings.MODEL_INPUT_WIDTH,
            'configured_height': settings.MODEL_INPUT_HEIGHT,
            'maintain_aspect_ratio': settings.MAINTAIN_ASPECT_RATIO,
            'input_shape': f"{actual_width}x{actual_height}x3",
            'size_override': actual_width != settings.MODEL_INPUT_WIDTH or actual_height != settings.MODEL_INPUT_HEIGHT
        }
    
    @staticmethod
    def draw_detections_with_overlay(frame, detections):
        """Draw detections with bounding boxes"""
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
    
    @staticmethod
    def create_verbose_overlay_lines(frame_metrics, detections, model_info, system_stats, logger_stats):
        """Create comprehensive technical overlay lines for video frame with frame skipping info"""
        lines = []
        
        if settings.SHOW_FRAME_DETAILS:
            lines.extend([
                f"=== FRAME DETAILS ===",
                f"Frame: #{frame_metrics.get('frame_number', 0)}",
                f"Timestamp: {frame_metrics.get('timestamp', time.time()):.3f}",
                f"Original: {frame_metrics.get('width', 0)}x{frame_metrics.get('height', 0)}",
                f"Model Input: {model_info.get('actual_width', 0)}x{model_info.get('actual_height', 0)}",
                f"Configured: {settings.MODEL_INPUT_WIDTH}x{settings.MODEL_INPUT_HEIGHT}",
                f"Size Override: {'Yes' if model_info.get('size_override', False) else 'No'}",
                f"Aspect Ratio: {'Maintained' if settings.MAINTAIN_ASPECT_RATIO else 'Stretched'}",
                f"Processing FPS: {logger_stats.get('fps', 0):.1f}",
                ""
            ])
        
        if settings.SHOW_FRAME_SKIP_STATS and settings.ENABLE_FRAME_SKIPPING:
            efficiency = logger_stats.get('processing_efficiency', 100)
            skip_pct = logger_stats.get('skip_percentage', 0)
            lines.extend([
                f"=== FRAME PROCESSING OPTIMIZATION ===",
                f"Frame Skipping: {'Enabled' if settings.ENABLE_FRAME_SKIPPING else 'Disabled'}",
                f"Process Every: {settings.PROCESS_EVERY_N_FRAMES} frames",
                f"Max FPS Limit: {settings.MAX_PROCESSING_FPS:.1f}",
                f"Processing Efficiency: {efficiency:.1f}%",
                f"Skip Rate: {skip_pct:.1f}%",
                f"Total Processed: {logger_stats.get('processed_frames', 0)}",
                f"Total Skipped: {logger_stats.get('skipped_frames', 0)}",
                f"Adaptive Skipping: {'On' if settings.ADAPTIVE_FRAME_SKIPPING else 'Off'}",
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
                f"Performance Mode: {'On' if settings.ENABLE_PERFORMANCE_MODE else 'Off'}",
                f"Power Saving: {'On' if settings.OPTIMIZE_FOR_POWER_SAVING else 'Off'}",
                ""
            ])
        
        if settings.SHOW_MODEL_DETAILS:
            lines.extend([
                f"=== MODEL INFO ===",
                f"Model: {model_info.get('name', 'Unknown')}",
                f"HEF Input: {model_info.get('actual_width', 0)}x{model_info.get('actual_height', 0)}",
                f"Config Input: {settings.MODEL_INPUT_WIDTH}x{settings.MODEL_INPUT_HEIGHT}",
                f"Output: {model_info.get('output_shape', 'N/A')}",
                f"Confidence: {settings.CONFIDENCE_THRESHOLD}",
                f"Classes: {model_info.get('num_classes', 80)}",
                f"Low Latency: {'On' if settings.LOW_LATENCY_MODE else 'Off'}",
                ""
            ])
        
        if settings.SHOW_DETECTION_DETAILS:
            avg_detections = logger_stats.get('detections_per_processed_frame', 0)
            lines.extend([
                f"=== DETECTIONS ({len(detections)}) ===",
                f"Avg per Frame: {avg_detections:.1f}",
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
            cpu_pct = system_stats.get('cpu_percent', 0)
            mem_pct = system_stats.get('memory_percent', 0)
            temp = system_stats.get('temperature', 0)
            
            # Add performance indicators
            cpu_status = "HIGH" if cpu_pct > settings.CPU_THRESHOLD_FOR_SKIPPING else "OK"
            mem_status = "HIGH" if mem_pct > settings.MEMORY_THRESHOLD_FOR_SKIPPING else "OK"
            
            lines.extend([
                f"=== SYSTEM METRICS ===",
                f"CPU: {cpu_pct:.1f}% ({cpu_status})",
                f"Memory: {mem_pct:.1f}% ({mem_status})",
                f"Temp: {temp:.1f}°C",
                f"CPU Threshold: {settings.CPU_THRESHOLD_FOR_SKIPPING:.1f}%",
                f"Memory Threshold: {settings.MEMORY_THRESHOLD_FOR_SKIPPING:.1f}%",
                f"Uptime: {logger_stats.get('uptime_seconds', 0):.0f}s",
                f"Total Frames: {logger_stats.get('total_frames', 0)}",
                f"Total Detections: {logger_stats.get('total_detections', 0)}",
            ])
        
        return lines
    
    @staticmethod
    def draw_verbose_technical_overlay(frame, overlay_lines):
        """Draw comprehensive technical overlay on video frame"""
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
            x_start, y_start = frame.shape[1] - 450, 20
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
            elif 'HIGH' in line or 'Override: Yes' in line:
                color = (0, 165, 255)  # Orange for warnings
            elif 'ERROR' in line or 'FATAL' in line:
                color = (0, 0, 255)    # Red for errors
            elif any(keyword in line for keyword in ['SUCCESS', 'DETECTION', 'Found', 'On', 'Enabled']):
                color = (0, 255, 0)    # Green for success/enabled
            elif any(keyword in line for keyword in ['Off', 'Disabled']):
                color = (128, 128, 128)  # Gray for disabled
            else:
                color = (255, 255, 255)  # White for regular text
            
            # Draw the text
            cv2.putText(frame, line, (x_start, y), font, font_scale, color, thickness, cv2.LINE_AA)
        
        return frame
    
    @staticmethod
    def should_update_overlay(frame_count):
        """Determine if overlay should be updated based on settings"""
        if not settings.REDUCE_OVERLAY_FREQUENCY:
            return True
        return frame_count % settings.OVERLAY_UPDATE_EVERY_N_FRAMES == 0
