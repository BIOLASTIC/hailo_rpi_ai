import time
import cv2
import redis
import numpy as np
import json
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
import traceback

try:
    from hailo_platform import VDevice, HEF, InferVStreams, InputVStreamParams, OutputVStreamParams, FormatType, FormatOrder
except ImportError as e:
    print("="*80)
    print("[FATAL] A critical HailoRT library class could not be imported.")
    print("       This indicates a potentially corrupted SDK installation or an incorrect venv setup.")
    print(f"       Original Error: {e}")
    print("="*80)
    exit(1)

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore', case_sensitive=False)
    REDIS_HOST: str
    REDIS_PORT: int
    RAW_FRAME_INPUT_CHANNEL: str
    ANNOTATED_FRAME_OUTPUT_CHANNEL: str
    AI_MODEL_PATH: str

settings = Settings()

COCO_LABELS = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}

def preprocess_frame_bgr(frame, target_height=640, target_width=640):
    """
    Preprocessing for Hailo YOLOv8 - BGR format, UINT8, no normalization.
    """
    resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    return resized

def preprocess_frame_rgb(frame, target_height=640, target_width=640):
    """
    Preprocessing for Hailo YOLOv8 - RGB format, UINT8, no normalization.
    """
    resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb_frame

def debug_input_output(input_data, results, frame_num):
    """Debug input and output data structures."""
    print(f"\n=== FRAME {frame_num} DEBUG ===")
    
    # Debug input
    for name, data in input_data.items():
        print(f"Input '{name}': shape={data.shape}, dtype={data.dtype}")
        print(f"  Value range: [{data.min()}, {data.max()}]")
        print(f"  Mean: {data.mean():.2f}, Std: {data.std():.2f}")
        
        if data.max() == 0 and data.min() == 0:
            print(f"  ⚠️  WARNING: Input data is all zeros!")
        else:
            print(f"  ✅ Input data looks valid")
    
    # Debug output - handle FLOAT32 NMS format
    for name, data in results.items():
        print(f"Output '{name}': type={type(data)}")
        if isinstance(data, list) and len(data) > 0:
            batch = data[0]
            print(f"  Batch: type={type(batch)}, length={len(batch)}")
            
            # Count detections per class
            detection_summary = []
            total_detections = 0
            
            for class_id, class_detections in enumerate(batch):
                if hasattr(class_detections, 'shape') and class_detections.shape[0] > 0:
                    num_dets = class_detections.shape[0]
                    total_detections += num_dets
                    class_name = COCO_LABELS.get(class_id, f"class_{class_id}")
                    detection_summary.append(f"{class_name}: {num_dets}")
                    
                    # Show sample detection for debugging
                    if len(detection_summary) <= 3:
                        sample_det = class_detections[0]
                        print(f"    {class_name} sample: {sample_det}")
            
            print(f"  Total detections: {total_detections}")
            if detection_summary:
                print(f"  Classes found: {', '.join(detection_summary[:5])}")
    
    print("=== END DEBUG ===\n")

def postprocess_hailo_nms_detections(raw_detections, original_height, original_width, model_height, model_width, confidence_threshold=0.25):
    """
    Process Hailo NMS BY_CLASS output format (FLOAT32).
    The model already applies NMS with 0.2 threshold, so we use slightly higher threshold.
    """
    detections = []
    
    if not raw_detections or len(raw_detections) == 0:
        return []
    
    try:
        # Extract detection batch - should be list of 80 class arrays
        detection_batch = raw_detections[0]
        
        if isinstance(detection_batch, list) and len(detection_batch) == 80:
            for class_id, class_detections in enumerate(detection_batch):
                
                # Check if this class has detections
                if hasattr(class_detections, 'shape') and class_detections.shape[0] > 0:
                    
                    # Process each detection for this class
                    for detection in class_detections:
                        if len(detection) >= 5:
                            # Hailo NMS format: [x_min, y_min, x_max, y_max, confidence]
                            # OR: [x_center, y_center, width, height, confidence]
                            # Let's handle both formats
                            
                            x1, y1, x2, y2, confidence = detection[:5]
                            
                            # Apply additional confidence filtering
                            if confidence < confidence_threshold:
                                continue
                            
                            # Determine if coordinates are normalized or absolute
                            if x2 <= 1.0 and y2 <= 1.0:
                                # Normalized coordinates (0-1)
                                if x2 - x1 < 0.1 and y2 - y1 < 0.1:
                                    # Likely center format
                                    x_center, y_center, width, height = x1, y1, x2, y2
                                    x_min = int((x_center - width/2) * original_width)
                                    y_min = int((y_center - height/2) * original_height)
                                    x_max = int((x_center + width/2) * original_width)
                                    y_max = int((y_center + height/2) * original_height)
                                else:
                                    # Corner format
                                    x_min = int(x1 * original_width)
                                    y_min = int(y1 * original_height)
                                    x_max = int(x2 * original_width)
                                    y_max = int(y2 * original_height)
                            else:
                                # Absolute coordinates relative to model size
                                x_min = int(x1 * original_width / model_width)
                                y_min = int(y1 * original_height / model_height)
                                x_max = int(x2 * original_width / model_width)
                                y_max = int(y2 * original_height / model_height)
                            
                            # Ensure proper coordinate order
                            if x_min > x_max:
                                x_min, x_max = x_max, x_min
                            if y_min > y_max:
                                y_min, y_max = y_max, y_min
                            
                            # Clamp coordinates to image bounds
                            x_min = max(0, min(x_min, original_width - 1))
                            y_min = max(0, min(y_min, original_height - 1))
                            x_max = max(x_min + 1, min(x_max, original_width))
                            y_max = max(y_min + 1, min(y_max, original_height))
                            
                            # Validate bounding box size
                            box_width = x_max - x_min
                            box_height = y_max - y_min
                            if box_width < 10 or box_height < 10:
                                continue
                                
                            # Skip boxes that are too large (likely false positives)
                            if box_width > original_width * 0.9 or box_height > original_height * 0.9:
                                continue
                            
                            detections.append({
                                'bbox': [x_min, y_min, x_max, y_max],
                                'confidence': float(confidence),
                                'class_id': int(class_id)
                            })
                            
                            class_name = COCO_LABELS.get(class_id, f"class_{class_id}")
                            print(f"🎯 [DETECTION] {class_name} ({confidence:.3f}) at [{x_min},{y_min},{x_max},{y_max}]")
    
    except Exception as e:
        print(f"[ERROR] Error in postprocess_hailo_nms_detections: {e}")
        traceback.print_exc()
    
    return detections

def draw_detections(frame, detections):
    """Draw detection results with enhanced visualization."""
    for detection in detections:
        x_min, y_min, x_max, y_max = detection['bbox']
        class_id = detection['class_id']
        confidence = detection['confidence']
        
        # Get class label
        label = COCO_LABELS.get(class_id, f"class_{class_id}")
        
        # Color coding based on confidence
        if confidence > 0.8:
            color = (0, 255, 0)    # Bright green for high confidence
        elif confidence > 0.6:
            color = (0, 255, 255)  # Yellow for medium confidence
        elif confidence > 0.4:
            color = (0, 165, 255)  # Orange for lower confidence
        else:
            color = (0, 100, 255)  # Red for low confidence
        
        # Draw thick bounding box
        thickness = 3
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, thickness)
        
        # Prepare label text
        label_text = f"{label} {confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        text_thickness = 2
        
        # Get text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, font, font_scale, text_thickness
        )
        
        # Draw label background
        label_y = y_min - 10
        if label_y < text_height + 10:
            label_y = y_max + text_height + 10
        
        cv2.rectangle(
            frame,
            (x_min, label_y - text_height - 5),
            (x_min + text_width + 10, label_y + 5),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            frame,
            label_text,
            (x_min + 5, label_y - 5),
            font,
            font_scale,
            (255, 255, 255),
            text_thickness
        )
    
    return frame

def main():
    print("🚀 === FINAL Hailo YOLOv8 Inference Service ===")
    print("📋 Model Info: HAILO8L, YOLOv8m, Built-in NMS (threshold: 0.2)")
    
    try:
        # Connect to Redis
        redis_client = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT)
        redis_client.ping()
        print(f"✅ Redis connection successful to {settings.REDIS_HOST}:{settings.REDIS_PORT}")
    except redis.exceptions.ConnectionError as e:
        print(f"❌ [FATAL] Could not connect to Redis: {e}")
        return

    # Verify model file
    hef_path = Path(settings.AI_MODEL_PATH)
    if not hef_path.exists():
        print(f"❌ [FATAL] AI Model file not found at: {hef_path.resolve()}")
        return
    
    print(f"✅ Model file found: {hef_path} ({hef_path.stat().st_size} bytes)")

    try:
        print("🔧 Initializing Hailo VDevice...")
        with VDevice() as vdevice:
            # Load and configure HEF
            hef = HEF(settings.AI_MODEL_PATH)
            network_groups = vdevice.configure(hef)
            network_group = network_groups[0]
            
            # Get model information
            input_vstream_info = hef.get_input_vstream_infos()[0]
            output_vstream_info = hef.get_output_vstream_infos()[0]
            
            model_height, model_width = input_vstream_info.shape[0], input_vstream_info.shape[1]
            print(f"📐 Model input: {model_width}x{model_height}x{input_vstream_info.shape[2]} ({input_vstream_info.format})")
            print(f"📤 Model output: {output_vstream_info.shape} ({output_vstream_info.format})")

            # Setup Redis subscription
            pubsub = redis_client.pubsub()
            pubsub.subscribe(settings.RAW_FRAME_INPUT_CHANNEL)
            print(f"📺 Subscribed to: '{settings.RAW_FRAME_INPUT_CHANNEL}'")
            print(f"📡 Publishing to: '{settings.ANNOTATED_FRAME_OUTPUT_CHANNEL}'")

            # Create VStream parameters
            input_vstreams_params = InputVStreamParams.make_from_network_group(network_group, quantized=False)
            output_vstreams_params = OutputVStreamParams.make_from_network_group(network_group, quantized=False)
            network_group_params = network_group.create_params()

            with network_group.activate(network_group_params):
                print("⚡ Network Group activated successfully")
                
                with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
                    print("🎯 InferVStreams pipeline created successfully")
                    print("🔄 Starting inference loop...\n")
                    
                    input_name = network_group.get_input_vstream_infos()[0].name
                    output_name = network_group.get_output_vstream_infos()[0].name

                    frame_count = 0
                    detection_count = 0
                    start_time = time.time()
                    use_rgb = False  # Start with BGR, switch to RGB if no detections
                    
                    while True:
                        message = pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                        if not message: 
                            continue

                        frame_count += 1
                        
                        # Decode frame from Redis
                        try:
                            np_array = np.frombuffer(message['data'], np.uint8)
                            frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
                            if frame is None:
                                continue
                        except Exception as e:
                            print(f"❌ Frame decode error: {e}")
                            continue
                        
                        original_height, original_width = frame.shape[:2]
                        
                        # Adaptive preprocessing - try RGB if BGR isn't working
                        if frame_count <= 100 and detection_count == 0 and frame_count % 50 == 0:
                            use_rgb = not use_rgb
                            format_name = "RGB" if use_rgb else "BGR"
                            print(f"🔄 Switching to {format_name} format at frame {frame_count}")
                        
                        # Preprocess frame
                        if use_rgb:
                            preprocessed_frame = preprocess_frame_rgb(frame, model_height, model_width)
                        else:
                            preprocessed_frame = preprocess_frame_bgr(frame, model_height, model_width)
                        
                        # Prepare input
                        input_data = {input_name: np.expand_dims(preprocessed_frame, axis=0)}
                        
                        try:
                            # Run inference
                            results = infer_pipeline.infer(input_data)
                            raw_detections = results[output_name]
                            
                            # Debug first few frames
                            if frame_count <= 3:
                                debug_input_output(input_data, results, frame_count)
                            
                            # Post-process detections
                            detections = postprocess_hailo_nms_detections(
                                raw_detections, original_height, original_width, model_height, model_width
                            )
                            
                            detection_count += len(detections)
                            
                            # Draw detections on frame
                            annotated_frame = draw_detections(frame.copy(), detections)
                            
                            # Add status overlay
                            elapsed_time = time.time() - start_time
                            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                            format_text = "RGB" if use_rgb else "BGR"
                            
                            status_lines = [
                                f"Frame: {frame_count} | FPS: {fps:.1f} | Format: {format_text}",
                                f"Detections: {len(detections)} | Total: {detection_count}",
                                f"Resolution: {original_width}x{original_height} -> {model_width}x{model_height}"
                            ]
                            
                            for i, status_text in enumerate(status_lines):
                                y_pos = 30 + (i * 30)
                                # Black outline
                                cv2.putText(annotated_frame, status_text, (10, y_pos), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                                # White text
                                cv2.putText(annotated_frame, status_text, (10, y_pos), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            
                            # Encode and publish result
                            ret, buffer = cv2.imencode('.jpg', annotated_frame, 
                                                     [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                            if ret:
                                redis_client.publish(settings.ANNOTATED_FRAME_OUTPUT_CHANNEL, buffer.tobytes())
                            else:
                                print(f"⚠️  Failed to encode frame {frame_count}")
                            
                            # Success notifications
                            if len(detections) > 0:
                                print(f"\n🎉 SUCCESS! Frame {frame_count}: Found {len(detections)} detections!")
                                for det in detections:
                                    class_name = COCO_LABELS.get(det['class_id'], f"class_{det['class_id']}")
                                    print(f"  ✅ {class_name}: {det['confidence']:.3f}")
                                print()
                            
                            # Periodic status
                            if frame_count % 60 == 0:
                                print(f"📊 Frame {frame_count}: {detection_count} total detections, {fps:.1f} FPS, using {format_text}")
                                
                        except Exception as e:
                            print(f"❌ Inference error on frame {frame_count}: {e}")
                            if frame_count <= 5:
                                traceback.print_exc()
                            continue

    except Exception as e:
        print(f"💥 [FATAL] Main loop error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
