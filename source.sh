import time
import cv2
import redis
import numpy as np
import json
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
import traceback

#
# --- THE FINAL, CORRECTED IMPORT BASED ON YOUR SYSTEM'S OUTPUT ---
# The class names are VDevice and HEF (all uppercase).
#
try:
    from hailo_platform import VDevice, HEF
except ImportError as e:
    print("="*80)
    print("[FATAL] A HailoRT library class could not be imported.")
    print(f"       Original Error: {e}")
    print("       This can indicate a corrupted SDK installation. Please try re-running")
    print("       the official Hailo installation scripts.")
    print("="*80)
    exit(1)

# --- Configuration ---
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore', case_sensitive=False)
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    AI_MODEL_PATH: str = "yolov8m.hef"
    AI_INPUT_CAMERA_ID: str = "usb"

settings = Settings()

# --- Constants ---
RAW_FRAME_CHANNEL = f"camera:frames:{settings.AI_INPUT_CAMERA_ID}"
AI_FRAME_CHANNEL = f"ai_stream:frames:{settings.AI_INPUT_CAMERA_ID}"
AI_DETECTIONS_CHANNEL = "ai:detections"
MODEL_NAME = Path(settings.AI_MODEL_PATH).stem

def draw_detections(frame, detections, labels):
    """Draws bounding boxes and labels on the frame."""
    for detection in detections:
        box = detection.get_bbox()
        label_id = detection.get_label()
        
        if label_id not in labels:
            print(f"[WARNING] Detected object with unknown label ID: {label_id}")
            continue

        label = labels[label_id]
        confidence = detection.get_confidence()
        
        cv2.rectangle(frame, (int(box.xmin()), int(box.ymin())),
                      (int(box.xmax()), int(box.ymax())), (0, 255, 0), 2)
        
        label_text = f"{label} ({confidence:.2f})"
        (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        cv2.rectangle(frame, (int(box.xmin()), int(box.ymin()) - text_height - 10),
                      (int(box.xmin()) + text_width, int(box.ymin())), (0, 255, 0), -1)
        
        cv2.putText(frame, label_text, (int(box.xmin()), int(box.ymin()) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return frame

def main():
    print("--- Hailo AI Inference Service Starting ---")
    
    try:
        redis_client = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT)
        redis_client.ping()
        print(f"[INFO] Redis connection successful to {settings.REDIS_HOST}:{settings.REDIS_PORT}")
    except redis.exceptions.ConnectionError as e:
        print(f"[FATAL] Could not connect to Redis: {e}")
        return

    hef_path = Path(settings.AI_MODEL_PATH)
    if not hef_path.exists():
        print("="*80)
        print(f"[FATAL] AI Model file not found at: {hef_path.resolve()}")
        print("="*80)
        return

    try:
        print("[INFO] Initializing Hailo VDevice...")
        with VDevice() as vdevice:
            # The correct logic: create HEF object first, using the uppercase class name.
            print(f"[INFO] Reading HEF file from: {settings.AI_MODEL_PATH}")
            hef = HEF(settings.AI_MODEL_PATH)
            
            # Then pass the object to configure.
            print("[INFO] Configuring VDevice with the loaded HEF...")
            network_groups = vdevice.configure(hef)
            network_group = network_groups[0]
            
            labels = network_group.get_labels()
            print(f"[INFO] Model '{MODEL_NAME}' loaded successfully.")
            
            pubsub = redis_client.pubsub()
            pubsub.subscribe(RAW_FRAME_CHANNEL)
            print(f"[INFO] Subscribed to Redis channel: '{RAW_FRAME_CHANNEL}'")
            
            with network_group.activate() as activated_network_group:
                print("[INFO] Network Group activated. Starting inference loop...")
                while True:
                    message = pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                    if not message:
                        continue

                    jpeg_bytes = message['data']
                    np_array = np.frombuffer(jpeg_bytes, np.uint8)
                    frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

                    if frame is None:
                        print("[WARNING] Failed to decode JPEG frame from Redis.")
                        continue
                    
                    with activated_network_group.infer_async(frame) as infer_pipeline:
                        detections = infer_pipeline.get_detections()

                    if not detections:
                        continue

                    detected_objects = []
                    for det in detections:
                        label_id = det.get_label()
                        if label_id not in labels: continue
                        
                        detected_objects.append({
                            "label": labels[label_id],
                            "confidence": float(det.get_confidence()),
                            "bbox": {"xmin": int(det.get_bbox().xmin()), "ymin": int(det.get_bbox().ymin()),
                                     "xmax": int(det.get_bbox().xmax()), "ymax": int(det.get_bbox().ymax())}
                        })
                    
                    redis_client.publish(AI_DETECTIONS_CHANNEL, json.dumps(detected_objects))
                    
                    annotated_frame = draw_detections(frame, detections, labels)
                    _, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                    redis_client.publish(AI_FRAME_CHANNEL, buffer.tobytes())

    except Exception as e:
        print(f"[FATAL] An error occurred in the main loop: {e}")
        traceback.print_exc()
    finally:
        print("--- Hailo AI Inference Service Shutting Down ---")

if __name__ == "__main__":
    main()