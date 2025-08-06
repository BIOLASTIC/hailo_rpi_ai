"""
Detection processing utilities for Hailo AI inference results
"""
import time
import numpy as np
from app.config.settings import settings
from app.services.frame_processor import COCO_LABELS


class DetectionProcessor:
    @staticmethod
    def postprocess_hailo_detections_with_timing(raw_detections, original_height, original_width, 
                                               model_height=None, model_width=None, confidence_threshold=None):
        """
        Post-processing with detailed timing and configurable model input size
        Now uses settings for model dimensions if not provided
        """
        start_time = time.time()
        detections = []
        
        # Use configurable model input size from settings
        if model_height is None:
            model_height = settings.MODEL_INPUT_HEIGHT
        if model_width is None:
            model_width = settings.MODEL_INPUT_WIDTH
        if confidence_threshold is None:
            confidence_threshold = settings.CONFIDENCE_THRESHOLD
        
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
                                
                                # Handle aspect ratio maintained preprocessing
                                if settings.MAINTAIN_ASPECT_RATIO:
                                    # Calculate the actual image area within the padded model input
                                    aspect_ratio = original_width / original_height
                                    target_aspect_ratio = model_width / model_height
                                    
                                    if aspect_ratio > target_aspect_ratio:
                                        # Image was fitted by width
                                        effective_width = model_width
                                        effective_height = int(model_width / aspect_ratio)
                                        x_offset = 0
                                        y_offset = (model_height - effective_height) // 2
                                    else:
                                        # Image was fitted by height
                                        effective_height = model_height
                                        effective_width = int(model_height * aspect_ratio)
                                        y_offset = 0
                                        x_offset = (model_width - effective_width) // 2
                                    
                                    # Adjust coordinates relative to the effective image area
                                    x_center_adj = (x_center * model_width - x_offset) / effective_width
                                    y_center_adj = (y_center * model_height - y_offset) / effective_height
                                    width_adj = (width * model_width) / effective_width
                                    height_adj = (height * model_height) / effective_height
                                    
                                    # Skip detections outside the effective image area
                                    if (x_center_adj < 0 or x_center_adj > 1 or 
                                        y_center_adj < 0 or y_center_adj > 1):
                                        continue
                                    
                                    # Convert to pixel coordinates in original image
                                    x_center_px = x_center_adj * original_width
                                    y_center_px = y_center_adj * original_height
                                    width_px = width_adj * original_width
                                    height_px = height_adj * original_height
                                    
                                else:
                                    # Direct conversion without aspect ratio consideration
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
