"""
Technical logging service for performance metrics with frame skipping stats
"""
import time
import numpy as np
from collections import deque


class TechnicalLogger:
    def __init__(self):
        self.frame_times = deque(maxlen=30)
        self.processing_times = deque(maxlen=30)
        self.inference_times = deque(maxlen=30)
        self.encoding_times = deque(maxlen=30)
        self.total_frames = 0
        self.processed_frames = 0
        self.skipped_frames = 0
        self.total_detections = 0
        self.start_time = time.time()
        self.last_fps_calculation = time.time()
        self.frames_since_last_calc = 0
        
        # Frame skipping statistics
        self.skip_reasons = {
            'manual': 0,
            'high_cpu': 0,
            'high_memory': 0,
            'fps_limit': 0,
            'adaptive': 0
        }
        
    def log_frame_metrics(self, processing_time, inference_time=None, encoding_time=None):
        """Log frame processing metrics"""
        current_time = time.time()
        self.frame_times.append(current_time)
        self.processing_times.append(processing_time)
        
        if inference_time:
            self.inference_times.append(inference_time)
        if encoding_time:
            self.encoding_times.append(encoding_time)
            
        self.processed_frames += 1
        self.total_frames += 1
        self.frames_since_last_calc += 1
    
    def log_skipped_frame(self, reason='manual'):
        """Log a skipped frame with reason"""
        self.skipped_frames += 1
        self.total_frames += 1
        if reason in self.skip_reasons:
            self.skip_reasons[reason] += 1
    
    def log_detections(self, count):
        """Log detection count"""
        self.total_detections += count
    
    def get_current_fps(self):
        """Get current FPS based on recent frames"""
        current_time = time.time()
        time_diff = current_time - self.last_fps_calculation
        
        if time_diff >= 1.0 and self.frames_since_last_calc > 0:  # Update every second
            fps = self.frames_since_last_calc / time_diff
            self.last_fps_calculation = current_time
            self.frames_since_last_calc = 0
            return fps
        return 0
    
    def get_stats(self):
        """Get current performance statistics with frame skipping info"""
        if len(self.frame_times) < 2:
            base_stats = {
                'fps': 0,
                'avg_processing_ms': 0,
                'avg_inference_ms': 0,
                'avg_encoding_ms': 0,
                'total_frames': self.total_frames,
                'processed_frames': self.processed_frames,
                'skipped_frames': self.skipped_frames,
                'total_detections': self.total_detections,
                'uptime_seconds': time.time() - self.start_time,
                'skip_reasons': self.skip_reasons.copy(),
                'processing_efficiency': 0
            }
            return base_stats
        
        # Calculate processing FPS (only processed frames)
        processing_fps = len(self.frame_times) / (self.frame_times[-1] - self.frame_times[0])
        
        # Calculate average times
        avg_processing = np.mean(self.processing_times) * 1000 if self.processing_times else 0
        avg_inference = np.mean(self.inference_times) * 1000 if self.inference_times else 0
        avg_encoding = np.mean(self.encoding_times) * 1000 if self.encoding_times else 0
        
        # Calculate efficiency metrics
        processing_efficiency = (self.processed_frames / self.total_frames * 100) if self.total_frames > 0 else 100
        skip_percentage = (self.skipped_frames / self.total_frames * 100) if self.total_frames > 0 else 0
        
        return {
            'fps': processing_fps,
            'avg_processing_ms': avg_processing,
            'avg_inference_ms': avg_inference,
            'avg_encoding_ms': avg_encoding,
            'total_frames': self.total_frames,
            'processed_frames': self.processed_frames,
            'skipped_frames': self.skipped_frames,
            'skip_percentage': skip_percentage,
            'total_detections': self.total_detections,
            'uptime_seconds': time.time() - self.start_time,
            'skip_reasons': self.skip_reasons.copy(),
            'processing_efficiency': processing_efficiency,
            'detections_per_processed_frame': (self.total_detections / self.processed_frames) if self.processed_frames > 0 else 0
        }
    
    def reset_skip_counters(self):
        """Reset skip reason counters"""
        self.skip_reasons = {key: 0 for key in self.skip_reasons}
    
    def get_performance_summary(self):
        """Get a summary of performance metrics"""
        stats = self.get_stats()
        return {
            'efficiency': f"{stats['processing_efficiency']:.1f}%",
            'skip_rate': f"{stats['skip_percentage']:.1f}%",
            'avg_detections_per_frame': f"{stats['detections_per_processed_frame']:.1f}",
            'processing_fps': f"{stats['fps']:.1f}",
            'most_common_skip_reason': max(stats['skip_reasons'].items(), key=lambda x: x[1])[0] if any(stats['skip_reasons'].values()) else 'none'
        }
