"""
System monitoring service for CPU, memory, and temperature
"""
import time
import psutil
import threading


class SystemMonitor:
    def __init__(self):
        self.cpu_percent = 0
        self.memory_percent = 0
        self.temperature = 0
        self.running = False
        self.thread = None
    
    def start_monitoring(self):
        """Start monitoring system metrics"""
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
    
    def _monitor_loop(self):
        """Main monitoring loop"""
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
        """Stop monitoring"""
        self.running = False
    
    def get_stats(self):
        """Get current system statistics"""
        return {
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'temperature': self.temperature
        }
