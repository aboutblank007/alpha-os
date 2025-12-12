import psutil
import os
import time
import logging
import threading

logger = logging.getLogger("ResourceMonitor")

class ResourceMonitor:
    """
    M2 Pro Resource Monitor & Protection
    """
    def __init__(self, memory_limit_gb=12.0, check_interval=5):
        self.memory_limit = memory_limit_gb * 1024 * 1024 * 1024 # Bytes
        self.check_interval = check_interval
        self.running = False
        self.process = psutil.Process(os.getpid())
        
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        logger.info("🛡️ Resource Monitor Started")

    def _monitor_loop(self):
        while self.running:
            mem_info = self.process.memory_info()
            rss = mem_info.rss # Resident Set Size
            
            # Convert to GB
            rss_gb = rss / (1024**3)
            
            if rss > self.memory_limit:
                logger.warning(f"⚠️ Memory CRITICAL: {rss_gb:.2f}GB > {self.memory_limit/(1024**3):.2f}GB")
                self._trigger_gc()
            
            time.sleep(self.check_interval)

    def _trigger_gc(self):
        import gc
        logger.info("🧹 Triggering Garbage Collection...")
        gc.collect()
        # Optional: Clear PyTorch Cache
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except:
            pass

    def stop(self):
        self.running = False
