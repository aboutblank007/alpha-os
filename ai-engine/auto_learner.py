import time
import logging
import subprocess
import sys
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Configure logging
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AutoLearner")

# Configuration
NEW_DATA_THRESHOLD = 100  # Retrain after 100 new samples
CHECK_INTERVAL = 60       # Check every 60 seconds
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

class AutoLearner:
    def __init__(self):
        self.last_count = 0
        self.supabase = None
        if SUPABASE_URL and SUPABASE_KEY:
            try:
                from supabase import create_client
                self.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
                logger.info("✅ AutoLearner connected to Supabase for polling")
            except Exception as e:
                logger.error(f"❌ AutoLearner failed to connect to Supabase: {e}")

    def run_command(self, command):
        try:
            logger.info(f"Running: {command}")
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"✅ Success: {command}")
                return True
            else:
                logger.error(f"❌ Failed: {command}\nOutput: {result.stdout}\nError: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Exception running {command}: {e}")
            return False

    def get_db_count(self):
        if not self.supabase: return 0
        try:
            # Count records with completed trades (where result_profit is not null)
            # using select with count='exact', head=True for efficiency
            res = self.supabase.table("training_signals") \
                .select("*", count="exact", head=True) \
                .not_.is_("result_profit", "null") \
                .execute()
            return res.count
        except Exception as e:
            logger.error(f"Error polling Supabase count: {e}")
            return 0
            return 0

    def trigger_retraining(self):
        logger.info("🚀 Starting Auto-Retraining Pipeline...")
        
        # 1. Ingest Data (Now polls DB)
        if not self.run_command("python3 src/ingest_mql_data.py"):
            return

        # 2. Enhance Features
        if not self.run_command("python3 enhance_features.py"):
            return

        # 3. Train Filter Model
        if not self.run_command("python3 train_filter.py"):
            return

        logger.info("✅ Retraining Complete. Model files updated.")
        # Note: client.py should be watching for model file changes to reload automatically.

    def start(self):
        logger.info(f"👀 AutoLearner Polling Mode Started. Threshold: {NEW_DATA_THRESHOLD}")
        
        # Initial count
        self.last_count = self.get_db_count()
        logger.info(f"Initial DB Record Count: {self.last_count}")
        
        try:
            while True:
                time.sleep(CHECK_INTERVAL)
                
                current_count = self.get_db_count()
                new_records = current_count - self.last_count
                
                if new_records > 0:
                    logger.info(f"📈 New Records Detected: {new_records} (Total: {current_count})")
                
                if new_records >= NEW_DATA_THRESHOLD:
                    logger.info(f"Threshold reached ({new_records} >= {NEW_DATA_THRESHOLD}). Triggering retraining...")
                    self.trigger_retraining()
                    self.last_count = current_count
                    
        except KeyboardInterrupt:
            logger.info("AutoLearner Stopped.")

if __name__ == "__main__":
    learner = AutoLearner()
    learner.start()

