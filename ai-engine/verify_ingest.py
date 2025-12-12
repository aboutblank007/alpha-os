
import os
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestIngest")

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'ai-engine/src'))

# Import the function to test
try:
    from ingest_mql_data import ingest_data
except ImportError:
    logger.error("Could not import ingest_data")
    sys.exit(1)

def test_ingestion():
    logger.info("🧪 Testing ingest_data()...")
    
    try:
        # We need to ensure SUPABASE_URL and SUPABASE_KEY are available.
        # Since we are running outside the container, we might need to source them.
        # But let's assume the user can run this with env vars.
        
        df = ingest_data(days=1, limit=10, source_dir=".")
        
        if df is not None and not df.empty:
            logger.info(f"✅ Successfully ingested {len(df)} rows")
            
            # Check for AI features
            ai_cols = ['dom_imbalance', 'volatility_shock', 'ai_score']
            found_cols = [c for c in ai_cols if c in df.columns]
            
            if found_cols:
                logger.info(f"✅ Found AI features: {found_cols}")
                print(df[found_cols].head())
            else:
                logger.error("❌ AI features NOT found in DataFrame columns!")
                logger.info(f"Available columns: {df.columns.tolist()}")
        else:
            logger.warning("⚠️ No data returned")
            
    except Exception as e:
        logger.error(f"❌ Ingestion failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ingestion()
