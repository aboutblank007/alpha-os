import asyncio
import logging
import time
import os
import numpy as np
from datetime import datetime, timedelta
from supabase import create_client, Client

logger = logging.getLogger("DQNTrainer")

class DQNTrainer:
    def __init__(self, dqn_agent, engine_ref):
        self.dqn = dqn_agent
        self.engine = engine_ref
        self.supabase_url = os.environ.get("SUPABASE_URL")
        self.supabase_key = os.environ.get("SUPABASE_KEY")
        self.supabase = None
        self.is_running = False
        self.learn_count = 0
        self.target_update_freq = 100
        self.save_interval = 50
        
        if self.supabase_url and self.supabase_key:
            try:
                self.supabase = create_client(self.supabase_url, self.supabase_key)
                logger.info("✅ DQNTrainer connected to Supabase")
            except Exception as e:
                logger.error(f"❌ Failed to connect to Supabase: {e}")

    async def start(self):
        self.is_running = True
        logger.info("🚀 DQNTrainer loop started")
        while self.is_running:
            try:
                if self.supabase:
                    await self.process_closed_trades()
            except Exception as e:
                logger.error(f"Error in trainer loop: {e}")
            
            await asyncio.sleep(30) # Check every 30s

    async def process_closed_trades(self):
        if not self.engine.pending_experiences:
            return

        # Snapshot of pending request IDs to check
        pending_ids = list(self.engine.pending_experiences.keys())
        
        # We need to match pending experiences (Timestamp) with Closed Trades.
        # Since we don't have a direct ID link guaranteed yet, we use Symbol + Time heuristic.
        
        for req_id in pending_ids:
            if req_id not in self.engine.pending_experiences:
                continue
                
            data = self.engine.pending_experiences[req_id]
            if len(data) < 5:
                continue
            
            state, action, timestamp, symbol, atr = data
            
            try:
                # Query trades table
                # status = 'closed'
                # symbol = symbol
                # created_at > timestamp
                
                ts_iso = datetime.utcfromtimestamp(timestamp).isoformat()
                end_iso = datetime.utcfromtimestamp(timestamp + 300).isoformat()  # 5 minutes window
                
                response = self.supabase.table("trades") \
                    .select("*") \
                    .eq("symbol", symbol) \
                    .eq("status", "closed") \
                    .gte("created_at", ts_iso) \
                    .lte("created_at", end_iso) \
                    .order("created_at", desc=False) \
                    .limit(1) \
                    .execute()
                
                if response.data:
                    trade = response.data[0]
                    
                    # Calculate Reward: risk-adjusted using stored ATR
                    pnl = float(trade.get("pnl_net", 0.0))
                    if atr <= 0:
                        atr = 1.0
                    reward = pnl / (atr + 1e-8)
                    
                    # Call engine to complete
                    # We pass volatility=None so it uses stored ATR
                    if self.engine.complete_experience(req_id, pnl, volatility=None):
                        # Train step
                        loss = self.dqn.learn()
                        if loss is not None:
                            self.learn_count += 1
                            if self.learn_count % 10 == 0:
                                logger.info(f"🧠 DQN Trained (Iter {self.learn_count}, Loss {loss:.6f})")
                            
                            if self.learn_count % self.target_update_freq == 0:
                                self.dqn.update_target()
                                logger.info("🔄 DQN Target Network Updated")
                                
                            if self.learn_count % self.save_interval == 0:
                                self.save_model()
                                
            except Exception as e:
                logger.error(f"Error checking trade for {req_id}: {e}")

    def save_model(self):
        try:
            path = "/app/models/dqn_weights.pth"
            import torch
            torch.save(self.dqn.policy_net.state_dict(), path)
            logger.info(f"💾 DQN Weights saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save DQN weights: {e}")

