//+------------------------------------------------------------------+
//|                                                         Test.mq5 |
//|                                  Copyright 2024, AlphaOS Project |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, AlphaOS Project"
#property version   "1.00"

int OnInit()
  {
   Print("Test EA Initialized");
   return(INIT_SUCCEEDED);
  }

void OnDeinit(const int reason)
  {
   Print("Test EA Deinitialized");
  }

void OnTick()
  {
   Print("Test EA Tick");
  }
