#!/bin/bash
# AlphaOS Health Check Script
#
# Usage:
#   ./scripts/health_check.sh
#   ./scripts/health_check.sh --port 9090

PORT=${1:-9090}

echo "Checking AlphaOS health on port $PORT..."

# Check if metrics endpoint is available
if curl -s "http://localhost:$PORT/metrics" > /dev/null 2>&1; then
    echo "✓ Metrics endpoint is up"
    
    # Check specific metrics
    METRICS=$(curl -s "http://localhost:$PORT/metrics")
    
    # Check tick count
    TICK_COUNT=$(echo "$METRICS" | grep "alphaos_ticks_total" | grep -v "#" | head -1 | awk '{print $2}')
    if [[ -n "$TICK_COUNT" ]]; then
        echo "✓ Ticks processed: $TICK_COUNT"
    fi
    
    # Check warmup
    WARMUP=$(echo "$METRICS" | grep "alphaos_warmup_progress" | grep -v "#" | head -1 | awk '{print $2}')
    if [[ -n "$WARMUP" ]]; then
        if (( $(echo "$WARMUP >= 1" | bc -l) )); then
            echo "✓ Warmup complete"
        else
            echo "⚠ Warming up: $(echo "$WARMUP * 100" | bc)%"
        fi
    fi
    
    # Check position
    POSITION=$(echo "$METRICS" | grep "alphaos_position_lots" | grep -v "#" | head -1)
    if [[ -n "$POSITION" ]]; then
        echo "✓ Position tracking active"
    fi
    
    echo ""
    echo "System is healthy!"
    exit 0
else
    echo "✗ Metrics endpoint not responding"
    echo "AlphaOS may not be running"
    exit 1
fi
