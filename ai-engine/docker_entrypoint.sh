#!/bin/bash
set -e

# Start Auto Learner in the background
echo "🚀 Starting Auto Learner Service..."
python auto_learner.py &
AUTO_LEARNER_PID=$!

# Start AI Engine Client (Foreground)
echo "🚀 Starting AI Engine Inference Client..."
python src/client.py &
CLIENT_PID=$!

# Wait for any process to exit
wait -n

# Exit with status of process that exited first
exit $?

