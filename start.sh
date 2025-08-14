#!/bin/bash
# Start Python backend
cd python-backend
source .venv/bin/activate
python api.py &
BACKEND_PID=$!
cd ../ui
# Start frontend
npm run dev &
FRONTEND_PID=$!
# Wait for both processes
wait $BACKEND_PID
wait $FRONTEND_PID
