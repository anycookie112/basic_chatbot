#!/bin/bash

# Start main FastAPI app on port 8000 in background
echo "Starting main API..."
python main.py &

# Start chat API server on port 8001 in background
echo "Starting chat API..."
python chat_api.py &

echo "Starting frontend server..."
python3 -m http.server 3000 &

wait
