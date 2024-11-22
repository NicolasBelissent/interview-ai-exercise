#!/bin/bash

# Start Redis server in the background
redis-server --daemonize yes

# Start FastAPI application
uvicorn src.main:app --host 0.0.0.0 --port 80 --reload
