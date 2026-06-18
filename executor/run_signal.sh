#!/bin/bash
# Auto-restart wrapper for signal service
# Restarts on crash with 10-second delay

cd /Volumes/Docker-SSD/projects/aaiwdbback/aai

while true; do
    echo "[$(date)] Starting signal service..."
    python3 -m executor.signal_service 2>&1 | tee -a /tmp/signal_live.log
    EXIT_CODE=$?
    echo "[$(date)] Signal service exited with code $EXIT_CODE. Restarting in 10s..."
    sleep 10
done
