#!/bin/bash
# AAI Trading Executor Launcher
# Starts Freqtrade (position manager) and Signal Service (5s inference)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load environment variables
if [ -f "$SCRIPT_DIR/.env" ]; then
    export $(grep -v '^#' "$SCRIPT_DIR/.env" | xargs)
fi

# Check prerequisites
if ! python3 -c "import freqtrade" 2>/dev/null; then
    echo "ERROR: freqtrade not found. Install: cd executor/freqtrade && pip install -e '.[all]'"
    exit 1
fi

# signal_service delegates to signal_generator (SSOT), which loads THIS checkpoint:
SSOT_CKPT="$PROJECT_DIR/experiments/v2_balanced/checkpoints/best.pt"
if [ ! -f "$SSOT_CKPT" ]; then
    echo "ERROR: SSOT model checkpoint not found at $SSOT_CKPT"
    exit 1
fi

MODE=$(python3 -c "import json; print('PAPER' if json.load(open('$SCRIPT_DIR/config.json')).get('dry_run', True) else 'LIVE')")
echo "============================================"
echo "  AAI Trading Executor"
echo "  Mode: $MODE"
echo "  Pairs: BTC/USDT:USDT, ETH/USDT:USDT"
echo "============================================"

# --- LIVE-money gate: refuse a live launch unless explicitly AND verifiably approved ---
if [ "$MODE" = "LIVE" ]; then
    if [ "${AAI_LIVE_CONFIRMED:-}" != "1" ] || [ ! -f "$SCRIPT_DIR/.LIVE_APPROVED" ]; then
        echo "REFUSING TO START LIVE: config.json dry_run=false but the live gate is NOT satisfied."
        echo "Live requires ALL of:"
        echo "  1. parity tests green        (executor/preflight_live.sh)"
        echo "  2. new exit validated in paper (executor/paper/AB_VERDICT.json)"
        echo "  3. dry-run proven end-to-end"
        echo "  4. sentinel file present:     executor/.LIVE_APPROVED"
        echo "  5. env set:                   AAI_LIVE_CONFIRMED=1"
        echo "Run ./preflight_live.sh first (it checks 1-3 and prints how to arm 4-5)."
        exit 1
    fi
    echo "  *** LIVE TRADING ARMED (gate satisfied) ***"
fi

# Start Freqtrade in background
echo "Starting Freqtrade..."
cd "$SCRIPT_DIR"
freqtrade trade \
    --config config.json \
    --strategy AAIStrategy \
    --strategy-path strategies/ \
    --db-url sqlite:///tradesv3.sqlite \
    &
FT_PID=$!
echo "  Freqtrade PID: $FT_PID"

# Wait for API server to be ready
echo "Waiting for Freqtrade API..."
for i in $(seq 1 30); do
    if curl -s http://127.0.0.1:8080/api/v1/ping > /dev/null 2>&1; then
        echo "  Freqtrade API ready"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "  WARNING: Freqtrade API not responding after 30s"
    fi
    sleep 1
done

# Start signal service in foreground
echo "Starting Signal Service..."
cd "$PROJECT_DIR"
python3 -m executor.signal_service &
SS_PID=$!
echo "  Signal Service PID: $SS_PID"

# Wait for either process to exit
echo ""
echo "Both processes running. Press Ctrl+C to stop."
echo ""

cleanup() {
    echo ""
    echo "Shutting down..."
    kill $SS_PID 2>/dev/null || true
    kill $FT_PID 2>/dev/null || true
    wait $SS_PID 2>/dev/null || true
    wait $FT_PID 2>/dev/null || true
    echo "Done."
}

trap cleanup SIGINT SIGTERM
wait -n $FT_PID $SS_PID 2>/dev/null
cleanup
