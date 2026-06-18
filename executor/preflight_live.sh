#!/bin/bash
# AAI live-trading PRE-FLIGHT GATE.
# Checks every automated safety condition and ONLY prints how to arm live if all pass.
# It NEVER flips dry_run and NEVER arms live itself.
set -u
cd "$(dirname "$0")/.."          # project root
PY=".venv/bin/python"
PASS=1

echo "=== AAI live-trading pre-flight ==="

# 1. parity: signal_service == paper path, and paper == training (the core correctness gate)
echo "[1] parity tests (signal_service==paper, paper==training) ..."
if $PY -m pytest executor/test_signal_parity_service.py executor/paper/test_signal_parity.py -q >/tmp/aai_parity.log 2>&1; then
    echo "    PASS"
else
    echo "    FAIL  (see /tmp/aai_parity.log)"; PASS=0
fi

# 2. new exit validated in paper: a recorded GO verdict from the live A/B book
echo "[2] paper A/B verdict (vol-target/signal-decay exit) ..."
VERD="executor/paper/AB_VERDICT.json"
if [ -f "$VERD" ] && $PY -c "import json,sys; sys.exit(0 if json.load(open('$VERD')).get('go') is True else 1)" 2>/dev/null; then
    echo "    PASS  ($($PY -c "import json;print(json.load(open('$VERD')).get('summary',''))" 2>/dev/null))"
else
    echo "    NOT MET  (need executor/paper/AB_VERDICT.json {\"go\": true}; run the live A/B long enough, then judge it)"; PASS=0
fi

# 3. dry_run is still true (flipping it is the LAST manual step, after 1-2 are green)
echo "[3] config dry_run still true ..."
if $PY -c "import json,sys; sys.exit(0 if json.load(open('executor/config.json'))['dry_run'] is True else 1)"; then
    echo "    OK (dry_run=true)"
else
    echo "    WARNING: dry_run already false — confirm 1-2 were green before it was flipped"
fi

echo "=================================="
if [ "$PASS" = "1" ]; then
    echo "ALL AUTOMATED CHECKS GREEN."
    echo "To ARM live (only after a clean end-to-end dry-run AND you accept the risk):"
    echo "  1) edit executor/config.json -> dry_run: false"
    echo "  2) touch executor/.LIVE_APPROVED"
    echo "  3) AAI_LIVE_CONFIRMED=1 ./executor/run.sh"
    echo "This script will NOT arm live for you."
else
    echo "BLOCKED: one or more checks not met -> live is not allowed. Fix the above and re-run."
    exit 1
fi
