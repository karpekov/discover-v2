#!/usr/bin/env bash
# HAR Discover Dashboard v2 – Longitudinal Analysis
# Usage: ./dashboard-v2/run.sh [--port 8001]

set -e
cd "$(dirname "$0")/.."

PORT=${1:-8001}
if [[ "$1" == "--port" ]]; then
  PORT="$2"
fi

echo "======================================================"
echo "  HAR Discover Dashboard v2"
echo "  http://localhost:${PORT}"
echo "======================================================"

conda run -n discover-v2-env python dashboard-v2/api/main.py --port "$PORT"
