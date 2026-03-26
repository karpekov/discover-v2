#!/usr/bin/env bash
# Start the HAR Discover dashboard
# Usage: bash dashboard/run.sh [port]

PORT=${1:-8000}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo ""
echo "  HAR Discover Dashboard"
echo "  http://localhost:$PORT"
echo "  Root: $ROOT"
echo ""

cd "$ROOT"
conda run -n discover-v2-env python dashboard/api/main.py --port "$PORT"
