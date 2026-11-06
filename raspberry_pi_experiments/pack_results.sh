#!/usr/bin/env bash
# Packs the results of both tests into one zip to send to Tair.
#   ./pack_results.sh
set -uo pipefail
cd "$(dirname "$0")"
OUT="results_$(date +%Y-%m-%d_%H-%M).zip"
zip -qr "$OUT" runs_test1 runs_test2 camera_test.jpg 2>/dev/null
echo "Created: $(pwd)/$OUT  ($(du -h "$OUT" | cut -f1))"
echo "Copy this file to a USB stick / Drive and send it to Tair."
