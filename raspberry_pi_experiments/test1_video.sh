#!/usr/bin/env bash
# Test 1: run the model on a recorded prawn video (no camera needed).
# Checks that detection works on the Pi and how fast. ~1-3 minutes.
# Output: runs_test1/<date_time>/
set -euo pipefail
cd "$(dirname "$0")"
./run.sh test_video/prawns_test.mp4 --record runs_test1
