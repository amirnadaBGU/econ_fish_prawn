#!/usr/bin/env bash
# Test 2: live from the USB camera for 5 minutes (prawns are not needed).
# Checks the camera, recording and files on the Pi. Stops by itself.
#   ./test2_camera.sh       (camera 0)
#   ./test2_camera.sh 1     (camera 1)
# Output: runs_test2/<date_time>/
set -euo pipefail
cd "$(dirname "$0")"
MINUTES=5 ./run.sh "${1:-0}" --record runs_test2
