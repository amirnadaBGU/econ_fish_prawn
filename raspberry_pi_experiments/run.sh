#!/usr/bin/env bash
# Live prawn detection + full recording from the USB camera.
#   ./run.sh                     camera 0, runs until Ctrl+C
#   ./run.sh 1                   camera 1
#   MINUTES=15 ./run.sh          stop by itself after 15 minutes
#   SHOW=1 ./run.sh              also show the video window (only with a screen connected)
#   ./run.sh video.mp4           run on a recorded video instead of the camera
# Output: runs/<date_time>/  (videos, CSVs, summary.json, status.json, snapshots)
set -euo pipefail
cd "$(dirname "$0")"
SRC=${1:-0}

ARGS=(--source "$SRC"
      --model models/prawn_yolo11n_pose_v0.1_416.onnx --imgsz 416 --device cpu
      --record runs --density-window 10 --video-fps 10)
[[ "$SRC" =~ ^[0-9]+$ ]] && ARGS+=(--width 1280 --height 720)
[ -n "${MINUTES:-}" ] && ARGS+=(--max-minutes "$MINUTES")
[ "${SHOW:-0}" = "1" ] || ARGS+=(--no-show)

exec .venv/bin/python live_prawn_detect.py "${ARGS[@]}" "${@:2}"
