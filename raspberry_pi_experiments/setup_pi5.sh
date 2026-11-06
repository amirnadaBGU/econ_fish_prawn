#!/usr/bin/env bash
# One-time setup on a Raspberry Pi 5 (Raspberry Pi OS Bookworm, 64-bit).
#   cd 2026-11_pi5_robot_demo && bash setup_pi5.sh
# Takes ~10 minutes (downloads PyTorch). Needs internet.
set -euo pipefail
cd "$(dirname "$0")"

if [ "$(uname -m)" != "aarch64" ]; then
    echo "WARNING: this is not a 64-bit ARM system ($(uname -m)). Use 64-bit Raspberry Pi OS." >&2
fi
[ -f /proc/device-tree/model ] && echo "Device: $(tr -d '\0' < /proc/device-tree/model)"

sudo apt-get update
sudo apt-get install -y python3-venv python3-pip libgl1 libglib2.0-0 v4l-utils zip unzip

python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

echo
echo "Testing the model on a blank image..."
.venv/bin/python - <<'EOF'
import numpy as np, time
from ultralytics import YOLO
m = YOLO("models/prawn_yolo11n_pose_v0.1_416.onnx", task="pose")
img = np.zeros((720, 1280, 3), np.uint8)
m.predict(img, imgsz=416, device="cpu", verbose=False)
t = time.perf_counter()
for _ in range(10):
    m.predict(img, imgsz=416, device="cpu", verbose=False)
print(f"Model OK - {10 / (time.perf_counter() - t):.1f} FPS (model only)")
EOF
chmod +x run.sh check_camera.sh test1_video.sh test2_camera.sh
echo
echo "Setup done. Next: ./check_camera.sh   then   ./test1_video.sh   and   ./test2_camera.sh"
