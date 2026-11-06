#!/usr/bin/env bash
# Lists the cameras and saves one test image (camera_test.jpg) from the chosen camera.
#   ./check_camera.sh        (camera 0)
#   ./check_camera.sh 1      (camera 1)
set -uo pipefail
cd "$(dirname "$0")"
CAM=${1:-0}

echo "=== Cameras connected:"
v4l2-ctl --list-devices 2>/dev/null || echo "(v4l2-ctl not found - run setup_pi5.sh)"
echo
echo "=== Formats of /dev/video$CAM:"
v4l2-ctl -d /dev/video$CAM --list-formats-ext 2>/dev/null | grep -E "\[|Size" | head -30
echo
.venv/bin/python - "$CAM" <<'EOF'
import sys, time, cv2
cam = int(sys.argv[1])
cap = cv2.VideoCapture(cam, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
    sys.exit(f"Could not open camera {cam}. Try another number: ./check_camera.sh 1")
frame = None
for _ in range(30):  # let exposure settle
    ok, f = cap.read()
    if ok:
        frame = f
if frame is None:
    sys.exit("Camera opened but sent no frames.")
t = time.perf_counter(); n = 0
while time.perf_counter() - t < 2:
    n += cap.read()[0]
cv2.imwrite("camera_test.jpg", frame)
print(f"Camera {cam} OK: {frame.shape[1]}x{frame.shape[0]}, ~{n / 2:.0f} FPS. Saved camera_test.jpg")
EOF
