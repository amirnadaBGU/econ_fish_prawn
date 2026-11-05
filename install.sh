#!/usr/bin/env bash
# One-time install for Linux, Raspberry Pi and NVIDIA Jetson.  Run:  bash install.sh
cd "$(dirname "$0")" || exit 1
echo "=== Prawn live detection: one-time install ==="
echo

fail() {
    echo
    echo "Install FAILED. Please send a screenshot of this window to the person who sent you this package."
    exit 1
}

if ! command -v python3 >/dev/null; then
    echo "python3 was not found. Install it with:  sudo apt install python3 python3-venv"
    exit 1
fi
if ! python3 -c "import venv, ensurepip" 2>/dev/null; then
    echo "The Python 'venv' module is missing. Install it with:  sudo apt install python3-venv"
    exit 1
fi

MODEL_STR=$(tr -d '\0' </proc/device-tree/model 2>/dev/null)
if [ -f /etc/nv_tegra_release ] || echo "$MODEL_STR" | grep -q "Jetson"; then
    PLATFORM=jetson
elif echo "$MODEL_STR" | grep -q "Raspberry Pi"; then
    PLATFORM=raspberrypi
else
    PLATFORM=linux
fi
echo "Platform: $PLATFORM ${MODEL_STR}"
python3 --version

if [ "$PLATFORM" = jetson ]; then
    # Jetson: see JetPack's own CUDA-enabled PyTorch and OpenCV (with GStreamer)
    # from inside the virtual environment, instead of the generic pip builds.
    python3 -m venv --system-site-packages .venv || fail
else
    python3 -m venv .venv || fail
fi
.venv/bin/python -m pip install --upgrade pip || fail
.venv/bin/python -m pip install ultralytics || fail

if [ "$PLATFORM" = jetson ]; then
    # pip pulls in its own opencv-python, which has no GStreamer and would hide
    # JetPack's OpenCV. Remove it when the system OpenCV has GStreamer.
    if /usr/bin/python3 -c "import cv2,re,sys; sys.exit(0 if re.search(r'GStreamer:\s*YES', cv2.getBuildInformation()) else 1)" 2>/dev/null; then
        echo "Using JetPack's OpenCV (with GStreamer)."
        .venv/bin/python -m pip uninstall -y opencv-python >/dev/null 2>&1
    fi
    if ! .venv/bin/python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        echo
        echo "WARNING: PyTorch cannot see the Jetson GPU, so detection will be slow."
        echo "Install the PyTorch build for your JetPack version, as described in:"
        echo "  https://docs.ultralytics.com/guides/nvidia-jetson/"
    fi
    echo
    echo "For best speed, make a TensorRT model ON THIS JETSON (takes several minutes):"
    echo "  .venv/bin/yolo export model=prawn_yolo11n_pose_v0.1.pt format=engine"
    echo "run.sh then uses it automatically."
elif [ "$PLATFORM" = raspberrypi ]; then
    echo
    echo "For best speed, make an NCNN model (takes a minute):"
    echo "  .venv/bin/yolo export model=prawn_yolo11n_pose_v0.1.pt format=ncnn"
    echo "run.sh then uses it automatically."
fi

echo
echo "Install finished. Now run:  bash run.sh"
