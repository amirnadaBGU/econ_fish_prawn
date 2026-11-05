#!/usr/bin/env bash
# Linux / Raspberry Pi / Jetson version of run.bat.  Run:  bash run.sh
cd "$(dirname "$0")" || exit 1

# ===================== Settings =====================
# Camera: a name (part of it is enough) or a number.
#   See3CAM = the e-con See3CAM_CU83 camera (found by name, whatever its number).
#   0, 1, 2 = camera number (/dev/video0, ...), for any other camera.
CAMERA=See3CAM
# Camera mode (See3CAM_CU83). Write one of these after MODE=
#   1080p  color 1920x1080, 30 fps.  Recommended - the model was trained on color images.
#   4k     color 3840x2160, 15 fps.  More detail, but much slower (about 4 fps on a laptop)
#          and very large videos - use with VIDEO_QUALITY=medium or low.
#   720p   color 1280x720, 30 fps.   For a slow computer (e.g. Raspberry Pi).
#   ir     infrared, grey 1920x1080, 30 fps.  See "About IR" below.
#   ir4k   infrared, grey 3840x2160, 15 fps.  Probably the same detail as ir, enlarged.
# About IR (infrared):
#   The camera also sees near-infrared light (invisible to the eye), so it can film in the
#   dark - but only with an infrared lamp (illuminator) shining on the scene. It is NOT a
#   thermal camera: it does not see heat. The standard camera has a 940 nm lens, so use a
#   940 nm lamp (an 850 nm lens version also exists - then use an 850 nm lamp).
#   The image is grey and its brightness is adjusted automatically in every frame.
#   The model learned from color images, so it may find fewer prawns in IR - test it first.
#   (The IR modes were tested on Windows only.)
# Not available here: the camera's combined color+IR modes (they need e-con's own software).
# For another camera (CAMERA=0, 1 ...) use 1080p, 4k or 720p only.
MODE=1080p
# Record the session into the "runs" folder (yes/no). no = only watch, nothing is saved.
RECORD=yes
# With RECORD=yes: also save the clean camera video (yes/no). Needed for training; about 1 GB per hour.
SAVE_RAW=yes
# Size of the saved videos: high = the full MODE size (e.g. 1920x1080),
#   medium = half (960x540, about 1/4 of the disk space), low = a quarter (480x270).
VIDEO_QUALITY=high
# A .engine (Jetson) or _ncnn_model folder (Raspberry Pi) next to it is used automatically.
MODEL=prawn_yolo11n_pose_v0.1.pt
# Camera controls. The camera remembers them until it is unplugged.
#   EXPOSURE: auto, or milliseconds (8, 16, 31, 63 ...)   empty = leave as is
#   GAIN:     1 = lowest (See3CAM); higher = brighter, noisier   empty = leave as is
#   WB:       auto, or white balance in Kelvin (e.g. 4500)   empty = leave as is
EXPOSURE=auto
GAIN=
WB=auto
# ====================================================

if [ ! -x .venv/bin/python ]; then
    echo "Please run install.sh first:  bash install.sh"
    exit 1
fi

EXTRA=()
[ "$RECORD" != "no" ] && EXTRA+=(--record runs)
[ "$SAVE_RAW" = "no" ] && EXTRA+=(--no-raw)
[ -n "$VIDEO_QUALITY" ] && EXTRA+=(--video-quality "$VIDEO_QUALITY")
[ -n "$EXPOSURE" ] && EXTRA+=(--exposure "$EXPOSURE")
[ -n "$GAIN" ] && EXTRA+=(--gain "$GAIN")
[ -n "$WB" ] && EXTRA+=(--wb "$WB")
# No screen attached (e.g. a Jetson in the field): run without a window, stop with Ctrl+C.
[ -z "$DISPLAY" ] && [ -z "$WAYLAND_DISPLAY" ] && EXTRA+=(--no-show) \
    && echo "No screen found: running without a window. Press Ctrl+C to stop."

echo "Starting the camera... press q or close the video window to stop."
.venv/bin/python live_prawn_detect.py --source "$CAMERA" --mode "$MODE" \
    --model "$MODEL" --kpts "${EXTRA[@]}"

echo
if [ "$RECORD" = "no" ]; then
    echo "Done. Nothing was recorded - RECORD=no in run.sh."
    exit 0
fi
echo "Done. The results are in the \"runs\" folder (the newest folder is this run)."
echo "Please zip that folder and send it back."
command -v xdg-open >/dev/null && [ -d runs ] && xdg-open runs >/dev/null 2>&1 &
