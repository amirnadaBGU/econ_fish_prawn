"""Platform-specific helpers so live_prawn_detect.py runs on Windows, Linux PCs,
Raspberry Pi and NVIDIA Jetson with the same code.

Everything that differs between systems lives here:
    detect_platform()        -> "windows" | "jetson" | "raspberrypi" | "linux" | "other"
    open_named_camera()      -> open a USB camera by (part of) its name
    apply_camera_controls()  -> exposure / gain / white balance / brightness
    read_camera_controls()   -> current values, for the session summary
    pick_model()             -> use a faster exported model when one sits next to the .pt
"""
import math
import platform
from pathlib import Path

import cv2


# ----------------------------------------------------------------------------- platform

def detect_platform():
    system = platform.system()
    if system == "Windows":
        return "windows"
    if system == "Linux":
        model = ""
        try:
            model = Path("/proc/device-tree/model").read_text(errors="ignore")
        except OSError:
            pass
        if "Jetson" in model or Path("/etc/nv_tegra_release").exists():
            return "jetson"
        if "Raspberry Pi" in model:
            return "raspberrypi"
        return "linux"
    return "other"


PLATFORM = detect_platform()


# ----------------------------------------------------------------------------- cameras

def list_cameras():
    """Return [(index, name)] of video capture devices, in the order OpenCV numbers them."""
    if PLATFORM == "windows":
        try:
            from pygrabber.dshow_graph import FilterGraph
        except ImportError:
            raise SystemExit("Finding a camera by name needs the 'pygrabber' package. "
                             "Run install.bat again, or use a camera number (0, 1, ...).")
        return list(enumerate(FilterGraph().get_input_devices()))

    # Linux: /sys/class/video4linux/videoN/name. One USB camera usually creates several
    # /dev/video nodes (video + metadata); the capture node has index 0.
    cams = []
    for node in sorted(Path("/sys/class/video4linux").glob("video*"),
                       key=lambda p: int(p.name[5:]) if p.name[5:].isdigit() else 0):
        try:
            if (node / "index").read_text().strip() != "0":
                continue
            cams.append((int(node.name[5:]), (node / "name").read_text().strip()))
        except (OSError, ValueError):
            continue
    return cams


def camera_backend():
    """OpenCV backend that matches the numbering of list_cameras()."""
    return cv2.CAP_DSHOW if PLATFORM == "windows" else cv2.CAP_V4L2


def open_named_camera(name):
    """Open the first camera whose name contains `name` (case-insensitive)."""
    cams = list_cameras()
    for index, dev in cams:
        if name.lower() in dev.lower():
            print(f"Camera: '{dev}' (index {index}, {PLATFORM})")
            cap = cv2.VideoCapture(index, camera_backend())
            # The See3CAM has no MJPG mode; it streams uncompressed UYVY
            # (1080p @ 30 fps, 4K @ ~15 fps over USB 3). Other cameras ignore this if unsupported.
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"UYVY"))
            return cap
    raise SystemExit(f"Camera '{name}' not found. Connected cameras: "
                     f"{[d for _, d in cams] or 'none'}\n"
                     "Check the USB cable (it must be USB 3) and try again.")


# ----------------------------------------------------------------------------- camera controls
#
# UVC cameras REMEMBER these settings until they are unplugged, even after the program
# closes. A control left out (None) is not touched, so it keeps whatever was set last.
# "auto" switches exposure / white balance back to automatic.

def _exposure_value(ms):
    """Convert milliseconds to the value the OpenCV backend expects."""
    if PLATFORM == "windows":
        # DirectShow: log2(seconds), whole steps only. -5 = 1/32 s = 31 ms.
        return round(math.log2(ms / 1000))
    # V4L2: exposure_time_absolute in units of 100 microseconds.
    return max(1, round(ms * 10))


def _exposure_ms(value):
    if PLATFORM == "windows":
        return round(1000 * 2 ** value, 2)
    return round(value / 10, 2)


def apply_camera_controls(cap, exposure=None, gain=None, wb=None, brightness=None):
    """Set camera controls. exposure: 'auto' or milliseconds; wb: 'auto' or Kelvin."""
    def put(prop, value, label):
        if not cap.set(prop, value):
            print(f"Warning: the camera did not accept {label}={value}")

    if exposure is not None:
        if str(exposure).lower() == "auto":
            # DirectShow: 1 = auto. V4L2: 3 = aperture priority (auto), 1 = manual.
            put(cv2.CAP_PROP_AUTO_EXPOSURE, 1 if PLATFORM == "windows" else 3, "auto exposure")
        else:
            if PLATFORM != "windows":
                put(cv2.CAP_PROP_AUTO_EXPOSURE, 1, "manual exposure")
            value = _exposure_value(float(exposure))
            put(cv2.CAP_PROP_EXPOSURE, value, "exposure")
            if PLATFORM == "windows" and abs(1000 * 2 ** value - float(exposure)) > 0.05 * float(exposure):
                print(f"Note: on Windows exposure moves in steps of x2; "
                      f"{exposure} ms -> {_exposure_ms(value)} ms")
    if gain is not None:
        put(cv2.CAP_PROP_GAIN, float(gain), "gain")
    if wb is not None:
        if str(wb).lower() == "auto":
            put(cv2.CAP_PROP_AUTO_WB, 1, "auto white balance")
        else:
            put(cv2.CAP_PROP_AUTO_WB, 0, "manual white balance")
            put(cv2.CAP_PROP_WB_TEMPERATURE, float(wb), "white balance")
    if brightness is not None:
        put(cv2.CAP_PROP_BRIGHTNESS, float(brightness), "brightness")


def read_camera_controls(cap):
    """Current control values as the camera reports them (for summary.json).

    Note: in auto exposure, DirectShow keeps reporting the last manual exposure value.
    """
    props = {"exposure_raw": cv2.CAP_PROP_EXPOSURE, "gain": cv2.CAP_PROP_GAIN,
             "auto_wb": cv2.CAP_PROP_AUTO_WB, "wb_temperature": cv2.CAP_PROP_WB_TEMPERATURE,
             "brightness": cv2.CAP_PROP_BRIGHTNESS, "contrast": cv2.CAP_PROP_CONTRAST,
             "saturation": cv2.CAP_PROP_SATURATION}
    out = {k: cap.get(p) for k, p in props.items()}
    if out["exposure_raw"] not in (0, -1):
        out["exposure_ms"] = _exposure_ms(out["exposure_raw"])
    return out


def open_settings_dialog(cap):
    """Windows only: the driver's own settings window. Returns False elsewhere."""
    if PLATFORM != "windows":
        return False
    cap.set(cv2.CAP_PROP_SETTINGS, 1)
    return True


# ----------------------------------------------------------------------------- models

def pick_model(path):
    """Prefer an exported model that suits this computer, if it sits next to the .pt file.

    Jetson:       <name>.engine       (TensorRT, export ON the Jetson: yolo export model=<name>.pt format=engine)
    Raspberry Pi: <name>_ncnn_model/  (NCNN:     yolo export model=<name>.pt format=ncnn)
    """
    p = Path(path)
    if p.suffix != ".pt":
        return str(p)
    candidates = {"jetson": p.with_suffix(".engine"),
                  "raspberrypi": p.with_name(p.stem + "_ncnn_model")}
    alt = candidates.get(PLATFORM)
    if alt is not None and alt.exists():
        print(f"Using {alt.name} (faster on {PLATFORM}) instead of {p.name}")
        return str(alt)
    return str(p)
