#!/usr/bin/env python3
"""Live prawn / fish detection: camera -> YOLO model -> on-screen boxes, count and FPS.

--target prawn (default) or fish picks the default model and the on-screen words.
Any YOLO detect or pose model works; its classes are read from the model file.

Examples:
    python live_prawn_detect.py --source See3CAM --target fish
    python live_prawn_detect.py --source See3CAM --model prawn_yolo11n_pose_v0.1.pt
    python live_prawn_detect.py --source 0 --model models/prawn_yolo11n_pose_v0.1.pt
    python live_prawn_detect.py --source 0 --model models/prawn_yolo11n_pose_v0.1.pt --record runs
    python live_prawn_detect.py --source pond.mp4 --model models/prawn.onnx --save out.mp4
    python live_prawn_detect.py --source 0 --model models/prawn.engine --no-show --save out.mp4

Press q (or Esc) in the window to quit, c to open the camera settings (Windows).

Runs on Windows, Linux, Raspberry Pi and NVIDIA Jetson; the differences live in
platform_utils.py. Camera controls (live cameras):
    python live_prawn_detect.py --source See3CAM --model ... --exposure 16 --gain 4 --wb 4500
    python live_prawn_detect.py --source See3CAM --model ... --exposure auto --wb auto

--record DIR saves a complete session into DIR/<timestamp>/:
    annotated.mkv    video with boxes, count and FPS
    raw.mkv          clean camera video (disable with --no-raw)
    (.mkv still plays if the program is killed; play with VLC or the Windows player)
    detections.csv   one row per detected object per frame (box, conf, class; keypoints
                     for pose models)
    per_frame.csv    one row per frame (count, per-class counts, inference ms, FPS)
    summary.json     model, settings, camera, computer, totals
    snapshots/       one annotated image every --snapshot-every seconds
"""
import argparse
import csv
import hashlib
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

import platform_utils

# --target: what to detect. Picks the default model (when --model is not given), the word
# on screen and the keypoint names of a pose model. Classes come from the model itself.
TARGETS = {
    "prawn": dict(model="prawn_yolo11n_pose_v0.1.pt", label="Prawns",
                  kpt_names=["carapace", "eyes", "rostrum", "tail"]),
    "fish": dict(model="fish_tolov26nano.pt", label="Fish", kpt_names=[]),
}

# Keys work in English and Hebrew keyboard layouts and with Caps Lock:
# on a Hebrew layout the q key types "/" and the c key types "ב"
# (0x5D1 as Unicode, 0xE1 in the Windows-1255 code page, 0xD1 = its low byte).
QUIT_KEYS = {ord("q"), ord("Q"), ord("/"), 27}  # 27 = Esc
SETTINGS_KEYS = {ord("c"), ord("C"), 0x5D1, 0xE1, 0xD1}

# --video-quality: how much the saved videos are shrunk (per side).
VIDEO_SCALE = {"high": 1, "medium": 2, "low": 4}


def video_size(shape, quality):
    """(width, height) of the saved video for a frame of this shape; kept even for the codec."""
    h, w = shape[:2]
    d = VIDEO_SCALE[quality]
    return max(2, w // d // 2 * 2), max(2, h // d // 2 * 2)


def fit(img, size):
    if (img.shape[1], img.shape[0]) == size:
        return img
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


# --mode: camera modes of the e-con See3CAM_CU83, as the camera reports them (DirectShow)
# and as e-con documents them. fourcc + requested size are what the camera is asked for;
# "out" is the image size the program works with.
#   Color modes are plain sizes, so they also work with other cameras.
#   IR modes: the camera sends a 10-bit image packed as MIPI RAW10 (4 pixels in 5 bytes)
#   inside a "Y16" stream whose advertised size is smaller (1920x675 -> 1920x1080).
#   Not offered: Y8 1920x1080 / 3840x2160 (OpenCV could not select them) and
#   Y16 3120x1080 / 4440x2160 (RGB + IR packed together; needs e-con's own software).
MODES = {
    "1080p": dict(fourcc="UYVY", w=1920, h=1080, out=(1920, 1080), fps=30, ir=False,
                  about="color 1920x1080, 30 fps"),
    "4k":    dict(fourcc="UYVY", w=3840, h=2160, out=(3840, 2160), fps=15, ir=False,
                  about="color 3840x2160, 15 fps"),
    "720p":  dict(fourcc="UYVY", w=1280, h=720, out=(1280, 720), fps=30, ir=False,
                  about="color 1280x720, 30 fps"),
    "ir":    dict(fourcc="Y16 ", w=1920, h=675, out=(1920, 1080), fps=30, ir=True,
                  about="infrared (near-IR), grey 1920x1080, 30 fps"),
    "ir4k":  dict(fourcc="Y16 ", w=3840, h=1350, out=(3840, 2160), fps=15, ir=True,
                  about="infrared, grey 3840x2160, 15 fps (likely upscaled from Full HD)"),
}

# Recording a live camera: most copies of one processed frame written to the video.
MAX_REPEATS = 3


def unpack_raw10(buf, w, h):
    """MIPI RAW10 buffer (4 pixels in 5 bytes) -> (h, w) uint16 image, values 0..1023."""
    b = buf.reshape(-1).view(np.uint8)
    if b.size != w * h * 5 // 4:
        raise SystemExit(f"IR mode: got {b.size} bytes per frame, expected {w * h * 5 // 4}. "
                         "Is this a See3CAM_CU83?")
    g = b.reshape(-1, 5).astype(np.uint16)
    px = np.empty((g.shape[0], 4), np.uint16)
    for k in range(4):
        px[:, k] = (g[:, k] << 2) | ((g[:, 4] >> (2 * k)) & 3)
    return px.reshape(h, w)


def ir_to_bgr(raw):
    """10-bit IR -> 8-bit 3-channel image for the model, the window and the videos.
    The range is stretched per frame (0.5%..99.5% of the pixels) so a dim scene is visible."""
    lo, hi = np.percentile(raw[::4, ::4], (0.5, 99.5))
    lut = np.clip((np.arange(1024) - lo) * (255.0 / max(hi - lo, 1)), 0, 255).astype(np.uint8)
    return cv2.cvtColor(lut[raw], cv2.COLOR_GRAY2BGR)       # lookup table: fast for 10-bit


def frame_reader(cap, mode):
    """Return a read() like cap.read() that also decodes the IR modes."""
    if not mode or not MODES[mode]["ir"]:
        return cap.read
    w, h = MODES[mode]["out"]

    def read():
        ok, buf = cap.read()
        return (ok, ir_to_bgr(unpack_raw10(buf, w, h)) if ok else None)
    return read


def parse_args():
    p = argparse.ArgumentParser(description="Live prawn / fish detection with a YOLO model.")
    p.add_argument("--source", default="0",
                   help="Camera name (e.g. See3CAM), camera index (0, 1, ...), "
                        "video file / URL, or a GStreamer pipeline string (anything "
                        "containing '!'). Default: 0")
    p.add_argument("--target", choices=list(TARGETS), default="prawn",
                   help="What to detect: "
                        + ", ".join(f"{k} (default model {v['model']})" for k, v in TARGETS.items())
                        + ". Default: prawn")
    p.add_argument("--model", default=None,
                   help="Model file: .pt, .onnx, .engine or an NCNN folder (detect or pose). "
                        "Default: the model of --target. With a .pt, a <name>.engine (Jetson) "
                        "or <name>_ncnn_model (Raspberry Pi) next to it is used automatically.")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default 0.25)")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size (default 640)")
    p.add_argument("--device", default=None,
                   help="Inference device: cpu, mps, 0 (CUDA GPU). Default: auto")
    p.add_argument("--mode", choices=list(MODES), default=None,
                   help="Camera mode (See3CAM_CU83): "
                        + "; ".join(f"{k} = {v['about']}" for k, v in MODES.items())
                        + ". The ir modes need the See3CAM; the model was trained on color "
                          "images, so detection in IR may be weaker. Overrides --width/--height.")
    p.add_argument("--width", type=int, default=None, help="Requested camera width")
    p.add_argument("--height", type=int, default=None, help="Requested camera height")
    cam = p.add_argument_group(
        "camera controls",
        "Live cameras only. The camera remembers these until it is unplugged; "
        "a control you leave out keeps its last value. In the window, press c to open "
        "the camera's settings dialog (Windows).")
    cam.add_argument("--exposure", default=None, metavar="MS|auto",
                     help="Exposure time in milliseconds, or 'auto'. On Windows it moves "
                          "in steps of x2 (..., 8, 16, 31, 63 ms, ...)")
    cam.add_argument("--gain", type=float, default=None,
                     help="Sensor gain (See3CAM: 1 = lowest). Higher = brighter but noisier")
    cam.add_argument("--wb", default=None, metavar="K|auto",
                     help="White balance in Kelvin (e.g. 4500), or 'auto'")
    cam.add_argument("--brightness", type=float, default=None, help="Image brightness")
    p.add_argument("--save", default=None, help="Save the annotated video to this path (.mp4, or .mkv to "
                        "keep the video playable if the program is killed)")
    p.add_argument("--record", default=None, metavar="DIR",
                   help="Record a full session (videos, CSVs, summary, snapshots) into DIR/<timestamp>/")
    p.add_argument("--no-raw", action="store_true", help="With --record: do not save raw.mkv")
    p.add_argument("--video-quality", choices=list(VIDEO_SCALE), default="high",
                   help="Size of the saved videos (--record and --save): high = camera "
                        "resolution, medium = half width and height (e.g. 1920x1080 -> 960x540), "
                        "low = a quarter (480x270). Detection always runs on the full image and "
                        "detections.csv stays in full-resolution pixels. Default: high")
    p.add_argument("--snapshot-every", type=float, default=10,
                   help="With --record: seconds between annotated snapshots (default 10, 0 = off)")
    p.add_argument("--kpts", action="store_true",
                   help="Also draw the pose keypoints (pose models only, e.g. prawn)")
    p.add_argument("--no-show", action="store_true", help="Do not open a window (headless)")
    return p.parse_args()


def is_camera_name(source):
    """A source that is not a number, file, URL or GStreamer pipeline is a camera name."""
    return not (source.isdigit() or "!" in source or "://" in source or Path(source).exists())


def is_live_camera(source):
    return source.isdigit() or is_camera_name(source)


def open_source(source, width=None, height=None, mode=None):
    if mode:
        if not is_live_camera(source):
            raise SystemExit("--mode is for cameras, not for video files / URLs.")
        m = MODES[mode]
        width, height = m["w"], m["h"]
        if m["ir"]:
            cap = (platform_utils.open_named_camera(source) if is_camera_name(source)
                   else cv2.VideoCapture(int(source), platform_utils.camera_backend()))
            # order matters: format first, then size; raw bytes, no color conversion
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*m["fourcc"]))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
            if not cap.isOpened():
                raise SystemExit(f"Could not open source: {source}")
            return cap
    if is_camera_name(source):
        # Camera chosen by name, e.g. "See3CAM" for the e-con See3CAM_CU83
        # (Windows: DirectShow, Linux / Raspberry Pi / Jetson: V4L2).
        cap = platform_utils.open_named_camera(source)
    elif source.isdigit():
        cap = cv2.VideoCapture(int(source))
        if width or height:
            # USB (UVC) cameras usually need MJPG to deliver HD at full frame rate.
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    elif "!" in source:
        cap = cv2.VideoCapture(source, cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(source)
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {source}")
    return cap


def class_counts(result, names):
    """Number of detections of each class, in the model's class order."""
    cls = result.boxes.cls.cpu().numpy().astype(int) if len(result.boxes) else []
    return [int((cls == c).sum()) if len(cls) else 0 for c in names]


def draw_overlay(img, counts, names, label, fps):
    # one class: "Prawns: 3"; several: "fish: 3  partial fish: 1"
    if len(names) == 1:
        what = f"{label}: {sum(counts)}"
    else:
        what = "  ".join(f"{names[c]}: {n}" for c, n in zip(names, counts))
    text = f"{what}   FPS: {fps:.1f}"
    scale = max(0.6, img.shape[1] / 1600)  # readable from 640 px up to 4K/5K frames
    thick = max(2, round(scale * 2))
    pad = round(10 * scale)
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.rectangle(img, (0, 0), (w + 2 * pad, h + 2 * pad), (0, 0, 0), -1)
    cv2.putText(img, text, (pad, h + pad), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 0), thick)


def model_fingerprint(model_path):
    """Short SHA-256 of the model file, or of all files in a model folder (NCNN)."""
    p = Path(model_path)
    h = hashlib.sha256()
    for f in sorted(p.rglob("*")) if p.is_dir() else [p]:
        if f.is_file():
            h.update(f.read_bytes())
    return h.hexdigest()[:16]


def system_info():
    info = {"platform": platform_utils.PLATFORM, "os": platform.platform(),
            "python": platform.python_version(),
            "cpu": platform.processor() or platform.machine()}
    try:
        import torch
        import ultralytics
        info["ultralytics"] = ultralytics.__version__
        info["torch"] = torch.__version__
        info["cuda_gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    return info


class SessionRecorder:
    """Writes everything about one run into a timestamped folder."""

    def __init__(self, root, args, cap, live, names, kpt_names):
        self.dir = Path(root) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.snap_dir = self.dir / "snapshots"
        if args.snapshot_every > 0:
            self.snap_dir.mkdir(exist_ok=True)
        self.args = args
        self.save_raw = not args.no_raw
        self.cam = {"width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    "reported_fps": round(cap.get(cv2.CAP_PROP_FPS) or 0, 2),
                    "mode": args.mode}
        if args.mode and MODES[args.mode]["ir"]:
            # the camera reports the packed stream size; the decoded image is this size.
            # IR videos are 8-bit, contrast-stretched per frame (not the raw 10-bit values).
            self.cam["width"], self.cam["height"] = MODES[args.mode]["out"]
        if live:
            self.cam["controls_start"] = platform_utils.read_camera_controls(cap)
        self.is_file = Path(args.source).is_file()
        self.started = datetime.now()
        self.writers = None
        self.t_first = None
        self.written = 0
        self.last_snap = None
        self.counts, self.infer_ms = [], []
        self.names, self.kpt_names = names, kpt_names
        self.class_totals = [0] * len(names)

        self.det_f = open(self.dir / "detections.csv", "w", newline="")
        self.det = csv.writer(self.det_f)
        kp_cols = [f"{k}_{c}" for k in kpt_names for c in ("x", "y", "conf")]
        # obj = number of the object within its frame (0, 1, ...); class = name from the model
        self.det.writerow(["frame", "time_s", "obj", "x1", "y1", "x2", "y2", "conf", "class"]
                          + kp_cols)
        self.frm_f = open(self.dir / "per_frame.csv", "w", newline="")
        self.frm = csv.writer(self.frm_f)
        # with several classes, also one count column per class (e.g. count_partial_fish)
        cls_cols = ([f"count_{names[c]}".replace(" ", "_") for c in names]
                    if len(names) > 1 else [])
        self.frm.writerow(["frame", "time_s", "count"] + cls_cols + ["infer_ms", "fps"])

    def _open_writers(self, fps, shape):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_fps = round(fps, 2)
        self.video_size = video_size(shape, self.args.video_quality)
        size = self.video_size
        # .mkv, not .mp4: an .mp4 gets its index only when it is closed, so a hard stop
        # (PyCharm Stop, closed console, power cut) leaves an unplayable file. An .mkv cut off
        # in the middle still plays up to about the last second.
        self.writers = [cv2.VideoWriter(str(self.dir / "annotated.mkv"), fourcc, fps, size)]
        if self.save_raw:
            self.writers.append(cv2.VideoWriter(str(self.dir / "raw.mkv"), fourcc, fps, size))

    def _write_video(self, raw, annotated, repeats):
        frames = [fit(annotated, self.video_size)]      # shrink once, write `repeats` times
        if self.save_raw:
            frames.append(fit(raw, self.video_size))
        for _ in range(repeats):
            for w, f in zip(self.writers, frames):
                w.write(f)

    def add(self, idx, t, raw, annotated, result, infer_ms, fps):
        boxes = result.boxes
        kpts = result.keypoints
        xyxy = boxes.xyxy.cpu().numpy() if len(boxes) else []
        confs = boxes.conf.cpu().numpy() if len(boxes) else []
        cls = boxes.cls.cpu().numpy().astype(int) if len(boxes) else []
        kxy = kpts.xy.cpu().numpy() if kpts is not None and len(boxes) else None
        kcf = kpts.conf.cpu().numpy() if kxy is not None and kpts.conf is not None else None
        for i in range(len(xyxy)):
            row = ([idx, f"{t:.3f}", i] + [f"{v:.1f}" for v in xyxy[i]]
                   + [f"{confs[i]:.3f}", self.names.get(cls[i], cls[i])])
            for k in range(len(self.kpt_names)):
                if kxy is not None and k < kxy.shape[1]:
                    row += [f"{kxy[i][k][0]:.1f}", f"{kxy[i][k][1]:.1f}",
                            f"{kcf[i][k]:.3f}" if kcf is not None else ""]
                else:
                    row += ["", "", ""]
            self.det.writerow(row)
        per_class = class_counts(result, self.names)
        self.class_totals = [a + b for a, b in zip(self.class_totals, per_class)]
        self.frm.writerow([idx, f"{t:.3f}", len(xyxy)] + (per_class if len(self.names) > 1 else [])
                          + [f"{infer_ms:.1f}", f"{fps:.1f}"])
        self.counts.append(len(xyxy))
        self.infer_ms.append(infer_ms)

        # Video: a file is written frame-for-frame at its own FPS. A camera is written at its
        # own FPS too, repeating each processed frame by wall-clock time so playback is
        # real-time. At most MAX_REPEATS copies per frame, and a larger gap is skipped, not
        # made up later: writing many copies of a 4K frame takes longer than the gap itself,
        # so catching up made the next gap bigger and the program slowed to a stop.
        if self.writers is None:
            self.t_first = t
            reported = self.cam["reported_fps"]
            mode = self.args.mode
            fps = (MODES[mode]["fps"] if mode else
                   reported if 1 <= reported <= 120 else 30.0)  # See3CAM reports 0
            self._open_writers(fps, raw.shape)
        repeats = due = 1
        if not self.is_file:
            due = max(0, int((t - self.t_first) * self.video_fps) + 1 - self.written)
            repeats = min(due, MAX_REPEATS)
        if repeats:
            self._write_video(raw, annotated, repeats)
        self.written += due
        self._snapshot(t, annotated)

    def _snapshot(self, t, annotated):
        every = self.args.snapshot_every
        if every > 0 and (self.last_snap is None or t - self.last_snap >= every):
            cv2.imwrite(str(self.snap_dir / f"t{t:07.1f}s.jpg"), annotated)
            self.last_snap = t

    def close(self, frames, elapsed, model_path, device, controls_end=None):
        for w in self.writers or []:
            w.release()
        self.det_f.close()
        self.frm_f.close()
        if controls_end is not None:
            # differs from controls_start if the settings dialog (key c) was used
            self.cam["controls_end"] = controls_end
        sha = model_fingerprint(model_path)
        n = len(self.counts)
        summary = {
            "started": self.started.isoformat(timespec="seconds"),
            "ended": datetime.now().isoformat(timespec="seconds"),
            "duration_s": round(elapsed, 1),
            "target": self.args.target,
            "model": {"file": Path(model_path).name, "sha256_16": sha,
                      "classes": [self.names[c] for c in self.names]},
            "settings": {"conf": self.args.conf, "imgsz": self.args.imgsz,
                         "device_requested": self.args.device, "device_used": device,
                         # what was asked for (None = left as the camera had it). With
                         # exposure "auto", the reported exposure value below is not reliable.
                         "camera_controls_requested": {
                             "exposure": self.args.exposure, "gain": self.args.gain,
                             "wb": self.args.wb, "brightness": self.args.brightness}},
            "source": self.args.source,
            "camera": self.cam,
            "video_fps_written": getattr(self, "video_fps", None),
            # detections.csv is in camera pixels; divide by video_scale to draw on the videos
            "video_quality": self.args.video_quality,
            "video_scale": VIDEO_SCALE[self.args.video_quality],
            "video_size_written": list(getattr(self, "video_size", ())) or None,
            "frames": frames,
            "avg_fps_end_to_end": round(frames / elapsed, 2) if elapsed > 0 else None,
            "avg_infer_ms": round(sum(self.infer_ms) / n, 1) if n else None,
            "objects_per_frame": {"mean": round(sum(self.counts) / n, 2) if n else None,
                                  "max": max(self.counts) if n else None,
                                  "frames_with_objects": sum(c > 0 for c in self.counts)},
            "detections_per_class": {self.names[c]: tot
                                     for c, tot in zip(self.names, self.class_totals)},
            "keypoints": self.kpt_names,
            "computer": system_info(),
        }
        (self.dir / "summary.json").write_text(json.dumps(summary, indent=2))
        return self.dir


def main():
    args = parse_args()
    target = TARGETS[args.target]
    model_arg = args.model or target["model"]
    if not Path(model_arg).exists() and (Path(__file__).parent / model_arg).exists():
        model_arg = str(Path(__file__).parent / model_arg)   # model next to this script
    model_path = platform_utils.pick_model(model_arg)
    model = YOLO(model_path)              # detect or pose: the task is read from the model
    names = model.names                   # e.g. {0: 'fish', 1: 'partial fish'}
    kpt_names = target["kpt_names"] if model.task == "pose" else []
    window = f"{target['label']} detection"
    print(f"Model: {Path(model_path).name} ({model.task}), classes: {list(names.values())}")
    cap = open_source(args.source, args.width, args.height, args.mode)
    read = frame_reader(cap, args.mode)
    live = is_live_camera(args.source)
    if live:
        platform_utils.apply_camera_controls(cap, exposure=args.exposure, gain=args.gain,
                                             wb=args.wb, brightness=args.brightness)
        for _ in range(5):          # let the new settings take effect
            cap.read()
        size = (MODES[args.mode]["out"] if args.mode and MODES[args.mode]["ir"] else
                (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        print("Camera:", f"mode {args.mode or '-'},", f"{size[0]}x{size[1]},",
              platform_utils.read_camera_controls(cap))
    rec = (SessionRecorder(args.record, args, cap, live, names, kpt_names)
           if args.record else None)
    if rec:
        print(f"Recording session to: {rec.dir}")

    writer = None
    controls_end = None
    fps_ema = None
    frames = 0
    t_start = time.perf_counter()
    try:
        while True:
            ok, frame = read()
            if not ok:
                break

            t0 = time.perf_counter()
            result = model.predict(frame, imgsz=args.imgsz, conf=args.conf,
                                   device=args.device, verbose=False)[0]
            dt = time.perf_counter() - t0
            fps_ema = 1 / dt if fps_ema is None else 0.9 * fps_ema + 0.1 / dt

            kpts = result.keypoints
            if not args.kpts:
                result.keypoints = None  # hide keypoints in the drawing only
            annotated = result.plot()
            result.keypoints = kpts
            draw_overlay(annotated, class_counts(result, names), names, target["label"], fps_ema)

            if rec:
                t = time.perf_counter() - t_start
                rec.add(frames, t, frame, annotated, result, dt * 1000, fps_ema)

            if args.save:
                if writer is None:
                    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
                    save_size = video_size(annotated.shape, args.video_quality)
                    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                             src_fps, save_size)
                writer.write(fit(annotated, save_size))

            if not args.no_show:
                cv2.imshow(window, annotated)
                key = cv2.waitKey(1)
                # Closing the window (X) = quit normally, so all files are saved.
                if key in QUIT_KEYS or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key in SETTINGS_KEYS and live and not platform_utils.open_settings_dialog(cap):
                    print("Settings dialog is Windows-only. On Linux use: v4l2-ctl -d /dev/videoN --list-ctrls")
            frames += 1
    except KeyboardInterrupt:
        pass
    finally:
        if live:
            controls_end = platform_utils.read_camera_controls(cap)
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - t_start
    if frames:
        print(f"Processed {frames} frames in {elapsed:.1f}s ({frames / elapsed:.1f} FPS end-to-end)")
    if args.save and writer is not None:
        print(f"Saved: {args.save}")
    if rec:
        device = getattr(model.predictor, "device", None) if model.predictor else None
        out = rec.close(frames, elapsed, model_path, str(device or args.device), controls_end)
        print(f"Session saved: {out}")


if __name__ == "__main__":
    main()
