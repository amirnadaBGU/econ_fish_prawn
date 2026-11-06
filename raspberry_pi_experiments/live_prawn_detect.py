#!/usr/bin/env python3
"""Live prawn detection: camera -> YOLO model -> on-screen boxes, count and FPS.

Raspberry Pi 5 version (Nov 2026 robot demo). Changes vs. the October demo:
  * --density-window: smoothed count ("density") over the last N seconds, shown on screen,
    written to per_frame.csv and to status.json (updated every second, for the feeder).
  * --max-minutes: stop by itself after N minutes.
  * Stops cleanly on Ctrl+C *and* on kill / systemd stop (SIGTERM), so the recording is
    always closed properly.
  * --video-fps: frame rate of the recorded videos (default 10) - keeps the Pi CPU for the model.
  * A camera that drops a few frames does not end the run; a status line is printed every 5 s.

Examples:
    python live_prawn_detect.py --source 0 --model models/prawn_yolo11n_pose_v0.1.pt
    python live_prawn_detect.py --source 0 --model models/prawn_yolo11n_pose_v0.1.pt --record runs
    python live_prawn_detect.py --source pond.mp4 --model models/prawn.onnx --save out.mp4
    python live_prawn_detect.py --source 0 --model models/prawn.engine --no-show --save out.mp4

Press q (or Esc) in the window to quit.

--record DIR saves a complete session into DIR/<timestamp>/:
    annotated.mp4    video with boxes, count and FPS
    raw.mp4          clean camera video (disable with --no-raw)
    detections.csv   one row per detected prawn per frame (box, conf, 4 keypoints, CL/TL)
    per_frame.csv    one row per frame (count, mean TL, inference ms, FPS)

Length: CL = carapace -> eyes, TL = rostrum -> tail, in pixels from the keypoints.
Pass --mm-per-px (from the camera height / calibration) to also get millimetres.
A length is left empty when either of its keypoints is below --kpt-conf.
    summary.json     model, settings, camera, computer, totals
    snapshots/       one annotated image every --snapshot-every seconds
"""
import argparse
import collections
import os
import signal
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

KPT_NAMES = ["carapace", "eyes", "rostrum", "tail"]
CL_KPTS = (0, 1)  # carapace start -> eyes
TL_KPTS = (2, 3)  # rostrum -> tail


def prawn_lengths(result, kpt_conf, mm_per_px=None):
    """Per detection: (cl_px, tl_px, cl_mm, tl_mm). None where a keypoint is unreliable."""
    kpts = result.keypoints
    n = len(result.boxes)
    if kpts is None or n == 0:
        return [(None, None, None, None)] * n
    xy = kpts.xy.cpu().numpy()
    cf = kpts.conf.cpu().numpy() if kpts.conf is not None else np.ones(xy.shape[:2])
    out = []
    for i in range(n):
        vals = []
        for a, b in (CL_KPTS, TL_KPTS):
            ok = cf[i][a] >= kpt_conf and cf[i][b] >= kpt_conf
            vals.append(float(np.linalg.norm(xy[i][a] - xy[i][b])) if ok else None)
        cl, tl = vals
        to_mm = lambda v: v * mm_per_px if (v is not None and mm_per_px) else None
        out.append((cl, tl, to_mm(cl), to_mm(tl)))
    return out


def mean_tl(lengths, use_mm):
    vals = [l[3] if use_mm else l[1] for l in lengths]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def parse_args():
    p = argparse.ArgumentParser(description="Live prawn detection with a YOLO model.")
    p.add_argument("--source", default="0",
                   help="Camera index (0, 1, ...), video file / URL, or a GStreamer "
                        "pipeline string (anything containing '!'). Default: 0")
    p.add_argument("--model", required=True, help="Model file: .pt, .onnx or .engine")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default 0.25)")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size (default 640)")
    p.add_argument("--device", default=None,
                   help="Inference device: cpu, mps, 0 (CUDA GPU). Default: auto")
    p.add_argument("--width", type=int, default=None, help="Requested camera width")
    p.add_argument("--height", type=int, default=None, help="Requested camera height")
    p.add_argument("--save", default=None, help="Save the annotated video to this .mp4 path")
    p.add_argument("--record", default=None, metavar="DIR",
                   help="Record a full session (videos, CSVs, summary, snapshots) into DIR/<timestamp>/")
    p.add_argument("--no-raw", action="store_true", help="With --record: do not save raw.mp4")
    p.add_argument("--snapshot-every", type=float, default=10,
                   help="With --record: seconds between annotated snapshots (default 10, 0 = off)")
    p.add_argument("--kpts", action="store_true", help="Also draw the 4 pose keypoints")
    p.add_argument("--mm-per-px", type=float, default=None,
                   help="Scale for converting lengths to mm (from camera height / calibration). "
                        "Without it, lengths are in pixels only")
    p.add_argument("--kpt-conf", type=float, default=0.5,
                   help="Minimum keypoint confidence for a length to be measured (default 0.5)")
    p.add_argument("--no-show", action="store_true", help="Do not open a window (headless)")
    p.add_argument("--density-window", type=float, default=10,
                   help="Seconds over which the prawn count is averaged into 'density' (default 10)")
    p.add_argument("--max-minutes", type=float, default=None,
                   help="Stop automatically after this many minutes (default: run until stopped)")
    p.add_argument("--video-fps", type=float, default=10,
                   help="With --record: frame rate of the saved camera videos (default 10)")
    p.add_argument("--status-file", default=None,
                   help="JSON file updated every second with the latest count and density "
                        "(default: status.json inside the --record folder)")
    return p.parse_args()


def open_source(source, width=None, height=None):
    if source.isdigit():
        if platform.system() == "Linux":
            cap = cv2.VideoCapture(int(source), cv2.CAP_V4L2)
        else:
            cap = cv2.VideoCapture(int(source))
    elif "!" in source:
        cap = cv2.VideoCapture(source, cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(source)
    if (source.isdigit() or source.startswith("/dev/video")) and (width or height):
        # USB (UVC) cameras usually need MJPG to deliver HD at full frame rate.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {source}")
    return cap


def draw_overlay(img, count, fps, tl=None, unit="px", density=None, window=None):
    text = f"Prawns: {count}   FPS: {fps:.1f}"
    if density is not None:
        text += f"   Density({window:.0f}s): {density:.1f}"
    if tl is not None:
        text += f"   Mean TL: {tl:.0f} {unit}"
    scale = max(0.6, img.shape[1] / 1600)  # readable from 640 px up to 4K/5K frames
    thick = max(2, round(scale * 2))
    pad = round(10 * scale)
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.rectangle(img, (0, 0), (w + 2 * pad, h + 2 * pad), (0, 0, 0), -1)
    cv2.putText(img, text, (pad, h + pad), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 0), thick)


def system_info():
    info = {"os": platform.platform(), "python": platform.python_version(),
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

    def __init__(self, root, args, cap):
        self.dir = Path(root) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.snap_dir = self.dir / "snapshots"
        if args.snapshot_every > 0:
            self.snap_dir.mkdir(exist_ok=True)
        self.args = args
        self.save_raw = not args.no_raw
        self.cam = {"width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    "reported_fps": round(cap.get(cv2.CAP_PROP_FPS) or 0, 2)}
        self.is_file = Path(args.source).is_file()
        self.started = datetime.now()
        self.writers = None
        self.t_first = None
        self.written = 0
        self.last_snap = None
        self.counts, self.infer_ms = [], []
        self.tl_px, self.tl_mm = [], []

        self.det_f = open(self.dir / "detections.csv", "w", newline="")
        self.det = csv.writer(self.det_f)
        kp_cols = [f"{k}_{c}" for k in KPT_NAMES for c in ("x", "y", "conf")]
        self.det.writerow(["frame", "time_s", "prawn", "x1", "y1", "x2", "y2", "conf"] + kp_cols
                          + ["cl_px", "tl_px", "cl_mm", "tl_mm"])
        self.frm_f = open(self.dir / "per_frame.csv", "w", newline="")
        self.frm = csv.writer(self.frm_f)
        self.frm.writerow(["frame", "time_s", "count", "mean_tl_px", "mean_tl_mm", "infer_ms", "fps", "density"])

    def _open_writers(self, fps, shape):
        h, w = shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_fps = round(fps, 2)
        self.writers = [cv2.VideoWriter(str(self.dir / "annotated.mp4"), fourcc, fps, (w, h))]
        if self.save_raw:
            self.writers.append(cv2.VideoWriter(str(self.dir / "raw.mp4"), fourcc, fps, (w, h)))

    def _write_video(self, raw, annotated):
        self.writers[0].write(annotated)
        if self.save_raw:
            self.writers[1].write(raw)

    def add(self, idx, t, raw, annotated, result, infer_ms, fps, lengths, density=None):
        boxes = result.boxes
        kpts = result.keypoints
        xyxy = boxes.xyxy.cpu().numpy() if len(boxes) else []
        confs = boxes.conf.cpu().numpy() if len(boxes) else []
        kxy = kpts.xy.cpu().numpy() if kpts is not None and len(boxes) else None
        kcf = kpts.conf.cpu().numpy() if kxy is not None and kpts.conf is not None else None
        for i in range(len(xyxy)):
            row = [idx, f"{t:.3f}", i] + [f"{v:.1f}" for v in xyxy[i]] + [f"{confs[i]:.3f}"]
            for k in range(len(KPT_NAMES)):
                if kxy is not None:
                    row += [f"{kxy[i][k][0]:.1f}", f"{kxy[i][k][1]:.1f}",
                            f"{kcf[i][k]:.3f}" if kcf is not None else ""]
                else:
                    row += ["", "", ""]
            row += ["" if v is None else f"{v:.1f}" for v in lengths[i]]
            self.det.writerow(row)
            if lengths[i][1] is not None:
                self.tl_px.append(lengths[i][1])
            if lengths[i][3] is not None:
                self.tl_mm.append(lengths[i][3])
        fmt = lambda v: "" if v is None else f"{v:.1f}"
        self.frm.writerow([idx, f"{t:.3f}", len(xyxy), fmt(mean_tl(lengths, False)),
                           fmt(mean_tl(lengths, True)), f"{infer_ms:.1f}", f"{fps:.1f}",
                           "" if density is None else f"{density:.2f}"])
        self.counts.append(len(xyxy))
        self.infer_ms.append(infer_ms)

        # Video: a file is written frame-for-frame at its own FPS. A camera is written at its
        # own FPS too, repeating each processed frame by wall-clock time so playback is
        # real-time however fast or slow the computer is.
        reported = self.cam["reported_fps"]
        if self.writers is None:
            self.t_first = t
            cam_fps = reported if 1 <= reported <= 120 else 30.0
            # From a camera, write at --video-fps (playback stays real-time); from a file, keep its FPS.
            self._open_writers(cam_fps if self.is_file else min(cam_fps, self.args.video_fps),
                               raw.shape)
        repeats = 1
        if not self.is_file:
            repeats = max(0, int((t - self.t_first) * self.video_fps) + 1 - self.written)
        for _ in range(repeats):
            self._write_video(raw, annotated)
        self.written += repeats
        self._snapshot(t, annotated)

    def _snapshot(self, t, annotated):
        every = self.args.snapshot_every
        if every > 0 and (self.last_snap is None or t - self.last_snap >= every):
            cv2.imwrite(str(self.snap_dir / f"t{t:07.1f}s.jpg"), annotated)
            self.last_snap = t

    def close(self, frames, elapsed, model_path, device):
        for w in self.writers or []:
            w.release()
        self.det_f.close()
        self.frm_f.close()
        sha = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()[:16]
        n = len(self.counts)
        summary = {
            "started": self.started.isoformat(timespec="seconds"),
            "ended": datetime.now().isoformat(timespec="seconds"),
            "duration_s": round(elapsed, 1),
            "model": {"file": Path(model_path).name, "sha256_16": sha},
            "settings": {"conf": self.args.conf, "imgsz": self.args.imgsz,
                         "density_window_s": self.args.density_window,
                         "device_requested": self.args.device, "device_used": device},
            "source": self.args.source,
            "camera": self.cam,
            "video_fps_written": getattr(self, "video_fps", None),
            "frames": frames,
            "avg_fps_end_to_end": round(frames / elapsed, 2) if elapsed > 0 else None,
            "avg_infer_ms": round(sum(self.infer_ms) / n, 1) if n else None,
            "prawns_per_frame": {"mean": round(sum(self.counts) / n, 2) if n else None,
                                 "max": max(self.counts) if n else None,
                                 "frames_with_prawns": sum(c > 0 for c in self.counts)},
            "keypoints": KPT_NAMES,
            "length": {
                "definition": "CL = carapace->eyes, TL = rostrum->tail (per detection, per frame)",
                "mm_per_px": self.args.mm_per_px,
                "kpt_conf_min": self.args.kpt_conf,
                "measured_detections": len(self.tl_px),
                "tl_px_median": round(float(np.median(self.tl_px)), 1) if self.tl_px else None,
                "tl_mm_median": round(float(np.median(self.tl_mm)), 1) if self.tl_mm else None,
            },
            "computer": system_info(),
        }
        (self.dir / "summary.json").write_text(json.dumps(summary, indent=2))
        return self.dir


class Density:
    """Mean prawn count over the last `window` seconds."""

    def __init__(self, window):
        self.window = window
        self.buf = collections.deque()

    def add(self, t, count):
        self.buf.append((t, count))
        while self.buf and t - self.buf[0][0] > self.window:
            self.buf.popleft()
        return sum(c for _, c in self.buf) / len(self.buf)


def write_status(path, data):
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)  # atomic: a reader never sees a half-written file


def _stop(signum, frame):
    raise KeyboardInterrupt


def main():
    args = parse_args()
    signal.signal(signal.SIGTERM, _stop)  # kill / systemd stop -> same clean exit as Ctrl+C
    model = YOLO(args.model, task="pose")
    cap = open_source(args.source, args.width, args.height)
    rec = SessionRecorder(args.record, args, cap) if args.record else None
    if rec:
        print(f"Recording session to: {rec.dir}")
    status_path = args.status_file or (rec.dir / "status.json" if rec else None)
    density = Density(args.density_window)
    is_camera = not Path(args.source).is_file()
    max_s = args.max_minutes * 60 if args.max_minutes else None
    failed_reads = 0
    last_status = last_print = -1e9
    print("Running. Stop with Ctrl+C." + (f" Auto-stop after {args.max_minutes} min." if max_s else ""))

    writer = None
    fps_ema = None
    frames = 0
    t_start = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                failed_reads += 1
                if is_camera and failed_reads < 50:  # a USB camera can drop a few frames
                    time.sleep(0.05)
                    continue
                if is_camera:
                    print("Camera stopped sending frames - ending the run.")
                break
            failed_reads = 0
            t_now = time.perf_counter() - t_start
            if max_s and t_now >= max_s:
                print(f"Reached --max-minutes ({args.max_minutes}) - stopping.")
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
            lengths = prawn_lengths(result, args.kpt_conf, args.mm_per_px)
            use_mm = args.mm_per_px is not None
            count = len(result.boxes)
            t = time.perf_counter() - t_start
            dens = density.add(t, count)
            draw_overlay(annotated, count, fps_ema, mean_tl(lengths, use_mm),
                         "mm" if use_mm else "px", dens, args.density_window)

            if rec:
                rec.add(frames, t, frame, annotated, result, dt * 1000, fps_ema, lengths, dens)
            if status_path and t - last_status >= 1:
                write_status(status_path, {
                    "time": datetime.now().isoformat(timespec="seconds"), "frame": frames,
                    "count": count, "density": round(dens, 2),
                    "density_window_s": args.density_window, "fps": round(fps_ema, 2)})
                last_status = t
            if t - last_print >= 5:
                print(f"[{t:7.1f}s] frames={frames}  FPS={fps_ema:.1f}  prawns={count}  "
                      f"density={dens:.1f}", flush=True)
                last_print = t

            if args.save:
                if writer is None:
                    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
                    h, w = annotated.shape[:2]
                    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                             src_fps, (w, h))
                writer.write(annotated)

            if not args.no_show:
                cv2.imshow("Prawn detection", annotated)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            frames += 1
    except KeyboardInterrupt:
        pass
    finally:
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
        out = rec.close(frames, elapsed, args.model, str(device or args.device))
        print(f"Session saved: {out}")


if __name__ == "__main__":
    main()
