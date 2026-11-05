"""
Live viewer for the e-con Systems 8MP 4K RGB-IR USB3 camera (See3CAM_CU83).

Just run this file (e.g. the Run button in PyCharm). Everything is controlled
from inside the camera window - no terminal arguments needed:

    1        - color 1920x1080 (30 fps)
    2        - color 4K 3840x2160 (~13-15 fps)
    3        - infrared 1920x1080 (30 fps)
    s        - save a snapshot to the current folder
               (IR snapshots are 16-bit PNG holding the raw 10-bit values 0..1023)
    c        - open the camera's settings dialog (exposure, gain, white balance...)
    h        - hide / show the on-screen help
    q / Esc  - quit
"""
import time

import cv2
import numpy as np

WIN = "See3CAM"

# IR-only mode: the camera sends a 1920x1080 10-bit image packed as MIPI RAW10
# (4 pixels in 5 bytes), advertised as a "Y16" stream of 1920x675.
IR_W, IR_H = 1920, 1080

MODES = {
    "1": dict(name="Color 1080p", fourcc="UYVY", w=1920, h=1080, ir=False),
    "2": dict(name="Color 4K",    fourcc="UYVY", w=3840, h=2160, ir=False),
    "3": dict(name="Infrared",    fourcc="Y16 ", w=1920, h=675,  ir=True),
}
HELP = "1: Color  2: 4K  3: IR  |  s: save  c: settings  h: help  q: quit"


def show_error(msg):
    """Show an error in a message box (works without a terminal)."""
    import tkinter as tk
    from tkinter import messagebox
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror("See3CAM", msg)
    root.destroy()


def find_camera(name_part="See3CAM"):
    """Return the DirectShow index of the e-con camera (matched by device name)."""
    from pygrabber.dshow_graph import FilterGraph

    for i, name in enumerate(FilterGraph().get_input_devices()):
        if name_part.lower() in name.lower():
            return i
    return None


def unpack_raw10(buf):
    """Unpack a MIPI RAW10 buffer into a (IR_H, IR_W) uint16 image with values 0..1023."""
    g = buf.view(np.uint8).reshape(-1, 5).astype(np.uint16)
    px = np.empty((g.shape[0], 4), np.uint16)
    for k in range(4):
        px[:, k] = (g[:, k] << 2) | ((g[:, 4] >> (2 * k)) & 3)
    return px.reshape(IR_H, IR_W)


def open_camera(index, mode):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*mode["fourcc"]))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, mode["w"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, mode["h"])
    cap.set(cv2.CAP_PROP_FPS, 30)
    if mode["ir"]:
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)   # give us the raw bytes
    return cap


def draw_overlay(img, lines):
    """Draw text lines on a copy of the image, scaled to the image width."""
    out = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    scale = out.shape[1] / 1920
    y = int(45 * scale)
    for text in lines:
        for color, thick in (((0, 0, 0), 6), ((255, 255, 255), 2)):   # outline + text
            cv2.putText(out, text, (int(20 * scale), y), cv2.FONT_HERSHEY_SIMPLEX,
                        1.1 * scale, color, max(1, int(thick * scale)), cv2.LINE_AA)
        y += int(50 * scale)
    return out


def main():
    index = find_camera()
    if index is None:
        show_error("See3CAM not found.\n\nCheck the USB-C cable (USB 3) and try again.")
        return

    mode_key = "1"
    cap = open_camera(index, MODES[mode_key])
    if not cap.isOpened():
        show_error("Could not open the camera.\nIs another program (e.g. e-CAMView) using it?")
        return

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 720)

    show_help = True
    message, message_until = "", 0.0
    fps, frames, t0 = 0.0, 0, time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            show_error("Lost connection to the camera.")
            break

        mode = MODES[mode_key]
        if mode["ir"]:
            raw = unpack_raw10(frame)                       # uint16, 0..1023
            # stretch the actual range to 0..255 so a dim IR scene is visible
            shown = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            raw = shown = frame

        frames += 1
        if time.time() - t0 >= 1:
            fps, frames, t0 = frames / (time.time() - t0), 0, time.time()

        lines = [f"{mode['name']}  {fps:.0f} fps"]
        if show_help:
            lines.append(HELP)
        if time.time() < message_until:
            lines.append(message)
        cv2.imshow(WIN, draw_overlay(shown, lines))

        key = cv2.waitKey(1) & 0xFF
        ch = chr(key) if key < 128 else ""
        if key in (ord("q"), 27):
            break
        elif ch in MODES and ch != mode_key:
            cap.release()
            mode_key = ch
            cap = open_camera(index, MODES[mode_key])
            fps, frames, t0 = 0.0, 0, time.time()
        elif ch == "s":
            name = time.strftime(("ir_" if mode["ir"] else "snapshot_") + "%Y%m%d_%H%M%S.png")
            cv2.imwrite(name, raw)                         # saved without the overlay
            message, message_until = f"Saved {name}", time.time() + 2
        elif ch == "c":
            cap.set(cv2.CAP_PROP_SETTINGS, 1)
        elif ch == "h":
            show_help = not show_help

        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
