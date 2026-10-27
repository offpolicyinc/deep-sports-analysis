"""Reading videos, and cutting clips (dsa.ingest.projects uses `cut` for the test clips)."""
from __future__ import annotations

import queue
import subprocess
import threading
from pathlib import Path

import numpy as np


def info(video: str) -> tuple[float, int, int]:
    """(fps, width, height)."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    out = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return out


def frame_at(video: str, i: int) -> np.ndarray:
    import cv2

    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, bgr = cap.read()
    cap.release()
    return bgr


def frames(video: str, limit: int = 0, start: int = 0):
    """Yield BGR frames from frame `start`, decoded by ffmpeg on all cores in a reader thread."""
    fps, w, h = info(video)
    cmd = ["ffmpeg", "-v", "error", "-threads", "0"] + (["-ss", f"{start / fps:.4f}"] if start else []) + ["-i", str(video)]
    if limit:
        cmd += ["-frames:v", str(limit)]
    p = subprocess.Popen(cmd + ["-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE, bufsize=w * h * 3 * 4)
    q: queue.Queue = queue.Queue(maxsize=64)

    def read():
        while (buf := p.stdout.read(w * h * 3)) and len(buf) == w * h * 3:
            q.put(np.frombuffer(buf, np.uint8).reshape(h, w, 3))
        q.put(None)

    threading.Thread(target=read, daemon=True).start()
    try:
        while (f := q.get()) is not None:
            yield f
    finally:                    # a caller that stops early must not leave ffmpeg holding the file open (on a Modal
        p.kill()                # volume, an open file makes volume.commit() wait forever)
        p.wait()


def frames_at(video: str, wanted) -> "Iterator[tuple[int, np.ndarray]]":
    """(i, BGR frame) for each frame i in `wanted`, in order: ffmpeg still decodes every frame, but converts and
    sends only these (byte-identical to frames()'s), which is most of frames()'s cost when few are wanted."""
    import tempfile

    fps, w, h = info(video)
    runs: list[list[int]] = []
    for i in sorted(wanted):
        if runs and i == runs[-1][1] + 1:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    if not runs:
        return
    with tempfile.NamedTemporaryFile("w", suffix=".txt") as script:    # a filter script: thousands of frames
        script.write("select=" + "+".join(f"between(n\\,{a}\\,{b})" for a, b in runs))
        script.flush()
        p = subprocess.Popen(["ffmpeg", "-v", "error", "-threads", "0", "-i", str(video), "-filter_script:v",
                              script.name, "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                             stdout=subprocess.PIPE, bufsize=w * h * 3 * 4)
        try:
            for i in (i for a, b in runs for i in range(a, b + 1)):
                buf = p.stdout.read(w * h * 3)
                if len(buf) < w * h * 3:
                    break
                yield i, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
        finally:
            p.kill()
            p.wait()


def cut(src: str, start: float, seconds: float, out: Path) -> None:
    """A clip at 30 fps, at most 1080p, no sound."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".part" + out.suffix)     # named `out` only once whole
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-i", src, "-t", str(seconds), "-r", "30",
                    "-vf", "scale=-2:'min(1080,ih)'", "-an", "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
                    str(tmp)], check=True)
    tmp.rename(out)

