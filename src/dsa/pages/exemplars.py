"""Side-by-side exemplar videos for the forehand findings: a typical early stroke next to a typical late one,
cropped on the player, his skeleton drawn on top with the measurement the finding is about, synced on contact
and slowed down, holding the moment that is measured.

    render(project, early, late, kind, out)   early/late: dicts with video, frame, value, label, pose (len(GRID), 13, 3)
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from dsa.measure.technique import GRID, HZ
from dsa.paths import video_dir

SIZE, GAP, BAR = 440, 6, 34                     # panel px; gap between panels; caption bar height
T0, T1, SLOW, HOLD = -0.6, 0.45, 0.4, 1.2       # seconds shown around contact; playback speed; hold at the moment
BONES = [(1, 2), (1, 3), (3, 5), (2, 4), (4, 6), (1, 7), (2, 8), (7, 8), (7, 9), (9, 11), (8, 10), (10, 12)]
WHITE, ACCENT, SOFT, DARK = (240, 240, 240), (235, 205, 140), (170, 170, 170), (24, 24, 24)  # BGR; accent pale blue
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text(img, s, org, scale=0.5, color=WHITE, thick=1):
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def _band(img, y0, y1, alpha=0.55):
    img[y0:y1] = (img[y0:y1] * (1 - alpha)).astype(np.uint8)


def _frames(spec: str, frame: int, fps: float, t0: float, t1: float = T1) -> dict[int, np.ndarray]:
    d = video_dir(spec)
    cap = cv2.VideoCapture(str(d / "kept.mp4" if (d / "kept.mp4").exists() else d / "source.mp4"))   # phone: no cut
    a, b = frame + int(np.floor(t0 * fps)) - 1, frame + int(np.ceil(t1 * fps)) + 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(a, 0))
    out = {}
    for g in range(max(a, 0), b + 1):
        ok, img = cap.read()
        if not ok:
            break
        out[g] = img
    return out


def _panel(project: str, s: dict, kind: str, moment: float, t0: float, t1: float = T1,
           held: bool = True) -> list[np.ndarray]:
    """One stroke's frames at HZ from t0 to t1 (the measured moment held, if held), drawn."""
    spec = f"{s.get('project', project)}/{s['video']}"           # a stroke may come from another project
    fps = json.loads((video_dir(spec) / "players.json").read_text())["fps"]
    imgs = _frames(spec, s["frame"], fps, t0, t1)
    pose, hz = s["pose"], s.get("hz", HZ)              # the pose's time grid: GRID's span at hz
    zero = int(round(-GRID[0] * hz))
    ts = np.round(np.arange(t0, t1 + 1e-9, 1 / hz), 4)
    # the crop follows his hips (smoothed), sized to his height, so he stays large while he runs
    def hip_at(t):
        q = pose[zero + int(round(t * hz)), 7:9, :2].mean(0)
        return q if np.isfinite(q).all() else None
    hips = np.array([hip_at(t) if hip_at(t) is not None else [np.nan, np.nan] for t in ts], float)
    for c in range(2):
        ok_ = np.isfinite(hips[:, c])
        hips[:, c] = np.interp(np.arange(len(ts)), np.flatnonzero(ok_), hips[ok_, c])
    k = np.ones(7) / 7
    hips = np.stack([np.convolve(np.pad(hips[:, c], 3, mode="edge"), k, "valid") for c in range(2)], 1)
    tall = np.nanmedian([np.nanmax(pose[zero + int(round(t * hz)), :, 1]) - np.nanmin(pose[zero + int(round(t * hz)), :, 1])
                         for t in ts])
    side = tall * 2.0
    sc = SIZE / side
    trail, out, hold = [], [], int(np.argmin(abs(ts - moment)))
    for n_, t in enumerate(ts):
        ox, oy = hips[n_, 0] - side / 2, hips[n_, 1] - side * 0.45
        ih, iw = next(iter(imgs.values())).shape[:2]     # kept inside the picture where it fits (a phone films him
        ox = min(max(ox, 0), iw - side) if side <= iw else ox            # near its bottom edge): no black edges
        oy = min(max(oy, 0), ih - side) if side <= ih else oy
        P = lambda p: (int(round((p[0] - ox) * sc)), int(round((p[1] - oy) * sc)))      # noqa: E731
        gi = zero + int(round(t * hz))
        g = s["frame"] + int(round(t * fps))
        src = imgs.get(g, next(iter(imgs.values())))
        m = np.float32([[sc, 0, -ox * sc], [0, sc, -oy * sc]])
        img = cv2.warpAffine(src, m, (SIZE, SIZE), borderValue=(0, 0, 0))
        img = cv2.addWeighted(img, 0.78, np.zeros_like(img), 0.22, 0)
        p = pose[gi]
        ok = np.isfinite(p[:, :2]).all(1)
        trail.append((t, p.copy()))
        hip = p[7:9, :2].mean(0)
        after = n_ >= hold
        # the measurement, under the skeleton
        if kind in ("finish_side", "finish_height", "wrist_speed", "takeback", "swing_time"):
            w = [P(q[6]) for tt, q in trail if np.isfinite(q[6, :2]).all() and tt >= t - 0.5]
            for a, b in zip(w, w[1:]):
                cv2.line(img, a, b, ACCENT, 2, cv2.LINE_AA)
        if kind in ("toss_arm", "trophy_time"):
            w = [P(q[5]) for tt, q in trail if np.isfinite(q[5, :2]).all()]
            for a, b in zip(w, w[1:]):
                cv2.line(img, a, b, ACCENT, 2, cv2.LINE_AA)
            if s.get("peak") is not None and t >= s["peak"]:
                cv2.circle(img, P(s["peak_pt"]), 9, ACCENT, 2, cv2.LINE_AA)
                if kind == "toss_arm":
                    hy = P(s["peak_head"])[1]
                    cv2.line(img, (0, hy), (SIZE, hy), SOFT, 1, cv2.LINE_AA)
        if kind == "lean" and after:
            c = s["at"]
            fx, fy = P(c[11:13].mean(0))
            sx, sy = P(c[1:3].mean(0))
            cv2.line(img, (fx, 0), (fx, SIZE), SOFT, 1, cv2.LINE_AA)
            cv2.line(img, (fx, sy), (sx, sy), ACCENT, 3, cv2.LINE_AA)
        if kind == "into_court":
            w = [P(q[11:13, :2].mean(0)) for tt, q in trail if np.isfinite(q[11:13, :2]).all()]
            for a, b in zip(w, w[1:]):
                cv2.line(img, a, b, ACCENT, 3, cv2.LINE_AA)
        if kind == "head_still":
            w = [P(q[0]) for tt, q in trail if np.isfinite(q[0, :2]).all() and -0.1 <= tt <= 0.1]
            for a, b in zip(w, w[1:]):
                cv2.line(img, a, b, ACCENT, 3, cv2.LINE_AA)
        if kind == "contact_side" and after and np.isfinite(hip).all():
            hx = P(hip)[0]
            c = s["at"]                                  # pose at the measured moment
            wx, wy = P(c[6])
            cv2.line(img, (hx, 0), (hx, SIZE), SOFT, 1, cv2.LINE_AA)
            cv2.line(img, (hx, wy), (wx, wy), ACCENT, 2, cv2.LINE_AA)
        if kind in ("finish_side",) and after:
            c = s["at"]
            hx = P(c[7:9, :2].mean(0))[0]
            cv2.line(img, (hx, 0), (hx, SIZE), SOFT, 1, cv2.LINE_AA)
            cv2.circle(img, P(c[6]), 9, ACCENT, 2, cv2.LINE_AA)
        if kind == "finish_height" and after:
            c = s["at"]
            sy = P(c[1:3, :2].mean(0))[1]
            cv2.line(img, (0, sy), (SIZE, sy), SOFT, 1, cv2.LINE_AA)
            cv2.circle(img, P(c[6]), 9, ACCENT, 2, cv2.LINE_AA)
        if kind == "takeback" and s.get("tb") is not None and t >= s["tb"]:
            cv2.circle(img, P(s["tb_pt"]), 9, ACCENT, 2, cv2.LINE_AA)
        # skeleton
        for a, b in BONES:
            if ok[a] and ok[b]:
                hot = (kind in ("shoulder_width", "shoulder_tilt", "shoulder_coil") and (a, b) == (1, 2)) or \
                      (kind in ("stance_fwd_m", "stance_side_m") and False) or \
                      (kind == "knee_bend" and (a, b) in ((7, 9), (9, 11), (8, 10), (10, 12)))
                cv2.line(img, P(p[a]), P(p[b]), ACCENT if hot else WHITE, 3 if hot else 1, cv2.LINE_AA)
        for j in range(13):
            if ok[j]:
                cv2.circle(img, P(p[j]), 2, WHITE, -1, cv2.LINE_AA)
        if kind in ("stance_m", "stance_fwd_m", "stance_side_m") and ok[11] and ok[12]:
            cv2.line(img, P(p[11]), P(p[12]), ACCENT, 3, cv2.LINE_AA)
        bi, fi = s.get("feet", (12, 11))                 # back and front ankle
        if kind == "stance_angle" and ok[bi] and ok[fi]:
            back, front = np.array(P(p[bi]), float), np.array(P(p[fi]), float)
            if after and s.get("baseline") is not None:
                # the baseline's direction on screen where he stands, from the back foot toward the front foot's side
                d = np.array(s["baseline"], float)
                d = d / np.linalg.norm(d) * (1 if d @ (front - back) >= 0 else -1)
                r = np.linalg.norm(front - back)
                cv2.line(img, tuple(back.astype(int)), tuple((back + d * r * 1.15).astype(int)), SOFT, 1, cv2.LINE_AA)
                a0, a1 = np.degrees(np.arctan2(d[1], d[0])), np.degrees(np.arctan2(*(front - back)[::-1]))
                a1 = a0 + (a1 - a0 + 180) % 360 - 180            # the short way round
                cv2.ellipse(img, tuple(back.astype(int)), (int(r * 0.45),) * 2, 0, min(a0, a1), max(a0, a1), ACCENT, 1, cv2.LINE_AA)
            cv2.line(img, tuple(back.astype(int)), tuple(front.astype(int)), ACCENT, 3, cv2.LINE_AA)
        # words
        _band(img, 0, 50)
        _text(img, s["label"], (12, 21), 0.5)
        if s.get("where"):
            _text(img, s["where"], (12, 41), 0.42, (200, 200, 200))
        if held:
            _text(img, f"{t:+.2f} s" if abs(t) > 0.005 else "contact", (SIZE - 78, 21), 0.5)
        if after and s.get("value_text"):
            _band(img, SIZE - 34, SIZE)
            _text(img, s["value_text"], (12, SIZE - 12), 0.5)
        out.append(img)
        if n_ == hold and held:
            out += [img] * int(HOLD * hz * SLOW)
    return out


def render(project: str, early: dict, late: dict, kind: str, moment: float, out: Path, t0: float = T0,
           titles: tuple[str, str] = ("2003-07", "2017-19")) -> Path:
    a, b = _panel(project, early, kind, moment, t0), _panel(project, late, kind, moment, t0)
    n = min(len(a), len(b))
    w, h = 2 * SIZE + GAP, SIZE + BAR
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                            "-r", f"{early.get('hz', HZ) * SLOW:g}", "-i", "-", "-c:v", "libx264", "-crf", "21", "-preset", "veryfast",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    for i in range(n):
        canvas = np.full((h, w, 3), 24, np.uint8)
        canvas[BAR:, :SIZE], canvas[BAR:, SIZE + GAP:] = a[i], b[i]
        _text(canvas, titles[0], (12, 23), 0.55)
        _text(canvas, titles[1], (SIZE + GAP + 12, 23), 0.55)
        enc.stdin.write(canvas.tobytes())
    enc.stdin.close()
    enc.wait()
    return out


def single(project: str, s: dict, out: Path, t0: float, t1: float) -> Path:
    """One stroke from t0 to t1 at real speed, no hold, every frame a keyframe: a page scrubs it in step with its
    skeleton (video time = seconds from t0)."""
    frames = _panel(project, s, "none", 0.0, t0, t1, held=False)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{SIZE}x{SIZE}",
                            "-r", f"{s.get('hz', HZ):g}", "-i", "-", "-c:v", "libx264", "-crf", "24", "-preset", "veryfast",
                            "-g", "1", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    for img in frames:
        enc.stdin.write(img.tobytes())
    enc.stdin.close()
    enc.wait()
    return out
