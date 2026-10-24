"""Every hit and its stroke: F3ED, the published model of F3Set (ICLR 2025), trained on TV tennis. Run by the GPU
step (dsa.gpu.app) on kept.mp4, piece by piece (runs of kept frames, from kept.json).

Per hit: side (near / far), position (deuce / middle / ad), shot (serve / return / stroke), and for returns and
strokes the hand (fh / bh) and technique (gs, slice, volley, smash, drop, lob); direction and outcome.
Frames are resized to 224 high and the model reads the centre 224 x 224, as it was trained.

    python -m dsa.gpu.strokes federer_usopen/2015              # list the hits
    python -m dsa.gpu.strokes federer_usopen/2015 --preview    # strokes/preview.mp4, labels on screen
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from dsa.paths import F3ED, F3SET, video_dir

ELEMENTS = ("near far deuce middle ad serve return stroke fh bh gs slice volley smash drop lob "
            "T B W CC DL DM II IO approach in winner forced-err unforced-err").split()
GROUPS = {"side": (0, 2), "position": (2, 5), "shot": (5, 8), "direction": (16, 24), "outcome": (25, 29)}
STROKE_GROUPS = {"hand": (8, 10), "technique": (10, 16)}      # not for serves
APPROACH, NMS = 24, 5


def f3ed(video: str, lengths: list[int], fps: float) -> dict:
    """On the GPU: F3ED's hit and label scores per 2-frame step, for each piece of `video` (`lengths` frames each,
    back to back)."""
    import cv2
    import timm
    import torch
    from torch.utils.data import DataLoader

    tc = timm.create_model                                    # the backbone's weights come from the checkpoint
    timm.create_model = lambda n, pretrained=False, **k: tc(n, pretrained=False, **k)
    # beside the video's own copy: dsa.gpu.gcp runs two videos at once on one VM, and a shared /tmp/frames gave
    # one video the other's strokes
    frames, pieces = f"{video}.frames", f"{video}.pieces.json"
    cap = cv2.VideoCapture(video)
    for k, n in enumerate(lengths):
        Path(f"{frames}/{k}").mkdir(parents=True, exist_ok=True)
        for i in range(n):
            ok, f = cap.read()
            assert ok, f"{video} ends early at piece {k}, frame {i}"
            h, w = f.shape[:2]
            cv2.imwrite(f"{frames}/{k}/{i:06d}.jpg", cv2.resize(f, (w * 224 // h, 224)))
    here = os.getcwd()
    os.chdir(F3SET)
    sys.path.insert(0, str(F3SET))
    from dataset.frame_process import ActionSeqVideoDataset
    from train_f3set_f3ed import F3Set
    from util.dataset import load_classes

    cfg = json.loads((F3ED / "config.json").read_text())
    classes = load_classes("data/f3set-tennis/elements.txt")
    model = F3Set(len(classes), cfg["feature_arch"], cfg["temporal_arch"], clip_len=cfg["clip_len"],
                  step=cfg["stride"], window=cfg["window"], use_ctx=cfg["use_ctx"], multi_gpu=False)
    model.load(torch.load(F3ED / "checkpoint_049.pt"))
    Path(pieces).write_text(json.dumps([{"video": str(k), "fps": fps, "num_frames": n, "far_hand": "RH",
                                       "near_hand": "RH", "events": []} for k, n in enumerate(lengths)]))
    ds = ActionSeqVideoDataset(classes, pieces, frames, cfg["clip_len"], overlap_len=cfg["clip_len"] // 2,
                               crop_dim=cfg["crop_dim"], stride=cfg["stride"])
    st = cfg["stride"]
    hit = [np.zeros((n // st, 2), np.float32) for n in lengths]
    el = [np.zeros((n // st, len(classes)), np.float32) for n in lengths]
    sup = [np.zeros(n // st, np.int32) for n in lengths]
    for b in DataLoader(ds, num_workers=8, batch_size=8):
        _, coarse, fine = model.predict(b["frame"], b["hand"])
        for j in range(len(coarse)):
            k, c, f, s = int(b["video"][j]), coarse[j], fine[j], int(b["start"][j])
            if s < 0:
                c, f, s = c[-s:], f[-s:], 0
            e = min(s + len(c), len(hit[k]))
            hit[k][s:e] += c[:e - s]
            el[k][s:e] += f[:e - s]
            sup[k][s:e] += 1
    os.chdir(here)
    shutil.rmtree(frames)
    return {"hit": [h / np.maximum(u, 1)[:, None] for h, u in zip(hit, sup)],
            "elements": [x / np.maximum(u, 1)[:, None] for x, u in zip(el, sup)], "stride": st}


def decode(scores: dict, kept: dict) -> list[dict]:
    """One row per hit: frame in kept.mp4, source_frame and time (s) in the match video."""
    rows, st, fps, k0 = [], int(scores["stride"]), kept["fps"], 0
    for (a, b), hit, el in zip(kept["pieces"], scores["hit"], scores["elements"]):
        bg = hit[:, 0]
        for i in range(len(hit)):
            w = bg[max(i - NMS // 2, 0):i + NMS // 2 + 1]
            if hit[i].argmax() != 1 or bg[i] != w.min():       # a hit, and the most confident bin near it
                continue
            s = el[i]
            r = {"frame": k0 + i * st, "source_frame": a + i * st, "time": round((a + i * st) / fps, 2),
                 "score": round(float(hit[i, 1]), 3)}
            for g, (lo, hi) in GROUPS.items():
                r[g] = ELEMENTS[lo + int(s[lo:hi].argmax())]
            for g, (lo, hi) in STROKE_GROUPS.items():
                r[g] = None if r["shot"] == "serve" else ELEMENTS[lo + int(s[lo:hi].argmax())]
            r["approach"] = bool(s[APPROACH] > 0.5)
            rows.append(r)
        k0 += b - a
    return rows


def label(r: dict) -> str:
    return " ".join(x for x in (r["side"], r["shot"], r["hand"], r["technique"], r["direction"], r["outcome"]) if x)


def preview(spec: str, hold: float = 1.2) -> Path:
    """strokes/preview.mp4: the analysed part of kept.mp4, each hit's label on screen for `hold` s."""
    import cv2

    d = video_dir(spec)
    hits = json.loads((d / "strokes/strokes.json").read_text())
    fps, n, src = hits["fps"], hits["frames"], {}
    kept = json.loads((d / "kept.json").read_text())
    for a, b in kept["pieces"]:
        src.update({len(src) + i: a + i for i in range(b - a)})
    cap, (w, h) = cv2.VideoCapture(str(d / "kept.mp4")), (1280, 720)
    out = d / "strokes/preview.mp4"
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                            "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-crf", "23", "-preset", "veryfast",
                            "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)
    for f in range(n):
        img = cv2.resize(cap.read()[1], (w, h))
        m, s = divmod(src[f] / fps, 60)
        cv2.putText(img, f"{int(m)}:{s:05.2f}", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        for r in [r for r in hits["hits"] if 0 <= f - r["frame"] < hold * fps][-1:]:
            colour = (0, 220, 255) if r["side"] == "near" else (255, 160, 0)
            cv2.rectangle(img, (0, 0), (w, 64), (0, 0, 0), -1)
            cv2.putText(img, label(r).upper(), (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.3, colour, 3)
            if f == r["frame"]:
                cv2.rectangle(img, (0, 0), (w - 1, h - 1), colour, 12)
        enc.stdin.write(img.tobytes())
    enc.stdin.close()
    enc.wait()
    return out


if __name__ == "__main__":
    spec = sys.argv[1]
    if sys.argv[2:] == ["--preview"]:
        print(preview(spec))
        sys.exit()
    for r in json.loads((video_dir(spec) / "strokes/strokes.json").read_text())["hits"]:
        m, s = divmod(r["time"], 60)
        print(f"{int(m):2d}:{s:05.2f}  {label(r):40s} {r['score']:.2f}")
