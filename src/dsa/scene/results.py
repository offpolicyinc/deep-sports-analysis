"""A video's results after steps 2-4, as the renders and the measures read them, and its points."""
from __future__ import annotations

import json

import numpy as np

from dsa.paths import hits_file, video_dir
from dsa.scene.court import phone_saved
from dsa.scene.court import tv as court_tv


def points(hits: list[dict], pieces: list[tuple[int, int]], fps: float) -> list[tuple[int, int, int]]:
    """[(hits, first, end)] per point: a serve, then hits until the next serve, a cut or a 4 s pause; frames
    from 1.5 s before the serve to 2 s after the last hit, within its piece."""
    hs, out = sorted(hits, key=lambda h: h["frame"]), []
    piece = lambda f: next(k for k, (a, b) in enumerate(pieces) if a <= f < b)          # noqa: E731
    for i, h in enumerate(hs):
        if h["shot"] != "serve":
            continue
        j = i
        while (j + 1 < len(hs) and hs[j + 1]["shot"] != "serve" and piece(hs[j + 1]["frame"]) == piece(h["frame"])
               and hs[j + 1]["frame"] - hs[j]["frame"] < 4 * fps):
            j += 1
        a, b = pieces[piece(h["frame"])]
        out.append((j - i + 1, max(a, h["frame"] - int(1.5 * fps)), min(b, hs[j]["frame"] + int(2 * fps))))
    return out


def load(spec: str) -> dict:
    """A video's results after step 4, read once: its frames' boxes, skeletons and ball (raw.npz), the court,
    who is at each end (players.json), names, hits (dsa.paths.hits_file) and where its pieces start."""
    src = video_dir(spec)
    z = np.load(src / "raw.npz")
    cut = np.cumsum(z["n"])[:-1]
    pl = json.loads((src / "players.json").read_text())
    names = json.loads((src / "names.json").read_text()) if (src / "names.json").exists() else {}
    sj, kj = hits_file(src), src / "kept.json"
    total = len(pl["near_far"])
    starts = np.cumsum([0] + [b - a for a, b in json.loads(kj.read_text())["pieces"]]) if kj.exists() else np.array([0, total])
    return {"src": src, "video": str(src / ("kept.mp4" if kj.exists() else "source.mp4")),
            "boxes": np.split(z["boxes"], cut), "kps": np.split(z["kp"], cut), "ball": z["ball"],
            "court": court_tv(z["court"], total, starts[1:-1])[0] if "court" in z else phone_saved(src / "court", total)[0],
            "size": (int(z["width"]), int(z["height"])), "pl": pl, "fps": pl["fps"], "total": total,
            "starts": starts, "hits": json.loads(sj.read_text())["hits"] if sj.exists() else [],
            "name": {k: (names.get(str(k + 1)) or f"Player {k + 1}").split()[-1] for k in (0, 1)}}
