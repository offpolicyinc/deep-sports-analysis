"""Hits from something better than F3ED, which misses many strokes on phone footage (trained on TV): SwingVision's own
shot list for recordings that came from it, else strokes labelled by eye. Either goes in <video>/strokes/labels.json,
in F3ED's hits' format, which dsa.measure.shots and dsa.scene.results read instead of strokes.json.

    python -m dsa.ingest.labels others          # every video of the project cut from a SwingVision recording

Which recordings came from SwingVision, and their matches' share ids, are in CACHE/shares.json ({project: {recording:
share id}}), not here: a share id opens that player's match to anyone.

SwingVision's API answers a match's share id without signing in: the match (when its video starts) and every shot
(when it was hit, which end, serve / groundstroke / volley / overhead / feed, the wing: right a forehand, left a
backhand for a right-hander, checked by eye on 14 of player A's). A shot's time in the video is its timestamp minus
the video's start (its hit_frame counts the 60 fps recording, at a rate that drifts from match to match).
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from datetime import datetime
from pathlib import Path

from dsa.ingest.projects import PROJECTS
from dsa.paths import ROOT, video_dir

API = "https://api.swing.tennis/v1/matches/"
CACHE = ROOT / "raw" / "other_players" / "swingvision"


def fetch(share: str) -> tuple[dict, list[dict]]:
    """A SwingVision match and its shots, kept in CACHE once fetched."""
    CACHE.mkdir(parents=True, exist_ok=True)
    out = []
    for name, url in [("match", API + share), ("shots", API + share + "/shots?page=0")]:
        p = CACHE / f"{share}.{name}.json"
        if not p.exists():
            p.write_bytes(urllib.request.urlopen(url, timeout=60).read())
        out.append(json.loads(p.read_text()))
    match, shots = out
    assert shots["meta"]["total_pages"] == 1, f"{share}: more than one page of shots"
    return match.get("data", match), shots["data"]


def hit(frame: int, fps: float, side: str, shot: str, hand: str | None, technique: str | None, by: str,
        player: str | None = None) -> dict:
    """One hit as dsa.gpu.strokes writes them (no direction or outcome); player: who hit it, when the labels say
    (dsa.measure.shots.owners takes it over names.json, which a compilation of clips can get wrong)."""
    return {"frame": frame, "source_frame": frame, "time": round(frame / fps, 3), "score": 1.0, "side": side,
            "position": None, "shot": shot, "direction": None, "outcome": None, "hand": hand, "technique": technique,
            "approach": False, "by": by} | ({"player": player} if player else {})


def write(spec: str, hits: list[dict], source: str) -> Path:
    p = video_dir(spec) / "strokes" / "labels.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"source": source, "hits": sorted(hits, key=lambda h: h["frame"])}, indent=1))
    return p


def swingvision(project: str) -> None:
    """labels.json for each video cut from a SwingVision recording (CACHE/shares.json): every shot SwingVision
    tracked but feeds, at our frame."""
    t = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()             # noqa: E731
    shares = json.loads((CACHE / "shares.json").read_text()).get(project, {})
    for v, spec in PROJECTS[project]["videos"].items():
        share = shares.get(Path(spec[0]).name)
        if not share:
            continue
        match, shots = fetch(share)
        fps = json.loads((video_dir(f"{project}/{v}") / "players.json").read_text())["fps"]
        n = len(json.loads((video_dir(f"{project}/{v}") / "players.json").read_text())["near_far"])
        t0, hits = t(match["video_started_at"]), []
        for s in shots:
            f = int(round((t(s["timestamp"]) - t0 - spec[2]) * fps))
            if s.get("hidden") or s["hit_type"] == "feed" or not 0 <= f < n:
                continue
            if "serve" in s["hit_type"]:
                shot, hand, tech = "serve", None, None
            else:
                shot = "return" if s["shot_type"] == "return" else "stroke"
                hand = {"right": "fh", "left": "bh"}.get(s["hit_wing"])
                tech = {"ground_stroke": "slice" if s["spin_type"] == "slice" else "gs"}.get(s["hit_type"], s["hit_type"])
            hits.append(hit(f, fps, s["hit_court_side"], shot, hand, tech, "swingvision"))
        print(write(f"{project}/{v}", hits, f"swingvision {share}"), len(hits), "hits")


def by_eye(spec: str, rows: list[tuple[int, str]], note: str, by: str = "eye") -> Path:
    """labels.json from strokes labelled by eye: (frame, kind), all the sender's (the video's first player) at the
    near end; kind forehand / backhand (topspin or flat), slice forehand / backhand, serve or overhead."""
    fps = json.loads((video_dir(spec) / "players.json").read_text())["fps"]
    project, v = spec.split("/")
    who = PROJECTS[project]["videos"][v][4][0]
    kinds = {"forehand": ("stroke", "fh", "gs"), "backhand": ("stroke", "bh", "gs"), "serve": ("serve", None, None),
             "slice forehand": ("stroke", "fh", "slice"), "slice backhand": ("stroke", "bh", "slice"),
             "overhead": ("stroke", None, "overhead")}
    return write(spec, [hit(f, fps, "near", *kinds[k], by, who) for f, k in rows], f"{by}: {note}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("project")
    swingvision(ap.parse_args().project)
