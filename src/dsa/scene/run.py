"""Everything after the GPU step, on the laptop: from the analysed video (kept.mp4 for TV, else source.mp4) and
raw.npz (dsa.gpu.process) to who played, where, and when they hit the ball. TV is assumed when kept.json exists.

  people     tracks joined into people, still ones marked (dsa.scene.tracks)
  court      phone: Opus once per camera position; tv: the GPU step's court checks (dsa.scene.court)
  players    per frame, near and far: the person playing in each half (dsa.scene.players)
  identity   phone: Player 1 / Player 2 for the whole clip, from what they wear (dsa.scene.identity). TV: what
             each near-end person wears (dsa.scene.who.looks); who is who is settled over the whole match,
             all its pieces at once, by dsa.scene.who
  contacts   ball contacts (dsa.scene.events)

Writes <video>/players.json (per frame: the player at the near and far end, 1 / 2 / null, and their boxes;
contacts as [frame, player]; phone: each player's look, for dsa.scene.identity.line_up); TV: <video>/picks.json (the same with person ids) and looks.npz, for dsa.scene.who. --test works on the test run instead (<video>/test/). The overlay video comes
after step 4 (dsa.pages.render).

    python -m dsa.scene.run tests/own_drew
    python -m dsa.scene.run federer_usopen/2015 [--test]
"""
from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np

from dsa.paths import video_dir
from dsa.scene import court as C
from dsa.scene import events, identity, players, tracks, who
from dsa.video import frames_at


def analyse(video: str, npz: str | Path, work: Path, tv: bool = False, court_marked: bool = False) -> dict:
    z = np.load(npz)
    fps, W, H = float(z["fps"]), int(z["width"]), int(z["height"])
    cut = np.cumsum(z["n"])[:-1]
    boxes, ids, kp = np.split(z["boxes"], cut), np.split(z["track"], cut), np.split(z["kp"], cut)
    n = len(boxes)
    pp = tracks.people(boxes, ids, fps)
    if not (tv or court_marked):
        C.phone(video, work / "court")
    court, log = C.tv(z["court"], n) if tv else C.phone_saved(work / "court", n)
    sides, hits = players.pick(pp["people"], boxes, ids, pp["person_of"], court, z["ball"], kp, H if tv else None)
    if tv:
        look = who.looks(sides, boxes, ids, pp["person_of"], kp, partial(frames_at, video))
        ident = {"player": {}, "margin": {}, "frames": [[None, None]] * n}
    else:
        look = {}
        looks = {(int(i), int(j)): np.float32(d) for i, j, d in zip(z["look_frame"], z["look_box"], z["look"])} if "look" in z else None
        ident = identity.split(sides, boxes, ids, pp["person_of"], kp, partial(frames_at, video), tv, looks)
    return {"looks": look, "fps": fps, "size": (W, H), "boxes": boxes, "ids": ids, "kp": kp, "ball": z["ball"], "court": court,
            "court_log": log, "people": pp["people"], "person_of": pp["person_of"], "sides": sides, "hits": hits,
            "player": ident["player"], "margin": ident["margin"], "labels": ident["frames"], "centres": ident.get("centres"), "gpu_seconds": json.loads(str(z["seconds"])),
            "contacts": events.contacts(z["ball"], boxes, ids, pp["person_of"], kp)}


def summary(r: dict) -> list[str]:
    sides, labels, fps = r["sides"], r["labels"], r["fps"]
    n = len(sides)
    out = ["court: " + "; ".join(r["court_log"]),
           f"frames with both players: {sum(all(p is not None for p in s) for s in sides) / n:.0%}, "
           f"one: {sum(sum(p is not None for p in s) == 1 for s in sides) / n:.0%}"]
    for k in (0, 1):
        near, far = (sum(lb[e] == k for lb in labels) / fps for e in (0, 1))
        out.append(f"Player {k + 1}: {near:.0f} s at the near end, {far:.0f} s at the far end")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="<project>/<video>")
    ap.add_argument("--tv", action="store_true", help="TV footage without kept.json (court from the GPU step)")
    ap.add_argument("--test", action="store_true", help="the test run's output, in <video>/test/")
    ap.add_argument("--court-marked", action="store_true", help="phone: the court is marked already (dsa.scene.court)")
    a = ap.parse_args()

    src = video_dir(a.video)
    d = src / "test" if a.test else src
    kept = (src / "kept.json").exists()
    video = str(src / ("kept.mp4" if kept else "source.mp4"))
    r = analyse(video, d / "raw.npz", src, a.tv or kept, a.court_marked)
    print(a.video, r["gpu_seconds"])

    def box(i, p):
        j = [j for j in np.flatnonzero(r["ids"][i] >= 0) if r["person_of"][int(r["ids"][i][j])] == p]
        return [int(v) for v in r["boxes"][i][j[0], :4]] if j else None

    boxes = [[None if p is None else box(i, p) for p in s] for i, s in enumerate(r["sides"])]
    if kept:                        # who is who: over the whole match (dsa.scene.who)
        lengths = [b - a for a, b in json.loads((src / "kept.json").read_text())["pieces"]]
        bounds = np.r_[0, np.cumsum(lengths)].astype(int)
        (d / "picks.json").write_text(json.dumps({
            "fps": r["fps"], "picks": [[None if p is None else int(p) for p in s] for s in r["sides"]], "boxes": boxes,
            "contacts": [[int(t), int(p)] for t, p in r["contacts"]],
            "pieces": [[int(a), int(b)] for a, b in zip(bounds, bounds[1:])]}))
        lk = r["looks"]
        np.savez(d / "looks.npz", **{k: np.array([c[j] for c in lk], int) for j, k in
                                     enumerate(("person", "first", "last", "frames"))},
                 look=np.array([c[4] for c in lk]).reshape(len(lk), len(who.VARIANTS) * len(who.PARTS), who.STATS))
        print(f"  {len(lk)} near-end stretches described")
        return
    for line in summary(r):
        print(" ", line)
    per_frame = [[None if k is None else k + 1 for k in lb] for lb in r["labels"]]

    def label(t, p):
        s = r["sides"][t]
        return per_frame[t][s.index(p)] if p in s else (r["player"][p] + 1 if p in r["player"] else None)

    (d / "players.json").write_text(json.dumps({
        "fps": r["fps"], "near_far": per_frame,
        "boxes": boxes,
        "contacts": [[t, label(t, p)] for t, p in r["contacts"]],
        "looks": None if r["centres"] is None else [np.round(c, 5).tolist() for c in r["centres"]]}))


if __name__ == "__main__":
    main()
