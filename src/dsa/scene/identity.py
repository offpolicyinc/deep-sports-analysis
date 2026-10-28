"""Player 1 and Player 2 for a whole clip, whichever end they are at.

The players picked per frame (dsa.scene.players) come as person ids that break whenever someone
leaves the frame, sits down or is hidden. Each id is described by what its player wears, and the ids are
split into two people:

  describe   Lab colour histograms (L 4 x a 8 x b 8 bins) of the torso (between shoulders and hips) and the
             shorts (between hips and knees), cut out along the skeleton so no court or background gets in;
             averaged over up to SAMPLES frames of the id. The app's GPU step works these colours out while
             it runs and saves them in raw.npz (look); when they're there, they're used and the video isn't opened.
  split      the two ids seen together for longest seed Player 1 and 2; every id then joins the nearer of
             the two (chi-square distance), heaviest ids first, never the one an id was on screen together
             with; the two looks are re-averaged and the split repeated until it settles.
  frames     per frame. TV: the near player by the near-end ids alone, split in two by colour; the far player
             is the other one. Far-end colour (a small player, the court bleeding in, one id sometimes
             running across both players) decides only when nobody is at the near end. Phone: each id's
             player from the split above.
  line_up    the pieces of one session, each split on its own, renumbered so Player 1 is the same person in all:
             each later piece takes whichever numbering puts its players' looks nearer the first piece's.

margin = how much nearer the id is to its own player than to the other (0 = a coin toss).
"""
from __future__ import annotations

import json
from collections import defaultdict

import cv2
import numpy as np

from dsa.paths import video_dir

SAMPLES = 40
MIN_SCORE = 0.3
BINS = (4, 8, 8)
TORSO = (5, 6, 12, 11)          # shoulders, hips (COCO-WholeBody)
SHORTS = (11, 12, 14, 13)       # hips, knees


def _hist(lab: np.ndarray, poly: np.ndarray) -> np.ndarray | None:
    poly = np.int32(poly)                               # only the polygon's box is looked at: the same histogram
    x0, y0 = np.clip(poly.min(0), 0, None)
    lab, poly = lab[y0:poly[:, 1].max() + 1, x0:poly[:, 0].max() + 1], poly - (x0, y0)
    if lab.size == 0:
        return None
    mask = np.zeros(lab.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, poly, 1)
    if mask.sum() < 30:
        return None
    h = cv2.calcHist([lab], [0, 1, 2], mask, list(BINS), [0, 256, 0, 256, 0, 256]).ravel()
    return h / h.sum()


def describe(bgr: np.ndarray | None, kp: np.ndarray, lab: np.ndarray | None = None) -> np.ndarray | None:
    """Torso + shorts colour of one person in one frame, from their joints; None when not seen well enough. `lab`:
    the frame already in Lab, in place of `bgr` (several people in one frame)."""
    def good(idx):
        return np.isfinite(kp[list(idx)]).all() and (kp[list(idx), 2] >= MIN_SCORE).all()

    if not good(TORSO):
        return None
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab) if lab is None else lab
    torso = _hist(lab, kp[list(TORSO), :2])
    if torso is None:
        return None
    shorts = _hist(lab, kp[list(SHORTS), :2]) if good(SHORTS) else None
    return np.concatenate([torso, shorts if shorts is not None else torso])


def chi2(a: np.ndarray, b: np.ndarray) -> float:
    return float(0.5 * np.sum((a - b) ** 2 / (a + b + 1e-9)))


def split(picks: list[list], per_frame: list[np.ndarray], ids: list[np.ndarray], person_of: dict[int, int],
          kp: list[np.ndarray], read, tv: bool = False, looks: dict | None = None) -> dict:
    """picks[i]: the players' person ids in frame i (None for an empty side). `read(wanted)` yields (i, BGR frame)
    for the frame numbers it's given (dsa.video.frames_at); unused when `looks` has people described already
    ({(frame, box): describe's}). Returns {"player": {person: 0 or 1}, "margin":
    {person: float}, "looks": {...}, "frames": per frame [near, far] as 0 / 1 / None, "centres": the two players'
    looks}."""
    seen: dict[int, list[int]] = defaultdict(list)
    together: dict[tuple[int, int], int] = defaultdict(int)
    for i, pk in enumerate(picks):
        ps = [p for p in pk if p is not None]
        for p in ps:
            seen[p].append(i)
        if len(ps) == 2:
            together[tuple(sorted(ps))] += 1
    if not seen:
        return {"player": {}, "margin": {}, "looks": {}}

    def box(i: int, p: int) -> int | None:                 # p's box in frame i
        return next((j for j in np.flatnonzero(ids[i] >= 0) if person_of[int(ids[i][j])] == p), None)

    def spread(fs: list[int]) -> list[int]:                # up to SAMPLES of them, spread out
        return [fs[i] for i in np.unique(np.linspace(0, len(fs) - 1, min(SAMPLES, len(fs))).astype(int))] if fs else []

    got: dict[int, list[np.ndarray]] = defaultdict(list)
    if looks is not None:
        for p, fs in seen.items():
            for i in spread([i for i in fs if (i, box(i, p)) in looks]):
                got[p].append(looks[i, box(i, p)])
    else:
        want: dict[int, list[int]] = defaultdict(list)      # frame -> persons to describe there
        for p, fs in seen.items():                          # a GPU step may pose some frames only (dsa.gpu.process)
            for i in spread([i for i in fs if (j := box(i, p)) is not None and np.isfinite(kp[i][j, 0, 0])] or fs):
                want[i].append(p)
        for i, bgr in read(want):
            for p in want[i]:
                if (j := box(i, p)) is not None:
                    d = describe(bgr, kp[i][j].astype(np.float32))
                    if d is not None:
                        got[p].append(d)
    look = {p: np.mean(v, axis=0) for p, v in got.items()}
    weight = {p: len(seen[p]) for p in look}
    if len(look) < 2:
        return {"player": {p: 0 for p in look}, "margin": {p: 0.0 for p in look}, "looks": look}

    pairs = [(n, a, b) for (a, b), n in together.items() if a in look and b in look]
    if pairs:
        _, a, b = max(pairs)
    else:                                                   # never on screen together: the two most different
        big = sorted(look, key=lambda p: -weight[p])[:6]
        _, a, b = max((chi2(look[x], look[y]), x, y) for x in big for y in big if x < y)
    centre = [look[a], look[b]]
    apart = defaultdict(set)
    for (x, y) in together:
        apart[x].add(y)
        apart[y].add(x)
    player: dict[int, int] = {}
    for _ in range(10):
        new: dict[int, int] = {}
        for p in sorted(look, key=lambda p: -weight[p]):
            d = [chi2(look[p], c) for c in centre]
            k = int(np.argmin(d))
            if any(new.get(q) == k for q in apart[p]):
                k = 1 - k
            new[p] = k
        if new == player:
            break
        player = new
        for k in (0, 1):
            members = [p for p in player if player[p] == k]
            if members:
                centre[k] = np.average([look[p] for p in members], axis=0, weights=[weight[p] for p in members])
    # per frame: the near player from the near-end ids alone, split in two by colour (players are big there);
    # the far player is the other one. Far-end colour, where the player is small, the court's colour bleeds
    # in and one id can run across both players, only counts when nobody is at the near end.
    near = [p for p in look if sum(picks[i][0] == p for i in seen[p]) > len(seen[p]) / 2] if tv else []
    near_of = _two(near, look, weight) if len(near) >= 2 else {}
    if near_of:                                    # keep the numbering of the first split
        same = sum(weight[p] for p in near_of if near_of[p] == player[p])
        if same < sum(weight[p] for p in near_of) / 2:
            near_of = {p: 1 - k for p, k in near_of.items()}
    per_frame = []
    for n, f in picks:
        k = near_of.get(n, player.get(n))
        far = (1 - k if k is not None else player.get(f)) if tv else player.get(f)
        per_frame.append([k, None if f is None else far])
    margin = {p: chi2(look[p], centre[1 - player[p]]) - chi2(look[p], centre[player[p]]) for p in player}
    return {"player": player, "margin": margin, "looks": look, "frames": per_frame, "centres": centre}


def _two(ids: list[int], look: dict, weight: dict) -> dict[int, int]:
    """ids split into two by look (2-means, chi-square), from the two most different of the 10 longest."""
    big = sorted(ids, key=lambda p: -weight[p])[:10]
    _, a, b = max((chi2(look[x], look[y]), x, y) for x in big for y in big if x < y)
    c, lab = [look[a], look[b]], {}
    for _ in range(20):
        new = {p: int(chi2(look[p], c[1]) < chi2(look[p], c[0])) for p in ids}
        if new == lab:
            break
        lab = new
        for k in (0, 1):
            m = [p for p in ids if lab[p] == k]
            if m:
                c[k] = np.average([look[p] for p in m], axis=0, weights=[weight[p] for p in m])
    return lab


def line_up(specs: list[str]) -> list[str]:
    """The later videos of one session (dsa.scene.run's players.json each) renumbered to match the first: Player 1
    and 2 swapped where the crossed pairing of the two players' looks is the nearer. A video without both looks
    (one player only) is left as it is. Returns the videos swapped."""
    def load(spec: str) -> dict:
        return json.loads((video_dir(spec) / "players.json").read_text())

    first, other, swapped = load(specs[0]).get("looks"), {1: 2, 2: 1}, []
    for spec in specs[1:]:
        pl = load(spec)
        if first is None or pl.get("looks") is None:
            continue
        (a1, a2), (b1, b2) = np.array(first), np.array(pl["looks"])
        if chi2(a1, b1) + chi2(a2, b2) <= chi2(a1, b2) + chi2(a2, b1):
            continue
        pl["near_far"] = [[other.get(k) for k in ends] for ends in pl["near_far"]]
        pl["contacts"] = [[t, other.get(k)] for t, k in pl["contacts"]]
        pl["looks"] = pl["looks"][::-1]
        (video_dir(spec) / "players.json").write_text(json.dumps(pl))
        swapped.append(spec)
    return swapped
