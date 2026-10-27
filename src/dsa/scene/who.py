"""Player 1 and Player 2 for a whole TV match, over all its pieces (the first video, "_b", "_x<k>": dsa.ingest.pad),
and their names, once per match.

dsa.scene.run picks the player at each end per frame (dsa.scene.players) as person ids that break whenever someone
is hidden or the camera cuts, and now and then run across both players. It describes each id's near-end frames
(where players are big) in chunks of CHUNK frames, at up to SAMPLES moments each, part by part (looks), and writes
<video>/picks.json and <video>/looks.npz. Here:

  look        per part (head: cap, hat, bandana, hair; torso; shorts; shins: socks, sleeves) the median Lab colour,
              with the frame made lighter or darker so the court under the player's feet is mid grey; as FEATURES: the
              shirt's colour, each other part's colour, lightness, and lightness against the shirt (the same light
              falls on all of one man, so "black shorts under a white shirt" holds in sun and shade).
  scoreboard  the serve marker read (Opus) shortly before READS serves spread over the match: who serves, and so
              who is at the serving end. Every other read teaches the split: its chunk is that player's
              ("anchors"); the features are weighted by how well they part the anchors' two players, and every
              chunk joins the nearer look (fit). The other reads check it: names.json's "why".
  no reads    (no scoreboard, or too few markers) the chunks are split in two without anchors (split: every feature
              that splits them in two on its own starts a 2-means, the best kept), and Opus names the two from
              NAMES crops each.
  neither     a chunk far from both looks (beyond FAR_OUT of the typical distance): someone else who got through
              dsa.scene.players; its frames are left unlabelled.
  frames      per video the chunks in time order are labelled together (Viterbi): the near player changes inside
              a kept piece only at a changeover the broadcast did not cut away from (costs SWITCH). The far
              player is the other one (the far end is not posed; two players never share an end).
  hands       with one left-hander in the match, every near-end serve votes on which hand the player serves with
              (the higher wrist up to contact; groundstrokes do not vote: dsa.gpu.strokes often calls a left-hander's
              forehand a backhand), and each vote against a player's hand costs his label HAND there: looks alone
              mix two players dressed alike (2024 US Open, Shelton v Tiafoe).

Writes for every piece <video>/players.json (as dsa.scene.run did for phone videos) and <video>/names.json.

    python -m dsa.scene.who atp_hardcourt/2023_uso_djokovic_medvedev     # any piece of the match
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict

import cv2
import numpy as np

from dsa.ingest.projects import PROJECTS, match, pair
from dsa.opus import ask
from dsa.paths import video_dir
from dsa.video import frame_at

SAMPLES, CHUNK, MIN_FRAMES, FAR_OUT, NAMES, HEIGHT = 8, 90, 10, 12.0, 12, 300
READS, MIN_READS, SURE_READS = 40, 8, 0.85      # scoreboard serve markers read per match; needed; agreeing
STATS = 5                       # per part: median L a b, L at the 15th / 85th percentile
MIN_SCORE = 0.3
PARTS = ("head", "torso", "shorts", "shins")
VARIANTS = (False, True)        # the court-grey correction: one gain, each channel (_balanced)


# ---------------------------------------------------------------- describing one person (dsa.scene.run calls this)

def _stats(lab: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """Median L, a, b and the 15th / 85th percentile of L of the masked pixels."""
    px = lab[mask > 0].astype(float)
    if len(px) < 40:
        return None
    return np.r_[np.median(px, axis=0), np.percentile(px[:, 0], [15, 85])]


def _poly(shape, pts) -> np.ndarray:
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(m, np.int32(pts), 1)
    return m


def _balanced(bgr: np.ndarray, box: np.ndarray, each: bool) -> np.ndarray:
    """The frame around the person corrected against the court under their feet: made lighter or darker so it is
    mid grey (one gain for all three channels), or with `each`, each channel balanced so it is grey. Sun, shade and
    each broadcast's exposure change the clothes much less; balancing each channel also takes out the blue of the
    shade, but against a blue court it washes pink and red out, so both are kept (VARIANTS)."""
    x1, y1, x2, y2 = (int(v) for v in box[:4])
    h = y2 - y1
    H, W = bgr.shape[:2]
    band = bgr[min(y2, H - 1):min(y2 + max(h // 10, 3), H), max(x1, 0):min(x2, W)]
    if band.size < 30:
        band = bgr[max(y2 - max(h // 20, 2), 0):y2, max(x1 - h // 4, 0):max(x1, 1)]
    if band.size < 30:
        return bgr
    court = np.median(band.reshape(-1, 3), axis=0)
    gain = 128.0 / np.maximum(court, 8) if each else np.clip(128.0 / max(float(court.mean()), 8.0), 0.5, 3.0)
    a, b = max(y1 - h // 5, 0), min(y2 + 1, H)
    out = bgr.copy()
    out[a:b] = np.clip(bgr[a:b] * gain, 0, 255).astype(np.uint8)
    return out


def describe(bgr: np.ndarray, kp: np.ndarray, box: np.ndarray) -> np.ndarray | None:
    """(len(VARIANTS) * len(PARTS), STATS) colour of each part (_stats) in each correction of the frame, NaN for a
    part not seen; None without shoulders and hips."""
    def good(*idx):
        return all(np.isfinite(kp[i]).all() and kp[i, 2] >= MIN_SCORE for i in idx)

    if not good(5, 6, 11, 12):
        return None
    ls, rs, lh, rh = (kp[i, :2] for i in (5, 6, 11, 12))
    sw = np.linalg.norm(ls - rs)
    top = (ls[1] + rs[1]) / 2
    cx = kp[0, 0] if good(0) else (ls[0] + rs[0]) / 2
    x1, x2 = int(cx - 0.45 * sw), int(cx + 0.45 * sw)
    y1, y2 = int(max(box[1], top - 1.1 * sw)), int(top - 0.1 * sw)
    masks: list[np.ndarray | None] = [None] * len(PARTS)
    if x2 > x1 and y2 > y1:
        m = np.zeros(bgr.shape[:2], np.uint8)
        m[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)] = 1
        masks[0] = m
    masks[1] = _poly(bgr.shape, (ls, rs, rh, lh))
    if good(13, 14):
        lk, rk = kp[13, :2], kp[14, :2]
        masks[2] = _poly(bgr.shape, (lh, rh, rh + 0.4 * (rk - rh), lh + 0.4 * (lk - lh)))
        if good(15, 16):
            la, ra = kp[15, :2], kp[16, :2]
            masks[3] = _poly(bgr.shape, (lk + 0.3 * (la - lk), rk + 0.3 * (ra - rk), ra, la))
    out = np.full((len(VARIANTS) * len(PARTS), STATS), np.nan)
    for v, each in enumerate(VARIANTS):
        lab = cv2.cvtColor(_balanced(bgr, box, each), cv2.COLOR_BGR2Lab)
        for p, m in enumerate(masks):
            h = None if m is None else _stats(lab, m)
            if h is not None:
                out[v * len(PARTS) + p] = h
    return out


def looks(picks: list[list], per_frame: list[np.ndarray], ids: list[np.ndarray], person_of: dict[int, int],
          kp: list[np.ndarray], read) -> list[tuple[int, int, int, int, np.ndarray]]:
    """[(near person id, first frame, last frame, frames picked, median look (describe))] per CHUNK of
    an id's near-end frames (one id now and then runs across both players: at a changeover without a cut), for
    chunks of at least MIN_FRAMES, described at up to SAMPLES moments. `read(wanted)` yields (i, BGR frame) for the frame
    numbers it's given (dsa.video.frames_at)."""
    seen: dict[int, list[int]] = defaultdict(list)
    for i, pk in enumerate(picks):
        if pk[0] is not None:
            seen[pk[0]].append(i)
    chunks = []
    for p, fs in seen.items():
        cut = [0] + [k for k in range(1, len(fs)) if fs[k] - fs[k - 1] > CHUNK or k % CHUNK == 0] + [len(fs)]
        parts = [fs[x:y] for x, y in zip(cut, cut[1:])]
        merged = []
        for part in parts:
            if merged and len(part) < MIN_FRAMES and part[0] - merged[-1][-1] <= CHUNK:
                merged[-1] = merged[-1] + part
            else:
                merged.append(part)
        chunks += [(p, part) for part in merged if len(part) >= MIN_FRAMES]
    want: dict[int, list[int]] = defaultdict(list)
    for c, (p, fs) in enumerate(chunks):
        for i in np.unique(np.linspace(0, len(fs) - 1, min(SAMPLES, len(fs))).astype(int)):
            want[fs[i]].append(c)
    got: dict[int, list[np.ndarray]] = defaultdict(list)
    for i, bgr in read(want):
        for c in want[i]:
            p = chunks[c][0]
            j = [j for j in np.flatnonzero(ids[i] >= 0) if person_of[int(ids[i][j])] == p]
            if j:
                d = describe(bgr, kp[i][j[0]].astype(np.float32), per_frame[i][j[0]])
                if d is not None:
                    got[c].append(d)
    out = []
    for c, v in sorted(got.items()):
        if len(v) >= 3:
            p, fs = chunks[c]
            with np.errstate(all="ignore"):
                out.append((p, fs[0], fs[-1], len(fs), np.nanmedian(np.stack(v), axis=0)))
    return out


# ---------------------------------------------------------------- the match

def pieces_of(spec: str) -> list[str]:
    """Every video of the match `spec` is part of that has been through dsa.scene.run, in order."""
    project, v = spec.split("/")
    m = match(v)
    vids = [w for w in PROJECTS[project]["videos"] if match(w) == m and (video_dir(f"{project}/{w}") / "picks.json").exists()]

    def order(w):
        k = re.search(r"_x(\d+)$", w)
        return (0 if w == m else 1 if w.endswith("_b") else 2, int(k.group(1)) if k else 0)
    return sorted(vids, key=order)


FEATURES = [f"{'wb ' if each else ''}{p}" for each in VARIANTS
            for p in ["torso L", "torso a", "torso b"] + [f"{p} {q}" for p in PARTS if p != "torso"
                                                         for q in ("L", "dark", "bright", "a", "b")]]
MIN_SPREAD = 2.0                # Lab units
CAP = 0.35                      # no feature carries more than this share of the weight
TRIM = 0.1                      # share of ids left out of the looks while splitting
JUDGED = 0.75                   # the scoreboard picks the split when one agrees with it at least this well
PURE = 4                        # how much keeping each kept piece to one player counts
SWITCH = 3.0                    # cost of the near player changing inside a kept piece (in chunk distances)
HAND = 8.0                      # cost of one hand vote against the player (in chunk distances)
BALANCE = 0.15                  # each player is at the near end at least this share of the frames


def features(look: np.ndarray) -> np.ndarray:
    """A look ((len(VARIANTS) * len(PARTS), STATS)) as FEATURES, per correction (VARIANTS): the shirt's colour,
    and per other part its colour (a, b), its lightness ("bright", the court under the player's feet set to mid
    grey) and how much lighter or darker it is than the shirt (median and darkest L): the same light falls on all
    of one man, so "black shorts under a white shirt" or "a white cap" holds in sun and in shade alike; blue shorts
    under a white shirt and black ones under a pink one are told apart by their own lightness."""
    out = []
    for v in range(len(VARIANTS)):
        part = look[v * len(PARTS):(v + 1) * len(PARTS)]
        t = part[PARTS.index("torso")]
        out += [t[0], t[1], t[2]]
        for p in PARTS:
            if p != "torso":
                x = part[PARTS.index(p)]
                out += [x[0] - t[0], x[3] - t[0], x[0], x[1], x[2]]
    return np.array(out)


def _dist(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    """Weighted mean squared difference over the features both have."""
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.sum(w[ok] * (a[ok] - b[ok]) ** 2) / max(np.sum(w[ok]), 1e-9)) if ok.any() else np.inf


def _centre(xs: np.ndarray, ws: np.ndarray) -> np.ndarray:
    ok = np.isfinite(xs)
    w = ws[:, None] * ok
    return np.where(ok.any(0), np.nansum(np.where(ok, xs, 0) * w, 0) / np.maximum(w.sum(0), 1e-9), np.nan)


def _fisher(a: np.ndarray, b: np.ndarray, wa: np.ndarray, wb: np.ndarray) -> float:
    """How well one feature parts two groups: squared distance of their weighted means over their spread."""
    ma, mb = np.average(a, weights=wa), np.average(b, weights=wb)
    va = np.average((a - ma) ** 2, weights=wa)
    vb = np.average((b - mb) ** 2, weights=wb)
    return float((ma - mb) ** 2 / max((wa.sum() * va + wb.sum() * vb) / (wa.sum() + wb.sum()), 0.05))


def _purity(lab: np.ndarray, n: np.ndarray, group: np.ndarray) -> float:
    """Share of the near-end frames whose id has the label most of its kept piece's frames have."""
    agree = 0.0
    for g in np.unique(group):
        m = group == g
        agree += max(n[m & (lab == 0)].sum(), n[m & (lab == 1)].sum())
    return float(agree / n.sum())


def _refine(x: np.ndarray, n: np.ndarray, lab: np.ndarray):
    """(score, looks, weights, labels) after trimmed, re-weighted 2-means from labels `lab` (-1: none yet);
    None when one side ends up with under BALANCE of the frames."""
    w = np.ones(x.shape[1]) / x.shape[1]
    c = [_centre(x[lab == k], n[lab == k]) for k in (0, 1)]
    for _ in range(40):
        new = np.array([int(_dist(v, c[1], w) < _dist(v, c[0], w)) for v in x])
        near = np.array([_dist(v, c[k], w) for v, k in zip(x, new)])
        keep = near <= np.quantile(near[np.isfinite(near)], 1 - TRIM)
        for k in (0, 1):
            if (keep & (new == k)).any():
                c[k] = _centre(x[keep & (new == k)], n[keep & (new == k)])
        fish = np.zeros(x.shape[1])
        for f in range(x.shape[1]):
            m = keep & np.isfinite(x[:, f])
            if (m & (new == 0)).sum() > 1 and (m & (new == 1)).sum() > 1:
                fish[f] = _fisher(x[m & (new == 0), f], x[m & (new == 1), f], n[m & (new == 0)], n[m & (new == 1)])
        w = fish + 1e-3
        for _ in range(5):
            w = np.minimum(w / w.sum(), CAP)
        if (new == lab).all():
            break
        lab = new
    if min(n[lab == 0].sum(), n[lab == 1].sum()) < BALANCE * n.sum():
        return None
    return float(np.sort(np.minimum(fish, 100))[-3:].sum()), c, w, lab


def _zscores(units: list[tuple]) -> tuple[np.ndarray, np.ndarray]:
    x = np.stack([features(u[2]) for u in units])
    med = np.nanmedian(x, 0)
    # robust z-scores; the spread at least MIN_SPREAD Lab units, or half the standard deviation (a feature most
    # chunks share exactly, e.g. white clipped at full lightness, would otherwise blow a few others up)
    spread = np.maximum.reduce([np.nanmedian(np.abs(x - med), 0) * 1.4826, 0.5 * np.nanstd(x, 0),
                                np.full(x.shape[1], MIN_SPREAD)])
    return (x - med) / spread, np.array([u[1] for u in units], float)


def fit(units: list[tuple], anchors: dict[int, int]) -> tuple[list[int | None], np.ndarray, np.ndarray]:
    """The split taught by the scoreboard: `anchors` {unit: 0 / 1} (the server's player at a read serve) keep their
    label; the feature weights are how well each feature parts the anchors' two players (Fisher); the looks start
    from the anchors, every unit joins the nearer and the looks are re-made (TRIM of the farthest left out) until
    it settles. Returns as split does."""
    x, n = _zscores(units)
    lab = np.full(len(x), -1)
    for u, k in anchors.items():
        lab[u] = k
    fixed = lab >= 0
    # the feature weights from the anchors alone (what the scoreboard shows parts the two), never from guesses
    fish = np.zeros(x.shape[1])
    for f in range(x.shape[1]):
        m = fixed & np.isfinite(x[:, f])
        if (m & (lab == 0)).sum() > 1 and (m & (lab == 1)).sum() > 1:
            fish[f] = _fisher(x[m & (lab == 0), f], x[m & (lab == 1), f], n[m & (lab == 0)], n[m & (lab == 1)])
    w = fish + 1e-3
    for _ in range(5):
        w = np.minimum(w / w.sum(), CAP)
    c = [_centre(x[fixed & (lab == k)], n[fixed & (lab == k)]) for k in (0, 1)]
    for _ in range(40):
        new = np.array([int(_dist(v, c[1], w) < _dist(v, c[0], w)) for v in x])
        new[fixed] = lab[fixed]
        near = np.array([_dist(v, c[k], w) for v, k in zip(x, new)])
        keep = fixed | (near <= np.quantile(near[np.isfinite(near)], 1 - TRIM))
        c = [_centre(x[keep & (new == k)], n[keep & (new == k)]) for k in (0, 1)]
        if (new == lab).all():
            break
        lab = new
    d = np.array([[_dist(v, c[0], w), _dist(v, c[1], w)] for v in x])
    own = d[np.arange(len(x)), lab]
    scale = np.median(own[np.isfinite(own)]) if np.isfinite(own).any() else 1.0
    out = [None if not np.isfinite(o) or o > FAR_OUT * scale else int(l) for l, o in zip(lab, own)]
    return out, w, d / max(scale, 1e-9)


def split(units: list[tuple], judge=None) -> tuple[list[int | None], np.ndarray, np.ndarray, float | None]:
    """units: (key, frames, look, kept piece). Returns per unit 0 / 1 / None (neither), the feature weights, per unit
    its distance to each look (in units of the typical distance to its own), and the judge's verdict.

    Every feature that splits the units in two on its own (the best threshold, both sides holding at least BALANCE
    of the near-end frames) starts a split, refined (_refine). `judge` (labels per unit -> how well they agree with
    the scoreboard, 0.5 to 1, or None) picks among them; without it, or when it cannot tell, the split whose
    features part the two looks best and keep each kept piece to one player (PURE weighs that)."""
    x, n = _zscores(units)
    group = np.array([hash(u[3]) for u in units])
    cands = []
    for f0 in range(x.shape[1]):
        ok = np.isfinite(x[:, f0])
        order = np.argsort(np.where(ok, x[:, f0], np.inf))[:ok.sum()]
        cum = np.cumsum(n[order])
        good = [(_fisher(x[order[:k], f0], x[order[k:], f0], n[order[:k]], n[order[k:]]), k)
                for k in range(1, len(order)) if min(cum[k - 1], cum[-1] - cum[k - 1]) >= BALANCE * n.sum()]
        if not good:
            continue
        _, k = max(good)
        lab = np.zeros(len(x), int)
        lab[order[k:]] = 1
        lab[~ok] = -1
        got = _refine(x, n, lab)
        if got is not None and not any(min((got[3] == c[3]).mean(), (got[3] != c[3]).mean()) < 0.02 for c in cands):
            cands.append(got)
    looks = [c[0] * _purity(c[3], n, group) ** PURE for c in cands]
    verdict = [judge(c[3]) if judge else None for c in cands]
    known = [v for v in verdict if v is not None]
    if known and max(known) >= JUDGED:
        best = max(range(len(cands)), key=lambda i: ((verdict[i] or 0) >= max(known) - 0.03, looks[i]))
    else:
        best = int(np.argmax(looks))
    _, c, w, lab = cands[best]
    d = np.array([[_dist(v, c[0], w), _dist(v, c[1], w)] for v in x])
    own = d[np.arange(len(x)), lab]
    scale = np.median(own[np.isfinite(own)]) if np.isfinite(own).any() else 1.0
    out = [None if not np.isfinite(o) or o > FAR_OUT * scale else int(l) for l, o in zip(lab, own)]
    return out, w, d / max(scale, 1e-9), verdict[best]


def units_of(spec: str) -> tuple[list[tuple], dict]:
    """Every near-end id of the match described by dsa.scene.run: (key (video, id), frames, look, kept piece); and
    each video's picks.json."""
    project = spec.split("/")[0]
    units, pk = [], {}
    for v in pieces_of(spec):
        d = video_dir(f"{project}/{v}")
        pk[v] = json.loads((d / "picks.json").read_text())
        z = np.load(d / "looks.npz")
        start = np.array([a for a, _ in pk[v]["pieces"]])
        for p, f0, f1, n, x in zip(z["person"], z["first"], z["last"], z["frames"], z["look"]):
            units.append(((v, int(p), int(f0), int(f1)), int(n), x, (v, int(np.searchsorted(start, f0, "right")))))
    return units, pk


def reads(spec: str, n: int = READS) -> list[dict]:
    """The scoreboard's serve marker (dsa.pages.quiz.server, Opus) shortly before n serves spread over every piece
    of the match: [{"video", "frame", "side", "name"}], name None when no marker was read."""
    from concurrent.futures import ThreadPoolExecutor

    from dsa.pages.quiz import server

    project = spec.split("/")[0]
    both = list(pair(spec))
    serves = []
    for v in pieces_of(spec):
        d = video_dir(f"{project}/{v}")
        pk = json.loads((d / "picks.json").read_text())
        start = [a for a, _ in pk["pieces"]]
        for h in json.loads((d / "strokes/strokes.json").read_text())["hits"]:
            if h["shot"] == "serve" and h["side"] in ("near", "far"):
                f0 = max(a for a in start if a <= h["frame"])
                serves.append((v, h["frame"], h["side"], f0, {"src": d, "video": str(d / "kept.mp4"), "fps": pk["fps"]}))
    pick = [serves[int(i)] for i in np.unique(np.linspace(0, len(serves) - 1, min(n, len(serves))).astype(int))]

    def one(sv):
        v, frame, side, f0, d = sv
        return {"video": v, "frame": frame, "side": side, "name": server(d, f0, frame, both)}
    with ThreadPoolExecutor(8) as ex:
        return list(ex.map(one, pick))


def read_unit(units: list[tuple], pk: dict, rd: dict) -> int | None:
    """The unit of the near player around the serve of read `rd` (nearest frame first); None when nobody near."""
    v, frame = rd["video"], rd["frame"]
    picks = pk[v]["picks"]
    for i in sorted(range(max(0, frame - 30), min(len(picks), frame + 31)), key=lambda i: abs(i - frame)):
        q = picks[i][0]
        if q is None:
            continue
        for u, unit in enumerate(units):
            key = unit[0]
            if key[0] == v and key[1] == q and key[2] <= i <= key[3]:
                return u
    return None


def hands(units: list[tuple], pk: dict, spec: str) -> np.ndarray:
    """Per unit (votes left-handed, votes right-handed) from its near-end serves (see hands, above). The near player
    is seen from behind, so his right is the image's right."""
    project = spec.split("/")[0]
    votes = np.zeros((len(units), 2))
    at = defaultdict(list)                       # (video, id) -> [(unit, first, last)]
    for u, unit in enumerate(units):
        at[unit[0][:2]].append((u, unit[0][2], unit[0][3]))
    for v in pieces_of(spec):
        d = video_dir(f"{project}/{v}")
        z = np.load(d / "raw.npz")
        cut = np.cumsum(z["n"])[:-1]
        boxes, kps = np.split(z["boxes"], cut), np.split(z["kp"], cut)
        picks, pb, fps = pk[v]["picks"], pk[v]["boxes"], pk[v]["fps"]

        def kp(i, q):                            # the near player's keypoints at frame i, if he is id q there
            if not 0 <= i < len(picks) or picks[i][0] != q or pb[i][0] is None:
                return None
            j = np.flatnonzero((boxes[i][:, :4].astype(int) == np.array(pb[i][0]).astype(int)).all(1))
            return kps[i][j[0]] if len(j) else None
        for h in json.loads((d / "strokes/strokes.json").read_text())["hits"]:
            f = h["frame"]
            if h["side"] != "near" or f >= len(picks) or picks[f][0] is None:
                continue
            q = picks[f][0]
            u = next((u for u, a, b in at[(v, q)] if a <= f <= b), None)
            if u is None:
                continue
            if h["shot"] == "serve":                 # the hitting wrist is the higher one up to contact
                ks = [kp(i, q) for i in range(f - int(0.2 * fps), f + 3)]
                ys = [k[[9, 10], 1] for k in ks if k is not None and (k[[9, 10], 2] > MIN_SCORE).all()]
                if ys:
                    top = np.min(ys, axis=0)
                    votes[u, int(top[1] < top[0])] += 1
    return votes


def agreement(pairs: list[tuple[int, str]], both: list[str]) -> tuple[float, int]:
    """(share of the serves whose server's label and scoreboard name agree under the better of the two namings,
    label 0's name index under it)."""
    if not pairs:
        return 0.5, 0
    same = sum((k == 0) == (nm == both[0]) for k, nm in pairs) / len(pairs)
    return (same, 0) if same >= 0.5 else (1 - same, 1)


def label(spec: str, verbose: bool = True, scoreboard: bool = True) -> tuple[dict, dict | None]:
    """players.json for every piece of the match, and the naming from the scoreboard (names.json; None when too
    few serve markers were read)."""
    project = spec.split("/")[0]
    vids = pieces_of(spec)
    units, pk = units_of(spec)
    both = list(pair(spec))
    rds = [r for r in (reads(spec) if scoreboard else []) if r["name"] in both]

    # every other read teaches the split; the rest check it (the naming's "why" counts those alone)
    teach, check = (rds[::2], rds[1::2]) if len(rds) >= 2 * MIN_READS else (rds, rds)
    at = [(read_unit(units, pk, r_), r_) for r_ in teach]
    votes: dict[int, Counter] = defaultdict(Counter)       # label 0 = both[0], 1 = both[1]
    for u, r_ in at:
        if u is not None:
            k = both.index(r_["name"])
            votes[u][k if r_["side"] == "near" else 1 - k] += 1
    anchors = {u: c.most_common(1)[0][0] for u, c in votes.items()}
    if len(anchors) >= MIN_READS and len(set(anchors.values())) == 2:
        lab, w, dist = fit(units, anchors)
        taught = True
    else:
        lab, w, dist, _ = split(units)
        taught = False
    if verbose:
        print("weights:", ", ".join(f"{k} {v:.2f}" for k, v in sorted(zip(FEATURES, w / w.sum()), key=lambda t: -t[1])[:6]))
    from dsa.measure.technique import lefty
    left = [lefty(n, spec.split("/")[0]) for n in both]
    extra = np.zeros((len(units), 2))            # per unit and label: the hand votes against that player
    if sum(left) == 1:
        votes = hands(units, pk, spec)
        extra = HAND * np.stack([votes[:, int(left[k])] for k in (0, 1)], axis=1)
        if verbose:
            print(f"  hands: {int(votes.sum())} serves vote on the near player's hand")
    out = {}
    for v in vids:
        p = pk[v]
        # per video, the chunks in time order (one near player at a time) labelled together (Viterbi): each chunk
        # costs its distance to the player's look, and the near player changing costs SWITCH inside a kept piece
        # (it happens there only at a changeover the broadcast did not cut away from), little between pieces
        mine = sorted((j for j, u in enumerate(units) if u[0][0] == v and lab[j] is not None), key=lambda j: units[j][0][2])
        per_frame = [None] * len(p["picks"])
        if mine:
            cost = np.minimum(dist[mine], 50.0) + extra[mine]
            acc, back = cost[0].copy(), []
            for a, b in zip(mine, mine[1:]):
                step = SWITCH if units[a][3] == units[b][3] else SWITCH / 10
                stay, move = acc, acc[::-1] + step
                back.append(np.where(stay <= move, [0, 1], [1, 0]))
                acc = np.minimum(stay, move) + cost[mine.index(b)]
            k = int(np.argmin(acc))
            path = [k]
            for bk in reversed(back):
                k = int(bk[k])
                path.append(k)
            for j, k in zip(mine, reversed(path)):
                _, q, f0, f1 = units[j][0]
                for i in range(f0, f1 + 1):
                    if p["picks"][i][0] == q:
                        per_frame[i] = k
        far_of: dict[int, Counter] = defaultdict(Counter)    # far ids: the other player of the near one
        for i, s in enumerate(p["picks"]):
            if s[1] is not None and per_frame[i] is not None:
                far_of[s[1]][1 - per_frame[i]] += 1
        labels = []
        for i, s in enumerate(p["picks"]):
            k = per_frame[i]
            f = None if s[1] is None else (1 - k if k is not None else
                                           (far_of[s[1]].most_common(1)[0][0] if far_of[s[1]] else None))
            labels.append([None if k is None else k + 1, None if f is None else f + 1])

        def at(t, q):
            s = p["picks"][t]
            return labels[t][s.index(q)] if q in s else None
        out[v] = {"fps": p["fps"], "near_far": labels, "boxes": p["boxes"],
                  "contacts": [[t, at(t, q)] for t, q in p["contacts"]]}
        if verbose:
            n = len(labels)
            got = [sum(lb[0] == k for lb in labels) / max(n, 1) for k in (1, 2)]
            nb = sum(s[0] is not None and lb[0] is None for s, lb in zip(p["picks"], labels)) / max(n, 1)
            print(f"  {v:34s} near: Player 1 {got[0]:.0%}, Player 2 {got[1]:.0%}, picked but neither {nb:.0%}")
    # the naming: the final labels against the scoreboard
    pairs = []
    for r_ in check:
        pl = out[r_["video"]]
        near = [x[0] for x in pl["near_far"][max(0, r_["frame"] - 30):r_["frame"] + 31] if x[0]]
        if near:
            k = Counter(near).most_common(1)[0][0] - 1
            pairs.append((k if r_["side"] == "near" else 1 - k, r_["name"]))
    if len(pairs) < MIN_READS:
        return out, None
    share, flip = agreement(pairs, both)
    if taught:                                         # label 0 is both[0]: how often the final labels agree
        flip = 0
        share = sum((k == 0) == (nm == both[0]) for k, nm in pairs) / len(pairs)
    first = 1 if flip == 0 else 2                      # Player number of both[0]
    names = {"who": first, "sure": bool(share >= SURE_READS), "by_match": True,
             "why": f"scoreboard serve marker: {round(share * len(pairs))} of {len(pairs)} serves agree"
                    + (" (serves not used to split)" if check is not teach else "")}
    names["1"], names["2"] = both if first == 1 else both[::-1]
    if verbose:
        print(f"  scoreboard: {names['why']} ({'taught by the scoreboard' if taught else 'split by looks alone'})")
    return out, names


# ---------------------------------------------------------------- names, once per match

SCHEMA = {
    "type": "object",
    "properties": {"who": {"type": "integer", "enum": [1, 2]}, "player_1": {"type": "string"},
                   "player_2": {"type": "string"}, "sure": {"type": "boolean"}, "why": {"type": "string"}},
    "required": ["who", "player_1", "player_2", "sure", "why"],
    "additionalProperties": False,
}


def sheet(project: str, shots: list[tuple[str, int, list[int]]]) -> np.ndarray:
    tiles = []
    for v, i, (x1, y1, x2, y2) in shots:
        f = frame_at(str(video_dir(f"{project}/{v}") / "kept.mp4"), i)
        m = int(0.15 * (y2 - y1))
        c = f[max(0, y1 - m):y2 + m, max(0, x1 - m):x2 + m]
        tiles.append(cv2.resize(c, (int(c.shape[1] * HEIGHT / c.shape[0]), HEIGHT)))
    return np.hstack(tiles)


def moments(players: dict[str, dict], k: int, n: int = NAMES) -> list[tuple[str, int, list[int]]]:
    """n near-end moments of player k spread over every piece of the match."""
    at = [(v, i, pl["boxes"][i][0]) for v, pl in players.items() for i, s in enumerate(pl["near_far"])
          if s[0] == k and pl["boxes"][i][0]]
    return [at[int(j)] for j in np.linspace(0, len(at) - 1, n)] if len(at) >= n else at


def name(spec: str, players: dict[str, dict]) -> dict:
    project = spec.split("/")[0]
    first = next(iter(players))
    d = video_dir(f"{project}/{first}")
    both = pair(f"{project}/{first}")
    imgs = []
    for k in (1, 2):
        imgs.append(d / f"who_player{k}.jpg")
        cv2.imwrite(str(imgs[-1]), sheet(project, moments(players, k)), [cv2.IMWRITE_JPEG_QUALITY, 90])
    info = PROJECTS[project]["videos"][first]
    prompt = (f"These are two tennis players from a TV broadcast, {info[1]}. The first image shows Player 1 at "
              f"{NAMES} moments of the match, the second Player 2; each image should show one man throughout "
              f"(say so in why if it does not). The two players are {both[0]} and {both[1]}. Which is Player 1 and "
              f"which Player 2? Set who to the player number of {both[0]}, name both players, say whether you are "
              f"sure, and say briefly what you went by (clothing, build, hair, technique, handedness).")
    a, _ = ask(prompt, imgs, SCHEMA, d / "who_cache")
    out = {"who": a["who"], "sure": a["sure"], "why": a["why"], "by_match": True}
    out["1"], out["2"] = both if a["who"] == 1 else both[::-1]
    return out


def run(spec: str, verbose: bool = True) -> dict:
    project = spec.split("/")[0]
    players, names = label(spec, verbose)
    if names is None:                               # too few serve markers: Opus names the two from their crops
        names = name(spec, players)
    for v, pl in players.items():
        d = video_dir(f"{project}/{v}")
        (d / "players.json").write_text(json.dumps(pl))
        (d / "names.json").write_text(json.dumps(names, indent=1))
    if verbose:
        print(f"  Player 1 = {names['1']}, Player 2 = {names['2']} ({'sure' if names['sure'] else 'unsure'}): "
              f"{names['why']}")
    return names


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="<project>/<video>, any piece of the match")
    a = ap.parse_args()
    run(a.video)
