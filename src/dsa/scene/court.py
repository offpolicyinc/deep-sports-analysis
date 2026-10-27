"""The court in every frame: 14 points where court lines cross, in the order of POINTS.

  phone   the camera is fixed but gets bumped. Opus marks the court once, OPUS_IN_S in, snapped to the painted
          lines (dsa.scene.court_snap; kept as marked when the snap fails). Every WINDOW_S a frame is aligned to
          the marked one by what doesn't move (ORB features, a RANSAC homography: the players moving through
          don't count, nor do shadows, which have no corners); when the camera has moved the court moves with
          it. Opus is asked again only when a frame can't be aligned at all: the view itself changed.
  tv      the camera pans and cuts. TennisCourtDetector runs every COURT_EVERY frames on the GPU
          (dsa.gpu.process); its points are kept when they form a plausible whole court, else snapped to the lines.
          A check that jumps away from the ones around it is replaced by their median (`steady`).
          A failed check keeps the last court for up to COURT_HOLD frames (a player across a line), then
          drops it (a cut to a close-up).

Reference court (TennisCourtDetector's court_reference.py): the 14 points in REF units, METRE per metre,
near baseline at the bottom. `homography` maps image points onto it; `in_area` is where players play.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

from dsa.video import frame_at, info

POINTS = [
    "far baseline, left doubles corner", "far baseline, right doubles corner",
    "near baseline, left doubles corner", "near baseline, right doubles corner",
    "left singles sideline at the far baseline", "left singles sideline at the near baseline",
    "right singles sideline at the far baseline", "right singles sideline at the near baseline",
    "far service line at the left singles sideline", "far service line at the right singles sideline",
    "near service line at the left singles sideline", "near service line at the right singles sideline",
    "far T (centre service line meets far service line)", "near T (centre service line meets near service line)",
]
REF = [(286, 561), (1379, 561), (286, 2935), (1379, 2935), (423, 561), (423, 2935), (1242, 561),
       (1242, 2935), (423, 1110), (1242, 1110), (423, 2386), (1242, 2386), (832, 1110), (832, 2386)]
LINES = [(0, 1), (2, 3), (0, 2), (1, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13)]
LEFT, RIGHT, FAR, NEAR, NET, MID = 286, 1379, 561, 2935, 1748, 832
METRE = 1117 / 10.97

WINDOW_S, OPUS_IN_S = 5.0, 2.0      # phone: the camera checked every 5 s; Opus marks the court 2 s in
ALIGN_W, RANSAC_PX = 960, 3.0       # frames aligned at 960 px wide, a feature agreeing within 3 px
MIN_MATCHED, STILL_PX = 60, 6.0     # fewer features agreeing: the view changed; a court within 6 px: not moved
ORB = cv2.ORB_create(4000)
MATCHER = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
CONSISTENT = 0.02                   # of the frame's width: a marked point further off the court the rest make is wrong
COURT_EVERY, COURT_HOLD = 5, 15     # frames (tv): a check every 5 frames; a failed one keeps the last court
JUMPY, JUMPY_WINDOW = 0.04, 9        # tv: a check off the median of the 9 around it by 4% of the court's width

SCHEMA = {
    "type": "object",
    "properties": {"points": {"type": "array", "minItems": 14, "maxItems": 14, "items": {
        "type": "object",
        "properties": {"in_view": {"type": "boolean"}, "x": {"type": "number"}, "y": {"type": "number"}},
        "required": ["in_view", "x", "y"], "additionalProperties": False}}},
    "required": ["points"],
    "additionalProperties": False,
}


def prompt(w: int, h: int, what: str) -> str:
    """A w x h frame and the numbered diagram in, the 14 points out."""
    return (
        f"The first image is a {w} x {h} frame of a {what}. The second is a diagram of a tennis "
        "court seen from above, the near baseline at the bottom, with 14 numbered points where court lines "
        "cross. Return the pixel coordinates (x from the left edge, y from the top edge) of each point in "
        "the first image, in order 1 to 14:\n"
        + "\n".join(f"{i + 1}. {p}" for i, p in enumerate(POINTS))
        + "\nLeft and right are as seen from the near baseline, looking at the far one. Place each point at "
        "the centre of the crossing lines, as precisely as you can. Set in_view to false for a point outside "
        "the frame or hidden (x and y then your best estimate). Look at the images directly; do not run any "
        "commands."
    )


def diagram(path: Path) -> None:
    s = 0.25
    img = np.full((int(3484 * s), int(1665 * s), 3), (60, 120, 60), np.uint8)
    ref = [(int(x * s), int(y * s)) for x, y in REF]
    for a, b in LINES:
        cv2.line(img, ref[a], ref[b], (255, 255, 255), 2)
    cv2.line(img, (ref[0][0], int(1748 * s)), (ref[1][0], int(1748 * s)), (200, 200, 200), 1)
    for i, p in enumerate(ref):
        cv2.circle(img, p, 6, (0, 0, 255), -1)
        cv2.putText(img, str(i + 1), (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imwrite(str(path), img)


# --- geometry -----------------------------------------------------------------------------------------------

def homography(pts: np.ndarray) -> np.ndarray | None:
    """Image -> reference court, from the finite points."""
    ok = np.isfinite(pts).all(1)
    if ok.sum() < 4:
        return None
    m, _ = cv2.findHomography(np.float32(pts[ok, :2]), np.float32(np.array(REF, float)[ok]), 0)
    return m


def in_area(xy: np.ndarray) -> np.ndarray:
    """Reference-court points in the playing area: the court, 3.5 m beside it, 8 m behind each baseline,
    minus the umpire's band within 2 m of the net outside the doubles sidelines."""
    x, y = xy[:, 0], xy[:, 1]
    area = (x >= LEFT - 3.5 * METRE) & (x <= RIGHT + 3.5 * METRE) & (y >= FAR - 8 * METRE) & (y <= NEAR + 8 * METRE)
    return area & ~((np.abs(y - NET) < 2 * METRE) & ~((x >= LEFT) & (x <= RIGHT)))


def consistent(pts: np.ndarray, w: int) -> np.ndarray:
    """Marked points made one court: a homography fitted without the points that disagree with the rest (RANSAC,
    within CONSISTENT of the frame's width), each point then put where it says. Opus now and then labels a line
    wrong (the far service line, squeezed behind the net on a low phone), which would skew the whole court."""
    m, agree = cv2.findHomography(np.float32(REF), np.float32(pts[:, :2]), cv2.RANSAC, CONSISTENT * w)
    if m is None or agree.sum() < 8:
        return pts
    out = pts.copy()
    out[:, :2] = cv2.perspectiveTransform(np.float32(REF)[None], m)[0]
    return out


def court_ok(pts: np.ndarray | None, size: tuple[int, int]) -> bool:
    """A plausible whole-court view: far baseline in frame, near baseline below it and wider, court area a
    reasonable share of the frame, left / right not flipped."""
    if pts is None or not np.isfinite(pts).all():
        return False
    w, h = size
    fl, fr, nl, nr = pts[0], pts[1], pts[2], pts[3]
    far_in = all(0 <= p[0] < w and 0 <= p[1] < h for p in (fl, fr))
    order = fl[0] < fr[0] and nl[0] < nr[0] and max(fl[1], fr[1]) < min(nl[1], nr[1])
    wider = np.linalg.norm(nr - nl) >= 0.9 * np.linalg.norm(fr - fl)
    area = cv2.contourArea(np.float32([fl, fr, nr, nl])) / (w * h)
    return bool(far_in and order and wider and 0.05 <= area <= 1.5)


MAX_TILT, CENTRE, MIN_DEPTH = 0.08, (0.3, 0.7), 0.4


def main_camera(pts: np.ndarray, w: int, min_depth: float = MIN_DEPTH) -> bool:
    """TV's main camera: high behind one end, looking straight down the court. Both baselines near level
    (rise over run <= MAX_TILT: not a side camera or a court fitted to the wrong lines), the court's centre
    within CENTRE of the frame width, and the far baseline at least MIN_DEPTH as wide as the near one (0.5
    on the 2015 US Open final; the low camera used for serve replays shows it a third as wide). Outer courts'
    main cameras sit lower: a project can lower min_depth for such a video (dsa.ingest.projects)."""
    fl, fr, nl, nr = pts[0], pts[1], pts[2], pts[3]
    tilt = max(abs(fr[1] - fl[1]) / max(abs(fr[0] - fl[0]), 1), abs(nr[1] - nl[1]) / max(abs(nr[0] - nl[0]), 1))
    cx = (fl[0] + fr[0] + nl[0] + nr[0]) / 4 / w
    depth = abs(fr[0] - fl[0]) / max(abs(nr[0] - nl[0]), 1)
    return bool(tilt <= MAX_TILT and CENTRE[0] <= cx <= CENTRE[1] and depth >= min_depth)


# --- phone --------------------------------------------------------------------------------------------------

def samples(video: str, first: int, step: int) -> np.ndarray:
    """Frames first, first + step, ... grey and ALIGN_W wide: (k, h, ALIGN_W) uint8."""
    w, h = info(video)[1:]
    ah = round(h * ALIGN_W / w / 2) * 2
    pick = f"select='gte(n\\,{first})*not(mod(n-{first}\\,{step}))',scale={ALIGN_W}:{ah},format=gray"
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf", pick, "-fps_mode", "passthrough",
                          "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, ah, ALIGN_W)


def align(ref: tuple, grey: np.ndarray, scale: float) -> np.ndarray | None:
    """The homography taking full-size points of the frame `ref` (orb.detectAndCompute's) was made from to the
    same points in `grey`, from what doesn't move between them (RANSAC drops the players); None when too little
    matches: the view changed, not the camera."""
    kp, des = ORB.detectAndCompute(grey, None)
    if des is None or ref[1] is None:
        return None
    pairs = MATCHER.match(ref[1], des)
    if len(pairs) < MIN_MATCHED:
        return None
    a = np.float32([ref[0][p.queryIdx].pt for p in pairs])
    b = np.float32([kp[p.trainIdx].pt for p in pairs])
    m, ok = cv2.findHomography(a, b, cv2.RANSAC, RANSAC_PX)
    if m is None or ok.sum() < MIN_MATCHED:
        return None
    s = np.diag([scale, scale, 1.0])
    return s @ m @ np.linalg.inv(s)


def snap(bgr: np.ndarray, rough: np.ndarray) -> tuple[np.ndarray, bool]:
    """Rough (14, 2+) points snapped to the painted lines; the rough points themselves if the snap fails."""
    from dsa.scene.court_snap import refine

    res = refine(bgr, np.float32(rough))
    return (np.float32(rough)[:, :2] if res is None else np.asarray(res["points"], np.float32)), res is not None


def ask_opus(video: str, frame: int, work: Path, what: str = "tennis video filmed on a phone") -> tuple[np.ndarray, bool]:
    """Opus marks the court on `frame`, snapped to the lines where possible: ((14, 2), snapped)."""
    from dsa.opus import ask

    work.mkdir(parents=True, exist_ok=True)
    if not (work / "diagram.png").exists():
        diagram(work / "diagram.png")
    img = work / f"{Path(video).stem}_{frame}.jpg"
    bgr = frame_at(video, frame)
    cv2.imwrite(str(img), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    h, w = bgr.shape[:2]
    ans, _ = ask(prompt(w, h, what), [img.resolve(), (work / "diagram.png").resolve()], SCHEMA, work / "cache")
    raw = np.array([[q["x"], q["y"], 2.0 if q["in_view"] else 0.0] for q in ans["points"]], np.float32)
    return snap(bgr, consistent(raw, w))


def phone(video: str, fps: float, n: int, work: Path) -> tuple[list[np.ndarray | None], list[str]]:
    """The court of every frame, and a line per camera position saying how it was found: Opus marks it once, and
    it follows the camera (see the module's docstring)."""
    first, step = min(round(OPUS_IN_S * fps), n - 1), round(WINDOW_S * fps)
    scale = info(video)[1] / ALIGN_W
    segs: list[list] = []                                       # [first frame, court, how]
    ref = court = None                                          # the marked frame's features, and its court
    for k, g in enumerate(samples(video, first, step)):
        f = first + k * step
        m = align(ref, g, scale) if ref else None
        if m is None:                                           # the start, or a view nothing matches: Opus
            court, ok = ask_opus(video, f, work)
            ref = ORB.detectAndCompute(g, None)
            segs.append([0 if k == 0 else f, court, "Opus" + (", snapped" if ok else ", as marked")])
            continue
        moved = cv2.perspectiveTransform(np.float32(court)[:, None, :2], m).reshape(-1, 2)
        if np.median(np.hypot(*(moved - segs[-1][1][:, :2]).T)) > STILL_PX:
            segs.append([f, moved, f"camera moved {np.median(np.hypot(*(moved - court[:, :2]).T)):.0f} px, followed"])
    ends = [s[0] for s in segs[1:]] + [n]
    out: list[np.ndarray | None] = [None] * n
    for (s0, pts, _), s1 in zip(segs, ends):
        out[s0:s1] = [pts] * (s1 - s0)
    np.savez(work / "court.npz", start=np.array([s[0] for s in segs]), end=np.array(ends),
             points=np.array([s[1][:, :2] for s in segs], float))  # read back by dsa.scene.results
    return out, [f"{s[0] / fps:.0f}-{e / fps:.0f} s: {s[2]}" for s, e in zip(segs, ends)]


def phone_saved(work: Path, n: int) -> list[np.ndarray | None] | None:
    """The court of every frame as dsa.scene.court.phone found it (<video>/court/court.npz), or None."""
    f = work / "court.npz"
    if not f.exists():
        return None
    z, out = np.load(f), [None] * n
    for a, b, p in zip(z["start"], z["end"], z["points"]):
        out[int(a):int(b)] = [p] * (int(b) - int(a))
    return out


# --- tv -----------------------------------------------------------------------------------------------------

def check_tv(tcd, bgr: np.ndarray) -> np.ndarray:
    """One TennisCourtDetector check (on the GPU): (14, 2), all NaN when no plausible court."""
    from dsa.scene.court_snap import refine

    h, w = bgr.shape[:2]
    raw = tcd.locate(bgr)
    if np.isfinite(raw).all() and court_ok(raw, (w, h)):
        return raw.astype(np.float32)
    if np.isfinite(raw).all(1).sum() >= 4:
        r = refine(bgr, raw)
        if r is not None and court_ok(r["points"], (w, h)):
            return np.asarray(r["points"], np.float32)
    return np.full((14, 2), np.nan, np.float32)


def steady(checks: np.ndarray, breaks=()) -> np.ndarray:
    """checks with each one that jumps away from the JUMPY_WINDOW checks around it (same piece) replaced by
    their median: the detector now and then puts a perfect court in the wrong place for a few checks, while
    the camera only pans smoothly, which the median follows."""
    out = checks.copy()
    good = np.flatnonzero(np.isfinite(checks).all((1, 2)))
    piece = np.searchsorted(np.asarray(sorted(breaks), int), good, side="right")
    h = JUMPY_WINDOW // 2
    for k, i in enumerate(good):
        near = [good[j] for j in range(max(0, k - h), min(len(good), k + h + 1)) if piece[j] == piece[k]]
        med = np.median(checks[near], 0)
        if np.max(np.linalg.norm(checks[i] - med, axis=1)) > JUMPY * np.linalg.norm(med[3] - med[2]):
            out[i] = med
    return out


def tv(checks: np.ndarray, n: int, breaks=()) -> tuple[list[np.ndarray | None], list[str]]:
    """checks: (n, 14, 2) from the GPU step, a check every few frames (NaN: failed or not checked), made
    steady first. The court of every frame: moved in a straight line between two good checks of the same
    piece (the camera pans between them), else the last good one held for COURT_HOLD frames, or 2.5 check
    spacings when the checks are sparser (older GPU runs checked once a second)."""
    checks = steady(checks, breaks)
    good = np.flatnonzero(np.isfinite(checks).all((1, 2)))
    gap = int(np.median(np.diff(good))) if len(good) > 1 else COURT_EVERY
    hold = max(COURT_HOLD, int(2.5 * gap))
    out: list[np.ndarray | None] = [None] * n
    last, last_f = None, -10 ** 9
    for i in range(n):
        c = checks[i]
        if np.isfinite(c).all():
            last, last_f = c, i
        elif i - last_f > hold:
            last = None
        out[i] = last
    for a, b in zip(good, good[1:]):
        if 1 < b - a <= 2 * gap and not any(a < k <= b for k in breaks):
            t = (np.arange(a + 1, b) - a) / (b - a)
            for i, w in zip(range(a + 1, b), t):
                out[i] = (1 - w) * checks[a] + w * checks[b]
    return out, [f"court in {sum(c is not None for c in out) / max(n, 1):.0%} of frames"]
