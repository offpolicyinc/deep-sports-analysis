"""How the project player's technique changed across the years, per category: forehand, backhand, serve and
movement (court position on rally shots, on returns, and coming to the net). Every near-court shot of his, and
of his opponents as a control for what the broadcast changes on its own (camera, resolution).

  extract   per hit: the hitter's 13 body joints from -PRE to +POST seconds around the labelled contact,
            cleaned (dsa.scene.clean) and resampled to 30 Hz, where his feet are on the court (metres behind the near
            baseline, metres right of the centre line) and the camera measured from the court lines. Left-handed
            opponents are mirrored to right-handed. -> projects/<project>/technique/<kind>/hits.npz, hits.json
  analyse   the kind's measures per stroke, routine strokes only, compared by era with the camera and the
            situation held fixed (robust). -> projects/<project>/technique/<kind>/analysis.json

    python -m dsa.measure.technique extract federer_usopen forehand
    python -m dsa.measure.technique analyse federer_usopen forehand
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from dsa.ingest.projects import PROJECTS
from dsa.measure.shots import owners
from dsa.paths import PROJECTS as ROOT_PROJECTS
from dsa.paths import video_dir
from dsa.scene import clean
from dsa.scene.court import METRE, MID, NEAR, homography

PRE, POST, HZ = 0.9, 0.6, 30                 # seconds kept around the labelled contact; resampled rate (a project's
                                             # "hz" in dsa.ingest.projects instead: 60 with dsa.gpu.fine's 50 / 60 fps skeletons)


def grid(hz: int = HZ) -> np.ndarray:
    return np.round(np.arange(-PRE * hz, POST * hz + 1) / hz, 4)


GRID = grid()
SMOOTH_S = 0.17                              # seconds: dsa.scene.clean's smoothing span at 30 fps, for dsa.gpu.fine's frames
FEED = 0.3                                   # forehand / backhand: the ball seen at least this many of his heights
                                             # above his head before contact (from the other end), else a feed
JOINTS = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]            # dsa.gpu.models.KP: nose, then body pairs L/R
NAMES = ["nose", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_hip", "r_hip",
         "l_knee", "r_knee", "l_ankle", "r_ankle"]
MIRROR = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11]                 # left <-> right, for left-handers
LEFTY = {"Feliciano Lopez", "Fernando Verdasco", "Donald Young", "Adrian Mannarino", "Rafael Nadal", "Ben Shelton",
         "Denis Shapovalov", "Jack Draper", "Player F"}


def lefty(player: str) -> bool:
    """Left-handed, by full name or (Federer's opponents, named from the match title) surname alone."""
    return player in LEFTY or any(player == n.split()[-1] for n in LEFTY)


KINDS = {                     # which hits; the front foot at contact for a right-hander; mirrored for analysis
    "forehand": {"select": lambda h: (h["shot"], h["hand"], h["technique"]) == ("stroke", "fh", "gs"),
                 "front": "left", "flip": False},
    "backhand": {"select": lambda h: (h["shot"], h["hand"], h["technique"]) == ("stroke", "bh", "gs"),
                 "front": "right", "flip": True},
    "serve": {"select": lambda h: h["shot"] == "serve", "front": "left", "flip": False, "win": (-0.9, 0.6)},
    "rally": {"select": lambda h: h["shot"] == "stroke", "front": "left", "flip": False},
    "return": {"select": lambda h: h["shot"] == "return", "front": "left", "flip": False},
}


def hits(spec: str, kind: str) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """(meta, pose (N, len(grid), 13, 3) as x, y, score in image px, x mirrored for left-handers,
    feet (N, len(grid), 2) in court metres) for the near-court hits of `kind` in `spec`. The skeleton comes from
    dsa.gpu.fine's 50 / 60 fps frames where the video has them (fine.npz) and the project measures from them
    ("fine_measures", default on), else from the 25 / 30 fps ones."""
    from dsa.scene import results

    project, v = spec.split("/")
    who = PROJECTS[project].get("who")          # the project player, if it has one: "his" in the rows
    G = grid(PROJECTS[project].get("hz", HZ))
    d = results.load(spec)
    fine = None
    if PROJECTS[project].get("fine_measures", True) and (d["src"] / "fine.npz").exists():
        fz, fj = np.load(d["src"] / "fine.npz"), json.loads((d["src"] / "fine.json").read_text())
        fine = {"kp": fz["kp"], "box": fz["box"], "frames": np.array(fj["frames"]), "fps": fj["fps"],
                "ratio": fj["ratio"], "windows": fj["windows"]}
        fine["smooth"] = int(round(SMOOTH_S * fine["fps"])) // 2 * 2 + 1         # odd, the same span in seconds
    fps, starts, nf, pb = d["fps"], [int(a) for a in d["starts"]], d["pl"]["near_far"], d["pl"]["boxes"]
    phone = PROJECTS[project].get("phone", False)

    def chosen(h: dict) -> bool:
        if KINDS[kind]["select"](h):
            return True
        if phone and kind in ("forehand", "backhand"):
            return (h["shot"], h["hand"], h["technique"]) == ("stroke", kind[0] + "h", "lob")
        return phone and kind == "serve" and (h["shot"], h["technique"]) == ("stroke", "smash")

    meta, poses, feet = [], [], []
    for h, player in owners(spec):
        if player is None or h["side"] != "near" or not chosen(h):
            continue
        f = h["frame"]
        lo = max(a for a in starts[:-1] if a <= f)
        hi = min(a for a in starts[1:] if a > f)
        k = nf[f][0]
        f0, f1 = max(lo, f - int(np.ceil(PRE * fps)) - 2), min(hi, f + int(np.ceil(POST * fps)) + 3)
        n = f1 - f0
        joint, height, court = np.full((n, 65, 3), np.nan), np.full(n, np.nan), np.full((n, 2), np.nan)
        for i in range(n):
            g, b = f0 + i, pb[f0 + i][0]
            if nf[g][0] != k or b is None:
                continue
            j = np.flatnonzero((d["boxes"][g][:, :4].astype(int) == b).all(1))
            if len(j):
                joint[i], height[i] = d["kps"][g][j[0]], b[3] - b[1]
        joint = clean.joints(joint, height)[:, JOINTS]
        t = (np.arange(f0, f1) - f) / fps
        tp, jp = t, joint                             # the skeleton the stroke is measured on
        if fine is not None:
            c = h["source_frame"] * fine["ratio"]
            a, b = next(((a, b) for a, b in fine["windows"] if a <= c < b), (0, 0))
            sel = np.flatnonzero((fine["frames"] >= max(a, c - (PRE + 0.1) * fine["fps"]))
                                 & (fine["frames"] < min(b, c + (POST + 0.1) * fine["fps"])))
            if len(sel) and np.isfinite(fine["box"][sel, 0]).mean() > 0.5:
                hb = fine["box"][sel, 3] - fine["box"][sel, 1]
                jp = clean.joints(fine["kp"][sel].astype(float), hb, smooth=fine["smooth"])[:, JOINTS]
                tp = (fine["frames"][sel] - c) / fine["fps"]
        for i in range(n):
            m = homography(d["court"][f0 + i]) if d["court"] is not None and d["court"][f0 + i] is not None else None
            a = joint[i, 11:13, :2]
            if m is not None and np.isfinite(a).all():
                p = (m @ np.append(a.mean(0), 1))
                court[i] = ((p[1] / p[2] - NEAR) / METRE, (p[0] / p[2] - MID) / METRE)
        pose = np.full((len(G), 13, 3), np.nan)
        for q in range(13):
            for c in range(3):
                ok = np.isfinite(jp[:, q, c])
                if ok.sum() >= 2:
                    pose[:, q, c] = np.interp(G, tp[ok], jp[ok, q, c], left=np.nan, right=np.nan)
        foot = np.full((len(G), 2), np.nan)
        for c in range(2):
            ok = np.isfinite(court[:, c])
            if ok.sum() >= 2:
                foot[:, c] = np.interp(G, t[ok], court[ok, c], left=np.nan, right=np.nan)
        cam, ground = None, None
        c = d["court"][f] if d["court"] is not None else None
        if c is not None and np.isfinite(c[:4]).all():
            nw = float(np.linalg.norm(c[3] - c[2]))
            cam = {"zoom": nw / d["size"][0], "angle": float(np.linalg.norm(c[1] - c[0])) / nw,
                   "tilt": float((c[2, 1] + c[3, 1] - c[0, 1] - c[1, 1]) / 2) / nw,
                   "baseline_y": float((c[2, 1] + c[3, 1]) / 2) / d["size"][1], "width": d["size"][0]}
            m = homography(c)
            i = f - f0
            a = joint[i, 11:13, :2] if 0 <= i < n else None
            if m is not None and a is not None and np.isfinite(a).all():
                ground = np.array([(m @ np.append(q, 1))[:2] / (m @ np.append(q, 1))[2] for q in a]) / METRE
                mid = ground.mean(0) * METRE                 # pixels per metre across the court where he stands
                inv = np.linalg.inv(m)
                ends = [inv @ np.array([mid[0] + d * METRE / 2, mid[1], 1]) for d in (-1, 1)]
                cam["ppm"] = float(np.linalg.norm(ends[0][:2] / ends[0][2] - ends[1][:2] / ends[1][2]))
        stance = None
        if ground is not None:                       # ankles on the court, metres: x right, y toward the near baseline
            swap = (lefty(player)) != (KINDS[kind]["front"] == "right")      # ground[0] is the left ankle
            front, back = (ground[1], ground[0]) if swap else (ground[0], ground[1])
            # across: how far the front foot has stepped across the back one, toward the other foot's side (negative:
            # out to its own side, an open stance); the near player faces the net, so the court's right is his right
            across = (back[0] - front[0]) if swap else (front[0] - back[0])      # swap: the front foot is the right one
            stance = {"stance_m": float(np.linalg.norm(front - back)), "stance_side_m": float(abs(front[0] - back[0])),
                      "stance_fwd_m": float(back[1] - front[1]), "stance_across_m": float(across)}
        if lefty(player):
            pose = pose[:, MIRROR]
            pose[..., 0] = -pose[..., 0]
            foot[:, 1] = -foot[:, 1]
        if kind in ("forehand", "backhand"):         # dsa.gpu.strokes' forehand / backhand only where the contact
            if "by" not in h:                        # (labelled strokes are trusted: a camera off to the side
                x = pose[int(np.argmin(abs(G))), :, 0]   # puts the contact on either side of his hips)
                side = x[6] - x[7:9].mean() if kind == "forehand" else x[5:7].mean() - x[7:9].mean()
                if np.isfinite(side) and (side > 0) != (kind == "forehand"):    # (it often swaps a left-hander's)
                    continue
            b = pb[f][0]                             # and only a ball that came from the other end: seen above his
            ball = d["ball"][max(lo, f - int(1.2 * fps)):f - int(0.15 * fps)]   # head in the second before, not
            ys = ball[ball[:, 0] > 0, 2]             # a feed (dropped and hit, it never rises above him)
            if (b is not None and len(ys) and (b[1] - ys.min()) / (b[3] - b[1]) < FEED
                    and v not in PROJECTS[project].get("no_feed_check", []) and "by" not in h):   # labelled: no feeds
                continue
        meta.append({"video": v, "year": int(v[:4]), "frame": f, "player": player, "his": player == who,
                     "lefty": lefty(player), "direction": h["direction"], "outcome": h["outcome"],
                     "shot": h["shot"], "hand": h["hand"], "technique": h["technique"],
                     "camera": cam, "stance": stance,
                     "height": float(np.nanmedian(height)) if np.isfinite(height).any() else None})
        poses.append(pose)
        feet.append(foot)
    return meta, np.array(poses).reshape(-1, len(G), 13, 3), np.array(feet).reshape(-1, len(G), 2)


def extract(project: str, kind: str) -> None:
    specs = [f"{project}/{v}" for v in PROJECTS[project]["videos"]
             if (video_dir(f"{project}/{v}") / "names.json").exists()]   # through step 4 (not a piece still running)
    meta, poses, feet = [], [], []
    with ProcessPoolExecutor(6) as ex:
        for spec, (m, p, f) in zip(specs, ex.map(hits, specs, [kind] * len(specs))):
            print(f"{spec.split('/')[1]:24s} his {sum(x['his'] for x in m):3d}  other {sum(not x['his'] for x in m):3d}",
                  flush=True)
            meta, poses, feet = meta + m, poses + [p], feet + [f]
    out = ROOT_PROJECTS / project / "technique" / kind
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "hits.npz", pose=np.concatenate(poses), feet=np.concatenate(feet),
                        grid=grid(PROJECTS[project].get("hz", HZ)),
                        names=np.array(NAMES))
    (out / "hits.json").write_text(json.dumps(meta))
    print(f"{len(meta)} {kind}: his {sum(x['his'] for x in meta)}, other {sum(not x['his'] for x in meta)}")


ERAS = [(2003, 2007), (2008, 2012), (2013, 2015), (2016, 2019)]
ERA_NAMES = ["2003-07", "2008-12", "2013-15", "2017-19"]
WIN = (-0.6, 0.4)                            # seconds analysed around the aligned contact
QUALITY = 0.9                                # share of joint samples seen in WIN, else the stroke is left out
STANDARD = {"depth": (-1.0, 3.0),            # routine strokes only: metres behind the baseline (- inside),
            "run": 0.8,                      # the slower share by running speed (per player group),
            "extreme": 5.0,                  # and no measure this many robust SDs from the player's median
            "rise": 0.3}                     # forehand / backhand: the wrist this many torso lengths higher 0.3 s
                                             # after contact than 0.15 s before (low to high: no slices, drop shots,
                                             # blocks or stretched defensive swings, which the labels miss)
PHONE_DEPTH = (-3.0, 3.0)                    # projects with "phone": amateurs rally from up to 3 m inside the baseline
                                             # (forehand / backhand); and dsa.gpu.strokes on a low phone behind the
                                             # baseline calls many forehands / backhands "lob" and most serves
                                             # "stroke, smash", so hits() takes those too (a real smash, near the
                                             # net, fails the serve's depth)


def era(year: int) -> int:
    return next(i for i, (a, b) in enumerate(ERAS) if a <= year <= b)


def normalise(pose: np.ndarray) -> np.ndarray:
    """(N, T, 13, 3) image px -> (N, T, 13, 2): centred on the mid-hip each frame, y up, in torso lengths
    (mid-shoulder to mid-hip, the median over the stroke)."""
    xy = pose[..., :2].copy()
    xy[..., 1] = -xy[..., 1]
    hip = xy[:, :, 7:9].mean(2, keepdims=True)
    torso = np.linalg.norm(xy[:, :, 1:3].mean(2) - xy[:, :, 7:9].mean(2), axis=-1)
    return (xy - hip) / np.nanmedian(torso, 1)[:, None, None, None]


def fill(x: np.ndarray) -> np.ndarray:
    """NaNs along time (axis 1) filled linearly, ends held."""
    x = x.copy()
    t = np.arange(x.shape[1])
    flat = x.reshape(x.shape[0], x.shape[1], -1)
    for i in range(flat.shape[0]):
        for j in range(flat.shape[2]):
            ok = np.isfinite(flat[i, :, j])
            if ok.any() and not ok.all():
                flat[i, :, j] = np.interp(t, t[ok], flat[i, ok, j])
    return flat.reshape(x.shape)


def align(xy: np.ndarray, grid: np.ndarray, win: tuple = WIN) -> np.ndarray:
    """(N, len(window), 13, 2) cut to `win` around the labelled contact."""
    hz = round(1 / (grid[1] - grid[0]))
    n = int(round((win[1] - win[0]) * hz)) + 1
    a = int(np.flatnonzero(np.isclose(grid, 0))[0]) + int(round(win[0] * hz))
    return xy[:, a:a + n]


def angle(a, b, c) -> np.ndarray:
    """Angle at b, degrees, in the image plane."""
    u, v = a - b, c - b
    cos = (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


STROKE = {                                    # name: (unit, what it measures)
    "contact_height": ("torso", "hitting wrist above the hips at contact"),
    "contact_side": ("torso", "hitting wrist out to the side of the hips at contact (backhand: to the backhand side)"),
    "elbow_angle": ("deg", "hitting elbow at contact (180 straight arm)"),
    "shoulder_width": ("deg", "how far the shoulders have opened at contact: 0 side-on, 90 square to the net, over 90 "
                              "past square"),
    "shoulder_coil": ("deg", "most the shoulders turn away from the net in the take-back: 90 side-on, over 90 back "
                             "turned past side-on"),
    "hip_coil": ("deg", "most the hips turn away from the net in the take-back: 90 side-on"),
    "separation": ("deg", "shoulders' turn minus the hips' at the shoulders' most-turned moment (shoulders turned "
                          "further than the hips)"),
    "shoulder_tilt": ("deg", "shoulder line from level at contact (negative: hitting shoulder low)"),
    "knee_bend": ("deg", "most-bent knee angle in the 0.4 s before contact (lower: more bend)"),
    "takeback": ("torso", "hitting wrist out to the side of the hips at the end of the take-back (the wrist's slowest "
                          "moment before the forward swing)"),
    "swing_time": ("s", "from the end of the take-back (the forward swing's start) to contact"),
    "wrist_speed": ("torso/s", "peak hitting-wrist speed through contact"),
    "finish_height": ("torso", "wrist above the shoulders 0.3 s after contact"),
    "finish_side": ("torso", "wrist out to the hitting side of the hips 0.3 s after contact (negative: wrapped across the body)"),
    "head_still": ("torso", "head movement relative to the hips in the 0.2 s around contact"),
    "finish_arm": ("deg", "hitting arm (shoulder to wrist) from upright 0.3 s after contact, seen from behind (positive: "
                          "out to the hitting side, negative: across the body, beyond -90 below the shoulder)"),
    "finish_open": ("deg", "how far the shoulders have opened 0.3 s after contact: 0 side-on, 90 square, over 90 past"),
    "takeback_turn": ("deg", "shoulder turn from square at the forward swing's start: 0 square, 90 side-on"),
    "stance_m": ("m", "feet apart at contact, both ankles mapped onto the court surface through its lines"),
    "stance_side_m": ("m", "feet apart sideways (along the baseline) at contact, on the court surface"),
    "stance_fwd_m": ("m", "front foot ahead of the back foot toward the net at contact, on the court surface "
                          "(0: open stance, side by side)"),
    "stance_angle": ("deg", "angle of the line from back foot to front foot against the baseline at contact, on the "
                            "court surface: 90 the front foot straight ahead (neutral), below 90 stepped across the "
                            "body (closed), above 90 out to its own side (open)"),
}
GROUND = ["stance_m", "stance_side_m", "stance_fwd_m", "stance_angle"]


def ground(stance: dict | None, k: str) -> float:
    """A stance measure on the court; stance_angle is the line from back foot to front foot against the
    baseline, 0 to 180 degrees: 90 the front foot straight ahead, less the further it steps across (closed), more the
    further out to its own side (open)."""
    if not stance:
        return np.nan
    if k == "stance_angle":
        across = stance.get("stance_across_m", stance["stance_side_m"])     # strokes extracted before it: unsigned
        return float(np.degrees(np.arctan2(stance["stance_fwd_m"], across)))
    return stance[k]
STROKE_LENGTHS = ["contact_height", "contact_side", "takeback",
           "wrist_speed", "finish_height", "finish_side", "head_still"]       # in torso lengths: depend on the scale          # from the court, not the pose: appended last
SERVE = {
    "reach": ("torso", "hitting wrist above the ankles at contact"),
    "toss_arm": ("torso", "tossing wrist's highest point above the head before contact"),
    "toss_straight": ("deg", "tossing elbow at the tossing wrist's highest point (180 straight arm)"),
    "toss_tilt": ("deg", "tossing arm (shoulder to wrist) from vertical at its highest point, seen from behind "
                         "(positive: leaning in over the head, negative: out to the side)"),
    "trophy_time": ("s", "from the tossing wrist's highest point to contact"),
    "trophy_tilt": ("deg", "shoulder line from level when the tossing wrist is highest (tossing shoulder up)"),
    "lean": ("torso", "shoulders right of the ankles at contact, seen from behind"),
    "trunk_lean": ("deg", "trunk (mid-hip to mid-shoulder) from upright at contact, seen from behind (positive: to the "
                          "tossing side)"),
    "hips_over": ("torso", "hips right of the ankles at contact, seen from behind"),
    "wrist_speed": ("torso/s", "peak hitting-wrist speed through contact"),
    "into_court": ("m", "how far his feet travel toward the net from 0.6 s before to 0.4 s after contact, on the court"),
}
SERVE_LENGTHS = ["reach", "toss_arm", "lean", "hips_over", "wrist_speed"]
RALLY = {
    "depth": ("m", "where his feet are at contact: metres behind the baseline, on the court"),
    "run_speed": ("m/s", "how fast his feet move on the court from 0.5 to 0.1 s before contact"),
    "net": ("share", "share of his shots that are volleys or smashes"),
}
RETURN = {"return_depth": ("m", "where his feet are at contact on a return: metres behind the baseline, on the court")}
MEASURES = {"forehand": (STROKE, STROKE_LENGTHS), "backhand": (STROKE, STROKE_LENGTHS),
            "serve": (SERVE, SERVE_LENGTHS), "rally": (RALLY, []), "return": (RETURN, [])}
CAMERA = ["zoom", "angle", "tilt", "baseline_y", "quality"]
CONTEXT = ["depth", "lateral", "run_speed"]


def turn(w: np.ndarray, a: int, b: int) -> tuple[np.ndarray, np.ndarray]:
    """How far the line between joints a and b (the shoulders 1-2, the hips 7-8) is turned away from facing the net,
    in degrees, per frame: seen from behind, the line looks widest when square to the net and shrinks as it turns, to
    nothing side-on, and its ends swap sides past that. So: arccos(signed width / the stroke's widest), 0 square,
    90 side-on, over 90 turned past side-on. Also the frame it is widest (square). The line's own length cancels,
    so a broad or narrow build doesn't count."""
    d = w[:, :, b, 0] - w[:, :, a, 0]
    sq = np.nanargmax(np.abs(np.nan_to_num(d)), 1)
    rows = np.arange(len(w))
    widest = np.abs(d[rows, sq])
    ang = np.degrees(np.arccos(np.clip(d * np.sign(d[rows, sq])[:, None] / widest[:, None], -1, 1)))
    return ang, sq


FWD = (-0.55, -0.05)         # the forward swing starts in here, s from contact


def phases(w: np.ndarray, t: np.ndarray) -> dict[str, np.ndarray]:
    """Each stroke's phases (frame indices into w's time axis t), from its own motion rather than fixed times:
      fwd      the start of the forward swing = the end of the take-back: walking back from the hitting wrist's
               fastest moment through contact (speed smoothed over 5 frames) until it is under half that, then on
               to where it stops slowing: the last real dip in speed before the swing, where the wrist turns from
               going back to coming through (kept within FWD)
      coil     the shoulders' most-turned moment (turn: furthest from square) in the take-back: -0.6 s to fwd
      hipcoil  the same for the hips
    w: (N, T, 13, 2) on t, contact at t = 0."""
    hz = round((len(t) - 1) / (t[-1] - t[0]))
    at = lambda s: int(np.argmin(abs(t - s)))                       # noqa: E731
    speed = np.linalg.norm(np.diff(w[:, :, 6], axis=1), axis=-1) * hz             # between frames f and f + 1
    k = np.ones(5) / 5
    sm = np.stack([np.convolve(np.pad(np.nan_to_num(v, nan=np.nanmedian(v)), 2, mode="edge"), k, "valid") for v in speed])
    peak = np.argmax(sm[:, at(-0.15):at(0.05) + 1], 1) + at(-0.15)
    fwd = peak.copy()
    for n, f in enumerate(peak):
        top = sm[n, f]
        while f > at(FWD[0]) and (sm[n, f] > 0.5 * top or sm[n, f - 1] < sm[n, f]):   # below half its peak, then
            f -= 1                                                                      # on down to the dip
        fwd[n] = min(f, at(FWD[1]))
    frames = np.arange(len(t))
    inside = (frames[None] >= at(-0.6)) & (frames[None] <= fwd[:, None])      # the take-back, per stroke
    most = lambda i, j: np.argmax(np.where(inside, np.nan_to_num(turn(w, i, j)[0]), -1), 1)   # noqa: E731
    return {"fwd": fwd, "coil": most(1, 2), "hipcoil": most(7, 8)}


def features(w: np.ndarray, feet: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """w: (N, T, 13, 2) aligned to WIN; feet (N, T, 2) court metres on the same window.
    -> (N, len(FEATURES)), (N, len(CONTEXT))."""
    t = np.round(np.linspace(WIN[0], WIN[1], w.shape[1]), 4)
    hz = round((w.shape[1] - 1) / (WIN[1] - WIN[0]))
    at = lambda s: int(np.argmin(abs(t - s)))                       # noqa: E731
    c, fin = at(0), at(0.3)
    (sa, ssq), (ha, _) = turn(w, 1, 2), turn(w, 7, 8)              # shoulders' and hips' turn from square, degrees
    ph = phases(w, t)
    fwd, coil_t = ph["fwd"], ph["coil"]
    rows = np.arange(len(w))
    # open at contact: 0 side-on, 90 square to the net, over 90 rotated past square (square came before contact)
    opened = np.where(ssq < c, 90 + sa[rows, c], 90 - sa[rows, c])
    sh_mid = w[:, :, 1:3].mean(2)
    knee = (angle(w[:, :, 7], w[:, :, 9], w[:, :, 11]) + angle(w[:, :, 8], w[:, :, 10], w[:, :, 12])) / 2
    speed = np.linalg.norm(np.diff(w[:, :, 6], axis=1), axis=-1) * hz
    f = np.stack([
        w[:, c, 6, 1], w[:, c, 6, 0], angle(w[:, c, 2], w[:, c, 4], w[:, c, 6]), opened,
        sa[rows, coil_t], ha[rows, ph["hipcoil"]], sa[rows, coil_t] - ha[rows, coil_t],
        np.degrees(np.arctan2(w[:, c, 2, 1] - w[:, c, 1, 1], np.abs(w[:, c, 2, 0] - w[:, c, 1, 0]))),   # |dx|: a
        # mirrored backhand's shoulder line points the other way; tilt from level, either way (was near +-180)
        np.nanmin(knee[:, at(-0.4):c + 1], 1), w[rows, fwd, 6, 0], (c - fwd) / hz,
        np.nanmax(speed[:, at(-0.15):at(0.1)], 1), w[:, fin, 6, 1] - sh_mid[:, fin, 1], w[:, fin, 6, 0],
        np.linalg.norm(w[:, at(0.1), 0] - w[:, at(-0.1), 0], axis=-1),
        np.degrees(np.arctan2(w[:, fin, 6, 0] - w[:, fin, 2, 0], w[:, fin, 6, 1] - w[:, fin, 2, 1])),
        np.where(ssq < fin, 90 + sa[:, fin], 90 - sa[:, fin]), sa[rows, fwd]], 1)
    run = np.linalg.norm(feet[:, at(-0.1)] - feet[:, at(-0.5)], axis=-1) / 0.4
    return f, np.stack([feet[:, c, 0], feet[:, c, 1], run], 1)


def serve_features(w: np.ndarray, feet: np.ndarray, win: tuple) -> tuple[np.ndarray, np.ndarray]:
    """SERVE per serve; context: where he stands (depth, lateral) before the serve, and no running."""
    t = np.round(np.linspace(win[0], win[1], w.shape[1]), 4)
    hz = round((w.shape[1] - 1) / (win[1] - win[0]))
    at = lambda s: int(np.argmin(abs(t - s)))                       # noqa: E731
    c, rows = at(0), np.arange(len(w))
    toss = slice(at(-0.9), at(-0.1))
    peak = np.nanargmax(w[:, toss, 5, 1], 1) + toss.start
    speed = np.linalg.norm(np.diff(w[:, :, 6], axis=1), axis=-1) * hz
    sh = w[rows, peak][:, 1:3]
    f = np.stack([
        w[:, c, 6, 1] - w[:, c, 11:13, 1].mean(1), w[rows, peak, 5, 1] - w[rows, peak, 0, 1],
        angle(w[rows, peak, 1], w[rows, peak, 3], w[rows, peak, 5]),
        np.degrees(np.arctan2(w[rows, peak, 5, 0] - w[rows, peak, 1, 0], w[rows, peak, 5, 1] - w[rows, peak, 1, 1])),
        (c - peak) / hz,
        np.degrees(np.arctan2(sh[:, 0, 1] - sh[:, 1, 1], sh[:, 1, 0] - sh[:, 0, 0])),
        w[:, c, 1:3, 0].mean(1) - w[:, c, 11:13, 0].mean(1),
        -np.degrees(np.arctan2(w[:, c, 1:3, 0].mean(1) - w[:, c, 7:9, 0].mean(1), w[:, c, 1:3, 1].mean(1) - w[:, c, 7:9, 1].mean(1))),
        w[:, c, 7:9, 0].mean(1) - w[:, c, 11:13, 0].mean(1), np.nanmax(speed[:, at(-0.15):at(0.1)], 1),
        feet[:, at(-0.6), 0] - feet[:, at(0.4), 0]], 1)
    return f, np.stack([feet[:, at(-0.6), 0], feet[:, at(-0.6), 1], np.zeros(len(w))], 1)


def court_features(kind: str, feet: np.ndarray, meta: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """RALLY or RETURN per shot, all from where his feet are on the court; no context held fixed (where he
    stands is the measure)."""
    t = np.round(np.linspace(WIN[0], WIN[1], feet.shape[1]), 4)
    at = lambda s: int(np.argmin(abs(t - s)))                       # noqa: E731
    depth = feet[:, at(0), 0]
    if kind == "return":
        f = depth[:, None]
    else:
        run = np.linalg.norm(feet[:, at(-0.1)] - feet[:, at(-0.5)], axis=-1) / 0.4
        net = np.array([m["technique"] in ("volley", "smash") for m in meta], float)
        f = np.stack([depth, run, net], 1)
    return f, np.zeros((len(feet), 1))


def robust(f: np.ndarray, ctx: np.ndarray, cam: np.ndarray, his: np.ndarray, eras: np.ndarray,
           groups: np.ndarray, dirs: list, years: np.ndarray, rng, boots: int = 2000) -> dict:
    """The test behind each finding, per feature. First, from all strokes (his and his opponents'), the
    measured camera (zoom, angle, tilt and framing from the court lines; picture quality from the pose model's
    confidence; the player's size on screen) and the situation (court position, running speed, direction)
    are regressed out, their effects estimated within each era and player so that real change over the years
    isn't taken for camera. Then each match gives one number, his mean, and the matches of the last era are
    compared with those of the first by Welch's t-test:
      own      his change (last era minus first), t, degrees of freedom, p, and p Holm-adjusted over the features
      tour     the same for his opponents: did the players he faced change too?
      vs       his per-match mean minus his opponent's in the same match: his change beyond theirs
    plus per-year means and era means with 95% intervals (matches resampled) for him and his opponents, for the
    charts."""
    from scipy.stats import ttest_ind

    levels = sorted(set(dirs))
    nuis = np.column_stack([ctx, cam] + [np.array([d == lv for d in dirs], float) for lv in levels[1:]])
    nuis = nuis - nuis.mean(0)
    cells = np.column_stack([(eras == k) & (his == h) for k in range(len(ERAS)) for h in (True, False)]).astype(float)
    beta = np.linalg.lstsq(np.column_stack([cells, nuis]), f, rcond=None)[0]     # nuisance effects within era
    res = f - nuis @ beta[cells.shape[1]:]

    def per_match(sel, other=None):
        ms = [g for g in dict.fromkeys(groups) if (sel & (groups == g)).sum() >= 2
              and (other is None or (other & (groups == g)).sum() >= 2)]
        v = np.stack([res[sel & (groups == g)].mean(0) - (0 if other is None else res[other & (groups == g)].mean(0))
                      for g in ms])
        e = np.array([eras[groups == g][0] for g in ms])
        return v, [np.flatnonzero(e == k) for k in range(len(ERAS))]

    def test(v, by):
        late, early = v[by[-1]], v[by[0]]
        t = ttest_ind(late, early, equal_var=False)
        p = np.asarray(t.pvalue)
        order = np.argsort(p)                                  # Holm: the k-th smallest p times (features - k)
        holm = np.empty_like(p)
        holm[order] = np.minimum(1, np.maximum.accumulate(p[order] * (len(p) - np.arange(len(p)))))
        return {"change": (late.mean(0) - early.mean(0)).round(4).tolist(), "t": np.round(t.statistic, 3).tolist(),
                "df": np.round(t.df, 1).tolist(), "p": p.tolist(), "holm": holm.tolist(),
                "matches": [len(b) for b in by]}

    own, tour, vs = test(*per_match(his)), test(*per_match(~his)), test(*per_match(his, ~his))
    yrs = sorted(set(years.tolist()))
    per_year = {who: {"mean": [], "se": [], "n": []} for who in ("his", "other")}
    for y in yrs:
        for who, sel in (("his", his), ("other", ~his)):
            x = f[sel & (years == y)]
            per_year[who]["mean"].append(np.round(x.mean(0), 4).tolist())
            per_year[who]["se"].append(np.round(x.std(0) / np.sqrt(max(len(x), 1)), 4).tolist())
            per_year[who]["n"].append(len(x))
    raw = {who: np.stack([f[sel & (eras == k)].mean(0) for k in range(len(ERAS))]).round(4).tolist()
           for who, sel in (("his", his), ("other", ~his))}
    era_ci = {}                                        # 95% interval of each era's mean, resampling its matches
    for who, sel in (("his", his), ("other", ~his)):
        ci = []
        for k in range(len(ERAS)):
            ms = [g for g in dict.fromkeys(groups[sel & (eras == k)])]
            rows = [f[sel & (groups == g)] for g in ms]
            bs = [np.concatenate([rows[j] for j in rng.choice(len(rows), len(rows))]).mean(0) for _ in range(boots // 4)]
            ci.append(np.percentile(bs, [2.5, 97.5], axis=0))
        era_ci[who] = np.round(np.stack(ci, 1), 4).tolist()     # (2, eras, F)
    return {"own": own, "era_ci": era_ci, "tour": tour, "vs": vs, "sd": f[his].std(0).round(4).tolist(), "years": yrs,
            "per_year": per_year, "era_means": raw}


def measure(kind: str, w: np.ndarray, fw: np.ndarray, meta: list[dict], win: tuple) -> tuple[np.ndarray, np.ndarray]:
    """The kind's measures and context per stroke."""
    if kind == "serve":
        return serve_features(w, fw, win)
    if kind in ("rally", "return"):
        return court_features(kind, fw, meta)
    f, ctx = features(w, fw)
    return np.column_stack([f] + [[ground(x["stance"], k) for x in meta] for k in GROUND]), ctx


def camera(meta: list[dict], score: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Per stroke: the camera measured from the court lines (CAMERA), picture quality (the pose model's
    confidence in the analysed window) and the player's height on screen; gaps filled with the median."""
    quality = np.nanmean(score[:, lo:hi + 1], (1, 2))
    cam = np.array([[(x["camera"] or {}).get(k, np.nan) for k in CAMERA[:-1]] + [q, x["height"] or np.nan]
                    for x, q in zip(meta, quality)], float)
    return np.where(np.isfinite(cam), cam, np.nanmedian(cam, 0))


def prepare(project: str, kind: str, group, yscale: float = 1.0) -> dict:
    """The kind's strokes, measured: hits.npz / hits.json (extract), well seen, routine (STANDARD: a routine
    situation and no measure far from that group's usual; groups: group(meta row), e.g. him / his opponents, or
    each player). Returns keep (indices into hits), w (aligned poses), f (measures), ctx (situation), score
    (pose confidence), meta, grid, lo / hi (the analysed window in grid steps), win, res_std (what was dropped),
    n (strokes extracted), and z (hits.npz). yscale: the skeletons' heights in the picture scaled by it before
    anything is measured (a camera that foreshortens the body less than another, made to look like it)."""
    out_dir = ROOT_PROJECTS / project / "technique" / kind
    z = np.load(out_dir / "hits.npz")
    meta = json.loads((out_dir / "hits.json").read_text())
    pose = z["pose"].copy()
    pose[..., 1] *= yscale
    grid, xy, feet = z["grid"], normalise(pose), z["feet"]
    if KINDS[kind]["flip"]:                     # a backhand mirrored to look like a forehand: out to the side is +
        xy[..., 0] = -xy[..., 0]
    win = KINDS[kind].get("win", WIN)
    lo, hi = int(np.argmin(abs(grid - win[0]))), int(np.argmin(abs(grid - win[1])))
    seen = np.isfinite(xy[:, lo:hi + 1, :, 0]).mean((1, 2))
    ctx_ok = np.isfinite(feet[:, int(np.argmin(abs(grid)))]).all(1)
    keep = np.flatnonzero((seen >= QUALITY) & ctx_ok)
    score = z["pose"][keep, :, :, 2]
    xy, feet, meta = fill(xy[keep]), fill(feet[keep]), [meta[i] for i in keep]
    w = align(xy, grid, win)
    fw = feet[:, lo:][:, :w.shape[1]]
    f, ctx = measure(kind, w, fw, meta, win)

    # routine strokes only: a routine situation (STANDARD) and no measure far from that group's usual
    g = np.array([group(m) for m in meta])
    std = np.ones(len(f), bool)
    extreme = np.zeros(len(f), bool)
    if kind in ("forehand", "backhand", "serve"):
        depth = PHONE_DEPTH if PROJECTS[project].get("phone") and kind != "serve" else STANDARD["depth"]
        std = (ctx[:, 0] >= depth[0]) & (ctx[:, 0] <= depth[1])
        if kind != "serve":
            rise = xy[:, int(np.argmin(abs(grid - 0.3))), 6, 1] - xy[:, int(np.argmin(abs(grid + 0.15))), 6, 1]
            std &= rise >= STANDARD["rise"]
        for sel in (g == x for x in dict.fromkeys(g)):
            if kind != "serve":
                std[sel] &= ctx[sel, 2] <= np.nanquantile(ctx[sel, 2], STANDARD["run"])
            med = np.nanmedian(f[sel], 0)
            mad = 1.4826 * np.nanmedian(abs(f[sel] - med), 0)
            mad = np.maximum(mad, 0.25 * np.nanstd(f[sel], 0))   # a measure often exactly 0 (coil: the line
            # edge-on) has MAD 0, which made every other value extreme
            extreme[sel] = (abs(f[sel] - med) > STANDARD["extreme"] * mad).any(1)
    ok = std & ~extreme
    res_std = {"situation": int((~std).sum()), "extreme": int((std & extreme).sum()), "kept": int(ok.sum()),
               "rule": STANDARD}
    return {"keep": keep[ok], "w": w[ok], "f": f[ok], "ctx": ctx[ok], "score": score[ok],
            "meta": [m for m, k in zip(meta, ok) if k], "grid": grid, "lo": lo, "hi": hi, "win": win,
            "res_std": res_std, "n": len(z["pose"]), "z": z}


def analyse(project: str, kind: str, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    out_dir = ROOT_PROJECTS / project / "technique" / kind
    names, lengths_ = MEASURES[kind]
    d = prepare(project, kind, lambda m: m["his"])
    keep, w, f, ctx, score, meta = d["keep"], d["w"], d["f"], d["ctx"], d["score"], d["meta"]
    lo, hi, win, res_std, z = d["lo"], d["hi"], d["win"], d["res_std"], d["z"]
    his = np.array([m["his"] for m in meta])

    def run(f, ctx, meta, score, his, boots=2000):
        f = np.where(np.isfinite(f), f, np.nanmedian(f, 0))
        ctx = np.where(np.isfinite(ctx), ctx, np.nanmedian(ctx, 0))
        return robust(f, ctx, camera(meta, score, lo, hi), his, np.array([era(m["year"]) for m in meta]),
                      np.array([m["video"] for m in meta]), [m["direction"] or "?" for m in meta],
                      np.array([m["year"] for m in meta]), rng, boots=boots)

    eras = np.array([era(m["year"]) for m in meta])
    res = {"kind": kind, "kept": int(len(keep)), "dropped": int(len(z["pose"]) - len(keep)), "standard": res_std,
           "eras": ERA_NAMES, "features": names, "lengths": lengths_, "win": win,
           "counts": {who: [int(((his == (who == "his")) & (eras == k)).sum()) for k in range(len(ERAS))]
                      for who in ("his", "other")}}
    cam = camera(meta, score, lo, hi)
    res["camera"] = {k: [float(np.median(cam[his & (np.array([x["year"] for x in meta]) == y), j]))
                         for y in sorted({x["year"] for x in meta})] for j, k in enumerate(CAMERA + ["size"])}
    res["robust"] = run(f, ctx, meta, score, his)
    # lengths in real metres from the court instead of torso lengths, for display
    px = z["pose"][keep][:, lo:hi + 1]
    mid = lambda a, b: (px[:, :, a, :2] + px[:, :, b, :2]) / 2                            # noqa: E731
    torso = np.nanmedian(np.linalg.norm(mid(1, 2) - mid(7, 8), axis=-1), 1)
    ppm = np.array([(x["camera"] or {}).get("ppm", np.nan) for x in meta])
    idx = [list(names).index(n) for n in lengths_]
    res["scales"] = {}
    for name, scale in (("metres", ppm),) if idx else ():
        g = f.copy()
        g[:, idx] = f[:, idx] * (torso / scale)[:, None]
        rb = run(g, ctx, meta, score, his, 1000)
        res["scales"][name] = {"own": rb["own"], "sd": rb["sd"], "era_means": rb["era_means"], "era_ci": rb["era_ci"]}
    res["strokes"] = {"index": keep.tolist(), "features": np.round(f, 4).tolist(),
                      "metres": np.round(torso / ppm, 4).tolist(),     # torso lengths -> metres, per stroke
                      "context": np.round(ctx, 3).tolist(), "quality": np.round(cam[:, CAMERA.index("quality")], 3).tolist()}
    res["robust"]["mean_pose"] = {who: [np.round(np.nanmean(w[sel & (eras == k)], 0), 3).tolist()
                                        for k in range(len(ERAS))] for who, sel in (("his", his), ("other", ~his))}
    (out_dir / "analysis.json").write_text(json.dumps(res))
    print(f"{kind}: {len(keep)} kept of {len(z['pose'])}", flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["extract", "analyse"])
    ap.add_argument("project")
    ap.add_argument("kind", choices=list(KINDS))
    a = ap.parse_args()
    extract(a.project, a.kind) if a.step == "extract" else analyse(a.project, a.kind)
