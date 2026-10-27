"""How the project player moves, across the years: every point played at the near end of the court, tracked
continuously from the serve to the end of the point, for him and, as a control, his opponents.

  extract   per point (dsa.scene.results.points): the near player's 13 body joints and 6 foot points (toes and heel
            of each foot) per frame, cleaned (dsa.scene.clean) and resampled to 30 Hz, both ankles and heels placed
            on the court through its lines (metres behind the near baseline, metres right of the centre mark),
            and every hit of the point at either end. Left-handed opponents are mirrored.
            -> projects/<project>/movement/tracks.npz, points.json
  analyse   measures of how he moves, compared by era like dsa.measure.technique (robust: matches resampled, both
            halves, false-discovery rate, his opponents as the control), on three kinds of rows:
              rally      per point with a rally: speed, metres per shot, how much is sideways, where he stands
              to_ball    per rally shot of his: how far and fast he runs to it, steps, adjustment steps, step length
              recover    per rally shot of his that comes back: how central and deep he is when the opponent
                         hits, how fast he gets there, and how much of his sideways movement crosses his feet
            -> projects/<project>/movement/analysis.json
            (his movement syllables, unsupervised, come from dsa.measure.syllables; the page from dsa.pages.movement_report)

    python -m dsa.measure.movement extract federer_usopen
    python -m dsa.measure.movement analyse federer_usopen
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from dsa.ingest.projects import PROJECTS, pair
from dsa.measure.shots import player_of
from dsa.measure.technique import JOINTS, MIRROR, lefty
from dsa.paths import PROJECTS as ROOT_PROJECTS
from dsa.paths import video_dir
from dsa.scene import clean
from dsa.scene.court import METRE, MID, NEAR, homography

HZ = 30
FEET = [17, 18, 19, 20, 21, 22]               # dsa.gpu.models.KP: left big toe, small toe, heel; right the same
POINTS = JOINTS + FEET                        # 19: nose, body pairs L/R, then the feet
MIRROR_ALL = MIRROR + [16, 17, 18, 13, 14, 15]
GROUND = {"l_ankle": 11, "r_ankle": 12, "l_heel": 15, "r_heel": 18}   # indices into POINTS placed on the court


def track(spec: str) -> list[dict]:
    """Every point in `spec` whose near player is known: his joints, feet on the court and the hits."""
    from dsa.scene import results

    project, v = spec.split("/")
    who, both = PROJECTS[project].get("who"), pair(spec)     # who: the project player, if it has one ("his")
    d = results.load(spec)
    names = json.loads((d["src"] / "names.json").read_text())
    fps, starts, nf, pb = d["fps"], [int(a) for a in d["starts"]], d["pl"]["near_far"], d["pl"]["boxes"]
    pieces = list(zip(starts[:-1], starts[1:]))
    out = []
    for n_hits, f0, f1 in results.points(d["hits"], pieces, fps):
        piece = next(k for k, (a, b) in enumerate(pieces) if a <= f0 < b)
        ids = Counter(nf[f][0] for f in range(f0, f1) if nf[f][0])
        if not ids:
            continue
        k = ids.most_common(1)[0][0]
        player = player_of(names, piece, k, both)
        if player is None or ids[k] < 0.8 * (f1 - f0):   # unnamed, or not tracked through most of the point
            continue
        n = f1 - f0
        joint, height = np.full((n, len(POINTS), 3), np.nan), np.full(n, np.nan)
        for i in range(n):
            g, b = f0 + i, pb[f0 + i][0]
            if nf[g][0] != k or b is None:
                continue
            j = np.flatnonzero((d["boxes"][g][:, :4].astype(int) == b).all(1))
            if len(j):
                joint[i], height[i] = d["kps"][g][j[0]][POINTS], b[3] - b[1]
        joint = clean.joints(joint, height)
        ground = np.full((n, len(GROUND), 2), np.nan)
        for i in range(n):
            c = d["court"][f0 + i] if d["court"] is not None else None
            m = homography(c) if c is not None else None
            if m is None:
                continue
            for q, idx in enumerate(GROUND.values()):
                if np.isfinite(joint[i, idx, :2]).all():
                    p = m @ np.append(joint[i, idx, :2], 1)
                    ground[i, q] = ((p[1] / p[2] - NEAR) / METRE, (p[0] / p[2] - MID) / METRE)
        # resampled to HZ
        t = np.arange(n) / fps
        grid = np.arange(0, t[-1] + 1e-9, 1 / HZ)
        rs = lambda x: np.stack([np.interp(grid, t[np.isfinite(c)], c[np.isfinite(c)], left=np.nan, right=np.nan)  # noqa: E731
                                 if np.isfinite(c).sum() >= 2 else np.full(len(grid), np.nan)
                                 for c in x.reshape(n, -1).T], 1).reshape((len(grid),) + x.shape[1:])
        pose, feet, hgt = rs(joint), rs(ground), rs(height[:, None])[:, 0]
        if lefty(player, spec.split("/")[0]):
            pose = pose[:, MIRROR_ALL]
            pose[..., 0] = -pose[..., 0]
            feet = feet[:, [1, 0, 3, 2]]
            feet[..., 1] = -feet[..., 1]
        hits = [{"t": round((h["frame"] - f0) / fps, 3), "side": h["side"], "shot": h["shot"], "hand": h["hand"],
                 "technique": h["technique"], "direction": h["direction"], "outcome": h["outcome"]}
                for h in d["hits"] if f0 <= h["frame"] < f1]
        c = d["court"][(f0 + f1) // 2] if d["court"] is not None else None
        out.append({"video": v, "year": int(v[:4]), "his": player == who, "player": player, "lefty": lefty(player, spec.split("/")[0]),
                    "frame": f0, "hits": hits, "pose": pose.astype(np.float32), "feet": feet.astype(np.float32),
                    "height": hgt.astype(np.float32),
                    "zoom": float(np.linalg.norm(c[3] - c[2]) / d["size"][0]) if c is not None else None})
    return out


def extract(project: str) -> None:
    specs = [f"{project}/{v}" for v in PROJECTS[project]["videos"]
             if (video_dir(f"{project}/{v}") / "names.json").exists()]   # through step 4 (not a piece still running)
    points = []
    with ProcessPoolExecutor(6) as ex:
        for spec, ps in zip(specs, ex.map(track, specs)):
            print(f"{spec.split('/')[1]:24s} points: his {sum(p['his'] for p in ps):3d}  other "
                  f"{sum(not p['his'] for p in ps):3d}", flush=True)
            points += ps
    out = ROOT_PROJECTS / project / "movement"
    out.mkdir(parents=True, exist_ok=True)
    lengths = np.array([len(p["pose"]) for p in points])
    np.savez_compressed(out / "tracks.npz", pose=np.concatenate([p["pose"] for p in points]),
                        feet=np.concatenate([p["feet"] for p in points]),
                        height=np.concatenate([p["height"] for p in points]), lengths=lengths)
    (out / "points.json").write_text(json.dumps([{k: v for k, v in p.items() if k not in ("pose", "feet", "height")}
                                                 for p in points]))
    print(f"{len(points)} points, {lengths.sum() / HZ / 60:.0f} minutes: his {sum(p['his'] for p in points)}, "
          f"other {sum(not p['his'] for p in points)}")


SMOOTH = 3                                    # frames (Gaussian sigma) for positions on the court
PLANT = 0.6                                   # m/s: a foot slower than this is on the ground
TOP = 8.0                                     # m/s: faster than any player runs, a tracking jump
LATERAL = 1.5                                 # m/s: sideways movement, for crossover steps

RALLY = {"mean_speed": ("m/s", "average speed of his feet on the court from the return to the last shot"),
         "top_speed": ("m/s", "95th percentile of that speed within the point"),
         "metres_per_shot": ("m", "distance his feet travel on the court during the rally, per shot of his"),
         "sideways": ("share", "share of that distance that is sideways (along the baseline)"),
         "depth": ("m", "average distance behind the baseline during the rally")}
TO_BALL = {"run": ("m", "distance his feet travel from the opponent's shot to his contact"),
           "peak_speed": ("m/s", "top speed of his feet on the way to the ball"),
           "steps": ("count", "foot plants from the opponent's shot to his contact"),
           "adjust": ("count", "foot plants in the last 0.6 s before contact"),
           "step_length": ("m", "distance per foot plant on the way to the ball")}
RECOVER = {"off_centre": ("m", "his distance from the centre mark, sideways, when the opponent hits back"),
           "ready_depth": ("m", "his distance behind the baseline when the opponent hits back"),
           "recover_speed": ("m/s", "his average speed from his shot to the opponent's"),
           "crossover": ("share", "share of fast sideways movement after his shot with his feet crossed")}


def split(z: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    off = np.r_[0, np.cumsum(z["lengths"])]
    return [(z["pose"][a:b], z["feet"][a:b]) for a, b in zip(off[:-1], off[1:])]


def kinematics(feet: np.ndarray) -> dict:
    """Smoothed positions on the court (metres behind the baseline, right of centre) of his feet and of each
    ankle, velocities and speeds; feet: (T, 4, 2) l/r ankle, l/r heel."""
    from scipy.ndimage import gaussian_filter1d

    f = feet.astype(float)
    for q in range(f.shape[1]):
        for c in range(2):
            ok = np.isfinite(f[:, q, c])
            if ok.sum() >= 2:
                f[:, q, c] = np.interp(np.arange(len(f)), np.flatnonzero(ok), f[ok, q, c])
    ank = gaussian_filter1d(f[:, :2], SMOOTH, axis=0)
    pos = ank.mean(1)
    vel = np.gradient(pos, axis=0) * HZ
    fv = np.linalg.norm(np.gradient(gaussian_filter1d(f[:, :2], 1, axis=0), axis=0), axis=-1) * HZ   # per ankle
    return {"pos": pos, "vel": vel, "speed": np.linalg.norm(vel, axis=1), "ankles": ank, "foot_speed": fv}


def plants(foot_speed: np.ndarray, a: int, b: int) -> int:
    """Foot plants in frames [a, b): either foot coming to rest."""
    still = foot_speed[a:b] < PLANT
    return int(sum(((~s[:-1]) & s[1:]).sum() for s in still.T))


def measures(points: list[dict], tracks: list) -> dict:
    """The three tables: rows of (point index, values)."""
    out = {"rally": [], "to_ball": [], "recover": []}
    for i, (p, (pose, feet)) in enumerate(zip(points, tracks)):
        if not np.isfinite(feet[:, :2]).any(axis=(1, 2)).mean() > 0.9:
            continue
        k = kinematics(feet)
        T = len(feet)
        fr = lambda t: min(T - 1, max(0, int(round(t * HZ))))                    # noqa: E731
        hs = sorted(p["hits"], key=lambda h: h["t"])
        mine = [h for h in hs if h["side"] == "near" and h["shot"] == "stroke"]
        if len(hs) >= 4:                                     # a rally: serve, return and more
            a, b = fr(hs[1]["t"]), fr(hs[-1]["t"])
            step = np.linalg.norm(np.diff(k["pos"][a:b + 1], axis=0), axis=1)
            side = np.abs(np.diff(k["pos"][a:b + 1, 1]))
            n_mine = max(1, sum(h["side"] == "near" for h in hs[1:]))
            out["rally"].append((i, [k["speed"][a:b].mean(), np.percentile(k["speed"][a:b], 95),
                                     step.sum() / n_mine, side.sum() / max(step.sum(), 1e-6),
                                     k["pos"][a:b, 0].mean()]))
        for h in mine:
            j = hs.index(h)
            prev = hs[j - 1] if j and hs[j - 1]["side"] == "far" else None
            nxt = hs[j + 1] if j + 1 < len(hs) and hs[j + 1]["side"] == "far" else None
            c = fr(h["t"])
            if prev and h["t"] - prev["t"] < 3:
                a = fr(prev["t"])
                run = np.linalg.norm(np.diff(k["pos"][a:c + 1], axis=0), axis=1).sum()
                n = plants(k["foot_speed"], a, c + 1)
                out["to_ball"].append((i, [run, k["speed"][a:c + 1].max(), n,
                                           plants(k["foot_speed"], max(a, c - fr(0.6)), c + 1), run / max(n, 1)]))
            if nxt and nxt["t"] - h["t"] < 3:
                e = fr(nxt["t"])
                lat = k["vel"][c:e, 1]
                fast = np.abs(lat) > LATERAL
                crossed = k["ankles"][c:e, 0, 1] > k["ankles"][c:e, 1, 1]          # left ankle right of right
                out["recover"].append((i, [abs(k["pos"][e, 1]), k["pos"][e, 0], k["speed"][c:e].mean(),
                                           crossed[fast].mean() if fast.sum() >= 3 else np.nan]))
    return out


def analyse(project: str, seed: int = 0) -> dict:
    from dsa.measure.technique import ERA_NAMES, era, robust

    rng = np.random.default_rng(seed)
    out_dir = ROOT_PROJECTS / project / "movement"
    z = dict(np.load(out_dir / "tracks.npz"))
    points = json.loads((out_dir / "points.json").read_text())
    tracks = split(z)
    res = {"eras": ERA_NAMES, "points": {"his": sum(p["his"] for p in points),
                                         "other": sum(not p["his"] for p in points)}, "tables": {}}

    def compare(rows, names):
        idx = np.array([i for i, _ in rows])
        f = np.array([v for _, v in rows], float)
        f = np.where(np.isfinite(f), f, np.nanmedian(f, 0))
        meta = [points[i] for i in idx]
        his = np.array([m["his"] for m in meta])
        cam = np.array([[m["zoom"] or np.nan, np.nanmedian(z["height"][np.r_[0, np.cumsum(z["lengths"])][i]:
                                                                       np.r_[0, np.cumsum(z["lengths"])][i + 1]])]
                        for m, i in zip(meta, idx)], float)
        cam = np.where(np.isfinite(cam), cam, np.nanmedian(cam, 0))
        rb = robust(f, np.zeros((len(f), 1)), cam, his, np.array([era(m["year"]) for m in meta]),
                    np.array([m["video"] for m in meta]), ["?"] * len(f), np.array([m["year"] for m in meta]), rng)
        return {"names": names, "n": {"his": int(his.sum()), "other": int((~his).sum())}, "robust": rb,
                "rows": {"index": idx.tolist(), "values": np.round(f, 4).tolist()}}

    m = measures(points, tracks)
    for name, spec in (("rally", RALLY), ("to_ball", TO_BALL), ("recover", RECOVER)):
        res["tables"][name] = compare(m[name], spec)
        print(f"{name}: {len(m[name])} rows", flush=True)

    (out_dir / "analysis.json").write_text(json.dumps(res))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["extract", "analyse"])
    ap.add_argument("project")
    a = ap.parse_args()
    extract(a.project) if a.step == "extract" else analyse(a.project)
