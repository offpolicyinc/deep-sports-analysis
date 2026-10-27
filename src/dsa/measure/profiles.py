"""How each player's strokes differ from the other players' (a project without one project player, such as
atp_hardcourt): the measures of dsa.measure.technique, each player against everyone else in the project.

Per kind (forehand, backhand, serve): the routine strokes of each player (dsa.measure.technique.prepare, grouped by
player); the measured camera, the event (AO / USO) and the situation regressed out, their effects estimated within each player so that
a player's style isn't taken for camera; then each player's match gives one number, his mean. For each player and
measure, his matches against every other player's matches by Welch's t-test, Holm over the measures within the
player. A finding: Holm-adjusted p below ALPHA and a difference of at least MIN_SD standard deviations of the
match means. Lengths are in torso lengths (zoom and the player's size cancel).

How they move (dsa.measure.movement's rally, run-to-the-ball and recovery measures) is compared the same way.

    python -m dsa.measure.profiles atp_hardcourt    -> projects/<project>/profiles/<kind>.json, movement.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from dsa.ingest.projects import PROJECTS, match
from dsa.measure.technique import KINDS as WINDOWS, MEASURES, WIN, NoStrokes, camera, prepare
from dsa.paths import PROJECTS as ROOT_PROJECTS

KINDS = ["forehand", "backhand", "serve"]
ALPHA, MIN_SD = 0.05, 0.4
MIN_STROKES = 2                     # a player's match counts with at least this many routine strokes


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)                                  # the k-th smallest p times (tests - k)
    out = np.empty_like(p)
    out[order] = np.minimum(1, np.maximum.accumulate(p[order] * (len(p) - np.arange(len(p)))))
    return out


def adjusted(f: np.ndarray, ctx: np.ndarray, cam: np.ndarray, player: np.ndarray, dirs: list) -> np.ndarray:
    """f with the camera and the situation (court position, running speed, direction) regressed out, their
    effects estimated within each player."""
    levels = sorted(set(dirs))
    nuis = np.column_stack([ctx, cam] + [np.array([d == lv for d in dirs], float) for lv in levels[1:]])
    nuis = nuis - nuis.mean(0)
    cells = np.column_stack([player == p for p in dict.fromkeys(player)]).astype(float)
    beta = np.linalg.lstsq(np.column_stack([cells, nuis]), f, rcond=None)[0]
    return f - nuis @ beta[cells.shape[1]:]


COURT = ["stance_m", "stance_side_m", "stance_fwd_m", "into_court"]      # metres on the court: made body-relative too


def torso_m(z: dict, keep: np.ndarray, lo: int, hi: int, meta: list, yscale: float = 1.0) -> np.ndarray:
    """Per stroke, his torso (mid-shoulder to mid-hip) in metres as the camera sees it: the torso in pixels (heights
    scaled by yscale, as prepare's) over the court's pixels per metre across the court where he stands."""
    px = z["pose"][keep][:, lo:hi + 1, :, :2] * [1, yscale]
    mid = lambda a, b: (px[:, :, a] + px[:, :, b]) / 2                                   # noqa: E731
    torso = np.nanmedian(np.linalg.norm(mid(1, 2) - mid(7, 8), axis=-1), 1)
    return torso / np.array([(m["camera"] or {}).get("ppm", np.nan) for m in meta])


def body_relative(f: np.ndarray, names: list[str], z: dict, keep: np.ndarray, lo: int, hi: int, meta: list,
                  yscale: float = 1.0) -> np.ndarray:
    """Every length in his own torso lengths (mid-shoulder to mid-hip), so a tall player's long reach isn't taken for
    style: the pose's lengths already are; the court's (metres) are divided by his torso in metres."""
    m = torso_m(z, keep, lo, hi, meta, yscale)
    m = np.where(np.isfinite(m), m, np.nanmedian(m))
    g = f.copy()
    for j, n in enumerate(names):
        if n in COURT:
            g[:, j] = f[:, j] / m
    return g


def camera_share(f: np.ndarray, ctx: np.ndarray, cam: np.ndarray, player: np.ndarray) -> np.ndarray:
    """Per measure: the share of a player's stroke-to-stroke variation the camera explains beyond the situation
    (each column taken around its player's mean, so it is the same player under different cameras)."""
    def within(x):
        x = x.copy()
        for p in dict.fromkeys(player):
            x[player == p] -= x[player == p].mean(0)
        return x
    fd, xd, cd = within(f), within(ctx), within(cam)
    rss = lambda X: ((fd - X @ np.linalg.lstsq(X, fd, rcond=None)[0]) ** 2).sum(0)          # noqa: E731
    return 1 - rss(np.column_stack([xd, cd])) / rss(xd)


def event(video: np.ndarray) -> np.ndarray:
    """1 for a US Open match, 0 for the Australian Open: their broadcasts differ (camera height, court colour),
    taken out with the camera."""
    return np.array(["_uso_" in v for v in video], float)


def compare(res: np.ndarray, player: np.ndarray, video: np.ndarray) -> dict:
    """Per player: the mean of his match means, the difference from the other players' matches, Welch's t, df,
    p and Holm-adjusted p per measure; and the SD of all match means (the scale for MIN_SD)."""
    from scipy.stats import ttest_ind

    units = [(p, v) for p, v in dict.fromkeys(zip(player, video))
             if ((player == p) & (video == v)).sum() >= MIN_STROKES]
    means = np.stack([res[(player == p) & (video == v)].mean(0) for p, v in units])
    who = np.array([p for p, _ in units])
    out = {}
    for p in dict.fromkeys(who):
        mine, rest = means[who == p], means[who != p]
        if len(mine) < 2:
            continue
        t = ttest_ind(mine, rest, equal_var=False)
        pv = np.asarray(t.pvalue)
        out[str(p)] = {"matches": int(len(mine)), "mean": mine.mean(0).round(4).tolist(),
                  "diff": (mine.mean(0) - rest.mean(0)).round(4).tolist(), "t": np.round(t.statistic, 3).tolist(),
                  "df": np.round(t.df, 1).tolist(), "p": pv.tolist(), "holm": holm(pv).tolist()}
    return {"players": out, "sd": means.std(0).round(4).tolist(), "pool": means.mean(0).round(4).tolist(),
            "units": len(units), "match_means": {"player": [str(p) for p, _ in units], "match": [str(v) for _, v in units],
                                                 "mean": means.round(4).tolist()}}


def box(values: np.ndarray, player: np.ndarray) -> dict:
    """Per player and measure, over all his strokes (rows): 5th, 25th, 50th, 75th and 95th percentiles."""
    return {str(p): np.nanpercentile(values[player == p], [5, 25, 50, 75, 95], axis=0).T.round(4).tolist()
            for p in dict.fromkeys(player)}


def analyse(project: str, kind: str) -> dict:
    names, lengths = list(MEASURES[kind][0]), MEASURES[kind][1]
    d = prepare(project, kind, lambda m: m["player"])
    mine = np.array([m["player"] in PROJECTS[project]["players"] for m in d["meta"]])  # not the opponents
    for k in ("keep", "w", "f", "ctx", "score"):
        d[k] = d[k][mine]
    d["meta"] = [m for m, k in zip(d["meta"], mine) if k]
    f, ctx, meta, z, lo, hi = d["f"], d["ctx"], d["meta"], d["z"], d["lo"], d["hi"]
    player = np.array([m["player"] for m in meta])
    video = np.array([match(m["video"]) for m in meta])          # one unit per player and match
    f = body_relative(np.where(np.isfinite(f), f, np.nanmedian(f, 0)), names, z, d["keep"], lo, hi, meta)
    ctx = np.where(np.isfinite(ctx), ctx, np.nanmedian(ctx, 0))
    cam = np.column_stack([camera(meta, d["score"], lo, hi), event(video)])
    dirs = [m["direction"] or "?" for m in meta]
    out = {"kind": kind, "features": names, "lengths": lengths, "kept": int(len(f)), "extracted": d["n"],
           "standard": d["res_std"], "alpha": ALPHA, "min_sd": MIN_SD,
           "strokes": {p: int((player == p).sum()) for p in dict.fromkeys(player)},
           "test": compare(adjusted(f, ctx, cam, player, dirs), player, video),
           "camera_share": camera_share(f, ctx, cam, player).round(4).tolist()}
    findings(out["test"], names)
    # for the page's figures: each player's average skeleton (torso units, y up, mirrored to a right-hander's
    # forehand), and per stroke its measures as shown (lengths in torso lengths, camera and situation out)
    out["win"] = list(d["win"])
    out["mean_pose"] = {str(p): np.nanmean(d["w"][player == p], 0).round(4).tolist() for p in dict.fromkeys(player)}
    shown = adjusted(f, ctx, cam, player, dirs)
    quality = np.nanmean(d["score"][:, lo:hi + 1], (1, 2))
    out["box"] = box(shown, player)
    np.savez_compressed(ROOT_PROJECTS / project / "profiles" / f"{kind}_strokes.npz", hit=d["keep"], f=shown,
                        ctx=ctx, quality=quality, player=player)
    return out


def findings(t: dict, names: list[str]) -> None:
    """Each player's findings (compare's output, in place): Holm below ALPHA and at least MIN_SD apart."""
    for r in t["players"].values():
        r["findings"] = [names[j] for j in range(len(names))
                         if r["holm"][j] < ALPHA and abs(r["diff"][j]) >= MIN_SD * t["sd"][j]]


def movement(project: str) -> dict:
    """The same for how they move (dsa.measure.movement's measures: per rally, per run to the ball, per recovery), the
    camera's zoom and the player's size on screen regressed out."""
    from dsa.measure.movement import RALLY, RECOVER, TO_BALL, measures, split

    d = ROOT_PROJECTS / project / "movement"
    z = dict(np.load(d / "tracks.npz"))
    points = json.loads((d / "points.json").read_text())
    off, m = np.r_[0, np.cumsum(z["lengths"])], measures(points, split(z))
    out = {}
    for table, spec in (("rally", RALLY), ("to_ball", TO_BALL), ("recover", RECOVER)):
        rows = [(i, v) for i, v in m[table] if points[i]["player"] in PROJECTS[project]["players"]]
        idx = np.array([i for i, _ in rows])
        f = np.array([v for _, v in rows], float)
        f = np.where(np.isfinite(f), f, np.nanmedian(f, 0))
        meta = [points[i] for i in idx]
        cam = np.array([[p["zoom"] or np.nan, np.nanmedian(z["height"][off[i]:off[i + 1]])] for p, i in zip(meta, idx)])
        cam = np.where(np.isfinite(cam), cam, np.nanmedian(cam, 0))
        cam = np.column_stack([cam, event(np.array([p["video"] for p in meta]))])
        player = np.array([p["player"] for p in meta])
        res = adjusted(f, np.zeros((len(f), 1)), cam, player, ["?"] * len(f))
        t = compare(res, player, np.array([match(p["video"]) for p in meta]))
        findings(t, list(spec))
        t["box"] = box(res, player)
        out[table] = {"features": list(spec), "units": {k: u for k, (u, _) in spec.items()},
                      "about": {k: a for k, (_, a) in spec.items()}, "rows": int(len(f)), "test": t}
    return out


def stature(project: str, kind: str, yscale: float = 1.0) -> dict[str, float]:
    """Per player with a known height ("heights"), his median torso as the camera sees it over his height; none
    without a stroke of the kind seen well."""
    try:
        d = prepare(project, kind, lambda m: m["player"], yscale)
    except NoStrokes:
        return {}
    m = torso_m(d["z"], d["keep"], d["lo"], d["hi"], d["meta"], yscale)
    player = np.array([x["player"] for x in d["meta"]])
    heights = PROJECTS[project].get("heights", {})
    return {p: float(np.nanmedian(m[player == p]) / heights[p]) for p in dict.fromkeys(player) if p in heights}


def foreshortening(project: str, kind: str) -> dict | None:
    """How much longer a body looks, for its height, on this project's camera than on compare_with's: the median of
    its players' torso-to-height ratios over the median of the other project's (each player's own median first). A
    phone behind the baseline sits lower than a TV camera and sees the body less foreshortened, so its torsos look
    longer: a length out to the side, in torso lengths, comes out short, and an angle from upright small.
    k: what the skeletons' heights are divided by so the medians match. A leaning torso shortens by less than the
    heights do, so k is found by a few rounds of rescaling, not taken from the ratio itself. None when none of
    its players' strokes of this kind is routine (a short video, or one filmed mostly from the far end)."""
    ours = stature(project, kind)
    if not ours:
        return None
    theirs = stature(PROJECTS[project]["compare_with"], kind)
    target, k = np.median(list(theirs.values())), np.median(list(ours.values())) / np.median(list(theirs.values()))
    for _ in range(4):
        k *= np.median(list(stature(project, kind, 1 / k).values())) / target
    return {"k": float(k), "ratio": float(np.median(list(ours.values())) / target),
            "ours": {p: round(v, 4) for p, v in ours.items()}, "theirs": {p: round(v, 4) for p, v in theirs.items()}}


def guests(project: str, kind: str) -> dict:
    """A project measured against another's players ("compare_with": our own phone sessions against
    atp_hardcourt): its players' routine strokes as boxes in the same units as analyse's (lengths in torso lengths).
    The pros' camera effects, estimated within TV broadcasts, don't carry to a phone (twice the zoom, half the
    angle), so they aren't regressed out; instead the skeletons' heights are scaled down by the measured
    foreshortening, so the body looks as foreshortened as on TV. A camera this low sees less of anything pointing
    toward the net, which no scaling of the picture fixes."""
    names, lengths = list(MEASURES[kind][0]), MEASURES[kind][1]
    fs = foreshortening(project, kind)
    if fs is None:
        return nothing(project, kind)
    d = prepare(project, kind, lambda m: m["player"], 1 / fs["k"])
    f, meta, z, lo, hi = d["f"], d["meta"], d["z"], d["lo"], d["hi"]
    player = np.array([m["player"] for m in meta])
    g = body_relative(f, names, z, d["keep"], lo, hi, meta, 1 / fs["k"])
    quality = np.nanmean(d["score"][:, lo:hi + 1], (1, 2))
    (ROOT_PROJECTS / project / "profiles").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ROOT_PROJECTS / project / "profiles" / f"{kind}_strokes.npz", hit=d["keep"], f=g,
                        ctx=d["ctx"], quality=quality, player=player)
    cam = {p: {k: round(float(np.nanmedian([(m["camera"] or {}).get(k, np.nan) for m, q in zip(meta, player) if q == p])), 3)
               for k in ("zoom", "angle", "tilt")} | {"height_px": round(float(np.nanmedian(
                   [m["height"] or np.nan for m, q in zip(meta, player) if q == p])))} for p in dict.fromkeys(player)}
    return {"kind": kind, "features": names, "lengths": lengths, "kept": int(len(f)), "extracted": d["n"],
            "strokes": {p: int((player == p).sum()) for p in dict.fromkeys(player)},
            "videos": {p: sorted({m["video"] for m, q in zip(meta, player) if q == p}) for p in dict.fromkeys(player)},
            "camera": cam, "foreshortening": fs, "after": stature(project, kind, 1 / fs["k"]),
            "box": box(np.where(np.isfinite(g), g, np.nan), player),
            "median": {p: np.nanmedian(g[player == p], 0).round(4).tolist() for p in dict.fromkeys(player)},
            "win": list(d["win"]),
            "mean_pose": {str(p): np.nanmean(d["w"][player == p], 0).round(4).tolist() for p in dict.fromkeys(player)}}


def nothing(project: str, kind: str) -> dict:
    """guests' result for a kind none of the project's players has a routine stroke of: no strokes, same shape."""
    names, lengths = list(MEASURES[kind][0]), MEASURES[kind][1]
    hits = json.loads((ROOT_PROJECTS / project / "technique" / kind / "hits.json").read_text())
    (ROOT_PROJECTS / project / "profiles").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ROOT_PROJECTS / project / "profiles" / f"{kind}_strokes.npz", hit=np.zeros(0, int),
                        f=np.zeros((0, len(names))), ctx=np.zeros((0, 3)), quality=np.zeros(0),
                        player=np.zeros(0, str))
    return {"kind": kind, "features": names, "lengths": lengths, "kept": 0, "extracted": len(hits), "strokes": {},
            "videos": {}, "camera": {}, "foreshortening": None, "after": {}, "box": {}, "median": {},
            "win": list(WINDOWS[kind].get("win", WIN)), "mean_pose": {}}


def run(project: str) -> None:
    out_dir = ROOT_PROJECTS / project / "profiles"
    out_dir.mkdir(parents=True, exist_ok=True)
    if PROJECTS[project].get("compare_with"):
        for kind in KINDS:
            r = guests(project, kind)
            (out_dir / f"{kind}.json").write_text(json.dumps(r))
            print(f"{kind}: {r['strokes']}", flush=True)
        return
    for kind in KINDS:
        r = analyse(project, kind)
        (out_dir / f"{kind}.json").write_text(json.dumps(r))
        print(f"{kind}: {r['kept']} routine strokes of {r['extracted']}, {r['test']['units']} player-matches",
              flush=True)
        for p, x in r["test"]["players"].items():
            print(f"  {p:20s} {x['matches']} matches  {', '.join(x['findings']) or '-'}", flush=True)
    if (ROOT_PROJECTS / project / "movement" / "tracks.npz").exists():
        r = movement(project)
        (out_dir / "movement.json").write_text(json.dumps(r))
        for table, x in r.items():
            print(f"{table}: {x['rows']} rows, {x['test']['units']} player-matches", flush=True)
            for p, y in x["test"]["players"].items():
                print(f"  {p:20s} {y['matches']} matches  {', '.join(y['findings']) or '-'}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("project")
    run(ap.parse_args().project)
