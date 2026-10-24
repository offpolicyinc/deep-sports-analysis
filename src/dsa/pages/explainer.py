"""The heatmap's companion, level with the stroke being shown: the most average clip and a player's typical clip,
both held at the moment a measure is taken, his average skeleton (blue) over everyone's (grey) with the measure drawn
on it, and every player's strokes on that measure as boxes.

On its own it steps through the heatmap's rows top to bottom, showing whoever stands out most on each; pointing at a
square shows that player and measure, clicking holds it (outlined) until clicked again.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np

from dsa.paths import PROJECTS as ROOT_PROJECTS

KINDS = ["serve", "forehand", "backhand"]
WORDS = {"contact_height": ("higher", "lower"), "finish_height": ("higher", "lower"), "reach": ("higher", "lower"),
         "toss_arm": ("higher", "lower"), "swing_time": ("longer", "shorter"), "trophy_time": ("longer", "shorter"),
         "takeback": ("further out", "closer in"), "contact_side": ("further out", "closer in"),
         "finish_side": ("further out", "further across"), "stance_side_m": ("wider", "narrower"),
         "stance_angle": ("more open", "more closed"),
         "lean": ("further right", "further left"),
         "toss_straight": ("straighter", "more bent"), "trunk_lean": ("more tilted", "more upright"),
         "hips_over": ("further to the hitting side", "further to the toss side"), "finish_arm": ("higher", "lower"), "toss_tilt": ("further in", "further out"),
         "trophy_tilt": ("more tilted", "less tilted"), "shoulder_width": ("more", "less", ""), "finish_open": ("more", "less", "")}   # (higher, lower, verb)
# who the finding sentence is about, where the row's name reads badly there ("Peter's trunk is 5° more tilted")
SUBJ = {"toss_straight": ("toss arm", "is"), "toss_tilt": ("toss arm", "leans"), "trophy_tilt": ("shoulders at the trophy", "are"),
        "trunk_lean": ("trunk", "is"), "hips_over": ("hips", "are"), "finish_arm": ("finish arm", "is"),
        "finish_side": ("finish", "is"), "contact_side": ("contact", "is")}
# a plain sentence per measure, read under the figure
SAY = {
    "contact_side": "How far out to the side of his hips he meets the ball.",
    "elbow_angle": "How straight the hitting arm is at contact (180° is fully straight).",
    "shoulder_width": "How far his shoulders have turned toward the net at contact: 0° side-on, 90° square to the net, "
                      "over 90° already past square.",
    "shoulder_coil": "How far he turns his shoulders away from the net before the forward swing: 90° is side-on, more "
                     "is turned past side-on.",
    "hip_coil": "How far he turns his hips away from the net before the forward swing: 90° is side-on.",
    "takeback_turn": "How far his shoulders are turned away from the net when the forward swing starts: 90° is side-on, "
                     "more is turned past side-on.",
    "finish_open": "How far his shoulders have turned toward the net 0.3 s after contact: 0° side-on, 90° square to the "
                   "net, over 90° past square.",
    "separation": "How much further the shoulders turn than the hips, at the shoulders' most-turned moment.",
    "takeback": "Where the hitting wrist is, out to the side, at the end of the take-back: the moment it stops going back "
                "and the forward swing starts.",
    "swing_time": "Time from the start of the forward swing (the end of the take-back) to contact.",
    "wrist_speed": "The hitting wrist's top speed through contact: wider gaps between the dots mean faster.",
    "finish_height": "How high the wrist finishes, above the shoulders, 0.3 s after contact.",
    "finish_side": "Where the wrist finishes 0.3 s after contact: out to the side, or wrapped across the body.",
    "head_still": "How much the head moves relative to the hips around contact.",
    "contact_height": "How high above the hips he meets the ball.",
    "shoulder_tilt": "How far the shoulder line tips from level at contact.",
    "knee_bend": "How much the knees bend before contact (smaller angle, more bend).",
    "stance_m": "How far apart the feet are at contact, seen from above.",
    "stance_side_m": "How far apart the feet are along the baseline at contact, seen from above.",
    "stance_fwd_m": "How far the front foot is ahead toward the net at contact (0: open stance).",
    "stance_angle": "The angle of the line from back foot to front foot against the baseline: 90° is the front foot "
                    "straight ahead of the back one, less when it steps across the body (closed), more when it steps out to its "
                    "own side (open).",
    "reach": "How high the hitting wrist is above the feet at contact.",
    "toss_arm": "How high the tossing arm reaches above the head.",
    "trunk_lean": "How far the trunk tilts from upright at contact, seen from behind, toward the tossing side "
                  "(as all of them do).",
    "hips_over": "How far the hips are out to the side of the feet at contact (positive: toward the hitting side).",
    "finish_arm": "Where the hitting arm points 0.3 s after contact: 0° straight up, -90° across the body at shoulder "
                  "height, beyond that lower (a wrap around the waist).",
    "toss_straight": "How straight the tossing arm is at the top of the toss (180° is fully straight).",
    "toss_tilt": "How far the tossing arm leans from upright at the top of the toss: positive in over the head, "
                 "negative out to the side.",
    "trophy_time": "Time from the tossing arm's highest point to contact.",
    "trophy_tilt": "How far the shoulder line tips (tossing shoulder up) when the tossing arm is highest.",
    "lean": "How far the shoulders are out to the side of the feet at contact.",
    "into_court": "How far his feet travel toward the net through the serve, seen from above.",
}


def picks(project: str, players: list[str]) -> dict[str, dict]:
    """Per stroke kind, the strokes the video shows (indices into its strokes npz): the most average one (nearest the
    median of the players' medians on every measure shown, each in SDs of a player's own strokes; clearly seen, not
    on the run; serves from the deuce side) and each player's EXAMPLES most typical ones (nearest his own medians,
    most typical first)."""
    from dsa.pages.profiles_report import UNSURE, kept, load, surname, strokes

    res, _ = load(project)
    out = {}
    for k in KINDS:
        s, _, _ = strokes(project, k)
        cols = [res[k]["features"].index(n) for n in kept(project, k)]
        f, pl, q = s["f"][:, cols], s["player"], s["quality"]
        within = np.sqrt(np.nanmean(np.stack([np.nanvar(f[pl == p], 0) for p in players]), 0))
        meds = {p: np.nanmedian(f[pl == p], 0) for p in players}
        calm = s["ctx"][:, 2] <= np.nanquantile(s["ctx"][:, 2], 0.6) if k != "serve" else deuce(project, s)
        def nearest(sel, target, n=1):
            idx = np.flatnonzero(sel)
            order = (int(j) for j in idx[np.argsort(np.nansum(((f[idx] - target) / within) ** 2, 1))])
            near = [j for j in order if lit(project, k, j)][:n] if n > 1 else [next(j for j in order if lit(project, k, j))]
            return near if n > 1 else near[0]
        ok = np.isin(pl, [p for p in players if p not in UNSURE]) & (q >= np.nanmedian(q)) & calm
        out[k] = {"avg": nearest(ok, np.median(np.stack(list(meds.values())), 0)),
                  "players": {surname(p): nearest((pl == p) & (q >= np.nanmedian(q[pl == p])) & calm, meds[p], EXAMPLES)
                              for p in players if p not in UNSURE}}
    return out


def guest_picks(project: str, players: list[str], extra: dict) -> dict[str, dict[str, int]]:
    """Per kind, each of our own players' EXAMPLES most typical strokes (indices into his project's strokes npz): of the
    SIX nearest his medians on the measures shown for phone footage (serves: from the deuce side) (in SDs of a pro's own
    strokes), the ones he stands stillest for (his feet travel least over the clip), stillest first: standard strokes,
    not ones on the move."""
    from dsa.ingest.projects import PROJECTS
    from dsa.pages.profiles_report import kept, load, strokes

    out = {k: {} for k in KINDS}
    for k in KINDS:
        ps, _, _ = strokes(project, k)
        for q, (g, gres) in extra.items():
            gs, meta, _ = strokes(g, k)
            names = [n for n in kept(project, k, phone=True) if n in gres[k]["features"]]
            pf = load(project)[0][k]["features"]
            pc, gc = [pf.index(n) for n in names], [gres[k]["features"].index(n) for n in names]
            within = np.sqrt(np.nanmean(np.stack([np.nanvar(ps["f"][ps["player"] == p][:, pc], 0) for p in players]), 0))
            hidden = set(PROJECTS[g].get("no_clips", []))
            fed = {tuple(x) for x in PROJECTS[g].get("fed", [])}
            mine = np.flatnonzero((gs["player"] == q)
                                  & np.array([meta[int(h)]["video"] not in hidden
                                              and (meta[int(h)]["video"], meta[int(h)]["frame"]) not in fed
                                              for h in gs["hit"]], bool))
            if k == "serve":
                mine = mine[deuce(g, gs)[mine]]
            if not len(mine):
                continue
            f = gs["f"][mine][:, gc]
            near = mine[np.argsort(np.nansum(((f - np.nanmedian(f, 0)) / within) ** 2, 1))][:SIX]
            _, _, pose = strokes(g, k)
            out[k][q] = [int(near[j]) for j in np.argsort([travel(pose[int(gs["hit"][i])]) for i in near])[:EXAMPLES]]
    return out


SIX = 6
EXAMPLES = 5    # clips per player and stroke, to flip through
LIT = 80        # the court round a clip's player at least this bright (0-255): not in a stand's shade, where a player
                # turns near black (the picks' court here: 101 and up; the shaded ones: 38-65)


@lru_cache(maxsize=None)
def lit(project: str, k: str, i: int) -> bool:
    """Stroke i of kind k (an index into its strokes npz) well lit: the court round the player's feet at contact, in
    the frame the clip shows, at least LIT bright. A dark shirt doesn't count against it; a stand's shade does."""
    import cv2
    from dsa.pages.profiles_report import MIRROR, strokes

    s, meta, pose = strokes(project, k)
    h = int(s["hit"][i])
    m, p = meta[h], pose[h].copy()
    if m["lefty"]:                                     # back to the picture: hits.npz mirrors left-handers
        p = p[:, MIRROR]
        p[..., 0] = -p[..., 0]
    q = p[int(np.argmin(abs(np.linspace(-0.9, 0.6, len(p))))), :, :2]
    d = ROOT_PROJECTS / project / m["video"]
    cap = cv2.VideoCapture(str(d / "kept.mp4" if (d / "kept.mp4").exists() else d / "source.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, m["frame"])
    ok, img = cap.read()
    if not ok or not np.isfinite(q).any():
        return True
    tall, feet, cx = np.nanmax(q[:, 1]) - np.nanmin(q[:, 1]), np.nanmax(q[11:13, 1]), np.nanmean(q[:, 0])
    y0, y1, x0, x1 = int(feet - 0.15 * tall), int(feet + 0.25 * tall), int(cx - 0.6 * tall), int(cx + 0.6 * tall)
    return float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[max(0, y0):y1, max(0, x0):x1].mean()) >= LIT


def deuce(project: str, s: dict) -> np.ndarray:
    """Per serve in s (a strokes npz): served from the deuce side, the feet right of the centre mark seen from
    behind (hits.npz mirrors left-handers' feet: back to the court), so every serve clip is seen the same way."""
    z = np.load(ROOT_PROJECTS / project / "technique" / "serve" / "hits.npz")
    meta = json.loads((ROOT_PROJECTS / project / "technique" / "serve" / "hits.json").read_text())
    x = z["feet"][:, int(np.argmin(abs(z["grid"]))), 1] * np.where([m["lefty"] for m in meta], -1, 1)
    return x[s["hit"].astype(int)] > 0


def travel(pose: np.ndarray) -> float:
    """How far the feet travel from 0.6 s before contact to 0.4 s after, in his heights (pose: image px)."""
    from dsa.pages.exemplars import GRID

    hz = round((len(pose) - 1) / (GRID[-1] - GRID[0]))
    zero = int(round(-GRID[0] * hz))
    p = pose[zero - int(0.6 * hz):zero + int(0.4 * hz) + 1]
    tall = np.nanmedian(np.nanmax(p[:, :, 1], 1) - np.nanmin(p[:, :, 1], 1))
    return float(np.nansum(np.linalg.norm(np.diff(p[:, 11:13, :2].mean(1), axis=0), axis=1)) / tall)


def videos(project: str, players: list[str], out_dir: Path, make: bool, extra: dict | None = None) -> dict[str, dict]:
    """The clips ([file name, label]) per kind: avg, and a list per player (and each of ours, extra), most typical
    first; rendered if make or missing. Each file is named for the stroke it shows."""
    from dsa.pages.exemplars import single
    from dsa.pages.profiles_report import label_of, load, stroke

    res, _ = load(project)
    extra = extra or {}
    chosen, jobs = picks(project, players), []
    mine = guest_picks(project, players, extra)
    out = {}
    for k, c in chosen.items():
        win = res[k]["win"]
        def clip(i, name, src=project):
            st = stroke(src, {"kind": k, "name": "", "j": 0, "unit": ""}, i)
            # the stroke shown is in the name: a new pick (the traits shown changed) is a new file, never an old video
            # under a new label, and nothing cached under an old name goes stale
            path = out_dir / f"ex_{k}_{name}_{st['video']}_{st['frame']}.mp4"
            st.pop("value_text", None)
            st["project"] = src
            if make or not path.exists():
                jobs.append((st, path, win))
            return path.name, st["label"]
        nth = lambda name, j: name if j == 0 else f"{name}_{j + 1}"            # noqa: E731
        out[k] = {"avg": clip(c["avg"], "average"),
                  "players": {q: [clip(i, nth(q.lower(), j)) for j, i in enumerate(ix)] for q, ix in c["players"].items()}}
        for q, ix in mine[k].items():
            out[k]["players"][label_of(q, extra)] = [clip(i, nth(label_of(q, extra).lower().replace(" ", "_"), j), extra[q][0])
                                                      for j, i in enumerate(ix)]
    print(len(jobs), "clips", flush=True)
    with ThreadPoolExecutor(6) as ex:
        list(ex.map(lambda j: single(project, j[0], j[1], *j[2]), jobs))
    return out


def moments(pose: np.ndarray, win: list) -> dict[str, int]:
    """The frames the figure stops at, from the pose's own motion (technique.phases) and its toss."""
    from dsa.measure.technique import phases

    t = np.round(np.linspace(win[0], win[1], len(pose)), 4)
    ph = {k: int(v[0]) for k, v in phases(pose[None], t).items()}
    a, b = int(np.argmin(abs(t + 0.9))), int(np.argmin(abs(t + 0.1)))
    ph["toss"] = a + int(np.nanargmax(pose[a:b + 1, 5, 1]))
    return ph


def panel_data(project: str, players: list[str], order: list[tuple[str, str]], clips: dict | None = None,
               extra: dict | None = None) -> tuple[dict, list[str]]:
    """What drives the panel (poses, measures, feet, clips, which are ours), and the players its menu offers."""
    from dsa.pages.profiles_report import SHORT, UNIT, UNSURE, guest_medians, heat_z, label_of, load, spread, tests, value

    extra = extra or {}
    surname = lambda q: label_of(q, extra)                                   # noqa: E731
    ours = {surname(q) for q in extra}
    res, _ = load(project)
    pros = [q for q in players if q not in extra]
    zs = heat_z(project, res, pros, extra)
    sd = {k: spread(project, k, pros) for k in KINDS}
    units = {(r["kind"], r["name"]): r["unit"] for r in tests(project)}
    kinds = {}
    for k in KINDS:
        r = res[k]
        mp = {surname(p): np.array(r["mean_pose"][p]) for p in pros if p in r["mean_pose"]}
        allp = np.nanmean(np.stack(list(mp.values())), 0)
        rest = {q: np.nanmean(np.stack([v for o, v in mp.items() if o != q]), 0) for q in mp}   # the other ten's
        for q, (_, gres) in extra.items():          # ours, onto the pros' time grid (phone 30 fps), against them all
            if q in gres[k]["mean_pose"]:
                g = np.array(gres[k]["mean_pose"][q], float)
                src, dst = np.linspace(0, 1, len(g)), np.linspace(0, 1, len(allp))
                mp[surname(q)] = np.stack([[np.interp(dst, src, g[:, j, c]) for c in range(g.shape[2])]
                                           for j in range(g.shape[1])]).transpose(2, 0, 1)
                rest[surname(q)] = allp
        kinds[k] = {"win": r["win"], "all": np.round(allp, 3).tolist(),
                    "players": {q: np.round(v, 3).tolist() for q, v in mp.items()},
                    "rest": {q: np.round(v, 3).tolist() for q, v in rest.items()},
                    "mom": {"all": moments(allp, r["win"]), **{q: moments(v, r["win"]) for q, v in mp.items()},
                            **{"rest:" + q: moments(v, r["win"]) for q, v in rest.items()}}}
    stance = {}                     # each player's median feet (sideways, forward; torso lengths), for the feet from above
    for k in KINDS:
        f = res[k]["features"]
        if "stance_side_m" in f:
            i, j = f.index("stance_side_m"), f.index("stance_fwd_m")
            stance[k] = {surname(q): [res[k]["box"][q][i][2], res[k]["box"][q][j][2]] for q in pros if q in res[k]["box"]}
            side, fwd = guest_medians(extra, k, "stance_side_m"), guest_medians(extra, k, "stance_fwd_m")
            ours_st = {surname(q): [side[q], fwd[q]] for q in extra if q in side and q in fwd}   # none: no NaN
            stance[k]["rest"] = {q: np.median(np.array([v for o, v in stance[k].items() if o != q]), 0).tolist()
                                 for q in stance[k]}
            stance[k]["rest"].update({q: np.median(np.array([v for o, v in stance[k].items() if o != "rest"]), 0).tolist()
                                      for q in ours_st})
            stance[k].update(ours_st)
    measures = []
    for k, n in order:
        r = res[k]
        j = r["features"].index(n)
        bx = {surname(q): [round(v, 4) for v in r["box"][q][j]] for q in pros
              if q in r["box"] and np.isfinite(r["box"][q][j][2])}
        meds = {q: v[2] for q, v in bx.items()}
        pro_med = float(np.median(list(meds.values())))
        u = units.get((k, n), "")
        mine = {surname(q): v for q, v in guest_medians(extra, k, n).items() if q in zs[k, n]}
        measures.append({"k": k, "n": n, "label": f"{k.capitalize()}: {SHORT.get(n, n)}", "say": SAY.get(n, ""),
                         "all": value(float(np.median(list(meds.values()))), u), "med": float(np.median(list(meds.values()))),
                         "vals": {q: value(v, u) for q, v in (meds | mine).items()}, "box": bx, "unit": UNIT.get(u, " " + u),
                         # the other ten's median, per player (all the pros' for ours)
                         "rest": {q: value(float(np.median([v for o, v in meds.items() if o != q])), u) for q in meds}
                                 | {q: value(pro_med, u) for q in mine},
                         "restmed": {q: float(np.median([v for o, v in meds.items() if o != q])) for q in meds}
                                    | {q: pro_med for q in mine},
                         # the finding in words: how far from the other ten, and which way
                         "short": SHORT.get(n, n), "words": WORDS.get(n, ("more", "less")), "subj": SUBJ.get(n),
                         "z": {surname(q): round(float(z), 2) for q, z in zs[k, n].items()},
                         # for comparing with one player (the menu): every median and a player's stroke spread
                         "meds": {q: round(float(v), 4) for q, v in (meds | mine).items()}, "sd": float(sd[k][j]),
                         "diff": {q: value(abs(v - float(np.median([w for o, w in meds.items() if o != q]))), u)
                                  for q, v in meds.items()} | {q: value(abs(v - pro_med), u) for q, v in mine.items()},
                         # who stands out most on it (shown when no one is pointed at)
                         "top": surname(max((q for q in zs[k, n] if q not in UNSURE and q in pros),
                                            key=lambda q: abs(zs[k, n][q])))})
    return ({"kinds": kinds, "measures": measures, "stance": stance, "clips": clips or {}, "ours": sorted(ours)},
            sorted(surname(p) for p in pros) + sorted(ours))


def panel(project: str, players: list[str], order: list[tuple[str, str]], clips: dict | None = None,
          extra: dict | None = None) -> str:
    """The 2 x 2 panel (two clips, the skeleton, the boxes) and the data and code that drive it. players: the pros
    and our own (extra: guests), who are set against all the pros."""
    D, names = panel_data(project, players, order, clips, extra)
    data = json.dumps(D, separators=(",", ":")).replace("</", "<\\/")
    pick = "".join(f'<option value="{q}">{q}</option>' for q in names)
    return (f'<div class="explain"><label class="ex-ref">Compare with <select id="ex-ref"><option value="">Average'
            f'</option>{pick}</select></label><div class="ex-cap"><b id="ex-title"></b>'
            f'<p id="ex-say"></p></div><div class="ex-grid">'
            f'<figure><video id="ex-me" muted playsinline preload="auto"></video><figcaption id="ex-mcap"></figcaption><div class="ex-pick" id="ex-pick"></div></figure>'
            f'<figure><video id="ex-avg" muted playsinline preload="auto"></video><figcaption id="ex-acap"></figcaption></figure>'
            f'<figure><svg id="ex-fig" viewBox="0 0 320 320" role="img"></svg><figcaption id="ex-fcap"></figcaption></figure>'
            f'<figure><svg id="ex-box" viewBox="0 0 320 320" role="img"></svg><figcaption id="ex-bcap"></figcaption></figure>'
            f'</div>'
            f'</div><script type="application/json" id="ex-data">{data}</script><script>{JS}</script>')


CSS = """
  .heatwrap { position: relative; display: flex; gap: 32px; justify-content: center; align-items: flex-start;
              max-width: 1060px; margin: 8px auto 0; padding: 0 16px; }
  .heatwrap .scroll { padding: 0; margin: 0; max-width: 100%; flex: none; }
  .explain { flex: 1; min-width: 0; max-width: 460px; background: var(--bg); transition: margin-top .45s ease; }
  .ex-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px 10px; margin-top: 10px; }
  .ex-grid figure { margin: 0; min-width: 0; }
  .ex-grid figcaption { font-size: 12px; line-height: 1.35; color: var(--muted); padding-top: 4px; min-height: 2.7em; }
  .ex-grid video { width: 100%; aspect-ratio: 1; display: block; background: #181818; border-radius: 8px; object-fit: cover; }
  #ex-fig, #ex-box { width: 100%; aspect-ratio: 1; display: block; background: #fafaf8; border: 1px solid var(--line);
            border-radius: 8px; box-sizing: border-box; }
  #ex-fig .b { stroke-width: 3.2; stroke-linecap: round; }
  #ex-fig .all { stroke: #b9b8b3; } #ex-fig .all.h { fill: #b9b8b3; }
  #ex-fig .me { stroke: #2563eb; } #ex-fig .me.h { fill: #2563eb; }
  #ex-fig .m { stroke: #c2410c; stroke-width: 2.4; fill: none; } #ex-fig .mf { fill: #c2410c; }
  #ex-fig .m.faint { stroke: #e3a582; } #ex-fig .ref { stroke: #c2410c; stroke-dasharray: 3 3; stroke-width: 1.2; }
  #ex-fig .note, #ex-box .note { font-size: 12px; fill: var(--faint); }
  #ex-box .b { stroke-width: 3; stroke-linecap: round; } #ex-box .all { stroke: #b9b8b3; } #ex-box .all.h { fill: #b9b8b3; }
  #ex-box .me { stroke: #2563eb; } #ex-box .me.h { fill: #2563eb; }
  #ex-box .m { stroke: #c2410c; stroke-width: 2.4; fill: none; } #ex-box .mf { fill: #c2410c; }
  #ex-box .ref { stroke: #c2410c; stroke-dasharray: 3 3; stroke-width: 1.2; fill: none; }
  #ex-box .nm { font-size: 13px; font-weight: 600; } #ex-box .val { font-size: 15px; font-weight: 700; }
  #ex-box text.me { fill: #2563eb; stroke: none; } #ex-box text.all { fill: #8a8983; stroke: none; }
  #ex-box .bar { stroke-width: 5; stroke-linecap: round; } #ex-box .bar.me { stroke: #2563eb; } #ex-box .bar.all { stroke: #b9b8b3; }
  #ex-box .dot.me { fill: #2563eb; } #ex-box .dot.all { fill: #b9b8b3; }
  .ex-cap { font-size: 15px; line-height: 1.45; min-height: 96px; }
  .ex-pick { display: flex; gap: 4px; flex-wrap: wrap; margin-top: 2px; }
  .ex-pick button { font: 500 12px Inter, sans-serif; color: var(--muted); background: transparent; border: 1px solid var(--line);
                    border-radius: 999px; min-width: 26px; padding: 2px 8px; cursor: pointer; }
  .ex-pick button[aria-pressed="true"] { color: var(--ink); border-color: var(--ink); }
  .ex-ref { display: block; font-size: 13px; color: var(--muted); margin: 0 0 8px; }
  .ex-ref select { font: inherit; color: var(--ink); margin-left: 6px; padding: 2px 6px; border: 1px solid var(--line);
                   border-radius: 6px; background: var(--bg); }
  svg.heat .tick.ref { fill: var(--ink); text-decoration: underline; }
  .ex-cap b { display: block; font-size: 17px; } .ex-cap span { display: block; color: var(--muted); font-size: 14px; margin-top: 2px; }
  .ex-cap p { margin: 6px 0 0; }
  svg.heat .band { fill: transparent; } svg.heat .hr.on .band { fill: #ecebe6; }
  svg.heat .hr.on .hrow { fill: var(--ink); font-weight: 600; } svg.heat .hr { cursor: pointer; }
  svg.heat .tick.on { fill: var(--ink); font-weight: 700; } svg.heat .cell { cursor: pointer; }
  svg.heat .held { fill: none; stroke: #17171a; stroke-width: 2.5; pointer-events: none; }
  @media (max-width: 960px) {
    .heatwrap { flex-direction: column; align-items: stretch; gap: 12px; padding: 0 12px; }
    .heatwrap .scroll { overflow-x: auto; flex: initial; }
    .explain { position: sticky; order: -1; max-width: none; z-index: 2; padding: 8px 0; border-bottom: 1px solid var(--line);
               margin-top: 0 !important; }
    .ex-grid { grid-template-columns: repeat(4, 1fr); gap: 6px; } .ex-grid figcaption { font-size: 10.5px; }
    .ex-cap { font-size: 13px; min-height: 0; } .ex-cap b { font-size: 14px; } .ex-cap span { font-size: 12.5px; }
  }
  /* a phone: the heatmap shrinks to the screen's width, every column showing, and the panel above it keeps only the two
     clips, so both fit on the screen together */
  @media (max-width: 640px) {
    .heatwrap { padding: 0; gap: 8px; }
    .heatwrap .scroll { overflow-x: visible; }
    .heatwrap svg.heat { max-width: 100%; height: auto; } svg.heat .hrow, svg.heat .tick { font-size: 13px; }
    .ex-pick { flex-wrap: nowrap; } .ex-pick button { min-width: 0; padding: 2px 7px; }
    .ex-grid { grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 6px; }
    .ex-grid figure:nth-child(n+3) { display: none; }
    .ex-grid figcaption { font-size: 11.5px; min-height: 0; }
    .ex-cap p { font-size: 12.5px; margin-top: 2px; }
    .ex-ref { margin-bottom: 4px; }
  }
"""

JS = r"""
(() => {
  const D = JSON.parse(document.getElementById("ex-data").textContent);
  const OTH = w => (D.ours || []).includes(w) ? "Pros" : "Others";     // ours are set against all the pros
  let ref = "";                            // compared with: "" the others' average, else one player (the menu)
  const fmt = (v, u) => (u === "°" || Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(2)) + u;
  // what `who` is set against, on measure m: the others' average, or the chosen player (not himself)
  function R(who, m, K) {
    const C = D.clips[m.k], st = D.stance[m.k] || {};
    if (!ref || ref === who || m.meds[ref] == null) {
      const o = OTH(who);
      return {lab: o, low: o.toLowerCase(), poss: o.toLowerCase() + "'", val: m.rest[who] ?? m.all,
              med: m.restmed[who] ?? m.med, pose: K.rest[who] || K.all, mom: K.mom["rest:" + who] || K.mom.all,
              clip: C?.avg, cap: `Average ${m.k}`, feet: st.rest?.[who], z: m.z[who], diff: m.diff[who]};
    }
    const a = m.meds[who], b = m.meds[ref];
    return {lab: ref, low: ref, poss: ref + "'s", val: m.vals[ref], med: b, pose: K.players[ref], mom: K.mom[ref],
            clip: C?.players[ref]?.[0], cap: `${ref}'s ${m.k}`, feet: st[ref],
            z: a == null ? null : (a - b) / m.sd, diff: a == null ? null : fmt(Math.abs(a - b), m.unit)};
  }
  const $ = id => document.getElementById(id), NS = "http://www.w3.org/2000/svg";
  const fig = $("ex-fig"), bxs = $("ex-box"), heat = document.querySelector("svg.heat");
  const BONES = [[1,2],[1,3],[3,5],[2,4],[4,6],[1,7],[2,8],[7,8],[7,9],[9,11],[8,10],[10,12]];
  const SLOW = 0.3, HOLD = 2.4;            // playback speed (of real time); seconds held on the measure
  const box = {};                          // one scale per stroke, from everyone's poses
  for (const [k, K] of Object.entries(D.kinds)) {
    let lo = [9, 9], hi = [-9, -9];
    for (const P of [K.all, ...Object.values(K.players)]) for (const f of P) for (const j of f) {
      const x = k === "backhand" ? -j[0] : j[0];
      lo = [Math.min(lo[0], x), Math.min(lo[1], j[1])]; hi = [Math.max(hi[0], x), Math.max(hi[1], j[1])];
    }
    const s = Math.min(300 / (hi[0] - lo[0]), 276 / (hi[1] - lo[1]));
    box[k] = {s, cx: 160 - (lo[0] + hi[0]) / 2 * s, cy: 32 + hi[1] * s};
  }
  // i: the row stepped to; held: {i, who} clicked (kept until clicked again); hov: pointed at
  let i = 0, held = null, hov = null, t = 0, hold = 0, last = null;
  const cur = () => hov ?? held ?? {i, who: null};
  const M = () => D.measures[cur().i];
  const WHO = () => cur().who ?? M().top;
  const hz = K => (K.all.length - 1) / (K.win[1] - K.win[0]);
  const fi = (K, s) => Math.max(0, Math.min(K.all.length - 1, Math.round((s - K.win[0]) * hz(K))));
  // per measure: [trail start | null, the frame it is taken], from that pose's own phases (mo)
  function when(n, K, mo) {
    const c = fi(K, 0);
    return ({shoulder_coil: [null, mo.coil], hip_coil: [null, mo.hipcoil], separation: [null, mo.coil],
      takeback: [fi(K, -0.6), mo.fwd], takeback_turn: [null, mo.fwd], finish_open: [null, fi(K, 0.3)], swing_time: [mo.fwd, c], wrist_speed: [fi(K, -0.15), fi(K, 0.1)],
      finish_height: [null, fi(K, 0.3)], finish_arm: [null, fi(K, 0.3)], finish_side: [null, fi(K, 0.3)], head_still: [fi(K, -0.1), fi(K, 0.1)],
      knee_bend: [null, fi(K, -0.15)], toss_arm: [null, mo.toss], toss_straight: [null, mo.toss], toss_tilt: [null, mo.toss], trophy_tilt: [null, mo.toss],
      trophy_time: [mo.toss, c], into_court: [null, fi(K, 0.4)]})[n] || [null, c];
  }
  const mk = (root, tag, at, txt) => { const e = document.createElementNS(NS, tag); for (const k in at) e.setAttribute(k, at[k]); if (txt != null) e.textContent = txt; root.appendChild(e); return e; };
  const lerp = (P, t) => { const a = Math.floor(t), b = Math.min(P.length - 1, a + 1), u = t - a; return P[a].map((j, q) => [j[0] + (P[b][q][0] - j[0]) * u, j[1] + (P[b][q][1] - j[1]) * u]); };
  const stopAt = () => { const m = M(), K = D.kinds[m.k], w = WHO(); return when(m.n, K, K.mom[K.players[w] ? w : "all"])[1]; };
  // one pose with the measure drawn on it (when t has reached the moment), on root, through X / Y
  function pose(root, m, K, P, mo, t, X, Y, fl, cls, strong) {
    const el = (tag, at, txt) => mk(root, tag, at, txt), mid = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
    const [f0, f1] = when(m.n, K, mo), q = lerp(P, Math.min(t, f1));
    for (const [a, b] of BONES) el("line", {x1: X(q[a]), y1: Y(q[a]), x2: X(q[b]), y2: Y(q[b]), class: "b " + cls});
    el("circle", {cx: X(q[0]), cy: Y(q[0]), r: 7, class: "h " + cls});
    const c = "m" + (strong ? "" : " faint");
    const tj = {takeback: 6, swing_time: 6, wrist_speed: 6, head_still: 0, trophy_time: 6}[m.n];   // a path, as it builds
    if (tj != null && f0 != null && t >= f0) {
      const pts = []; for (let f = f0; f <= Math.min(t, f1); f++) pts.push(P[f][tj]);
      el("polyline", {points: pts.map(p => X(p) + "," + Y(p)).join(" "), class: c});
      if (m.n === "wrist_speed") for (const p of pts) el("circle", {cx: X(p), cy: Y(p), r: 2.4, class: "mf"});
    }
    if (t < f1 - 0.01) return;
    const hip = mid(q[7], q[8]), sh = mid(q[1], q[2]), ank = mid(q[11], q[12]);
    const side = (from, to, y) => { el("line", {x1: X(from), x2: X(from), y1: Y([0, from[1] + 2.2]), y2: Y([0, from[1] - 2.6]), class: "ref"}); el("line", {x1: X([from[0], y]), y1: Y([0, y]), x2: X([to[0], y]), y2: Y([0, y]), class: c}); };
    const up = (base, to, x) => { el("line", {x1: X([x - 0.5, base[1]]), x2: X([x + 0.5, base[1]]), y1: Y(base), y2: Y(base), class: "ref"}); el("line", {x1: X([x, base[1]]), x2: X([x, to[1]]), y1: Y(base), y2: Y(to), class: c}); };
    const width = (a, b) => { el("line", {x1: X(q[a]), y1: Y(q[a]), x2: X(q[b]), y2: Y(q[b]), class: c, "stroke-width": 4.5}); };
    const arc = (a, b, c2) => { el("polyline", {points: [q[a], q[b], q[c2]].map(p => X(p) + "," + Y(p)).join(" "), class: c, "stroke-width": 4}); el("circle", {cx: X(q[b]), cy: Y(q[b]), r: 9, class: c}); };
    const tilt = () => { el("line", {x1: X(q[1]), y1: Y(q[1]), x2: X(q[2]), y2: Y(q[2]), class: c, "stroke-width": 4.5}); el("line", {x1: X(q[1]) - 18 * fl, x2: X(q[2]) + 18 * fl, y1: Y(sh), y2: Y(sh), class: "ref"}); };
    const feet = () => el("line", {x1: X(q[11]), y1: Y(q[11]), x2: X(q[12]), y2: Y(q[12]), class: c, "stroke-width": 4.5});
    ({contact_side: () => side(hip, q[6], q[6][1]), takeback: () => side(hip, q[6], q[6][1]), finish_side: () => side(hip, q[6], q[6][1]),
      lean: () => side(ank, sh, sh[1]), contact_height: () => up(hip, q[6], q[6][0]), finish_height: () => up(sh, q[6], q[6][0]),
      reach: () => up(ank, q[6], q[6][0]), toss_arm: () => up(q[0], q[5], q[5][0]),
      toss_straight: () => arc(1, 3, 5),
      trunk_lean: () => { el("line", {x1: X(hip), x2: X(hip), y1: Y(hip), y2: Y([0, hip[1] + 1.6]), class: "ref"});
                          el("line", {x1: X(hip), y1: Y(hip), x2: X(sh), y2: Y(sh), class: c, "stroke-width": 4}); },
      hips_over: () => side(ank, hip, hip[1]),
      finish_arm: () => { el("line", {x1: X(q[2]), x2: X(q[2]), y1: Y(q[2]), y2: Y([0, q[2][1] + 1.4]), class: "ref"});
                          el("line", {x1: X(q[2]), y1: Y(q[2]), x2: X(q[6]), y2: Y(q[6]), class: c, "stroke-width": 4}); },
      toss_tilt: () => { el("line", {x1: X(q[1]), x2: X(q[1]), y1: Y(q[1]), y2: Y([0, q[1][1] + 1.6]), class: "ref"});
                         el("line", {x1: X(q[1]), y1: Y(q[1]), x2: X(q[5]), y2: Y(q[5]), class: c, "stroke-width": 4}); },
      shoulder_width: () => width(1, 2), takeback_turn: () => width(1, 2), finish_open: () => width(1, 2), shoulder_coil: () => width(1, 2), hip_coil: () => width(7, 8),
      separation: () => { width(1, 2); width(7, 8); }, elbow_angle: () => arc(2, 4, 6),
      knee_bend: () => { arc(7, 9, 11); arc(8, 10, 12); }, shoulder_tilt: tilt, trophy_tilt: tilt,
      stance_m: feet, stance_side_m: feet, stance_fwd_m: feet, stance_angle: feet, into_court: feet,
      head_still: () => el("circle", {cx: X(q[0]), cy: Y(q[0]), r: 12, class: c}),
      swing_time: () => el("circle", {cx: X(q[6]), cy: Y(q[6]), r: 6, class: c}),
      trophy_time: () => el("circle", {cx: X(q[6]), cy: Y(q[6]), r: 6, class: c})}[m.n] || (() => {}))();
  }
  function draw() {
    const m = M(), K = D.kinds[m.k], B = box[m.k], fl = m.k === "backhand" ? -1 : 1, who = WHO();
    fig.textContent = "";
    const X = p => B.cx + fl * p[0] * B.s, Y = p => B.cy - p[1] * B.s;
    const r = R(who, m, K);
    pose(fig, m, K, r.pose, r.mom, t, X, Y, fl, "all", !K.players[who]);
    if (K.players[who]) pose(fig, m, K, K.players[who], K.mom[who], t, X, Y, fl, "me", true);
    const at = stopAt(), tsec = Math.min(t, at) / hz(K) + K.win[0];
    for (const v of [va, vm]) if (v.readyState >= 2 && !v.seeking && Math.abs(v.currentTime - (tsec - K.win[0])) > 0.015) v.currentTime = tsec - K.win[0];
    mk(fig, "text", {x: 10, y: 20, class: "note"}, Math.abs(tsec) < 0.005 ? "contact" : (tsec > 0 ? "+" : "") + tsec.toFixed(2) + " s");
  }
  // the result: him and everyone side by side, each with the measure drawn (as Federer's page: then vs now) and the
  // number under it; a timing as two bars, the stance as both feet from above
  const BARS = {swing_time: ["forward swing starts", "contact"], trophy_time: ["toss hand highest", "contact"],
                wrist_speed: ["", ""], head_still: ["", ""]};
  function result() {
    const m = M(), K = D.kinds[m.k], who = WHO(), el = (tag, at, txt) => mk(bxs, tag, at, txt);
    bxs.textContent = "";
    const r = R(who, m, K);
    const sides = [[who, "me", m.vals[who] ?? "–", K.players[who], K.mom[who], m.box[who]],
                   [r.lab, "all", r.val, r.pose, r.mom, null]];
    if (BARS[m.n]) {
      const o = r.med, v = [m.meds[who] ?? o, o], top = Math.max(...v) * 1.15, X = x => 30 + x / top * 260;
      sides.forEach(([lab, cls, val], r) => {
        const y = 110 + r * 100;
        el("text", {x: 30, y: y - 22, class: "nm " + cls}, lab);
        el("line", {x1: X(0), x2: X(v[r]), y1: y, y2: y, class: "bar " + cls});
        el("circle", {cx: X(0), cy: y, r: 6, class: "dot " + cls}); el("circle", {cx: X(v[r]), cy: y, r: 6, class: "dot " + cls});
        el("text", {x: X(v[r]), y: y + 28, "text-anchor": "middle", class: "val " + cls}, val);
      });
      const [a, b] = BARS[m.n];
      if (a) { el("text", {x: X(0), y: 300, "text-anchor": "start", class: "note"}, a); el("text", {x: 290, y: 300, "text-anchor": "end", class: "note"}, b); }
      return;
    }
    if (/^stance/.test(m.n) && D.stance[m.k]) {          // feet from above: back foot fixed, the front foot where it lands
      const st = D.stance[m.k], o = [210, 250], s = 70;
      el("line", {x1: 40, x2: 290, y1: o[1], y2: o[1], class: "ref"}); el("text", {x: 290, y: o[1] + 18, "text-anchor": "end", class: "note"}, "baseline");
      el("text", {x: 160, y: 22, "text-anchor": "middle", class: "note"}, "from above, net ↑");
      el("circle", {cx: o[0], cy: o[1], r: 7, class: "dot all"});
      // on the stance angle, the line is turned to the median angle: the angle of the median offsets can be far off it
      // over 90° the front foot is out to its own side (open), under 90° across the body (closed): seen from behind,
      // a right-hander's forehand front foot (the left) opens to the left, his backhand's (the right) to the right
      const side = m.k === "backhand" ? 1 : -1;
      [[who, "me", st[who], m.meds[who]], [r.lab, "all", r.feet, r.med]].forEach(([lab, cls, v, deg]) => {
        if (!v || v[0] == null) return;
        if (m.n === "stance_angle" && deg != null) { const L = Math.hypot(v[0], v[1]), a = deg * Math.PI / 180; v = [side * L * Math.cos(a), L * Math.sin(a)]; }
        const fx = o[0] - v[0] * s, fy = o[1] - v[1] * s;
        el("line", {x1: o[0], y1: o[1], x2: fx, y2: fy, class: "bar " + cls});
        el("circle", {cx: fx, cy: fy, r: 7, class: "dot " + cls});
        el("text", {x: fx - 12, y: fy + 4, "text-anchor": "end", class: "nm " + cls}, lab);
      });
      el("text", {x: 30, y: 46, class: "val me"}, `${who} ${m.vals[who] ?? "–"}`);     // the key on top, with the panel's title
      el("text", {x: 30, y: 64, class: "val all"}, `${r.low} ${r.val}`);
      return;
    }
    // a turn, from above with the net up: the shoulder (or hip) line at each one's median angle, the hitting side dotted.
    // Drawn from the number itself: widths on the average pose can point the other way
    const TURN = {hip_coil: "hips", shoulder_coil: "shoulders", takeback_turn: "shoulders", shoulder_width: "shoulders", finish_open: "shoulders"};
    if (TURN[m.n]) {
      const o = [160, 175], L = 95, sg = m.k === "backhand" ? -1 : 1, open = m.n === "shoulder_width" || m.n === "finish_open";
      el("line", {x1: 40, x2: 280, y1: o[1], y2: o[1], class: "ref"});
      el("text", {x: 280, y: o[1] - 8, "text-anchor": "end", class: "note"}, "square to the net");
      el("text", {x: 160, y: 22, "text-anchor": "middle", class: "note"}, `${TURN[m.n]} from above, net ↑`);
      [[who, "me", m.meds[who]], [r.lab, "all", r.med]].forEach(([lab, cls, v]) => {
        if (v == null) return;
        const a = (open ? 90 - v : v) * Math.PI / 180, dx = sg * L * Math.cos(a), dy = L * Math.sin(a);   // turning away from the net takes the hitting side back, down the page
        el("line", {x1: o[0] - dx, y1: o[1] - dy, x2: o[0] + dx, y2: o[1] + dy, class: "bar " + cls});
        el("circle", {cx: o[0] + dx, cy: o[1] + dy, r: 7, class: "dot " + cls});
        const e = cls === "me" ? 1 : -1;   // his name at the hitting end, the others' at the far end, so close lines don't collide
        el("text", {x: o[0] + e * (dx + 12 * sg), y: o[1] + e * dy + 4, "text-anchor": e * sg > 0 ? "start" : "end", class: "nm " + cls}, lab);
      });
      el("text", {x: 30, y: 46, class: "val me"}, `${who} ${m.vals[who] ?? "–"}`);     // the key on top, with the panel's title
      el("text", {x: 30, y: 64, class: "val all"}, `${r.low} ${r.val}`);
      return;
    }
    // two poses at the measure's moment, side by side, one scale
    const fl = m.k === "backhand" ? -1 : 1, B = box[m.k], s = B.s * 0.7, end = 1e9;
    sides.forEach(([lab, cls, val, P, mo], r) => {
      if (!P) return;
      const cx = 80 + r * 160, q0 = lerp(P, when(m.n, K, mo)[1]), hip = [(q0[7][0] + q0[8][0]) / 2, (q0[7][1] + q0[8][1]) / 2];
      const X = p => cx + fl * (p[0] - hip[0]) * s, Y = p => 150 - (p[1] - hip[1]) * s;
      pose(bxs, m, K, P, mo, end, X, Y, fl, cls, true);
      el("text", {x: cx, y: 276, "text-anchor": "middle", class: "nm " + cls}, lab);
      el("text", {x: cx, y: 300, "text-anchor": "middle", class: "val " + cls}, val);
    });
  }
  const va = $("ex-avg"), vm = $("ex-me"), blobs = {};
  const put = (v, file) => {                       // each clip fetched once, played from memory: scrubbing is instant
    if (v.dataset.file === file) return;
    v.dataset.file = file;
    (blobs[file] ??= fetch(file).then(r => r.blob()).then(b => URL.createObjectURL(b)))
      .then(u => { if (v.dataset.file === file) v.src = u; });
  };
  let outline = null, pick = 0, picked = "";      // which of the player's clips is showing, for whom and which stroke
  function pager(mine, key) {
    const el = $("ex-pick");
    el.textContent = "";
    if (!mine || mine.length < 2) return;
    mine.forEach(([, label], j) => {
      const b = document.createElement("button");
      b.textContent = j + 1; b.title = label; b.setAttribute("aria-pressed", j === pick);
      b.addEventListener("click", () => {          // holds what is showing, so the panel stops stepping on
        held ??= {i: cur().i, who: WHO()}; hov = null; i = held.i; pick = j; picked = key; restart();
      });
      el.appendChild(b);
    });
  }
  function caption() {
    const m = M(), C = D.clips[m.k], who = WHO(), r = R(who, m, D.kinds[m.k]);
    if (C) {
      va.style.visibility = r.clip ? "visible" : "hidden";
      if (r.clip) put(va, r.clip[0]);
      $("ex-acap").textContent = r.cap;
      const mine = C.players[who], key = who + " " + m.k;
      if (key !== picked) { pick = 0; picked = key; }
      vm.style.visibility = mine ? "visible" : "hidden";
      if (mine) put(vm, mine[Math.min(pick, mine.length - 1)][0]);
      $("ex-mcap").textContent = `${who}'s ${m.k}` + (mine?.length > 1 ? ` (${Math.min(pick, mine.length - 1) + 1} of ${mine.length})` : "");
      pager(mine, key);
    }
    $("ex-fcap").textContent = `${who}'s average vs ${r.poss}`;
    const z = r.z;
    const [noun, verb] = m.subj ?? [m.short, m.words[2] ?? "is"];
    $("ex-bcap").textContent = z == null ? "" : (Math.abs(z) < 0.3
      ? `${who}'s ${noun} ${verb} about the same as ${r.poss}.`
      : `${who}'s ${noun} ${verb} ${r.diff} ${z > 0 ? m.words[0] : m.words[1]} than ${r.poss}.`).replace(/ +/g, " ");
    $("ex-title").textContent = m.label;
    $("ex-say").textContent = m.say;
    heat.querySelectorAll(".hr").forEach(g => g.classList.toggle("on", g.dataset.k === m.k && g.dataset.n === m.n));
    heat.querySelectorAll(".tick[data-p]").forEach(e => e.classList.toggle("on", e.dataset.p === who));
    outline?.remove(); outline = null;               // the held square, outlined
    if (held) {
      const g = heat.querySelectorAll(".hr")[held.i], r = held.who && g.querySelector(`.cell[data-p="${held.who}"]`);
      if (r) outline = mk(g, "rect", {x: +r.getAttribute("x") - 1, y: +r.getAttribute("y") - 1, width: +r.getAttribute("width") + 2,
                                      height: +r.getAttribute("height") + 2, rx: 3, class: "held"});
    }
    result(); level();
  }
  // fixed beside the heatmap, the same height
  const ex = document.querySelector(".explain"), wrap = document.querySelector(".heatwrap");
  function level() {
    if (innerWidth <= 960) { heat.parentElement.style.marginTop = ""; ex.style.flex = ex.style.width = ""; ex.style.top = (document.querySelector("nav.cats")?.getBoundingClientRect().height || 0) + "px"; return; }
    // the panel exactly as tall as the heatmap, tops level: its height grows with its width (the 2 x 2's squares)
    heat.parentElement.style.marginTop = ex.style.marginTop = "0px";
    ex.style.flex = "none";
    const h = heat.getBoundingClientRect().height;
    for (let k = 0; k < 4; k++) ex.style.width = Math.max(240, ex.offsetWidth + h - ex.offsetHeight) + "px";
  }
  addEventListener("resize", level);
  const restart = () => { t = 0; hold = 0; caption(); };
  function tick(now) {
    const dt = last == null ? 0 : Math.min(0.1, (now - last) / 1000); last = now;
    if (fig.getBoundingClientRect().width) {            // its tab is showing
      const K = D.kinds[M().k], at = stopAt();
      if (t < at) t = Math.min(at, t + dt * hz(K) * SLOW);
      else if ((hold += dt) > HOLD) { if (!hov && !held) i = (i + 1) % D.measures.length; restart(); }
      draw();
    }
    requestAnimationFrame(tick);
  }
  const rows = [...heat.querySelectorAll(".hr")];
  const idx = g => D.measures.findIndex(m => m.k === g.dataset.k && m.n === g.dataset.n);
  rows.forEach(g => {
    const n = idx(g);
    g.addEventListener("mouseenter", () => { if (!held) { hov = {i: n, who: null}; restart(); } });
    g.querySelector(".hrow").addEventListener("click", () => { held = held && held.i === n && !held.who ? null : {i: n, who: null}; hov = null; i = n; restart(); });
    g.querySelectorAll(".cell").forEach(r => {
      const p = r.dataset.p;
      r.addEventListener("mouseenter", () => { if (!held) { hov = {i: n, who: p}; restart(); } });
      r.addEventListener("click", () => { held = held && held.i === n && held.who === p ? null : {i: n, who: p}; hov = null; i = n; restart(); });
    });
  });
  const tint = z => z > 0 ? `rgba(220,38,38,${Math.min(z / 1.5, 1).toFixed(2)})` : `rgba(37,99,235,${Math.min(-z / 1.5, 1).toFixed(2)})`;
  function recolour() {
    rows.forEach(g => {
      const m = D.measures[idx(g)];
      g.querySelectorAll(".cell").forEach(c => {
        c.dataset.f ??= c.getAttribute("fill");
        const q = c.dataset.p, a = m.meds[q], b = m.meds[ref];
        c.setAttribute("fill", !ref ? c.dataset.f : q === ref || a == null || b == null ? "transparent" : tint((a - b) / m.sd));
      });
    });
    heat.querySelectorAll(".tick[data-p]").forEach(e => e.classList.toggle("ref", e.dataset.p === ref));
    const zd = document.getElementById("zdef");
    if (zd) zd.textContent = `z: his median minus ${ref ? ref + "'s" : "the others'"}, in units of how`;
  }
  $("ex-ref").addEventListener("change", e => { ref = e.target.value; recolour(); restart(); });
  heat.addEventListener("mouseleave", () => { if (hov) { i = hov.i; hov = null; restart(); } });
  restart(); requestAnimationFrame(tick);
})();
"""
