"""Who hit each shot, by name, from every video's hits (dsa.gpu.strokes), who is at each end (dsa.scene.run) and which
player number is which player (dsa.scene.names; the two players of a video: dsa.ingest.projects.pair).

The first of the pair has names.json's "who" number (per kept piece when names.json has "by_piece": highlight
reels, whose clips each start the tracking afresh); the other player has the other number. Step 3 now and then swaps the two players for a
few frames, so the player at that end is the one there most often in the WINDOW seconds around the hit (within
the same kept piece: nobody changes ends inside one).

Writes projects/<project>/shots.json: one row per hit whose player is known, with the player, the video, its year
and the hit's fields (shot, hand, technique, direction, side, outcome, ...); prints a count per video and player.

    python -m dsa.measure.shots federer_usopen
    python -m dsa.measure.shots federer_usopen --check      # the naming against the scoreboard
    python -m dsa.measure.shots federer_usopen --pieces 2010_dabul 2008_andreev   # his number per piece, ditto
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

from dsa.ingest.projects import PROJECTS, pair
from dsa.paths import PROJECTS as ROOT_PROJECTS
from dsa.paths import hits_file, video_dir

WINDOW = 1.0                    # seconds either side of a hit
CHECKS = 8                      # serves read off the scoreboard per video


def player_at(near_far: list, frame: int, end: int, lo: int, hi: int, half: int) -> int | None:
    """The player id at `end` (0 near, 1 far) most often in frames [frame - half, frame + half] within [lo, hi)."""
    ids = Counter(x[end] for x in near_far[max(lo, frame - half):min(hi, frame + half + 1)] if x[end])
    return ids.most_common(1)[0][0] if ids else None


def player_of(names: dict, piece: int, k: int | None, both: tuple[str, str]) -> str | None:
    """The name of player number k in kept piece `piece` (names: names.json, both: dsa.ingest.projects.pair); None when
    the number or the naming of the piece is unknown."""
    by = names.get("by_piece")          # the first player's number per kept piece (highlight reels)
    who = names["who"] if by is None else by.get(str(piece))
    return None if not (k and who) else both[0] if k == who else both[1]


def owners(spec: str) -> list[tuple[dict, str | None]]:
    """Every hit in `spec` with the name of its player (None: unknown)."""
    d = video_dir(spec)
    pl = json.loads((d / "players.json").read_text())
    names = json.loads((d / "names.json").read_text())
    hits = json.loads(hits_file(d).read_text())["hits"]
    kj = d / "kept.json"
    lengths = [b - a for a, b in json.loads(kj.read_text())["pieces"]] if kj.exists() else [len(pl["near_far"])]
    bounds, s = [], 0
    for n in lengths:
        bounds.append((s, s + n))
        s += n
    half, both, out = int(WINDOW * pl["fps"]), pair(spec), []
    for h in hits:
        i, (lo, hi) = next((i, (a, b)) for i, (a, b) in enumerate(bounds) if a <= h["frame"] < b)
        k = player_at(pl["near_far"], h["frame"], 0 if h["side"] == "near" else 1, lo, hi, half)
        out.append((h, h.get("player") or player_of(names, i, k, both)))       # labels may name him
    return out


def shots(spec: str) -> tuple[list[dict], dict]:
    """(the hits in `spec` whose player is known, counts per player and unknown)."""
    names = json.loads((video_dir(spec) / "names.json").read_text())
    rows, count = [], Counter()
    for h, who in owners(spec):
        count[who or "unknown"] += 1
        if who:
            rows.append({"player": who, "video": spec.split("/")[1], "year": int(spec.split("/")[1][:4]), **h})
    return rows, dict(count) | {"sure": names["sure"]}


def check(spec: str, n: int = CHECKS) -> dict:
    """The scoreboard's serve marker (dsa.pages.quiz.server, read by Opus) shortly before n serves spread through the
    video, against the player numbers: agree / disagree / no marker. A swapped naming disagrees throughout, a
    mixed one in part."""
    from dsa.pages.quiz import server
    from dsa.scene import results

    d = results.load(spec)
    names = json.loads((d["src"] / "names.json").read_text())
    both, him = list(pair(spec)), names["who"]
    serves = [h for h in d["hits"] if h["shot"] == "serve"]
    out = Counter()
    for h in serves[::max(1, len(serves) // n)][:n]:
        f0 = max(a for a in d["starts"][:-1] if a <= h["frame"])
        k = player_at(d["pl"]["near_far"], h["frame"], 0 if h["side"] == "near" else 1, int(f0), h["frame"] + 1,
                      int(WINDOW * d["fps"]))
        who = server(d, int(f0), h["frame"], both)
        out["no marker" if who is None or k is None else "agree" if (who == both[0]) == (k == him) else "disagree"] += 1
    return dict(out)


def pieces(spec: str) -> dict[str, int]:
    """The first player's number per kept piece from the scoreboard serve marker at every serve, written to names.json as
    "by_piece" (for highlight reels, whose clips each start the tracking afresh). Pieces with no read serve, or
    whose serves disagree, are left out."""
    from concurrent.futures import ThreadPoolExecutor

    from dsa.pages.quiz import server
    from dsa.scene import results

    d, v = results.load(spec), spec.split("/")[1]
    both, starts = list(pair(spec)), [int(a) for a in d["starts"]]

    def one(h):
        i = max(j for j, a in enumerate(starts[:-1]) if a <= h["frame"])
        k = player_at(d["pl"]["near_far"], h["frame"], 0 if h["side"] == "near" else 1, starts[i], h["frame"] + 1,
                      int(WINDOW * d["fps"]))
        who = server(d, starts[i], h["frame"], both)
        return i, None if k is None or who is None else k if who == both[0] else 3 - k

    with ThreadPoolExecutor(8) as ex:
        seen = list(ex.map(one, [h for h in d["hits"] if h["shot"] == "serve"]))
    ids: dict[str, set] = {}
    for i, k in seen:
        if k:
            ids.setdefault(str(i), set()).add(k)
    by = {i: s.pop() for i, s in ids.items() if len(s) == 1}
    f = d["src"] / "names.json"
    names = json.loads(f.read_text()) | {"by_piece": by}
    f.write_text(json.dumps(names, indent=1))
    print(f"{v:24s} {len(by)} of {len(starts) - 1} pieces; {both[0]}'s number {dict(Counter(by.values()))}", flush=True)
    return by


def run(project: str) -> list[dict]:
    rows, per = [], Counter()
    for v in PROJECTS[project]["videos"]:
        got, count = shots(f"{project}/{v}")
        rows += got
        per.update(r["player"] for r in got)
        named = "  ".join(f"{k} {n:4d}" for k, n in count.items() if k not in ("unknown", "sure"))
        print(f"{v:24s} {named}  unknown {count.get('unknown', 0):3d}  {'' if count['sure'] else 'naming unsure'}")
    (ROOT_PROJECTS / project / "shots.json").write_text(json.dumps(rows, indent=1))
    print("per player:", dict(per.most_common()), " total", len(rows))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("project")
    ap.add_argument("--check", action="store_true", help="the naming against the scoreboard, per video")
    ap.add_argument("--pieces", nargs="+", metavar="VIDEO", help="his number per kept piece (highlight reels)")
    a = ap.parse_args()
    if a.pieces:
        for v in a.pieces:
            pieces(f"{a.project}/{v}")
    elif a.check:
        from concurrent.futures import ThreadPoolExecutor
        vs = list(PROJECTS[a.project]["videos"])
        with ThreadPoolExecutor(8) as ex:
            for v, r in zip(vs, ex.map(lambda v: check(f"{a.project}/{v}"), vs)):
                print(f"{v:24s} {r}", flush=True)
    else:
        run(a.project)
