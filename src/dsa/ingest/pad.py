"""atp_hardcourt: more footage, one piece at a time, until every player has TARGET routine strokes of each kind
(forehand, backhand, serve) or the Modal budget for this run is spent. After each piece the strokes are counted
again (dsa.measure.technique extract, dsa.measure.profiles) and the next piece is the most useful one:

  1. a new match's first hour (dsa.ingest.projects.MORE, in order; spares last), while one of its players is short
     (a match on the SSD already is used from there; otherwise the upload is downloaded, one at a time)
  2. otherwise the next PAD_S of a match already in, the one whose listed players are shortest
  3. otherwise a spare match with a short player

Each piece goes through dsa.batch (cut, models, players, names, renders) before the next one starts. What was
added is in projects/atp_hardcourt/pad.json (read by dsa.ingest.projects); what failed in pad_failed.json.

    python -m dsa.ingest.pad [--target 70] [--budget 75]     (ANTHROPIC_API_KEY set: names and starts read by Opus)
"""
from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
import time

from dsa.batch import HOLD
from dsa.ingest.projects import MORE, PAD, PAD_S, PROJECTS, piece
from dsa.paths import ROOT, video_dir

PROJECT = "atp_hardcourt"
KINDS = ["forehand", "backhand", "serve"]
SPARES = {"2023_uso_djokovic_fritz", "2022_ao_fritz_tiafoe", "2024_uso_dimitrov_rublev", "2024_ao_deminaur_rublev",
          "2022_uso_korda_paul"}
TAIL = 600                         # seconds left at a recording's end not used (handshakes, interviews)
FAILED = PAD.with_name("pad_failed.json")
PY = [sys.executable, "-m"]


def log(*a) -> None:
    print(time.strftime("%H:%M"), *a, flush=True)


def counts() -> dict[str, dict[str, int]]:
    """Routine strokes per listed player and kind, as dsa.measure.profiles keeps them."""
    for k in KINDS:
        subprocess.run(PY + ["dsa.measure.technique", "extract", PROJECT, k], check=True, capture_output=True)
    subprocess.run(PY + ["dsa.measure.profiles", PROJECT], check=True, capture_output=True)
    d = ROOT / "projects" / PROJECT / "profiles"
    got = {k: json.loads((d / f"{k}.json").read_text())["strokes"] for k in KINDS}
    return {p: {k: got[k].get(p, 0) for k in KINDS} for p in PROJECTS[PROJECT]["players"]}


def spend(since: str) -> float:
    """Modal dollars of this project's apps from the UTC hour `since` (e.g. 2026-10-05T06), by the hourly report:
    the daily one leaves out the current (UTC) day."""
    out = subprocess.run([str(sys.executable).rsplit("/", 1)[0] + "/modal", "billing", "report", "--start",
                          since[:10], "-r", "h", "--json"], capture_output=True, text=True).stdout
    try:
        rows = json.loads(out)
    except json.JSONDecodeError:
        return 0.0
    rows = rows if isinstance(rows, list) else rows.get("data", [])
    return sum(float(r.get("cost") or 0) for r in rows
               if (r.get("description") or r.get("app_name") or "").startswith("deep-sports")
               and r.get("interval_start", "")[:13] >= since[:13])


def duration(yt: str) -> float | None:
    f = next((ROOT / "raw").glob(f"*/{yt}.mp4"), None)
    if f is None:
        return None
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                 str(f)], capture_output=True, text=True).stdout)


def matches() -> dict[str, tuple]:
    """Every match in the project now: name -> (YouTube id, start s, listed players in it, next later piece)."""
    vids, out = PROJECTS[PROJECT]["videos"], {}
    for v, (yt, _, at, s, players) in vids.items():
        base = v.split("_x")[0].removesuffix("_b")
        if base not in out:
            first = vids[base]
            out[base] = [yt, first[2], [p for p in players if p in PROJECTS[PROJECT]["players"]], 1]
        if "_x" in v:
            out[base][3] = max(out[base][3], int(v.split("_x")[1]) + 1)
    return out


LOWER = {"Ben Shelton": {"backhand": 50}}   # targets below --target: Shelton hits few routine backhands (~6 an hour)


def short_by(c: dict, players: list[str], target: int) -> int:
    return sum(max(0, min(target, LOWER.get(p, {}).get(k, target)) - c[p][k]) for p in players for k in KINDS)


PER_PIECE = {"forehand": 9, "backhand": 7, "serve": 11}  # routine strokes a player gains from PAD_S of a match, about
ROUND = 6                                                # pieces added at once (dsa.batch runs them side by side)
TRIES = 3                                                # rounds a piece is tried before it is left out


def choose(c: dict, target: int, pad: list[str], failed: set[str], most: int = ROUND) -> list[str]:
    """The next pieces to add together: every new match (not a spare) with a short player; otherwise up to `most`
    later pieces of matches already in, each time the one whose players are shortest after what the pieces chosen
    before it should bring (PER_PIECE); otherwise one spare."""
    listed = PROJECTS[PROJECT]["players"]
    new = [m for m in MORE if m not in pad and m not in failed
           and short_by(c, [p for p in MORE[m][2] if p in listed], target)
           and not (HOLD.exists() and duration(MORE[m][0]) is None)]      # no downloads while held
    if first := [m for m in new if m not in SPARES]:
        return first
    c = {p: dict(v) for p, v in c.items()}
    nxt = {m: v for m, v in matches().items()}
    out = []
    while len(out) < most:
        best = None
        for m, (yt, at, players, k) in nxt.items():
            name, dur = f"{m}_x{k}", duration(yt)
            if name in failed or dur is None or at is None or at + 3600 + k * PAD_S > dur - TAIL:
                continue
            gain = short_by(c, players, target)
            if gain and (best is None or gain > best[0]):
                best = (gain, name, m)
        if best is None:
            break
        out.append(best[1])
        nxt[best[2]][3] += 1
        for p in nxt[best[2]][2]:
            for k in KINDS:
                c[p][k] += PER_PIECE[k]
    return out or new[:1]


def add(names: list[str]) -> list[str]:
    """Through dsa.batch together (downloads one at a time, and where play starts read from the upload); the ones
    that did not get through step 4 (names)."""
    pad = json.loads(PAD.read_text()) if PAD.exists() else []
    PAD.write_text(json.dumps(pad + [n for n in names if n not in pad], indent=1))
    log(", ".join(names), "added; batch running")
    with open(ROOT / "logs/batch" / f"{PROJECT}.log", "a") as fh:
        subprocess.run(PY + ["dsa.batch", PROJECT], stdout=fh, stderr=subprocess.STDOUT)
    return [n for n in names if not (video_dir(f"{PROJECT}/{n}") / "names.json").exists()]


def held(name: str) -> bool:
    """A piece whose recording is not on the SSD yet while downloads are held (logs/batch/no_youtube): it waits,
    and is not counted as failed."""
    p = piece(name)
    return HOLD.exists() and (p is None or duration(p[0]) is None)


def idle() -> None:
    """Wait for a batch already running (one left by an earlier run of this) to finish."""
    while subprocess.run(["pgrep", "-f", "dsa.batch atp_hardcourt"], capture_output=True).returncode == 0:
        time.sleep(30)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=int, default=70)
    ap.add_argument("--budget", type=float, default=75, help="Modal dollars for this run")
    ap.add_argument("--since", help="the run's start, a UTC hour such as 2026-10-05T06 (default: now)")
    a = ap.parse_args()
    since = a.since or datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H")
    tries: dict[str, int] = {}
    while True:
        idle()
        c = counts()
        log("strokes (fh / bh / sv):", "; ".join(f"{p.split()[-1]} {'/'.join(str(c[p][k]) for k in KINDS)}"
                                                 for p in c))
        used = spend(since)
        if used >= a.budget:
            log(f"budget spent (${used:.2f}); stopping")
            break
        pad = json.loads(PAD.read_text()) if PAD.exists() else []
        failed = set(json.loads(FAILED.read_text())) if FAILED.exists() else set()
        # pieces added before that did not get through (an outage, a failed step) go again first, TRIES rounds
        pending = [n for n in pad if not (video_dir(f"{PROJECT}/{n}") / "names.json").exists()
                   and not held(n)]
        names = pending or choose(c, a.target, pad, failed)
        if not names:
            log(f"nothing left to add (${used:.2f} spent)")
            break
        for n in add(names):
            if held(n):
                continue
            tries[n] = tries.get(n, 0) + 1
            if tries[n] < TRIES:
                log(n, "did not get through; again next round")
                continue
            log(n, f"did not get through {TRIES} times; left out")
            failed.add(n)
            FAILED.write_text(json.dumps(sorted(failed)))
            PAD.write_text(json.dumps([p for p in json.loads(PAD.read_text()) if p != n], indent=1))
        log(f"round done (${spend(since):.2f} spent so far)")


if __name__ == "__main__":
    main()
