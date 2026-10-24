"""Getting a video's footage to projects/<project>/<video>/source.mp4, from YouTube or a local recording.

    python -m dsa.ingest.download list
    python -m dsa.ingest.download start federer_usopen/2015_djokovic   # where play starts (Opus), for the catalog
    python -m dsa.ingest.download get federer_usopen/2015_djokovic     # YouTube: only the kept minutes from there
    python -m dsa.ingest.download get tests/own_drew                   # a 5-minute cut of a local recording
    python -m dsa.ingest.download get atp_hardcourt/2024_ao_sinner_medvedev   # whole upload kept in raw/<project>/
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from dsa.ingest.projects import MORE, PROJECTS
from dsa.paths import ROOT, video_dir


def upload(spec: str) -> Path:
    """Where the video's whole YouTube upload is kept (projects that keep them: quiz, atp_hardcourt); one another
    project downloaded already is used from there."""
    project, video = spec.split("/")
    yt = PROJECTS[project]["videos"][video][0]
    own = ROOT / "raw" / project / f"{yt}.mp4"
    return own if own.exists() else next((ROOT / "raw").glob(f"*/{yt}.mp4"), own)


FORMAT = "bv*[height<=720][ext=mp4]"
# YouTube's player needs a JavaScript runtime to solve its challenges; without one yt-dlp falls back to
# clients that get "confirm you're not a bot" after a burst of requests
JS = ["--js-runtimes", f"node:{shutil.which('node')}"] if shutil.which("node") else []
EVERY, LOOK = 30, 1500                  # start: a frame every EVERY s of the first LOOK s


def stream(yt: str) -> str:
    """The video stream's URL, for ffmpeg to read parts of without downloading the whole upload."""
    return subprocess.run([sys.executable, "-m", "yt_dlp", "--no-warnings", *JS, "-g", "-f", FORMAT,
                           f"https://www.youtube.com/watch?v={yt}"], check=True, capture_output=True,
                          text=True).stdout.split()[0]


def fetch(yt: str, full: Path) -> None:
    """The whole upload to `full` (one at a time: YouTube blocks bursts)."""
    full.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "yt_dlp", "--no-warnings", *JS, "-f", FORMAT, "-N", "8",
                    "-o", str(full), f"https://www.youtube.com/watch?v={yt}"], check=True)


def start(spec: str) -> int:
    """The second play starts in a YouTube upload (after walk-ons, coin toss and warm-up), from a frame every
    EVERY s of its first LOOK s, read by Opus. Frames in raw/starts/<id>/."""
    from concurrent.futures import ThreadPoolExecutor

    from dsa.opus import ask

    project, video = spec.split("/")
    yt, match = (PROJECTS[project]["videos"].get(video) or MORE[video])[:2]
    out = ROOT / "raw/starts" / yt
    out.mkdir(parents=True, exist_ok=True)
    if (out / "start.json").exists():
        return json.loads((out / "start.json").read_text())["start"]
    whole = next((ROOT / "raw").glob(f"*/{yt}.mp4"), None)       # read from the upload on disk when it's there
    url, ts = str(whole) if whole else stream(yt), range(0, LOOK + 1, EVERY)
    def grab(t):                        # a stalled read gives up; the frame is then left out
        try:
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-rw_timeout", "20000000", "-ss", str(t), "-i", url,
                            "-frames:v", "1", "-vf", "scale=640:-1", str(out / f"{t:05d}.jpg")], timeout=90)
        except subprocess.TimeoutExpired:
            pass
    if not all((out / f"{t:05d}.jpg").exists() for t in ts):
        with ThreadPoolExecutor(16) as ex:
            list(ex.map(grab, ts))
    ts = [t for t in ts if (out / f"{t:05d}.jpg").exists()]          # none past the end, or stalled
    imgs = [out / f"{t:05d}.jpg" for t in ts]
    schema = {"type": "object", "properties": {"first": {"type": "integer"}, "why": {"type": "string"}},
              "required": ["first", "why"], "additionalProperties": False}
    p = PROJECTS[project]
    of = f"{p['who']} at the US Open, {video[:4]}, {match}" if "who" in p else match   # e.g. "US Open 2024 QF, Sinner vs Medvedev"
    a, _ = ask(f"Frames from a YouTube upload of {of}: frame k (from 0, {len(imgs)} frames) is {', '.join(map(str, ts))} seconds in, in turn. Which is the first frame "
               f"of the match itself: a point being played, or the time between points, with the score on screen "
               f"or the match clearly under way? Not an intro, studio, walk-on, coin toss or the warm-up (players "
               f"rallying back and forth before the match, no score shown). Answer its k.", imgs, schema,
               out / "cache")
    print(spec, "starts at", ts[a["first"]], "s:", a["why"])
    (out / "start.json").write_text(json.dumps({"start": ts[a["first"]], "why": a["why"]}))
    return ts[a["first"]]


def get(spec: str) -> None:
    from dsa.video import cut
    from dsa.video import info as probe

    project, video = spec.split("/")
    p = PROJECTS[project]
    out = video_dir(spec) / "source.mp4"
    if out.exists():
        print(out, "exists")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    if project == "tests":
        src, start_s = p["videos"][video]
        cut(src, start_s, p["seconds"], out)
        return
    if project in ("own", "others"):
        src, _, start_s, seconds = p["videos"][video][:4]
        if src.startswith("yt:"):                   # one of ours on YouTube: the whole upload to raw/own/ first
            full = ROOT / "raw" / project / f"{src[3:]}.mp4"
            if not full.exists():
                fetch(src[3:], full)
            src = str(full)
        cut(src, start_s, seconds or 10 ** 6, out)
        return
    info = p["videos"][video]
    at = 0
    if project == "federer_usopen":
        # only the kept minutes, read straight from the stream (yt-dlp's --download-sections stalls on these)
        yt, _, at, seconds = info[:4]
        if at is None:
            sys.exit(f"{spec}: no start yet (python -m dsa.ingest.download start {spec})")
        full, part = out.with_name("window.mp4"), out.with_name("window.part.mp4")
        if not full.exists():                       # named window.mp4 only once whole
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(at), "-i", stream(yt),
                            *(["-t", str(seconds)] if seconds else []), "-c", "copy", "-an", str(part)], check=True)
            part.rename(full)
        seconds, at = 0, 0
    else:
        # the whole upload (kept in raw/<project>/: a window can grow later without downloading again; 8 fragments
        # at a time, tens of MB/s against about real time for a stream read), then `seconds` of it from `at`
        if project == "quiz":
            seconds = info[4]
        else:
            at, seconds = info[2], info[3]
        full = upload(spec)
        if not full.exists():
            fetch(info[0], full)
        if at is None:                              # where play starts, read from the upload now on disk
            at = start(spec)
    # F3ED (step 2) finds hits at 25-30 fps: 50 / 60 fps uploads are halved
    fps = probe(str(full))[0]
    # re-encoded when halved, and when cut from inside an upload: copied from there, the file's frame count
    # overstates the frames that decode (dsa.gpu.cuts needs them equal)
    enc = ["-c:v", "libx264", "-crf", "18", "-preset", "veryfast"]
    # and when not H.264: YouTube serves some uploads as AV1, which Modal's decoders can't read
    inside = at or seconds or (project == "federer_usopen" and info[2])   # a copied -t cut overstates them too
    codec = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name",
                            "-of", "csv=p=0", str(full)], capture_output=True, text=True).stdout.strip()
    rate = ["-vf", f"fps={fps / 2}", *enc] if fps > 31 else enc if inside or codec != "h264" else ["-c", "copy"]
    tmp = out.with_name("source.part.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", *(["-ss", str(at)] if at else []), "-i", str(full),
                    *(["-t", str(seconds)] if seconds else []), *rate, "-an", str(tmp)], check=True)
    tmp.rename(out)
    if project == "federer_usopen":
        full.unlink()
    print(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["list", "get", "start"])
    ap.add_argument("video", nargs="?", help="<project>/<video>")
    a = ap.parse_args()
    if a.cmd in ("get", "start"):
        {"get": get, "start": start}[a.cmd](a.video)
        return
    for name, p in PROJECTS.items():
        print(f"{name}: {p['what']}")
        for v, info in p["videos"].items():
            there = (video_dir(f"{name}/{v}") / "source.mp4").exists()
            print(f"  {'✓' if there else ' '} {name}/{v}  {info[1] if name != 'tests' else ''}")


if __name__ == "__main__":
    main()
