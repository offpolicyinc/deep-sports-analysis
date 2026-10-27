"""Step 4: which of Player 1 / Player 2 is the project's player ("who" in dsa.ingest.projects), asked of Opus once per
video; for a video whose two players are listed (the quiz), which is which. It sees CROPS moments of each
player, spread over the video and taken at the near end where they are biggest, with what the project and video
are (e.g. "Final vs Djokovic"), and names both.

Writes <video>/names.json: {"1": name, "2": name, "who": 1 or 2, "sure": bool, "why": ...}.

    python -m dsa.scene.names federer_usopen/2015 [--test]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from dsa.ingest.projects import PROJECTS, pair
from dsa.opus import ask
from dsa.paths import video_dir
from dsa.video import frame_at

CROPS, HEIGHT, NEAR_SHARE = 6, 360, 0.25        # moments per player; crop height in px

SCHEMA = {
    "type": "object",
    "properties": {"who": {"type": "integer", "enum": [1, 2]}, "player_1": {"type": "string"},
                   "player_2": {"type": "string"}, "sure": {"type": "boolean"}, "why": {"type": "string"}},
    "required": ["who", "player_1", "player_2", "sure", "why"],
    "additionalProperties": False,
}


def moments(players: dict, k: int) -> list[tuple[int, list[int]]]:
    """[(frame, box)] for player k: CROPS frames spread over the video, at the near end when they spend at
    least NEAR_SHARE of their time there (a few seconds at the near end are more likely a mix-up, e.g. a ball
    kid crossing in front of the net, than the player)."""
    at = [[i for i, (s, b) in enumerate(zip(players["near_far"], players["boxes"])) if s[side] == k and b[side]]
          for side in (0, 1)]
    side = 0 if len(at[0]) >= max(CROPS, NEAR_SHARE * (len(at[0]) + len(at[1]))) else 1
    pick = [at[side][int(j)] for j in np.linspace(0, len(at[side]) - 1, CROPS)] if at[side] else []
    return [(i, players["boxes"][i][side]) for i in pick]


def sheet(video: str, shots: list[tuple[int, list[int]]]) -> np.ndarray:
    tiles = []
    for i, (x1, y1, x2, y2) in shots:
        f = frame_at(video, i)
        m = int(0.15 * (y2 - y1))
        c = f[max(0, y1 - m):y2 + m, max(0, x1 - m):x2 + m]
        tiles.append(cv2.resize(c, (int(c.shape[1] * HEIGHT / c.shape[0]), HEIGHT)))
    return np.hstack(tiles)


def sheets(spec: str, test: bool = False) -> list[Path]:
    """names_player1.jpg and names_player2.jpg: each player at CROPS moments of the video."""
    src = video_dir(spec)
    d = src / "test" if test else src
    video = str(src / ("kept.mp4" if (src / "kept.json").exists() else "source.mp4"))
    players = json.loads((d / "players.json").read_text())
    imgs = []
    for k in (1, 2):
        imgs.append(d / f"names_player{k}.jpg")
        cv2.imwrite(str(imgs[-1]), sheet(video, moments(players, k)), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return imgs


def name(spec: str, test: bool = False) -> dict:
    project, v = spec.split("/")
    p = PROJECTS[project]
    src = video_dir(spec)
    d = src / "test" if test else src
    imgs = sheets(spec, test)
    info = p["videos"][v]
    both = None if "who" in p else pair(spec)
    match = info[1] if both else f"{v}, {info[1]}"
    shown = (f"These are two tennis players from a TV broadcast, {match}. The first image shows Player 1 at "
             f"{CROPS} moments of the match, the second Player 2. ")
    why = "say whether you are sure, and say briefly what you went by (clothing, build, hair, technique, handedness)."
    if both:
        prompt = (shown + f"The two players are {both[0]} and {both[1]}. Which is Player 1 and which Player 2? "
                  f"Set who to the player number of {both[0]}, name both players, " + why)
    else:
        prompt = (f"{p['what']} " + shown + f"Which of them is {p['who']}? Name both players, " + why)
    a, _ = ask(prompt, imgs, SCHEMA, src / "names_cache")
    out = {"1": a["player_1"], "2": a["player_2"], "who": a["who"], "sure": a["sure"], "why": a["why"]}
    if both:                                       # the listed spellings, in the order Opus put them
        out["1"], out["2"] = both if a["who"] == 1 else both[::-1]
    (d / "names.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="<project>/<video>")
    ap.add_argument("--test", action="store_true", help="the test run's output, in <video>/test/")
    a = ap.parse_args()
    print(json.dumps(name(a.video, a.test), indent=1))
