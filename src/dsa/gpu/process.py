"""Step 2, the frame loop: every model on the analysed video, saved once so nothing after it needs a GPU. The analysed
video is kept.mp4 (TV: the kept frames, from dsa.gpu.cuts) or source.mp4 (phone).

  strokes  (TV) F3ED: every hit and its stroke (dsa.gpu.strokes) -> strokes/scores.npz, decoded on the laptop to
           strokes/strokes.json.
  pass 1   ffmpeg decodes on all cores in a reader thread; RF-DETR person boxes (conf >= 0.2) in batches of
           BATCH; WASB ball on each frame with its neighbours. TV: TennisCourtDetector every 5 frames and at
           each cut (dsa.scene.court.check_tv), and RTMW-x skeletons right away for every box (conf >= 0.3) whose
           feet are in the playing area (dsa.scene.court.in_area). A project's "gpu" options (dsa.ingest.projects) can pose
           the near end only and check the court less often. Cached as raw.pass1.npz.
  people   ByteTrack, restarted at each cut between kept pieces, stitched, the still ones marked (dsa.scene.tracks).
  pass 2   phone only (its court comes later, from Opus): decode again; RTMW-x on every tracked, non-still
           box. Crops are prepared on a thread pool, POSE_BATCH per forward.

    raw.npz  n (boxes per frame), boxes (sum n x 5: x1 y1 x2 y2 conf), track (ByteTrack id, -1 none),
             kp (sum n x 65 x 3: x y score, NaN when not posed), ball (frames x 4: found x y score),
             court (frames x 14 x 2, TV only), fps, width, height, seconds (per stage)
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from dsa.paths import video_dir

BATCH, POSE_BATCH = 16, 128
CHUNK = 64                      # frames decoded before their ball inputs and court checks run in parallel


def process(video: str, out: str, tv: bool = False, device: str = "cuda", pose_device: str = "cuda",
            limit: int = 0, batch: int = BATCH, on_pass1=None, breaks=(), near_only: bool = False,
            court_every: int = 0, models: dict | None = None) -> dict:
    """TV options (a project's "gpu" in dsa.ingest.projects): near_only poses only people at the near end (a quarter of
    the skeletons, and all the stroke and movement analysis uses); court_every checks the court that often (frames;
    default COURT_EVERY). models: already loaded ones (load_models), shared by runs side by side on one GPU."""
    import cv2

    from dsa.gpu.models import KP, Pose
    from dsa.scene import tracks
    from dsa.scene.court import COURT_EVERY, COURT_HOLD, NET, check_tv, homography, in_area
    from dsa.video import frames, info

    t0 = time.time()
    breaks = set(breaks)
    every = court_every or COURT_EVERY
    hold = max(COURT_HOLD, int(2.5 * every))                     # as dsa.scene.court.tv holds a court
    fps, W, H = info(video)
    tm = {"detect": 0.0, "ball": 0.0, "court": 0.0, "track": 0.0, "pose": 0.0}
    boxes: list[np.ndarray] = []
    balls: list[np.ndarray] = []
    checks: list[np.ndarray] = []
    kp: list[np.ndarray] = []
    pool = ThreadPoolExecutor(8)
    pose = None
    crops: list[tuple[int, int, tuple]] = []                     # (frame, box, prepared crop)

    def flush():
        t = time.time()
        for (i, j, _), k in zip(crops, pose.run([c[2] for c in crops])):
            kp[i][j] = k
        crops.clear()
        tm["pose"] += time.time() - t

    # pass 1: boxes, ball; TV: court once a second, and skeletons of everyone in the playing area
    n = 0
    p1 = Path(out).with_suffix(".pass1.npz")
    z = np.load(p1) if p1.exists() else None    # pass 1 already done (a later step failed): reuse it
    if z is not None and (not tv or "kp" in z):
        cut = np.cumsum(z["n"])[:-1]
        boxes, balls, n = list(np.split(z["boxes"], cut)), list(z["ball"]), len(z["n"])
        checks = list(z["court"]) if tv else []
        kp = list(np.split(z["kp"], cut)) if tv else []
    else:
        m = models or load_models(tv, device, pose_device)
        det, ball, tcd = m["det"], m["ball"], m.get("tcd")
        if tv:
            pose = m["pose"]
        small: dict[int, np.ndarray] = {}
        court_h: list = []                                       # TV: image -> court homography per frame, or None
        last = (-10 ** 9, None)

        def run_det(fs: list[np.ndarray]):
            t = time.time()
            i0 = len(boxes)
            boxes.extend(det.people(fs))
            tm["detect"] += time.time() - t
            if tv:                                               # skeletons now, while the frames are in memory
                t = time.time()
                want = []
                for k, f in enumerate(fs):
                    i, b = i0 + k, boxes[i0 + k]
                    kp.append(np.full((len(b), len(KP), 3), np.nan, np.float16))
                    js = np.flatnonzero(b[:, 4] >= 0.3) if len(b) else np.zeros(0, int)
                    if len(js) and court_h[i] is not None:
                        feet = np.float32([[(b[j, 0] + b[j, 2]) / 2, b[j, 3]] for j in js])[:, None]
                        xy = cv2.perspectiveTransform(feet, court_h[i]).reshape(-1, 2)
                        js = js[in_area(xy) & ((xy[:, 1] > NET) if near_only else True)]
                    want += [(i, j, f) for j in js]
                crops.extend((i, j, c) for (i, j, _), c in
                             zip(want, pool.map(lambda w: pose.prepare(w[2], boxes[w[0]][w[1]]), want)))
                tm["pose"] += time.time() - t
                if len(crops) >= POSE_BATCH:
                    flush()

        def run_ball(upto: int):                # frames whose next neighbour has arrived (or the last frame)
            t = time.time()
            todo = list(range(len(balls), upto))
            if todo:
                balls.extend(ball.locate([[small[i - 1] if i and i not in breaks else small[i], small[i],
                                           small[i + 1] if i + 1 in small and i + 1 not in breaks else small[i]]
                                          for i in todo], W, H))
            for i in [k for k in small if k < upto - 2]:
                del small[i]
            tm["ball"] += time.time() - t

        none = np.full((14, 2), np.nan, np.float32)

        def prep(i0: int, fs: list[np.ndarray]):
            """A chunk's CPU work, run while the GPU works on the chunk before: the ball model's inputs and the
            court checks (a chunk of CHUNK frames holds several, run side by side)."""
            t = time.time()
            smalls = list(pool.map(ball.small, fs))
            got = {}
            if tv:
                due = [k for k in range(len(fs)) if (i0 + k) % every == 0 or i0 + k in breaks]
                got = dict(zip(due, pool.map(lambda k: check_tv(tcd, fs[k]), due)))
            return smalls, got, time.time() - t

        def take(i0: int, fs: list[np.ndarray], res) -> None:
            """prep's results into the running state, in frame order (each court holds for the frames after)."""
            nonlocal last
            smalls, got, took = res
            for k, sm in enumerate(smalls):
                small[i0 + k] = sm
            tm["court"] += took                                  # ball inputs and court checks, overlapped
            for k in range(len(fs) if tv else 0):
                i, c = i0 + k, got.get(k, none)
                checks.append(c)
                if i in breaks:
                    last = (-10 ** 9, None)                      # a new piece: the last court may not hold
                if np.isfinite(c).all():
                    last = (i, homography(c))
                court_h.append(last[1] if i - last[0] <= hold else None)

        def chunks():
            fs = []
            for f in frames(video, limit):
                fs.append(f)
                if len(fs) == CHUNK:
                    yield fs
                    fs = []
            if fs:
                yield fs

        ahead = ThreadPoolExecutor(1)                            # prep of the next chunk, one at a time
        it, i0 = chunks(), 0
        cur = next(it, None)
        job = ahead.submit(prep, 0, cur) if cur is not None else None
        while cur is not None:
            res = job.result()
            nxt = next(it, None)
            if nxt is not None:
                job = ahead.submit(prep, i0 + len(cur), nxt)
            take(i0, cur, res)
            for k in range(0, len(cur), batch):
                run_det(cur[k:k + batch])
                run_ball(len(boxes) - 1)
            i0 += len(cur)
            cur = nxt
        ahead.shutdown()
        n = i0
        run_ball(n)
        if crops:
            flush()
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        np.savez(p1, n=np.array([len(b) for b in boxes]), boxes=np.concatenate(boxes), ball=np.array(balls, np.float32),
                 **({"court": np.array(checks), "kp": np.concatenate(kp)} if tv else {}))
        if on_pass1:
            on_pass1()

    # people: track, stitch, mark the still ones
    t = time.time()
    ids = tracks.track(boxes, fps, breaks=breaks)
    pp = tracks.people(boxes, ids, fps)
    still = {p for p, v in pp["people"].items() if v["still"]}
    tm["track"] = time.time() - t

    # pass 2 (phone: the court is not known yet): skeletons of tracked, non-still people
    if tv:
        posed = int(sum(np.isfinite(k[:, 0, 0]).sum() for k in kp))
    else:
        kp = [np.full((len(b), len(KP), 3), np.nan, np.float16) for b in boxes]
        want: dict[int, list[int]] = {}
        for i in range(n):
            for j in np.flatnonzero(ids[i] >= 0):
                if pp["person_of"][int(ids[i][j])] not in still:
                    want.setdefault(i, []).append(j)
        posed = sum(map(len, want.values()))
        if want:
            pose = (models or {}).get("pose") or Pose(pose_device)
            for i, f in enumerate(frames(video, limit)):
                js = want.get(i, [])
                crops.extend((i, j, c) for j, c in zip(js, pool.map(lambda j: pose.prepare(f, boxes[i][j]), js)))
                if len(crops) >= POSE_BATCH:
                    flush()
            if crops:
                flush()

    pool.shutdown()
    tm = {k: round(v, 1) for k, v in tm.items()} | {"total": round(time.time() - t0, 1)}
    np.savez_compressed(out, n=np.array([len(b) for b in boxes]), boxes=np.concatenate(boxes),
                        track=np.concatenate(ids), kp=np.concatenate(kp), ball=np.array(balls, np.float32),
                        fps=fps, width=W, height=H, seconds=json.dumps(tm),
                        **({"court": np.array(checks)} if tv else {}))
    return {"clip": Path(video).stem, "frames": n, "people": len(pp["people"]), "still": len(still),
            "posed": posed, "ball_found": int(sum(b[0] for b in balls)), "seconds": tm}


def load_models(tv: bool = True, device: str = "cuda", pose_device: str = "cuda") -> dict:
    """The models process() runs, loaded once (several runs can share them: their calls are thread-safe)."""
    from dsa.gpu.models import Ball, Detector, Pose, Tcd

    m = {"det": Detector(device), "ball": Ball(device)}
    if tv:
        m |= {"tcd": Tcd(device), "pose": Pose(pose_device)}
    return m


def plan(spec: str, seconds: float = 0) -> tuple[str, list[int], float]:
    """(analysed video's name, frames per piece, fps); the first `seconds` of it only, when given."""
    d = video_dir(spec)
    if (d / "kept.json").exists():
        k = json.loads((d / "kept.json").read_text())
        name, lengths, fps = "kept.mp4", [b - a for a, b in k["pieces"]], k["fps"]
    else:
        import cv2

        from dsa.video import info

        fps = info(str(d / "source.mp4"))[0]
        n = int(cv2.VideoCapture(str(d / "source.mp4")).get(cv2.CAP_PROP_FRAME_COUNT))
        name, lengths = "source.mp4", [n]
    if seconds:
        left, cut = int(seconds * fps), []
        for n in lengths:
            if left > 0:
                cut.append(min(n, left))
            left -= n
        lengths = cut
    return name, lengths, fps
