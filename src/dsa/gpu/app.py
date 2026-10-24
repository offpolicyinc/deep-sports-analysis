"""The Modal app: the GPU steps, one container per video, results back to the laptop. Code, weights and the
video go up to the volume once (dsa.gpu.volume); parts already there are skipped. TV is assumed when kept.json
exists.

  cut        step 0 (dsa.gpu.cuts) on a T4: kept.mp4, kept.json, cuts/shots.json back
  analyse    step 2 (dsa.gpu.process, dsa.gpu.strokes) on an L40S: raw.npz, strokes/ back
  fine_pose  step 2b (dsa.gpu.fine) on an L40S, after `python -m dsa.gpu.fine clip`: fine.npz back

    modal run --detach src/dsa/gpu/app.py --videos atp_hardcourt/x --cuts    # step 0
    modal run --detach src/dsa/gpu/app.py --videos federer_usopen/2015       # step 2; again afterwards: collects
    modal run src/dsa/gpu/app.py --videos federer_usopen/2015 --seconds 60   # test run: first 60 s, into test/
                                                                        # (--defaults: without the project's gpu options)
    modal run src/dsa/gpu/app.py --videos tests/broadcast_usopen2015_final --tv
    modal run --detach src/dsa/gpu/app.py --videos atp_hardcourt/x --fine    # step 2b
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from dsa.gpu.process import plan, process
from dsa.ingest.projects import PROJECTS
from dsa.paths import ROOT, video_dir

try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    from dsa.gpu.volume import image, listing, pull, push, volume

    # the model steps (analyse, fine_pose). 60 s of kept video, all-in price (GPU + cores + memory): L40S with 6
    # cores, 24 GB: 55 s at 2.43 $/h; with 8 cores, 32 GB: 55 s at 2.58; 4 cores, 16 GB: 65 s at 2.27; an L4 with
    # 8 cores, 32 GB: 122 s at 1.43 (two runs side by side: 101 s each). The people detector is GPU-bound.
    GPU = os.environ.get("DSA_GPU", "L40S")
    app = modal.App("deep-sports-gpu", image=image("onnxruntime-gpu==1.30.0", "timm==0.6.13", "tabulate==0.9.0"))

    @app.function(gpu=GPU, volumes={"/vol": volume}, cpu=6, memory=24_576, timeout=4 * 60 * 60, max_containers=9)
    def analyse(spec: str, name: str, lengths: list[int], fps: float, out: str, tv: bool, strokes: bool,
                people: bool, defaults: bool = False) -> dict:
        return models(spec, name, lengths, fps, out, tv, strokes, people, defaults, volume.commit)

    @app.function(gpu="T4", volumes={"/vol": volume}, cpu=8, memory=16_384, timeout=2 * 60 * 60, max_containers=9)
    def cut(spec: str) -> dict:
        """TV: dsa.gpu.cuts on the volume's source.mp4 (TennisCourtDetector on the GPU, not the laptop's) -> kept.mp4,
        kept.json, cuts/shots.json."""
        from dsa.gpu import cuts

        t = time.time()
        m = cuts.run(spec, "cuda")
        volume.commit()
        keep = sum(r["end_frame"] - r["start_frame"] for r in m["shots"] if r["label"] == "keep")
        return {"kept_min": round(keep / m["fps"] / 60, 1), "of_min": round(m["frames"] / m["fps"] / 60, 1),
                "cut_s": round(time.time() - t)}

    def cut_all(videos: list[str]) -> None:
        """dsa.gpu.cuts on Modal for each video without kept.json, then kept.mp4 / kept.json / cuts/shots.json back
        (kept.mp4 stays on the volume for the GPU step)."""
        jobs = [v for v in videos if not (video_dir(v) / "kept.json").exists()]
        rels = {v: video_dir(v).relative_to(ROOT) for v in jobs}
        print(f"{len(jobs)} to cut: {jobs}", flush=True)
        if not jobs:
            return
        push(["code/tennis_court_detector", "models/tcd"] + [str(rels[v] / "source.mp4") for v in jobs])
        for v, r in zip(jobs, cut.map(jobs)):
            for f in ("cuts/shots.json", "kept.mp4", "kept.json"):     # kept.json last: it marks the step done
                pull(str(rels[v] / f))
            print(v, r, flush=True)

    @app.function(gpu=GPU, volumes={"/vol": volume}, cpu=6, memory=16_384, timeout=2 * 60 * 60, max_containers=9)
    def fine_pose(spec: str) -> dict:
        """Step 2b (dsa.gpu.fine): people and the tracked near player's skeleton on every frame of fine.mp4."""
        import onnxruntime as ort
        import torch  # noqa: F401  (CUDA libraries for onnxruntime)

        from dsa.gpu.fine import pose

        ort.preload_dlls()
        t = time.time()
        r = pose(spec)
        volume.commit()
        return r | {"fine_s": round(time.time() - t)}

    def fine_all(videos: list[str]) -> None:
        """Step 2b on Modal for each video with fine.json (dsa.gpu.fine clip) and no fine.npz; fine.npz back."""
        jobs = [v for v in videos if (video_dir(v) / "fine.json").exists() and not (video_dir(v) / "fine.npz").exists()]
        rels = {v: video_dir(v).relative_to(ROOT) for v in jobs}
        print(f"{len(jobs)} to pose at full frame rate: {jobs}", flush=True)
        if not jobs:
            return
        push([str(rels[v] / f) for v in jobs for f in ("fine.mp4", "fine.json")])
        for v, r in zip(jobs, fine_pose.map(jobs)):
            pull(str(rels[v] / "fine.npz"))
            print(v, r, flush=True)

    @app.local_entrypoint()
    def main(videos: str, tv: bool = False, seconds: float = 0, cuts: bool = False,
             defaults: bool = False, fine: bool = False):
        """Run what the volume lacks, then bring results back. Run with `modal run --detach` so a laptop that
        drops off the network doesn't stop it; once the run has finished (modal app list), the same command
        again just brings the results back."""
        if cuts:
            return cut_all(videos.split(","))
        if fine:
            return fine_all(videos.split(","))
        out = "test" if seconds else "."                          # a test never counts as done
        jobs, todo = plan_all(videos.split(","), seconds, tv, listing)
        print(f"{len(jobs)} to run: {[(j[0], 'strokes' * j[6], 'people' * j[7]) for j in jobs]}", flush=True)
        if jobs:
            push(inputs(jobs))
            for i, r in enumerate(analyse.starmap([(j[0], j[2], j[3], j[4], out, j[5], j[6], j[7], defaults)
                                                   for j in jobs])):
                print(jobs[i][0], r, flush=True)
        for v, (lengths, fps) in todo.items():
            print(v, "collected", collect(v, out, lengths, fps, listing, pull), flush=True)


def models(spec: str, name: str, lengths: list[int], fps: float, out: str, tv: bool, strokes: bool, people: bool,
           defaults: bool = False, commit=lambda: None) -> dict:
    """Step 2 on this machine's GPU, under the data root (Modal: the volume, committed by `commit`; dsa.gpu.gcp: the
    VM's disk): F3ED's scores (strokes/scores.npz) and dsa.gpu.process's raw.npz in video_dir(spec)/out."""
    import shutil

    import onnxruntime as ort
    import torch  # noqa: F401  (CUDA libraries for onnxruntime)

    from dsa.gpu.strokes import f3ed

    d, o = video_dir(spec), video_dir(spec) / out
    opts = {} if defaults else PROJECTS[spec.split("/")[0]].get("gpu", {})
    # read from the container's own disk: nothing on the volume stays open while results are committed
    video, r = shutil.copy(d / name, f"/tmp/{spec.replace('/', '_')}_{name}"), {}
    if strokes:
        t = time.time()
        sc = f3ed(video, lengths, fps)
        (o / "strokes").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(o / "strokes/scores.npz", stride=sc["stride"],
                            **{f"hit_{k}": h for k, h in enumerate(sc["hit"])},
                            **{f"elements_{k}": e for k, e in enumerate(sc["elements"])})
        commit()
        r["strokes_s"] = round(time.time() - t)
    if people:
        ort.preload_dlls()
        r |= process(video, str(o / "raw.npz"), tv=tv, on_pass1=commit, limit=sum(lengths),
                     breaks=np.cumsum(lengths)[:-1].tolist(), **opts)
        commit()
    return r


def plan_all(videos: list[str], seconds: float, tv: bool, listing) -> tuple[list[tuple], dict]:
    """The step 2 jobs `videos` still need, judged by what the remote store has (listing: dsa.gpu.volume's or
    dsa.gpu.gcp's): (spec, rel, name, lengths, fps, tv, strokes, people) each, and {spec: (lengths, fps)}."""
    jobs, todo = [], {}
    for v in videos:
        rel = video_dir(v).relative_to(ROOT)
        name, lengths, fps = plan(v, seconds)
        todo[v] = (lengths, fps)
        is_tv = tv or name == "kept.mp4"
        have = set() if seconds else listing(str(rel))
        on = is_tv or PROJECTS[v.split("/")[0]].get("strokes", False)   # F3ED: TV, or a project that asks
        s = on and str(rel / "strokes/scores.npz") not in have
        p = str(rel / "raw.npz") not in have
        if s or p:
            jobs.append((v, rel, name, lengths, fps, is_tv, s, p))
    return jobs, todo


def inputs(jobs: list[tuple]) -> list[str]:
    """What the jobs read from the remote store: code, weights and each video."""
    return (["code/f3set", "models/f3ed"] * any(j[6] for j in jobs) + ["code/wasb"] * any(j[7] for j in jobs)
            + ["code/tennis_court_detector", "models/tcd"] * any(j[7] and j[5] for j in jobs)
            + [str(j[1] / j[2]) for j in jobs])


def collect(v: str, out: str, lengths: list[int], fps: float, listing, pull) -> dict:
    """Bring back what the remote store has for `v` and the laptop does not; decode F3ED's scores."""
    from dsa.gpu.strokes import decode

    o, d, got = video_dir(v).relative_to(ROOT) / out, video_dir(v) / out, {}
    have = listing(str(o))
    if str(o / "raw.npz") in have and not (d / "raw.npz").exists():
        got["raw"] = str(pull(str(o / "raw.npz")))
    if str(o / "strokes/scores.npz") in have and not (d / "strokes/strokes.json").exists():
        z = np.load(pull(str(o / "strokes/scores.npz")))
        sc = {"stride": int(z["stride"]), "hit": [z[f"hit_{k}"] for k in range(len(lengths))],
              "elements": [z[f"elements_{k}"] for k in range(len(lengths))]}
        kj = video_dir(v) / "kept.json"
        kept = json.loads(kj.read_text()) if kj.exists() else {"fps": fps, "pieces": [[0, lengths[0]]]}
        hits = decode(sc, kept)
        (d / "strokes/strokes.json").write_text(json.dumps({"fps": fps, "frames": sum(lengths), "hits": hits},
                                                          indent=1))
        got["hits"] = len(hits)
    return got
