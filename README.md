# Deep Sports Analysis

Tennis video (phone or TV) → the two players over time (Player 1 / Player 2, near or far end), their
full skeletons, the ball, and their hits. Built only from off-the-shelf models (RF-DETR, WASB, RTMW,
TennisCourtDetector, F3ED) plus Opus for marking the court. How it works, costs and status:
[docs/PIPELINE.md](docs/PIPELINE.md).

The skeleton quiz (Skeleton Tennis, https://offpolicyinc.com/work/skeleton-tennis/) is built from this pipeline; where its
files are and how it was made: [docs/QUIZ.md](docs/QUIZ.md).

One video, step by step (atp_hardcourt in full, with GPUs and costs: [docs/ATP_PIPELINE.md](docs/ATP_PIPELINE.md)):

```
            package   step                      runs on       writes (projects/<project>/<video>/)
            ───────   ────                      ───────       ────────────────────────────────────
 YouTube ─► ingest    download, slice           laptop        source.mp4  (whole upload: raw/<project>/)
            gpu       0  cut: live play only    Modal T4      kept.mp4, kept.json
            gpu       2  models                 Modal L40S    raw.npz, strokes/
            scene     3  players at each end    laptop        players.json
            scene     4  names                  laptop, Opus  names.json
            pages        overlay renders        laptop        render.mp4, render_plain.mp4
            gpu       2b 50/60 fps skeletons    Modal L40S    fine.npz
            measure   5  shots, strokes, moves  laptop        <project>/shots.json, technique/, movement/
            measure   6  compare players        laptop        <project>/profiles/
            pages     7  player pages           laptop        <project>/review/

 batch.py runs every video of a project through steps 0-4 and 2b, each step as soon as its input exists.
```

```
src/dsa/
  ingest/    projects.py (the catalog: projects, videos, players), download.py (YouTube / local -> source.mp4)
  gpu/       everything that runs on a Modal GPU, never on the laptop
             app.py (the Modal app: cut, analyse, fine_pose), process.py (step 2's frame loop), models.py
             (RF-DETR, WASB, RTMW-x, TennisCourtDetector), strokes.py (F3ED), cuts.py (step 0, + cuts.html),
             fine.py (step 2b), volume.py (the Modal volume and image)
  scene/     steps 3-4 from raw.npz: run.py (entry), tracks, court, court_snap, players, identity, events,
             clean, names (Opus), results.py (a video's results, read once, and its points)
  measure/   step 5-6: shots, technique, movement, syllables (Modal T4), profiles (players compared)
  pages/     what people look at: render (overlay videos), quiz, technique_report, movement_report
             (+ movement_explorer.js), clips, exemplars, serve (local server)
  batch.py   every video of a project through every step
  paths.py   where everything lives, the same on the laptop and on Modal (docs/STORAGE.md)
  video.py, opus.py   decoding and cutting; cached Opus calls
tests/       one file per package, on made-up data (no models, no drive): uv run pytest
```

Setup: `uv sync --extra cloud`; Opus needs `ANTHROPIC_API_KEY`. Data lives
under `data/` (the external drive) or `DSA_DATA`; layout and Modal in [docs/STORAGE.md](docs/STORAGE.md).

License: MIT ([LICENSE](LICENSE)), for this repository's code. The models and code it downloads and runs
(RF-DETR, WASB, RTMW, TennisCourtDetector, F3Set / F3ED, keypoint-MoSeq) keep their own licenses.
