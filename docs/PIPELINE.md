# Pipeline

Tennis video in → players, skeletons, ball, and every hit with its stroke type. Off-the-shelf models only,
plus Opus. Where files live: [STORAGE.md](STORAGE.md).

## Terms

- **recording**: what we downloaded (TV: the first hour), `source.mp4`
- **kept video**: the frames step 0 keeps from the recording, `kept.mp4` (2015: 26.5 min)
- **test run**: the first 60 s of the kept video (`--seconds 60`), into `test/`

## Steps

| # | Step | Code | Runs on | Status |
|---|---|---|---|---|
| 0 | Keep live play (TV) | `dsa.gpu.cuts` (`app.py --cuts`) | Modal | done |
| 1 | Drop replays (TV) | – | laptop | none in 2015: no repeated frames, no slow hit spacing |
| 2 | All GPU models: hits + strokes, people, ball, court, skeletons | `dsa.gpu.app`, `dsa.gpu.process` | Modal | done on 2015 |
| 3 | Near/far player, P1/P2, contacts | `dsa.scene.run` | laptop | done on 2015 |
| 4 | Which player is which (Federer, or the two listed) | `dsa.scene.names` | laptop (Opus) | done |
| 5 | Stroke and movement measurements | `dsa.measure.technique`, `dsa.measure.movement`, `dsa.measure.syllables` | laptop (syllables: Modal) | done |
| 6 | Compare: one player's eras, or players with each other | `dsa.measure.technique analyse`, `dsa.measure.profiles` | laptop | done; profiles new |

Phone footage skips 0–1, and step 2 skips F3ED (unless the project sets `strokes`).

Step 2 also runs on Google Cloud (`dsa.gpu.gcp`, project wc3env-pw): one VM with an L4 takes a list of videos, two side by side, through the bucket gs://wc3env-pw-dsa (laid out as the data root), then deletes itself. The project may have one GPU at a time (raising it was refused in Oct 2026), so it suits a few videos, not a batch: phone footage runs at about a quarter of real time per video, two at once (5 min in 20 min).

## What each step does

0. **Keep live play.** Split at camera cuts. Keep a shot if the court detector sees the whole court from the
   main camera (level baselines, centred, seen from high up). The kept frames are saved as `kept.mp4`;
   everything after works on it. Review page: `dsa.gpu.cuts review`.
1. **Drop replays.** Slow-motion replays from the main camera would pass step 0. Two checks: repeated frames,
   and hits inside a rally more than ~2.5 s apart (2015: longest 2.3 s, 99% under 2.0 s).
2. **All GPU models, one GPU per video, on `kept.mp4`:** F3ED (hits, and for each: near/far,
   serve/return/stroke, forehand/backhand, technique, direction, outcome), RF-DETR (people), WASB (ball),
   TennisCourtDetector (court), ByteTrack (tracking, restarted at each cut), RTMW-x (65-point skeletons).
   Crowd who never move are skipped.
3. **Players.** In each half, the moving person nearest the singles court is the player (not the line judges
   or ball kids further out). P1/P2 are told apart by clothing colour, which carries across changes of end.
   Phone courts are marked by Opus.
4. **Names.** Opus sees 6 crops of each player (near end where possible) and names both, once per video:
   which is the project's player (Federer), or which is which of the two listed for the match.
5. **Measurements.** Per near-court stroke: contact point, body turn, wrist speed, footwork; per point: his
   movement on the court, and keypoint-MoSeq syllables of his whole body.
6. **Compare.** Federer: his eras against each other, his opponents as the control (`dsa.measure.technique analyse`).
   atp_hardcourt: each player against the others, matches as the unit (`dsa.measure.profiles`).

## Cost

- Step 2: about $2.20 per kept video (2015: 26.5 min, 50 GPU-min, $2.15 billed). atp_hardcourt poses the
  near end only and checks the court every 10 frames (`gpu` in its project entry): about a third less.
- Step 0 on Modal: one T4 per video for the cut detection and court checks.
- No model ever runs on the laptop; it does the rest (tracking, players, names via Opus, measurements).

## Known problems

- F3ED sees only the centre 56% of the frame.
- F3ED calls a serve a forehand when the camera cuts in just before it (2015, 1:28).
- Far-end player is ~70 px tall: fine for footwork, too small for swing detail.
- P1/P2 is wrong on ~0.4% of hits in 2015 (a few short track pieces put with the other player).
- 2003–2006 are standard definition: recheck step 0's settings there.
- F3ED misses most hits at 50/60 fps (a 60 fps US Open upload: 112 hits in 12 min, half called serves), so
  `dsa.ingest.download get` halves such uploads to 25/30 fps.
- Outer-court main cameras sit lower than step 0's main-camera rule allows (far baseline 0.32–0.37 of the
  near one, against 0.4); a project can lower it per video (`min_depth` in `dsa.ingest.projects`).

## YouTube downloads

What works, and what YouTube's errors mean (no browser cookies: a logged-in session got flagged before):

- **Setup.** yt-dlp with its extras (`yt-dlp[default]`: the `yt-dlp-ejs` challenge solver) and a JavaScript
  runtime (`--js-runtimes node:<path>`, `dsa.ingest.download.JS`). Keep both current; check against PyPI.
- **Whole uploads, 8 fragments at a time** (`-N 8`): tens of MB/s. Reading a window from the stream URL with
  ffmpeg, or `--download-sections`, runs at about real time.
- **"Sign in to confirm you're not a bot"**: a limit per IP address after a burst of requests to YouTube's
  pages (searches, `-g` stream lookups, `start`, downloads: about 50-100 within half an hour). Cookies don't
  fix it for long. Stop, wait (it lifts after a while, typically within hours), then one download at a time
  with a pause between them (`dsa.batch`: GET at a time, GET_GAP seconds apart).
- **"n challenge solving failed" / "The page needs to be reloaded"**: yt-dlp or its solver is out of date
  (or, with cookies, the session is flagged). Upgrade `yt-dlp[default]` and retry without cookies.

## Commands

A video is `<project>/<video>`, e.g. `federer_usopen/2015` (list: `python -m dsa.ingest.download list`).

```
python -m dsa.ingest.download get federer_usopen/2015                  # download (TV: the recording is the first hour)
modal run src/dsa/gpu/app.py --videos federer_usopen/2015 --cuts   # 0 on Modal, brings back kept.mp4
python -m dsa.gpu.cuts review federer_usopen/2015                   # check 0 by eye
modal run --detach src/dsa/gpu/app.py --videos federer_usopen/2015  # 2; again afterwards to collect
python -m dsa.gpu.gcp run others/2026_e,others/2026_g        # 2 on GCP instead; again afterwards to collect
python -m dsa.gpu.strokes federer_usopen/2015 --preview             # video with the stroke labels
python -m dsa.scene.run federer_usopen/2015                           # 3 (--test: on the test run)
python -m dsa.scene.names federer_usopen/2015                         # 4 (--test: on the test run)
python -m dsa.pages.render federer_usopen/2015                        # overlay video (--seconds 60: first minute)
python -m dsa.pages.render federer_usopen/2015 --plain                # the models' output alone, on black
python -m dsa.batch atp_hardcourt                               # 0-4 for every video of a project, restartable
python -m dsa.measure.shots atp_hardcourt                               # every hit by a named player
python -m dsa.measure.technique extract atp_hardcourt forehand          # 5 (then analyse: federer_usopen's eras)
python -m dsa.measure.profiles atp_hardcourt                            # 6: each player against the others
python -m dsa.pages.quiz points all                                   # quiz: a plain clip per 6+ shot point
python -m dsa.pages.quiz export ../offpolicysite/media/skeleton-tennis   # quiz: clips + manifest.json for the game
```
