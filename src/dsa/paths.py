"""Where everything lives. One layout under one root, the same on the laptop and on Modal, so a file keeps its
path when it moves between them (docs/STORAGE.md).

    code/<repo>/                    third-party code: wasb (with its tennis weights), tennis_court_detector, f3set
    models/<model>/                 weights: tcd/, f3ed/, rfdetr/ (RF-DETR's download cache)
    projects/<project>/<video>/     one folder per video; projects and their videos are listed in dsa.ingest.projects
        source.mp4                  the video as analysed (TV: its first hour)
        cuts/                       dsa.gpu.cuts: shots.json, review.json, thumbs/, proxy.mp4
        raw.npz                     dsa.gpu.process: people, ball, court, skeletons (raw.pass1.npz while it runs)
        strokes/                    dsa.gpu.strokes: input.mp4 (224 high), scores.npz, strokes.json; labels.json
                                    (dsa.ingest.labels: SwingVision's or by eye, read in place of strokes.json)
        court/                      dsa.scene.court: Opus court answers (phone)
        players.json                dsa.scene.run
        names.json, render.mp4      dsa.scene.names, dsa.pages.render

Root: DSA_DATA, else data/ (the external drive) on the laptop; /vol (the deep-sports volume) on Modal, where
DSA_DATA is set to it.
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("DSA_DATA", "data"))
CODE, MODELS, PROJECTS = ROOT / "code", ROOT / "models", ROOT / "projects"
WASB, TCD, F3SET = CODE / "wasb", CODE / "tennis_court_detector", CODE / "f3set"
TCD_WEIGHTS = MODELS / "tcd/model_tennis_court_det.pt"
F3ED = MODELS / "f3ed"


def video_dir(spec: str) -> Path:
    """'federer_usopen/2015' -> projects/federer_usopen/2015/."""
    project, video = spec.split("/")
    return PROJECTS / project / video


def hits_file(d: Path) -> Path:
    """A video folder's hits: strokes/labels.json (SwingVision's, or by eye) when there is one, else F3ED's."""
    return d / "strokes/labels.json" if (d / "strokes/labels.json").exists() else d / "strokes/strokes.json"
