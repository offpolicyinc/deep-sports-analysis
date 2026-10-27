"""The off-the-shelf models, each behind a small wrapper. Nothing here is trained by us.

  Detector   RF-DETR Medium (Roboflow, Apache 2.0) at 1152 px: person boxes.
  Ball       WASB-SBDT (NTT, MIT) HRNet with its tennis weights, batched: frames t-1, t, t+1 at
             512 x 288, ImageNet-normalised, stacked to 9 channels -> the heaviest heatmap blob above
             0.5 in t, mapped back to the frame. Run without WASB's CUDA-only runner and its tracker.
  Pose       RTMW-x (MMPose, ONNX via rtmlib) at 384 x 288: 133 COCO-WholeBody points, of which KP
             (body 17, feet 6, hands 42) are kept.
  Tcd        TennisCourtDetector (yastrebksv, MIT): 14 court points on a broadcast frame, with its own
             line-intersection refinement and its homography fit (re-done here: its own breaks on NumPy 2).
"""
from __future__ import annotations

import sys

import cv2
import numpy as np

from dsa.paths import TCD, TCD_WEIGHTS, WASB

KP = list(range(23)) + list(range(91, 133))
RTMW_URL = ("https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/"
            "rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip")


class Detector:
    def __init__(self, device: str):
        import torch
        from rfdetr import RFDETRMedium
        from rfdetr.assets.coco_classes import COCO_CLASSES

        self.model = RFDETRMedium(resolution=1152)
        if device == "cuda":
            self.model.inference(compile=False, dtype=torch.float16)
        self.torch, self.device = torch, device
        self.scale = torch.tensor(255.0, device=device)
        self.person = next(i for i, n in COCO_CLASSES.items() if n == "person")

    def people(self, bgr, threshold: float = 0.2) -> list[np.ndarray]:
        """Per frame (n, 5): x1 y1 x2 y2 conf, for BGR frames (a list, or an (n, H, W, 3) uint8 array or tensor,
        which may already be on the device). They become RF-DETR's input on the device, exactly as its predict()
        would on the CPU (the same boxes, bit for bit): 1.8x faster on an L40S, where preparing 1080p frames on the
        CPU was the bottleneck."""
        torch = self.torch
        from rfdetr.detr import _uint8_chw_to_float

        x = bgr if torch.is_tensor(bgr) else torch.from_numpy(np.stack(bgr))
        if self.device == "cuda" and not x.is_cuda:
            x = x.pin_memory().cuda(non_blocking=True)
        x = _uint8_chw_to_float(x.flip(-1).permute(0, 3, 1, 2).contiguous(), self.scale)      # BGR HWC -> RGB CHW
        ds = self.model.predict(list(x), threshold=threshold, include_source_image=False)
        return [np.c_[d.xyxy[d.class_id == self.person], d.confidence[d.class_id == self.person]].astype(np.float32)
                for d in (ds if isinstance(ds, list) else [ds])]


class _AttrDict(dict):
    """The OmegaConf access WASB's HRNet expects: keys as attributes too."""

    __getattr__ = dict.__getitem__

    @classmethod
    def wrap(cls, v):
        return cls({k: cls.wrap(x) for k, x in v.items()}) if isinstance(v, dict) else v


class Ball:
    def __init__(self, device: str):
        import torch
        import yaml

        sys.path.insert(0, str(WASB / "src"))
        from models import build_model

        self.model = build_model({"model": _AttrDict.wrap(yaml.safe_load(open(WASB / "src/configs/model/wasb.yaml")))})
        state = torch.load(WASB / "pretrained_weights/wasb_tennis_best.pth.tar", map_location="cpu")["model_state_dict"]
        self.model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()})
        self.model.to(device).eval()
        self.torch, self.device = torch, device
        self.mean, self.std = np.float32([0.485, 0.456, 0.406]), np.float32([0.229, 0.224, 0.225])

    def small(self, bgr: np.ndarray) -> np.ndarray:
        return (cv2.resize(bgr[:, :, ::-1], (512, 288)).astype(np.float32) / 255 - self.mean) / self.std

    def locate(self, triplets: list[list[np.ndarray]], W: int, H: int) -> np.ndarray:
        """triplets of small() frames (t-1, t, t+1) -> per t: found, x, y, score."""
        torch = self.torch
        up = {}                                         # each frame to the GPU once, the triplets stacked there
        for t in triplets:
            for a in t:
                if id(a) not in up:
                    up[id(a)] = torch.from_numpy(a).to(self.device, non_blocking=True)
        x = torch.stack([torch.cat([up[id(a)] for a in t], 2).permute(2, 0, 1) for t in triplets])
        # half precision on the GPU: twice as fast, heatmaps within 0.001 of full precision
        with torch.no_grad(), torch.autocast("cuda", torch.float16, enabled=self.device == "cuda"):
            hm = self.model(x)[0][:, 1].sigmoid().float().cpu().numpy()
        out = np.zeros((len(hm), 4), np.float32)
        for i, m in enumerate(hm):
            if m.max() <= 0.5:
                out[i, 3] = m.max()
                continue
            n, lab = cv2.connectedComponents((m > 0.5).astype(np.uint8))
            best = max(range(1, n), key=lambda k: m[lab == k].sum())
            ys, xs = np.nonzero(lab == best)
            w = m[ys, xs]
            out[i] = (1, (xs * w).sum() / w.sum() * W / 512, (ys * w).sum() / w.sum() * H / 288, w.sum())
        return out


class Pose:
    def __init__(self, device: str = "cuda", fp16: bool = False):
        import torch  # noqa: F401  (loads the CUDA libraries onnxruntime's CUDA provider needs; without, it runs on the CPU)
        from rtmlib import RTMPose

        self.m = RTMPose(RTMW_URL, model_input_size=(288, 384), backend="onnxruntime", device=device)
        if device == "cuda" and fp16:           # 1.6x the crops per second on a 3090 Ti
            import onnx
            import onnxruntime as ort
            from onnxconverter_common import float16

            half = float16.convert_float_to_float16(onnx.load(self.m.onnx_model), keep_io_types=True)
            self.m.session = ort.InferenceSession(half.SerializeToString(), providers=["CUDAExecutionProvider"])
        shape = self.m.session.get_inputs()[0].shape
        self.batched = not isinstance(shape[0], int) or shape[0] != 1

    def prepare(self, bgr: np.ndarray, box: np.ndarray):
        img, center, scale = self.m.preprocess(bgr, box[:4])
        return img.transpose(2, 0, 1).astype(np.float32), center, scale

    def run(self, prepared: list) -> list[np.ndarray]:
        """prepare() results -> per crop (len(KP), 3): x, y, score."""
        sess = self.m.session
        name = sess.get_inputs()[0].name
        pixels = np.stack([p[0] for p in prepared])
        if self.batched:
            outs = sess.run(None, {name: pixels})
        else:
            each = [sess.run(None, {name: p[None]}) for p in pixels]
            outs = [np.concatenate([o[i] for o in each]) for i in range(len(each[0]))]
        res = []
        for k, (_, center, scale) in enumerate(prepared):
            xy, sc = self.m.postprocess([o[k:k + 1] for o in outs], center, scale)
            res.append(np.concatenate([xy[0, KP], sc[0, KP, None]], 1))
        return res


class Tcd:
    def __init__(self, device: str):
        import torch

        sys.path.insert(0, str(TCD))
        from tracknet import BallTrackerNet

        self.model = BallTrackerNet(out_channels=15)
        self.model.load_state_dict(torch.load(TCD_WEIGHTS, map_location="cpu"))
        self.model.to(device).eval()
        self.torch, self.device = torch, device

    def locate(self, bgr: np.ndarray) -> np.ndarray:
        """(14, 2) court points in pixels of a frame of any size, NaN where none is found."""
        from postprocess import postprocess, refine_kps

        h, w = bgr.shape[:2]
        img = bgr if (w, h) == (1280, 720) else cv2.resize(bgr, (1280, 720), interpolation=cv2.INTER_AREA)
        inp = cv2.resize(img, (640, 360)).astype(np.float32) / 255.0
        t = self.torch.from_numpy(inp.transpose(2, 0, 1)[None]).to(self.device)
        with self.torch.no_grad():
            pred = self.torch.sigmoid(self.model(t)[0]).cpu().numpy()
        pts = []
        for k in range(14):
            x, y = postprocess((pred[k] * 255).astype(np.uint8), low_thresh=170, max_radius=25)
            if k not in (8, 9, 12) and x and y:
                x, y = refine_kps(img, int(y), int(x))
            pts.append((np.nan, np.nan) if x is None else (x, y))
        pts = np.array(pts, float)
        m = _fit_court(pts)
        if m is not None:
            from dsa.scene.court import REF
            pts = cv2.perspectiveTransform(np.float32(REF)[:, None], m).reshape(14, 2)
        return pts * np.array([w / 1280, h / 720])


def _fit_court(pts: np.ndarray) -> np.ndarray | None:
    """TCD's homography step: fit each of its 12 four-point court configurations and keep the one whose
    projection lies closest, on average, to the other detected points."""
    from homography import court_conf_ind

    from dsa.scene.court import REF

    best, best_d = None, np.inf
    for inds in court_conf_ind.values():
        if np.isnan(pts[inds]).any():
            continue
        m, _ = cv2.findHomography(np.float32(REF)[inds], np.float32(pts[inds]), method=0)
        if m is None:
            continue
        proj = cv2.perspectiveTransform(np.float32(REF)[:, None], m).reshape(14, 2)
        others = [i for i in range(12) if i not in inds and not np.isnan(pts[i]).any()]
        d = np.mean(np.linalg.norm(proj[others] - pts[others], axis=1)) if others else np.inf
        if d < best_d:
            best, best_d = m, d
    return best
