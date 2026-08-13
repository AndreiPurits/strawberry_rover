from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import logging
from PIL import Image as PILImage
from torchvision import models as tv_models
from torchvision import transforms
from ultralytics import YOLO

_LOG = logging.getLogger("strawberry_ensemble")


RipenessClass = Literal["green", "turning", "ripe", "rotten"]


@dataclass(frozen=True)
class Detection:
    bbox_xyxy: Tuple[int, int, int, int]
    detector_conf: float


@dataclass(frozen=True)
class InstanceResult:
    bbox_xyxy: Tuple[int, int, int, int]
    detector_conf: float
    ripeness_class: RipenessClass
    classifier_conf: float
    mask_fullres: Optional[np.ndarray]  # uint8 {0,255}, HxW
    contour_fullres: Optional[np.ndarray]  # Nx1x2 int32
    distance_m: Optional[float]
    center_xy: Tuple[int, int]
    mass_g: Optional[float] = None


@dataclass
class PipelineTimings:
    detector_ms: float = 0.0
    classifier_ms: float = 0.0
    segmentation_ms: float = 0.0
    depth_ms: float = 0.0
    total_ms: float = 0.0
    fps: float = 0.0


@dataclass
class PipelineStats:
    det_frames_total: int = 0
    det_positive_frames: int = 0
    cls_calls: int = 0
    seg_calls: int = 0
    cls_skipped_frames: int = 0
    seg_skipped_frames: int = 0


@dataclass(frozen=True)
class EnsembleConfig:
    detector_weights: str
    classifier_weights: str
    segmenter_weights: str
    device: str = "cuda"
    detector_imgsz: int = 640
    segmenter_imgsz: int = 384
    detector_conf: float = 0.35
    detector_iou: float = 0.6
    max_detections: int = 20
    max_rois: int = 8
    # Segmentation can be a major latency source; keep defaults identical to previous behavior.
    segment_every_n: int = 1  # run segmentation once per N frames (1 = every frame)
    segment_max_rois: int = 8  # run segmentation only for top-K detections (by conf)
    segment_min_det_conf: float = 0.0  # skip segmentation unless det_conf >= threshold
    crop_pad_frac: float = 0.15
    depth_sync_slop_s: float = 0.030
    mask_alpha: float = 0.35


def _clamp_xyxy(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x1c = max(0, min(w - 1, x1))
    y1c = max(0, min(h - 1, y1))
    x2c = max(0, min(w - 1, x2))
    y2c = max(0, min(h - 1, y2))
    if x2c <= x1c:
        x2c = min(w - 1, x1c + 1)
    if y2c <= y1c:
        y2c = min(h - 1, y1c + 1)
    return x1c, y1c, x2c, y2c


def _pad_xyxy(x1: int, y1: int, x2: int, y2: int, pad_frac: float, w: int, h: int) -> Tuple[int, int, int, int]:
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    px = int(round(bw * pad_frac))
    py = int(round(bh * pad_frac))
    return _clamp_xyxy(x1 - px, y1 - py, x2 + px, y2 + py, w=w, h=h)


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _load_classifier_ckpt(weights_path: str) -> Dict:
    ckpt = torch.load(weights_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise RuntimeError("Classifier checkpoint must be a dict")
    required = ("model_name", "state_dict")
    for k in required:
        if k not in ckpt:
            raise RuntimeError(f"Classifier checkpoint missing key '{k}'")
    return ckpt


def _make_classifier_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name == "mobilenet_v3_small":
        m = tv_models.mobilenet_v3_small(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m
    if model_name == "resnet18":
        m = tv_models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if model_name == "efficientnet_b0":
        m = tv_models.efficientnet_b0(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m
    raise ValueError(model_name)


def _build_classifier_eval_tf(img_size: int) -> transforms.Compose:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.15)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize,
        ]
    )


class RipenessClassifier:
    def __init__(self, weights_path: str, device: str) -> None:
        ckpt = _load_classifier_ckpt(weights_path)
        self.classes: List[str] = list(ckpt.get("classes") or ["green", "turning", "ripe", "rotten"])
        self.img_size: int = int(ckpt.get("img_size") or 224)
        model_name = str(ckpt.get("model_name") or "")
        state = ckpt.get("state_dict")
        if not model_name or not isinstance(state, dict):
            raise RuntimeError("Invalid classifier checkpoint: expected model_name and state_dict")

        cuda_ok = bool(torch.cuda.is_available())
        if str(device) == "cuda" and not cuda_ok:
            _LOG.warning("Classifier requested cuda but torch.cuda.is_available=False -> using CPU")
        dev = torch.device(device if (device != "cuda" or cuda_ok) else "cpu")
        self.device = dev
        self.model = _make_classifier_model(model_name, num_classes=len(self.classes))
        self.model.load_state_dict(state)
        self.model.to(dev)
        self.model.eval()
        self.tf = _build_classifier_eval_tf(self.img_size)
        _LOG.info(f"Classifier device={str(self.device)} model={model_name} img_size={int(self.img_size)}")

    @torch.no_grad()
    def infer_crop_bgr(self, crop_bgr: np.ndarray) -> Tuple[RipenessClass, float]:
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil = PILImage.fromarray(crop_rgb)
        x = self.tf(pil).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
        idx = int(np.argmax(probs))
        cls = str(self.classes[idx])
        conf = float(probs[idx])
        return cls, conf  # type: ignore[return-value]

    @torch.no_grad()
    def infer_crops_bgr(self, crops_bgr: List[np.ndarray]) -> List[Tuple[RipenessClass, float]]:
        if not crops_bgr:
            return []
        xs: List[torch.Tensor] = []
        for crop_bgr in crops_bgr:
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            pil = PILImage.fromarray(crop_rgb)
            xs.append(self.tf(pil))
        x = torch.stack(xs, dim=0).to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()  # (N,C)
        out: List[Tuple[RipenessClass, float]] = []
        for p in probs:
            idx = int(np.argmax(p))
            out.append((str(self.classes[idx]), float(p[idx])))  # type: ignore[arg-type]
        return out


class YoloDetector:
    def __init__(self, weights_path: str, device: str, imgsz: int, conf: float, iou: float, max_det: int) -> None:
        self.model = YOLO(weights_path)
        self.device = device
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.max_det = int(max_det)
        if "cuda" in str(device) and not bool(torch.cuda.is_available()):
            _LOG.warning("Detector requested cuda but torch.cuda.is_available=False -> ultralytics may fall back to CPU")
        _LOG.info(f"Detector device={str(device)} imgsz={int(self.imgsz)} conf={float(self.conf)} iou={float(self.iou)} max_det={int(self.max_det)}")

    def infer(self, frame_bgr: np.ndarray) -> List[Detection]:
        res = self.model.predict(
            source=frame_bgr,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            max_det=self.max_det,
            verbose=False,
        )[0]
        out: List[Detection] = []
        if res.boxes is None or res.boxes.xyxy is None:
            return out
        xyxy = res.boxes.xyxy.detach().cpu().numpy()
        confs = res.boxes.conf.detach().cpu().numpy()
        for (x1, y1, x2, y2), c in zip(xyxy, confs):
            out.append(
                Detection(
                    bbox_xyxy=(int(x1), int(y1), int(x2), int(y2)),
                    detector_conf=float(c),
                )
            )
        return out


class YoloSegmenterRoi:
    def __init__(self, weights_path: str, device: str, imgsz: int, conf: float = 0.25, iou: float = 0.7) -> None:
        self.model = YOLO(weights_path)
        self.device = device
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        if "cuda" in str(device) and not bool(torch.cuda.is_available()):
            _LOG.warning("Segmenter requested cuda but torch.cuda.is_available=False -> ultralytics may fall back to CPU")
        _LOG.info(f"Segmenter device={str(device)} imgsz={int(self.imgsz)} conf={float(self.conf)} iou={float(self.iou)}")

    def infer_mask_on_crop(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        res = self.model.predict(
            source=crop_bgr,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )[0]
        if res.masks is None or res.boxes is None:
            return None
        if res.masks.data is None or res.boxes.conf is None:
            return None
        masks = res.masks.data.detach().cpu().numpy()  # (N, H, W) float
        confs = res.boxes.conf.detach().cpu().numpy()  # (N,)
        if masks.size == 0 or confs.size == 0:
            return None
        best_i = int(np.argmax(confs))
        m = masks[best_i]
        m = (m > 0.5).astype(np.uint8) * 255
        return m


class DepthEstimator:
    def __init__(self) -> None:
        pass

    def estimate_distance_m(self, depth: Optional[np.ndarray], mask_fullres: Optional[np.ndarray]) -> Optional[float]:
        if depth is None or mask_fullres is None:
            return None
        if depth.ndim != 2:
            return None
        if mask_fullres.ndim != 2 or mask_fullres.shape != depth.shape:
            return None

        if depth.dtype == np.uint16:
            depth_m = depth.astype(np.float32) * 0.001
        else:
            depth_m = depth.astype(np.float32)

        sel = mask_fullres > 0
        if not bool(np.any(sel)):
            return None

        vals = depth_m[sel]
        vals = vals[np.isfinite(vals)]
        vals = vals[(vals > 0.05) & (vals < 20.0)]
        if vals.size == 0:
            return None
        return float(np.median(vals))


class StrawberryEnsemblePipeline:
    def __init__(self, cfg: EnsembleConfig) -> None:
        self.cfg = cfg
        try:
            cuda_ok = bool(torch.cuda.is_available())
            msg = f"Pipeline init device={str(cfg.device)} torch.cuda.is_available={cuda_ok}"
            if cuda_ok:
                try:
                    msg += f" cuda_device={int(torch.cuda.current_device())} name={str(torch.cuda.get_device_name(torch.cuda.current_device()))}"
                except Exception:
                    pass
            _LOG.info(msg)
        except Exception:
            pass
        self.detector = YoloDetector(
            weights_path=cfg.detector_weights,
            device=cfg.device,
            imgsz=cfg.detector_imgsz,
            conf=cfg.detector_conf,
            iou=cfg.detector_iou,
            max_det=cfg.max_detections,
        )
        self.classifier: Optional[RipenessClassifier] = None
        self.segmenter: Optional[YoloSegmenterRoi] = None
        if str(cfg.classifier_weights).strip():
            self.classifier = RipenessClassifier(cfg.classifier_weights, device=cfg.device)
        if str(cfg.segmenter_weights).strip():
            self.segmenter = YoloSegmenterRoi(cfg.segmenter_weights, device=cfg.device, imgsz=cfg.segmenter_imgsz)
        self.depth = DepthEstimator()

        self._last_frame_ms: Optional[float] = None
        self._frame_i: int = 0
        self.timings = PipelineTimings()
        self.stats = PipelineStats()
        self._last_cls_by_bbox: Dict[Tuple[int, int, int, int], Tuple[RipenessClass, float]] = {}
        self._last_mask_by_bbox: Dict[Tuple[int, int, int, int], Optional[np.ndarray]] = {}
        # Simple last-result cache (used on skip frames).
        self._last_cls: Optional[Tuple[RipenessClass, float]] = None
        self._last_mask: Optional[np.ndarray] = None  # uint8 {0,255}, HxW
        # HUD stats filled by runtime (RGB/Depth input fps).
        self._hud_rgb_fps: float = 0.0
        self._hud_depth_fps: float = 0.0

    def set_input_fps(self, *, rgb_fps: float, depth_fps: float) -> None:
        self._hud_rgb_fps = float(rgb_fps)
        self._hud_depth_fps = float(depth_fps)

    def infer(self, frame_bgr: np.ndarray, depth_aligned: Optional[np.ndarray]) -> List[InstanceResult]:
        self._frame_i += 1
        self.stats.det_frames_total += 1
        t0 = _now_ms()
        h, w = frame_bgr.shape[:2]

        t_det0 = _now_ms()
        detections = self.detector.infer(frame_bgr)
        t_det1 = _now_ms()

        results: List[InstanceResult] = []
        cls_ms = 0.0
        seg_ms = 0.0
        dep_ms = 0.0

        detections = sorted(detections, key=lambda d: float(d.detector_conf), reverse=True)
        detections = detections[: max(0, int(self.cfg.max_rois))]
        if detections:
            self.stats.det_positive_frames += 1

        # Decide whether we run segmentation on this frame.
        seg_every_n = max(1, int(self.cfg.segment_every_n))
        do_seg_this_frame = (self._frame_i % seg_every_n) == 0
        seg_max_rois = max(0, int(self.cfg.segment_max_rois))
        seg_min_det_conf = float(self.cfg.segment_min_det_conf)

        crops: List[Optional[np.ndarray]] = []
        pads: List[Tuple[int, int, int, int]] = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox_xyxy
            x1p, y1p, x2p, y2p = _pad_xyxy(x1, y1, x2, y2, self.cfg.crop_pad_frac, w=w, h=h)
            crop = frame_bgr[y1p:y2p, x1p:x2p]
            if crop.size == 0:
                crops.append(None)
                pads.append((x1p, y1p, x2p, y2p))
                continue
            crops.append(crop)
            pads.append((x1p, y1p, x2p, y2p))

        # Classifier every 8th frame (only if detector found something).
        do_cls_this_frame = (self.classifier is not None) and ((self._frame_i % 8) == 0)
        cls_by_i: Dict[int, Tuple[RipenessClass, float]] = {}
        if do_cls_this_frame:
            valid_idx = [i for i, c in enumerate(crops) if c is not None]
            valid_crops = [crops[i] for i in valid_idx if crops[i] is not None]  # type: ignore[list-item]
            cls_out: List[Tuple[RipenessClass, float]] = []
            if valid_crops:
                t_cls0 = _now_ms()
                assert self.classifier is not None
                cls_out = self.classifier.infer_crops_bgr(valid_crops)
                t_cls1 = _now_ms()
                cls_ms += (t_cls1 - t_cls0)
                self.stats.cls_calls += 1
            for j, i in enumerate(valid_idx):
                cls_by_i[i] = cls_out[j]
        else:
            if self.classifier is not None:
                self.stats.cls_skipped_frames += 1

        for i, det in enumerate(detections):
            crop = crops[i]
            x1, y1, x2, y2 = det.bbox_xyxy
            x1p, y1p, x2p, y2p = pads[i]
            if crop is None:
                continue
            bbox_key = (x1, y1, x2, y2)
            if do_cls_this_frame and i in cls_by_i:
                ripeness_cls, ripeness_conf = cls_by_i[i]
                self._last_cls_by_bbox[bbox_key] = (ripeness_cls, float(ripeness_conf))
                self._last_cls = (ripeness_cls, float(ripeness_conf))
            else:
                ripeness_cls, ripeness_conf = self._last_cls_by_bbox.get(
                    bbox_key, self._last_cls or ("green", 0.0)
                )

            mask_crop: Optional[np.ndarray] = None
            if (
                self.segmenter is not None
                and do_seg_this_frame
                and i < seg_max_rois
                and float(det.detector_conf) >= seg_min_det_conf
            ):
                t_seg0 = _now_ms()
                assert self.segmenter is not None
                mask_crop = self.segmenter.infer_mask_on_crop(crop)
                t_seg1 = _now_ms()
                seg_ms += (t_seg1 - t_seg0)
                self.stats.seg_calls += 1
            else:
                if self.segmenter is not None:
                    self.stats.seg_skipped_frames += 1

            mask_full: Optional[np.ndarray] = None
            contour: Optional[np.ndarray] = None
            if mask_crop is not None:
                mask_full = np.zeros((h, w), dtype=np.uint8)
                mh, mw = mask_crop.shape[:2]
                if (mw, mh) != (crop.shape[1], crop.shape[0]):
                    mask_crop = cv2.resize(mask_crop, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask_full[y1p:y2p, x1p:x2p] = mask_crop
                contours, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    contour = max(contours, key=cv2.contourArea)
                self._last_mask_by_bbox[bbox_key] = mask_full
                self._last_mask = mask_full
            else:
                # Reuse last mask on skip frames (prefer bbox-specific, else global last).
                mask_full = self._last_mask_by_bbox.get(bbox_key)
                if mask_full is None:
                    mask_full = self._last_mask
                if mask_full is not None:
                    contours, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        contour = max(contours, key=cv2.contourArea)

            t_dep0 = _now_ms()
            distance_m = self.depth.estimate_distance_m(depth_aligned, mask_full)
            t_dep1 = _now_ms()
            dep_ms += (t_dep1 - t_dep0)

            cx = int((x1 + x2) * 0.5)
            cy = int((y1 + y2) * 0.5)
            results.append(
                InstanceResult(
                    bbox_xyxy=(x1, y1, x2, y2),
                    detector_conf=float(det.detector_conf),
                    ripeness_class=ripeness_cls,
                    classifier_conf=float(ripeness_conf),
                    mask_fullres=mask_full,
                    contour_fullres=contour,
                    distance_m=distance_m,
                    center_xy=(cx, cy),
                )
            )

        t1 = _now_ms()
        self.timings.detector_ms = (t_det1 - t_det0)
        self.timings.classifier_ms = cls_ms
        self.timings.segmentation_ms = seg_ms
        self.timings.depth_ms = dep_ms
        self.timings.total_ms = (t1 - t0)

        now_ms = _now_ms()
        if self._last_frame_ms is not None:
            dt = max(1e-3, now_ms - self._last_frame_ms)
            self.timings.fps = 1000.0 / dt
        self._last_frame_ms = now_ms
        return results

    def render_overlay(self, frame_bgr: np.ndarray, instances: Iterable[InstanceResult]) -> np.ndarray:
        out = frame_bgr.copy()
        h, w = out.shape[:2]

        mask_acc = np.zeros((h, w), dtype=np.uint8)
        for inst in instances:
            if inst.mask_fullres is not None:
                mask_acc = cv2.bitwise_or(mask_acc, inst.mask_fullres)

        if bool(np.any(mask_acc)):
            color = np.zeros_like(out, dtype=np.uint8)
            color[:, :, 1] = 200  # green
            alpha = float(self.cfg.mask_alpha)
            m = (mask_acc > 0)[:, :, None]
            out = np.where(m, (out * (1.0 - alpha) + color * alpha).astype(np.uint8), out)

        for inst in instances:
            x1, y1, x2, y2 = inst.bbox_xyxy
            x1, y1, x2, y2 = _clamp_xyxy(x1, y1, x2, y2, w=w, h=h)
            cls = inst.ripeness_class
            if cls == "ripe":
                box_color = (20, 220, 20)
            elif cls == "turning":
                box_color = (0, 215, 255)
            elif cls == "green":
                box_color = (0, 180, 0)
            else:
                box_color = (30, 30, 255)

            cv2.rectangle(out, (x1, y1), (x2, y2), box_color, 2)
            if inst.contour_fullres is not None:
                cv2.drawContours(out, [inst.contour_fullres], -1, box_color, 2)

            dist_txt = f"{inst.distance_m:.2f} m" if inst.distance_m is not None else "None"
            mass_txt = f"{inst.mass_g:.0f} g" if inst.mass_g is not None else "None"
            txt = f"{cls} | {inst.classifier_conf:.2f} | {dist_txt} | {mass_txt}"
            cv2.putText(out, txt, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, txt, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # Top-left overlay (green), requested format.
        overlay_lines = [
            f"FPS: {self.timings.fps:.1f}",
            f"RGB: {self._hud_rgb_fps:.0f}",
            f"Depth: {self._hud_depth_fps:.0f}",
            f"{w}x{h}",
        ]
        x0, y0 = 12, 36
        dy = 34
        for j, line in enumerate(overlay_lines):
            y = y0 + j * dy
            cv2.putText(out, line, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(out, line, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (0, 255, 0), 2, cv2.LINE_AA)

        hud = (
            f"det={self.timings.detector_ms:.1f}ms  "
            f"cls={self.timings.classifier_ms:.1f}ms  "
            f"seg={self.timings.segmentation_ms:.1f}ms  "
            f"depth={self.timings.depth_ms:.1f}ms  "
            f"total={self.timings.total_ms:.1f}ms  "
            f"fps={self.timings.fps:.1f}"
        )
        cv2.putText(out, hud, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, hud, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return out


def default_production_config(repo_root: Path) -> EnsembleConfig:
    return EnsembleConfig(
        detector_weights=str(repo_root / "runs" / "detect_benchmark_v3" / "yolov8s_v3_lowdensity" / "weights" / "best.pt"),
        classifier_weights=str(repo_root / "runs" / "classification_benchmark_v2" / "efficientnet_b0" / "best.pt"),
        segmenter_weights=str(repo_root / "runs" / "segment_benchmark" / "yolov8n_seg_benchmark" / "weights" / "best.pt"),
    )

