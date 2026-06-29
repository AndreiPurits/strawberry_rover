"""A/B auto-tune stereo UVC brightness + gamma (RealSense D405)."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from rover_perception.stereo_brightness_mask import (
    RectNorm,
    include_mask_from_regions,
    masked_gray_mean,
)
from rover_perception.v4l2_controls import apply_v4l2_controls


class StereoAutoBrightness:
    """Keep mean gray in [target_min, target_max]; trial new params off-stream."""

    BRIGHTNESS_MIN = -64
    BRIGHTNESS_MAX = 64
    GAMMA_MIN = 100
    GAMMA_MAX = 500

    def __init__(
        self,
        *,
        device_index: int,
        target_min: float,
        target_max: float,
        brightness: int,
        gamma: int,
        trial_interval_sec: float = 2.0,
        logger=None,
    ) -> None:
        self.device_index = int(device_index)
        self.target_min = float(target_min)
        self.target_max = float(target_max)
        self.active_brightness = int(brightness)
        self.active_gamma = int(gamma)
        self.trial_interval_sec = max(0.5, float(trial_interval_sec))
        self.logger = logger

        self._exclude_regions: List[RectNorm] = []
        self._mask_ready = False
        self.last_excluded_pct: Optional[float] = None

        self.last_mean: Optional[float] = None
        self.tuning = False
        self._last_trial_at = 0.0
        self._axis = 0
        self._last_candidate: Optional[Tuple[int, int]] = None
        self._trial_brightness: Optional[int] = None
        self._trial_gamma: Optional[int] = None
        self._last_trial_mean: Optional[float] = None
        self._trials_total = 0
        self._tuning_visible_until = 0.0

    def set_exclude_regions(self, regions: List[RectNorm]) -> None:
        self._exclude_regions = list(regions)
        self._mask_ready = True

    def frame_mean(self, frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if not self._exclude_regions:
            self.last_excluded_pct = 0.0
            return float(gray.mean())

        include = include_mask_from_regions(gray, self._exclude_regions)
        self.last_excluded_pct = round((1.0 - include.mean()) * 100.0, 1)
        return masked_gray_mean(gray, include)

    def in_range(self, mean: float) -> bool:
        return self.target_min <= mean <= self.target_max

    def _distance(self, mean: Optional[float]) -> float:
        if mean is None:
            return float("inf")
        if self.in_range(mean):
            return 0.0
        if mean > self.target_max:
            return float(mean - self.target_max)
        return float(self.target_min - mean)

    def apply_active(self) -> None:
        apply_v4l2_controls(
            self.device_index,
            auto_exposure=True,
            exposure=-1,
            gain=-1,
            brightness=self.active_brightness,
            gamma=self.active_gamma,
        )

    def apply_values(self, brightness: int, gamma: int) -> None:
        apply_v4l2_controls(
            self.device_index,
            auto_exposure=True,
            exposure=-1,
            gain=-1,
            brightness=int(brightness),
            gamma=int(gamma),
        )

    def update_from_frame(self, frame: np.ndarray) -> None:
        self.last_mean = self.frame_mean(frame)

    def should_trial(self, now: float) -> bool:
        if self.tuning or self.last_mean is None:
            return False
        if self.in_range(self.last_mean):
            return False
        return (now - self._last_trial_at) >= self.trial_interval_sec

    def begin_trial(self, trial_b: int, trial_g: int) -> None:
        self.tuning = True
        self._trial_brightness = int(trial_b)
        self._trial_gamma = int(trial_g)
        self._last_candidate = (self._trial_brightness, self._trial_gamma)

    def next_candidate(self) -> Optional[Tuple[int, int]]:
        if self.last_mean is None:
            return None
        too_bright = self.last_mean > self.target_max
        too_dark = self.last_mean < self.target_min
        err = self._distance(self.last_mean)

        if self._axis == 0:
            if too_bright:
                if self.active_brightness > self.BRIGHTNESS_MIN + 2:
                    step = -max(8, min(30, int(round(err * 0.8))))
                else:
                    step = 0
            elif too_dark:
                if self.active_brightness < self.BRIGHTNESS_MAX - 2:
                    step = max(8, min(30, int(round(err * 0.8))))
                else:
                    step = 0
            else:
                step = 0
            trial_b = max(
                self.BRIGHTNESS_MIN,
                min(self.BRIGHTNESS_MAX, self.active_brightness + step),
            )
            trial_g = self.active_gamma
        else:
            if too_bright:
                step = -max(40, min(160, int(round(err * 4.0))))
            elif too_dark:
                step = max(40, min(160, int(round(err * 4.0))))
            else:
                step = 0
            trial_g = max(
                self.GAMMA_MIN,
                min(self.GAMMA_MAX, self.active_gamma + step),
            )
            trial_b = self.active_brightness

        self._axis = 1 - self._axis
        if trial_b == self.active_brightness and trial_g == self.active_gamma:
            if too_bright and self.active_gamma > self.GAMMA_MIN + 20:
                trial_g = max(self.GAMMA_MIN, self.active_gamma - 80)
            elif too_dark and self.active_gamma < self.GAMMA_MAX - 20:
                trial_g = min(self.GAMMA_MAX, self.active_gamma + 80)
            else:
                return None
        self.begin_trial(trial_b, trial_g)
        return self._last_candidate

    def finish_trial(self, trial_mean: Optional[float]) -> None:
        self._last_trial_at = time.monotonic()
        self._trials_total += 1
        self._last_trial_mean = trial_mean
        candidate = self._last_candidate
        active_mean = self.last_mean
        commit = False
        if trial_mean is not None and candidate is not None:
            if self.in_range(trial_mean):
                commit = True
            elif self._distance(trial_mean) + 0.5 < self._distance(active_mean):
                commit = True

        if commit and candidate is not None:
            self.active_brightness, self.active_gamma = candidate
            if self.logger is not None:
                self.logger.info(
                    "Auto-brightness committed "
                    f"brightness={self.active_brightness} gamma={self.active_gamma} "
                    f"mean={trial_mean:.1f} (was {active_mean:.1f})"
                )
        elif self.logger is not None and candidate is not None:
            self.logger.info(
                "Auto-brightness trial rejected "
                f"candidate={candidate} mean={trial_mean} active={active_mean}"
            )

        self.tuning = False
        self._trial_brightness = None
        self._trial_gamma = None
        self._tuning_visible_until = time.monotonic() + 2.0
        self.apply_active()

    def stats(self) -> Dict[str, Any]:
        mean = self.last_mean
        visible = self.tuning or (time.monotonic() < self._tuning_visible_until)
        out: Dict[str, Any] = {
            "brightness_mean": round(mean, 1) if mean is not None else None,
            "brightness_ok": mean is not None and self.in_range(mean),
            "brightness": self.active_brightness,
            "gamma": self.active_gamma,
            "tuning": visible,
            "target_min": self.target_min,
            "target_max": self.target_max,
            "trials_total": self._trials_total,
            "brightness_mask_ready": self._mask_ready,
            "brightness_mask_regions": len(self._exclude_regions),
        }
        if self._trial_brightness is not None:
            out["trial_brightness"] = self._trial_brightness
        if self._trial_gamma is not None:
            out["trial_gamma"] = self._trial_gamma
        if self._last_trial_mean is not None:
            out["last_trial_mean"] = round(self._last_trial_mean, 1)
        if self._last_candidate is not None:
            out["last_candidate"] = {
                "brightness": self._last_candidate[0],
                "gamma": self._last_candidate[1],
            }
        if self.last_excluded_pct is not None:
            out["brightness_excluded_pct"] = self.last_excluded_pct
        return out
