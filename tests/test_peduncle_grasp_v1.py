#!/usr/bin/env python3
"""Unit tests for peduncle grasp v1 (no GPU required for core geometry)."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pipelines.peduncle_grasp.association_v2 import (  # noqa: E402
    associate_candidates,
    choose_base_tip,
    direction_score,
    obb_axis_ends,
    score_candidate,
)
from pipelines.peduncle_grasp.calyx import estimate_calyx_geometry  # noqa: E402
from pipelines.peduncle_grasp.cut import propose_cut  # noqa: E402
from pipelines.peduncle_grasp.roi import build_closeup_roi  # noqa: E402
from pipelines.peduncle_grasp.temporal import TemporalVerifier  # noqa: E402
from pipelines.peduncle_grasp import PeduncleCandidate, PeduncleCandidateFeatures  # noqa: E402


ASSOC_CFG = {
    "far_tau": 0.85,
    "close_tau": 0.30,
    "max_berry_iou": 0.4,
    "max_obb_length_berry_scale": 0.7,
    "into_berry_max": 0.45,
    "w_confidence": 0.25,
    "w_proximity": 0.35,
    "w_contact": 0.2,
    "w_direction": 0.2,
    "w_continuity": 0.0,
    "w_depth": 0.0,
    "w_occlusion": 0.0,
    "w_crossing": 0.1,
    "min_score": 0.2,
    "min_model_conf": 0.1,
}


def make_obb(base, tip, half_w=4.0):
    bx, by = base
    tx, ty = tip
    dx, dy = tx - bx, ty - by
    n = math.hypot(dx, dy) or 1.0
    ux, uy = dx / n, dy / n
    px, py = -uy, ux
    return [
        (bx + px * half_w, by + py * half_w),
        (bx - px * half_w, by - py * half_w),
        (tx - px * half_w, ty - py * half_w),
        (tx + px * half_w, ty + py * half_w),
    ]


class TestObbEnds(unittest.TestCase):
    def test_axis_ends_and_base(self):
        pts = make_obb((100, 200), (100, 100))
        e0, e1, length = obb_axis_ends(pts)
        self.assertGreater(length, 90)
        base, tip, d = choose_base_tip(e0, e1, (100, 210))
        self.assertLess(math.hypot(base[0] - 100, base[1] - 200), 15)


class TestRoi(unittest.TestCase):
    def test_roi_maps(self):
        bgr = np.zeros((400, 400, 3), dtype=np.uint8)
        roi = build_closeup_roi(bgr, (200, 150), 80, (200, 200), {
            "width_berry_scale": 1.4,
            "height_above_calyx_scale": 0.55,
            "height_below_calyx_scale": 0.35,
            "min_size_px": 96,
            "max_size_px": 640,
        })
        fx, fy = roi.to_full(10, 20)
        self.assertEqual((fx, fy), (roi.x1 + 10, roi.y1 + 20))
        rx, ry = roi.to_roi(fx, fy)
        self.assertAlmostEqual(rx, 10)
        self.assertAlmostEqual(ry, 20)


class TestDirection(unittest.TestCase):
    def test_aligned_high(self):
        s = direction_score((100, 200), (100, 150), (100, 150), (100, 80))
        self.assertGreater(s, 0.85)

    def test_cross_low(self):
        s = direction_score((100, 200), (100, 150), (100, 150), (180, 150))
        self.assertLess(s, 0.6)


class TestAssociation(unittest.TestCase):
    def test_correct_peduncle(self):
        berry = (80, 160, 120, 220)
        calyx = (100, 160)
        center = (100, 190)
        pts = make_obb((100, 158), (100, 120), half_w=3.0)
        c = score_candidate(pts, 0.8, center, berry, 60, calyx, None, ASSOC_CFG, mode="close")
        self.assertIsNotNone(c)
        self.assertLess(c.features.normalized_base_distance, 0.3)

    def test_far_stem_rejected(self):
        berry = (80, 160, 120, 220)
        calyx = (100, 160)
        center = (100, 190)
        pts = make_obb((100, 40), (100, 10))  # far above
        c = score_candidate(pts, 0.9, center, berry, 60, calyx, None, ASSOC_CFG, mode="close")
        self.assertIsNone(c)

    def test_best_of_multiple(self):
        berry = (80, 160, 120, 220)
        calyx = (100, 160)
        center = (100, 190)
        good = make_obb((100, 158), (100, 120), half_w=3.0)
        bad = make_obb((180, 100), (220, 60))
        scored, best = associate_candidates(
            [(bad, 0.95), (good, 0.7)],
            center,
            berry,
            60,
            calyx,
            None,
            ASSOC_CFG,
            mode="close",
        )
        self.assertIsNotNone(best)
        self.assertLess(best.features.normalized_base_distance, 0.35)

    def test_low_conf_reject(self):
        berry = (80, 160, 120, 220)
        pts = make_obb((100, 158), (100, 120), half_w=3.0)
        c = score_candidate(pts, 0.05, (100, 190), berry, 60, (100, 160), None, ASSOC_CFG, mode="close")
        self.assertIsNone(c)


class TestCut(unittest.TestCase):
    def test_cut_offset_and_clearance(self):
        pts = make_obb((100, 150), (100, 80))
        feats = PeduncleCandidateFeatures(0.8, 5, 0.1, 0.9, 0.9, None, None, None, 0.0, 0.5)
        cand = PeduncleCandidate(pts, 0.8, (100, 150), (100, 80), feats, 0.7, 1.0)
        mask = np.zeros((240, 200), dtype=np.uint8)
        mask[160:220, 80:120] = 255
        cut = propose_cut(cand, (100, 160), 60, mask, {"offset_berry_scale": 0.22, "clearance_radius_berry_scale": 0.12})
        self.assertTrue(cut.clearance_ok)
        # put berry into clearance
        mask[100:160, 90:110] = 255
        cut2 = propose_cut(cand, (100, 160), 60, mask, {"offset_berry_scale": 0.22, "clearance_radius_berry_scale": 0.2})
        self.assertFalse(cut2.clearance_ok)


class TestTemporal(unittest.TestCase):
    def test_needs_three_frames(self):
        v = TemporalVerifier({"required_frames": 3, "max_base_shift_px": 18, "max_cut_shift_px": 22, "max_angle_change_deg": 25})
        for i in range(2):
            v.push(1, (100, 150), (100, 148), (100, 100), (100, 135), 0.6)
            ok, reason, n = v.evaluate(0.3)
            self.assertFalse(ok)
            self.assertIn("TEMPORAL_CONFIRMATION", reason)
        v.push(1, (100, 150), (100, 148), (100, 100), (100, 135), 0.6)
        ok, reason, n = v.evaluate(0.3)
        self.assertTrue(ok)
        self.assertEqual(n, 3)

    def test_unstable_cut(self):
        v = TemporalVerifier({"required_frames": 3, "max_base_shift_px": 18, "max_cut_shift_px": 5, "max_angle_change_deg": 25})
        v.push(1, (100, 150), (100, 148), (100, 100), (100, 135), 0.6)
        v.push(1, (100, 150), (100, 148), (100, 100), (100, 140), 0.6)
        v.push(1, (100, 150), (100, 148), (100, 100), (130, 135), 0.6)
        ok, reason, _ = v.evaluate(0.3)
        self.assertFalse(ok)
        self.assertEqual(reason, "CUT_UNSTABLE")


class TestCalyxGeometry(unittest.TestCase):
    def test_pca_tip(self):
        mask = np.zeros((200, 120), dtype=np.uint8)
        # vertical berry, narrower at top
        cv2 = __import__("cv2")
        cv2.ellipse(mask, (60, 120), (35, 50), 0, 0, 360, 255, -1)
        est = estimate_calyx_geometry((20, 60, 100, 180), mask, tip_band_frac=0.12, geometry_confidence=0.35)
        self.assertEqual(est.source, "geometry")
        self.assertLess(est.point_2d[1], 120)  # tip toward top of ellipse


if __name__ == "__main__":
    unittest.main()
