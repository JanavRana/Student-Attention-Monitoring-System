"""
gaze_estimator.py
=================
Phase 4 module — iris-ratio gaze direction estimator.

Algorithm
---------
For each eye the iris landmark (468 / 473) is positioned inside the eye's
own bounding geometry.  All distances are pixel-space but expressed as
ratios, so the result is scale-invariant.

Horizontal ratio
    h = (iris_x  −  min_corner_x) / (max_corner_x − min_corner_x)
    0.0 = iris at the camera-left  edge of the eye
    0.5 = iris centred
    1.0 = iris at the camera-right edge of the eye

    For both eyes the leftmost corner in image space has the smaller x
    coordinate and the rightmost has the larger, so min/max produces a
    direction-consistent ratio without any per-eye sign flip.

Vertical ratio
    top_y = mean(upper-outer lid y,  upper-inner lid y)   [idx 1, 2]
    bot_y = mean(lower-inner lid y,  lower-outer lid y)   [idx 4, 5]
    v = (iris_y − top_y) / (bot_y − top_y)
    0.0 = looking up   (iris near upper lid)
    0.5 = centred
    1.0 = looking down (iris near lower lid)

Left and right eye ratios are averaged to form the combined output.
EAR landmark index ordering (imported from LandmarkProcessor):
    idx 0 = outer corner   idx 3 = inner corner
    idx 1 = upper-outer    idx 2 = upper-inner
    idx 4 = lower-inner    idx 5 = lower-outer
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import cv2
import numpy as np

from landmark_processor import (
    LandmarkProcessor,
    LEFT_EYE_EAR,
    RIGHT_EYE_EAR,
    LEFT_IRIS_CENTER,
    RIGHT_IRIS_CENTER,
)


# ─────────────────────────────────────────────────────────────────────────────
# Configurable thresholds — replace with per-user values in Phase 5 calibration
# ─────────────────────────────────────────────────────────────────────────────

GAZE_H_LEFT_THRESHOLD:  float = 0.40
"""Horizontal ratio below this → gaze classified as LEFT."""

GAZE_H_RIGHT_THRESHOLD: float = 0.60
"""Horizontal ratio above this → gaze classified as RIGHT."""

GAZE_V_UP_THRESHOLD:    float = 0.40
"""Vertical ratio below this → gaze classified as UP."""

GAZE_V_DOWN_THRESHOLD:  float = 0.60
"""Vertical ratio above this → gaze classified as DOWN."""


# ─────────────────────────────────────────────────────────────────────────────
# Direction enumerations
# ─────────────────────────────────────────────────────────────────────────────

class HorizontalDirection(Enum):
    LEFT   = "LEFT"
    CENTER = "CENTER"
    RIGHT  = "RIGHT"


class VerticalDirection(Enum):
    UP     = "UP"
    CENTER = "CENTER"
    DOWN   = "DOWN"


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GazeResult:
    """
    Immutable snapshot of gaze state after processing one frame.

    Attributes
    ----------
    horizontal_ratio : float
        Average iris position across both eyes, 0.0 (camera-left) … 1.0
        (camera-right).
    vertical_ratio : float
        Average iris height inside the eye, 0.0 (up) … 1.0 (down).
    horizontal_direction : HorizontalDirection
        Classified horizontal gaze direction.
    vertical_direction : VerticalDirection
        Classified vertical gaze direction.
    left_eye_ratio : tuple[float, float]
        (h_ratio, v_ratio) for the left eye only.
    right_eye_ratio : tuple[float, float]
        (h_ratio, v_ratio) for the right eye only.
    """
    horizontal_ratio:      float
    vertical_ratio:        float
    horizontal_direction:  HorizontalDirection
    vertical_direction:    VerticalDirection
    left_eye_ratio:        tuple[float, float]
    right_eye_ratio:       tuple[float, float]

    def is_looking_center(self) -> bool:
        """True when both horizontal and vertical directions are CENTER."""
        return (
            self.horizontal_direction is HorizontalDirection.CENTER and
            self.vertical_direction   is VerticalDirection.CENTER
        )


# ─────────────────────────────────────────────────────────────────────────────
# Per-eye ratio computation (module-level pure function — easily unit-testable)
# ─────────────────────────────────────────────────────────────────────────────

def _eye_ratios(
    eye_pts:  np.ndarray,   # shape (6, 3)  from get_eye_landmarks()
    iris_pt:  np.ndarray,   # shape (3,)    from get_landmark()
) -> tuple[float, float]:
    """
    Return (h_ratio, v_ratio) for one eye, both clamped to [0.0, 1.0].
    Returns (0.5, 0.5) on degenerate geometry (eye too small to measure).
    """
    # ── Horizontal ────────────────────────────────────────────────────
    # idx 0 = outer corner x,  idx 3 = inner corner x
    # Use min/max so the formula is identical for both eyes regardless of
    # which corner is geometrically left vs right on that side of the face.
    x_a = eye_pts[0, 0]    # outer corner x
    x_b = eye_pts[3, 0]    # inner corner x
    x_lo, x_hi = (x_a, x_b) if x_a < x_b else (x_b, x_a)
    eye_w = x_hi - x_lo

    if eye_w < 1.0:
        return 0.5, 0.5

    h = float(np.clip((iris_pt[0] - x_lo) / eye_w, 0.0, 1.0))

    # ── Vertical ──────────────────────────────────────────────────────
    # idx 1, 2 = upper lid (p2, p3);  idx 4, 5 = lower lid (p5, p6)
    top_y = (eye_pts[1, 1] + eye_pts[2, 1]) * 0.5
    bot_y = (eye_pts[4, 1] + eye_pts[5, 1]) * 0.5
    eye_h = bot_y - top_y

    if eye_h < 1.0:
        return h, 0.5

    v = float(np.clip((iris_pt[1] - top_y) / eye_h, 0.0, 1.0))

    return h, v


# ─────────────────────────────────────────────────────────────────────────────
# GazeEstimator
# ─────────────────────────────────────────────────────────────────────────────

class GazeEstimator:
    """
    Estimates gaze direction from MediaPipe iris landmarks every frame.

    Usage
    -----
    ::

        estimator = GazeEstimator()

        # In the per-frame loop (after processor.update(result)):
        gaze = estimator.update(processor, timestamp_s)
        print(gaze.horizontal_direction, gaze.vertical_direction)
    """

    def __init__(
        self,
        h_left_threshold:  float = GAZE_H_LEFT_THRESHOLD,
        h_right_threshold: float = GAZE_H_RIGHT_THRESHOLD,
        v_up_threshold:    float = GAZE_V_UP_THRESHOLD,
        v_down_threshold:  float = GAZE_V_DOWN_THRESHOLD,
    ) -> None:
        """
        Parameters
        ----------
        h_left_threshold, h_right_threshold:
            Horizontal ratio boundaries.  Values between the two thresholds
            are classified CENTER.  Replace with calibrated values in Phase 5.
        v_up_threshold, v_down_threshold:
            Vertical ratio boundaries.
        """
        self._h_left  = h_left_threshold
        self._h_right = h_right_threshold
        self._v_up    = v_up_threshold
        self._v_down  = v_down_threshold

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def update(
        self,
        processor:   LandmarkProcessor,
        timestamp_s: float,
    ) -> GazeResult:
        """
        Compute gaze ratios and directions for the current frame.

        Parameters
        ----------
        processor:
            A LandmarkProcessor already updated for this frame.
        timestamp_s:
            Monotonic wall-clock time in seconds (reserved for Phase 5
            dwell-time filtering; not used in the ratio computation itself).

        Returns
        -------
        GazeResult
            Immutable snapshot.  Returns a neutral CENTER result when no
            face is detected or iris landmarks are unavailable.
        """
        if not processor.is_valid:
            return _NEUTRAL

        eye_lms    = processor.get_eye_landmarks()
        iris_left  = processor.get_landmark(LEFT_IRIS_CENTER)
        iris_right = processor.get_landmark(RIGHT_IRIS_CENTER)

        if eye_lms is None or iris_left is None or iris_right is None:
            return _NEUTRAL

        l_h, l_v = _eye_ratios(eye_lms["left"],  iris_left)
        r_h, r_v = _eye_ratios(eye_lms["right"], iris_right)

        avg_h = (l_h + r_h) * 0.5
        avg_v = (l_v + r_v) * 0.5

        return GazeResult(
            horizontal_ratio     = round(avg_h, 3),
            vertical_ratio       = round(avg_v, 3),
            horizontal_direction = self._classify_h(avg_h),
            vertical_direction   = self._classify_v(avg_v),
            left_eye_ratio       = (round(l_h, 3), round(l_v, 3)),
            right_eye_ratio      = (round(r_h, 3), round(r_v, 3)),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _classify_h(self, ratio: float) -> HorizontalDirection:
        if ratio < self._h_left:
            return HorizontalDirection.LEFT
        if ratio > self._h_right:
            return HorizontalDirection.RIGHT
        return HorizontalDirection.CENTER

    def _classify_v(self, ratio: float) -> VerticalDirection:
        if ratio < self._v_up:
            return VerticalDirection.UP
        if ratio > self._v_down:
            return VerticalDirection.DOWN
        return VerticalDirection.CENTER


# Sentinel returned when no face is present — avoids constructing a new object
# on every no-face frame.  Immutable so it is safe to share.
_NEUTRAL = GazeResult(
    horizontal_ratio     = 0.5,
    vertical_ratio       = 0.5,
    horizontal_direction = HorizontalDirection.CENTER,
    vertical_direction   = VerticalDirection.CENTER,
    left_eye_ratio       = (0.5, 0.5),
    right_eye_ratio      = (0.5, 0.5),
)


# ─────────────────────────────────────────────────────────────────────────────
# Visual debug — drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def draw_gaze_visualization(
    frame_bgr: np.ndarray,
    processor: LandmarkProcessor,
) -> None:
    """
    Draw iris centers, eye centers, and crosshair guides on the frame.

    Must be called after ``GazeEstimator.update()`` for the same frame.
    Has no effect when no face is detected.

    Colours
    -------
    Cyan    : iris center dot + crosshair guide lines
    Magenta : geometric eye center (average of 6 EAR landmarks)
    """
    if not processor.is_valid:
        return

    eye_lms    = processor.get_eye_landmarks()
    iris_left  = processor.get_landmark(LEFT_IRIS_CENTER)
    iris_right = processor.get_landmark(RIGHT_IRIS_CENTER)

    if eye_lms is None or iris_left is None or iris_right is None:
        return

    for eye_pts, iris_pt in (
        (eye_lms["left"],  iris_left),
        (eye_lms["right"], iris_right),
    ):
        iris_x = int(iris_pt[0])
        iris_y = int(iris_pt[1])

        # ── Iris center ────────────────────────────────────────────────
        cv2.circle(frame_bgr, (iris_x, iris_y), 3, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame_bgr, (iris_x, iris_y), 7, (0, 255, 255),  1, cv2.LINE_AA)

        # ── Geometric eye center (avg of 6 EAR points) ────────────────
        eye_cx = int(eye_pts[:, 0].mean())
        eye_cy = int(eye_pts[:, 1].mean())
        cv2.circle(frame_bgr, (eye_cx, eye_cy), 2, (255, 0, 255), -1, cv2.LINE_AA)

        # ── Horizontal guide — spans eye corners at iris height ────────
        x_lo = int(min(eye_pts[0, 0], eye_pts[3, 0])) - 2
        x_hi = int(max(eye_pts[0, 0], eye_pts[3, 0])) + 2
        cv2.line(frame_bgr, (x_lo, iris_y), (x_hi, iris_y),
                 (0, 200, 200), 1, cv2.LINE_AA)

        # ── Vertical guide — spans lid heights at iris x ───────────────
        top_y = int((eye_pts[1, 1] + eye_pts[2, 1]) * 0.5) - 2
        bot_y = int((eye_pts[4, 1] + eye_pts[5, 1]) * 0.5) + 2
        cv2.line(frame_bgr, (iris_x, top_y), (iris_x, bot_y),
                 (0, 200, 200), 1, cv2.LINE_AA)


def draw_gaze_overlay(
    frame_bgr: np.ndarray,
    result:    GazeResult,
) -> None:
    """
    Render the Phase 4 gaze debug HUD in the bottom-right corner.

    Four fixed-position text lines — no per-frame layout computation.
    Positioned to avoid overlap with Phase 1–3 overlays:
        top-left    : FPS / face status  (Phase 1)
        bottom-left : blink panel        (Phase 2)
        top-right   : head pose panel    (Phase 3)
        bottom-right: gaze panel         (Phase 4)  ← here
    """
    h, w = frame_bgr.shape[:2]
    x      = w - 195
    y_base = h - 108
    dy     = 22

    centre_color = (0, 220, 0) if result.is_looking_center() else (0, 140, 220)

    lines: list[tuple[str, tuple[int, int, int]]] = [
        (f"H-ratio: {result.horizontal_ratio:.2f}", (180, 230, 180)),
        (f"V-ratio: {result.vertical_ratio:.2f}",  (180, 230, 180)),
        (f"H-dir  : {result.horizontal_direction.value}", centre_color),
        (f"V-dir  : {result.vertical_direction.value}",   centre_color),
    ]

    for i, (text, color) in enumerate(lines):
        cv2.putText(
            frame_bgr,
            text,
            (x, y_base + i * dy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )
