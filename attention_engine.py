"""
attention_engine.py
====================
Phase 5 module — fuses BlinkResult, HeadPoseResult, and GazeResult into a
single smoothed attention score with session-level statistics.

Performs NO computer vision and never touches MediaPipe directly.
Consumes the immutable result objects already produced by:
    BlinkDetector.update()      -> BlinkResult
    HeadPoseEstimator.update()  -> HeadPoseResult
    GazeEstimator.update()      -> GazeResult
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from blink_detector import BlinkResult, EyeState, ClosureType
from head_pose_estimator import HeadPoseResult
from gaze_estimator import GazeResult, HorizontalDirection, VerticalDirection


# ─────────────────────────────────────────────────────────────────────────────
# Fixed component weights
# ─────────────────────────────────────────────────────────────────────────────

WEIGHT_FACE_PRESENCE: float = 0.30
WEIGHT_HEAD_POSE:     float = 0.25
WEIGHT_GAZE:          float = 0.30
WEIGHT_BLINK:         float = 0.15

# ─────────────────────────────────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────────────────────────────────

CALIBRATION_DURATION_S: float = 3.0

# ─────────────────────────────────────────────────────────────────────────────
# Smoothing
# ─────────────────────────────────────────────────────────────────────────────

EMA_ALPHA: float = 0.15

# ─────────────────────────────────────────────────────────────────────────────
# Score-tolerance constants (deviation from calibrated baseline before
# a component score starts dropping from 1.0)
# ─────────────────────────────────────────────────────────────────────────────

HEAD_PITCH_TOLERANCE_DEG: float = 15.0
HEAD_YAW_TOLERANCE_DEG:   float = 15.0
HEAD_RAMP_DEG:            float = 15.0   # additional degrees to fully decay to 0.0

GAZE_RATIO_TOLERANCE: float = 0.15
GAZE_RAMP:            float = 0.20

EAR_DROP_TOLERANCE: float = 0.05   # allowed drop below normal_EAR before penalty
EAR_RAMP:           float = 0.10


# ─────────────────────────────────────────────────────────────────────────────
# Attention state classification
# ─────────────────────────────────────────────────────────────────────────────

class AttentionState(Enum):
    HIGHLY_ATTENTIVE     = "HIGHLY_ATTENTIVE"
    ATTENTIVE            = "ATTENTIVE"
    PARTIALLY_ATTENTIVE  = "PARTIALLY_ATTENTIVE"
    DISTRACTED           = "DISTRACTED"
    HIGHLY_DISTRACTED    = "HIGHLY_DISTRACTED"
    NO_FACE              = "NO_FACE"


def _classify_state(score_pct: float, face_present: bool) -> AttentionState:
    if not face_present:
        return AttentionState.NO_FACE
    if score_pct >= 90.0:
        return AttentionState.HIGHLY_ATTENTIVE
    if score_pct >= 75.0:
        return AttentionState.ATTENTIVE
    if score_pct >= 50.0:
        return AttentionState.PARTIALLY_ATTENTIVE
    if score_pct >= 25.0:
        return AttentionState.DISTRACTED
    return AttentionState.HIGHLY_DISTRACTED


# ─────────────────────────────────────────────────────────────────────────────
# Calibration profile
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CalibrationProfile:
    """Personal baseline collected during the 3-second calibration stage."""
    neutral_head_pitch:      float = 0.0
    neutral_head_yaw:        float = 0.0
    neutral_gaze_horizontal: float = 0.5
    neutral_gaze_vertical:   float = 0.5
    normal_ear:              float = 0.30
    is_complete:             bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Session statistics
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SessionStatistics:
    sample_count:               int = 0
    score_sum:                  float = 0.0
    average_attention:          float = 0.0
    minimum_attention:          float = 100.0
    maximum_attention:          float = 0.0
    time_attentive_s:           float = 0.0
    time_distracted_s:          float = 0.0
    longest_distracted_streak_s: float = 0.0
    blink_count:                int = 0
    long_eye_closures:          int = 0
    face_loss_duration_s:       float = 0.0

    # internal tracking (not part of public stats output)
    _current_distracted_streak_s: float = field(default=0.0, repr=False)


# ─────────────────────────────────────────────────────────────────────────────
# Result object
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AttentionResult:
    raw_score:        float            # current frame composite, 0-100
    smoothed_score:   float            # EMA-smoothed, 0-100
    face_score:       float
    blink_score:      float
    head_pose_score:  float
    gaze_score:       float
    state:            AttentionState
    is_calibrated:    bool
    calibration_progress: float        # 0.0 - 1.0
    statistics:       SessionStatistics


# ─────────────────────────────────────────────────────────────────────────────
# AttentionEngine
# ─────────────────────────────────────────────────────────────────────────────

class AttentionEngine:
    """
    Combines BlinkResult, HeadPoseResult, and GazeResult into a single
    smoothed attention score, with a 3-second calibration stage and
    session-level statistics tracking.

    Call update() once per processed frame, after all upstream modules
    (BlinkDetector, HeadPoseEstimator, GazeEstimator) have run for that frame.
    """

    def __init__(self) -> None:
        self._calibration = CalibrationProfile()
        self._calib_start_s: Optional[float] = None
        self._calib_pitch_sum: float = 0.0
        self._calib_yaw_sum: float = 0.0
        self._calib_gh_sum: float = 0.0
        self._calib_gv_sum: float = 0.0
        self._calib_ear_sum: float = 0.0
        self._calib_samples: int = 0

        self._smoothed_score: float = 100.0
        self._ema_initialized: bool = False

        self._stats = SessionStatistics()

        self._prev_face_present: bool = True
        self._face_absent_start_s: Optional[float] = None
        self._last_blink_count_seen: int = 0
        self._last_long_closure_duration: float = -1.0

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def update(
        self,
        blink_result: BlinkResult,
        head_pose_result: HeadPoseResult,
        gaze_result: GazeResult,
        face_present: bool,
        timestamp_s: float,
        dt_s: float,
    ) -> AttentionResult:
        """
        Process one frame's worth of upstream results.

        Parameters
        ----------
        blink_result, head_pose_result, gaze_result:
            Outputs from the existing Phase 2-4 modules for this frame.
        face_present:
            Whether a face was detected this frame (from LandmarkProcessor.is_valid).
        timestamp_s:
            Monotonic session time in seconds.
        dt_s:
            Elapsed seconds since the previous update() call (for statistics
            accumulation).
        """
        # ── Calibration stage ───────────────────────────────────────────
        if not self._calibration.is_complete:
            self._run_calibration(blink_result, head_pose_result, gaze_result,
                                   face_present, timestamp_s)

        # ── Component scores ────────────────────────────────────────────
        face_score = 1.0 if face_present else 0.0

        if face_present and self._calibration.is_complete:
            blink_score     = self._score_blink(blink_result)
            head_pose_score = self._score_head_pose(head_pose_result)
            gaze_score      = self._score_gaze(gaze_result)
        elif face_present:
            # Face present but calibration not yet complete: neutral scores.
            blink_score, head_pose_score, gaze_score = 1.0, 1.0, 1.0
        else:
            blink_score, head_pose_score, gaze_score = 0.0, 0.0, 0.0

        # ── Weighted combination ────────────────────────────────────────
        raw_fraction = (
            WEIGHT_FACE_PRESENCE * face_score +
            WEIGHT_HEAD_POSE     * head_pose_score +
            WEIGHT_GAZE          * gaze_score +
            WEIGHT_BLINK         * blink_score
        )
        raw_score = raw_fraction * 100.0

        # ── EMA smoothing ───────────────────────────────────────────────
        if not self._ema_initialized:
            self._smoothed_score = raw_score
            self._ema_initialized = True
        else:
            self._smoothed_score = (
                EMA_ALPHA * raw_score + (1.0 - EMA_ALPHA) * self._smoothed_score
            )

        state = _classify_state(self._smoothed_score, face_present)

        # ── Statistics ───────────────────────────────────────────────────
        self._update_statistics(state, face_present, blink_result, dt_s)

        return AttentionResult(
            raw_score            = round(raw_score, 1),
            smoothed_score       = round(self._smoothed_score, 1),
            face_score           = round(face_score, 3),
            blink_score          = round(blink_score, 3),
            head_pose_score      = round(head_pose_score, 3),
            gaze_score           = round(gaze_score, 3),
            state                = state,
            is_calibrated        = self._calibration.is_complete,
            calibration_progress = self._calibration_progress(timestamp_s),
            statistics           = self._stats,
        )

    @property
    def calibration(self) -> CalibrationProfile:
        return self._calibration

    @property
    def statistics(self) -> SessionStatistics:
        return self._stats

    # ─────────────────────────────────────────────────────────────────────────
    # Calibration
    # ─────────────────────────────────────────────────────────────────────────

    def _run_calibration(
        self,
        blink_result: BlinkResult,
        head_pose_result: HeadPoseResult,
        gaze_result: GazeResult,
        face_present: bool,
        timestamp_s: float,
    ) -> None:
        if self._calib_start_s is None:
            self._calib_start_s = timestamp_s

        if not face_present:
            return  # skip frames with no face; do not pollute the baseline

        self._calib_pitch_sum += head_pose_result.pitch
        self._calib_yaw_sum   += head_pose_result.yaw
        self._calib_gh_sum    += gaze_result.horizontal_ratio
        self._calib_gv_sum    += gaze_result.vertical_ratio
        self._calib_ear_sum   += blink_result.average_ear
        self._calib_samples   += 1

        elapsed = timestamp_s - self._calib_start_s
        if elapsed >= CALIBRATION_DURATION_S and self._calib_samples > 0:
            n = self._calib_samples
            self._calibration.neutral_head_pitch      = self._calib_pitch_sum / n
            self._calibration.neutral_head_yaw        = self._calib_yaw_sum   / n
            self._calibration.neutral_gaze_horizontal  = self._calib_gh_sum   / n
            self._calibration.neutral_gaze_vertical    = self._calib_gv_sum   / n
            self._calibration.normal_ear               = self._calib_ear_sum  / n
            self._calibration.is_complete               = True

    def _calibration_progress(self, timestamp_s: float) -> float:
        if self._calibration.is_complete:
            return 1.0
        if self._calib_start_s is None:
            return 0.0
        elapsed = timestamp_s - self._calib_start_s
        return max(0.0, min(1.0, elapsed / CALIBRATION_DURATION_S))

    # ─────────────────────────────────────────────────────────────────────────
    # Component scores (each returns 0.0 - 1.0)
    # ─────────────────────────────────────────────────────────────────────────

    def _score_blink(self, blink_result: BlinkResult) -> float:
        if blink_result.eye_state is EyeState.CLOSED:
            if blink_result.closure_duration_s >= 2.0:
                return 0.0
            if blink_result.closure_duration_s >= 0.4:
                return 0.3
            return 1.0

        deviation = self._calibration.normal_ear - blink_result.average_ear
        return _ramp_score(deviation, EAR_DROP_TOLERANCE, EAR_RAMP)

    def _score_head_pose(self, head_pose_result: HeadPoseResult) -> float:
        pitch_dev = abs(head_pose_result.pitch - self._calibration.neutral_head_pitch)
        yaw_dev   = abs(head_pose_result.yaw   - self._calibration.neutral_head_yaw)

        pitch_score = _ramp_score(pitch_dev, HEAD_PITCH_TOLERANCE_DEG, HEAD_RAMP_DEG)
        yaw_score   = _ramp_score(yaw_dev,   HEAD_YAW_TOLERANCE_DEG,   HEAD_RAMP_DEG)

        return min(pitch_score, yaw_score)

    def _score_gaze(self, gaze_result: GazeResult) -> float:
            h_dev = abs(gaze_result.horizontal_ratio - self._calibration.neutral_gaze_horizontal)
            v_dev = abs(gaze_result.vertical_ratio   - self._calibration.neutral_gaze_vertical)

            h_score = _ramp_score(h_dev, GAZE_RATIO_TOLERANCE, GAZE_RAMP)
            v_score = _ramp_score(v_dev, GAZE_RATIO_TOLERANCE, GAZE_RAMP)
            ratio_score = min(h_score, v_score)

            # GazeEstimator's own horizontal/vertical direction classification
            # uses fixed geometric thresholds independent of the calibrated
            # baseline above. When GazeEstimator has already classified gaze
            # as off-center, the calibration-relative ratio score must not be
            # allowed to remain near 1.0 just because the deviation hasn't yet
            # crossed this engine's separate tolerance+ramp window.
            if not gaze_result.is_looking_center():
                return 0.0

            return ratio_score

    # ─────────────────────────────────────────────────────────────────────────
    # Statistics
    # ─────────────────────────────────────────────────────────────────────────

    def _update_statistics(
        self,
        state: AttentionState,
        face_present: bool,
        blink_result: BlinkResult,
        dt_s: float,
    ) -> None:
        s = self._stats

        s.sample_count += 1
        s.score_sum += self._smoothed_score
        s.average_attention = s.score_sum / s.sample_count
        s.minimum_attention = min(s.minimum_attention, self._smoothed_score)
        s.maximum_attention = max(s.maximum_attention, self._smoothed_score)

        is_distracted = state in (
            AttentionState.DISTRACTED,
            AttentionState.HIGHLY_DISTRACTED,
            AttentionState.NO_FACE,
        )

        if is_distracted:
            s.time_distracted_s += dt_s
            s._current_distracted_streak_s += dt_s
            s.longest_distracted_streak_s = max(
                s.longest_distracted_streak_s, s._current_distracted_streak_s
            )
        else:
            s.time_attentive_s += dt_s
            s._current_distracted_streak_s = 0.0

        # ── Face loss duration ──────────────────────────────────────────
        if not face_present:
            if self._face_absent_start_s is None:
                self._face_absent_start_s = 0.0  # marker; accumulate via dt
            s.face_loss_duration_s += dt_s
        else:
            self._face_absent_start_s = None

        # ── Blink count (cumulative counter already tracked by BlinkDetector) ──
        if blink_result.blink_count > self._last_blink_count_seen:
            s.blink_count = blink_result.blink_count
            self._last_blink_count_seen = blink_result.blink_count

        # ── Long eye closures (count transitions into LONG_CLOSURE / POSSIBLE_SLEEP) ──
        if blink_result.eye_state is EyeState.OPEN and blink_result.last_closure_type in (
            ClosureType.LONG_CLOSURE, ClosureType.POSSIBLE_SLEEP
        ):
            # Only increment once per completed event: guard with closure_duration_s
            # transitioning from >0 to the same value across frames is avoided by
            # tracking the duration value that triggered this branch.
            if blink_result.closure_duration_s != self._last_long_closure_duration:
                s.long_eye_closures += 1
                self._last_long_closure_duration = blink_result.closure_duration_s


# ─────────────────────────────────────────────────────────────────────────────
# Pure scoring helper
# ─────────────────────────────────────────────────────────────────────────────

def _ramp_score(deviation: float, tolerance: float, ramp: float) -> float:
    """
    Linear decay score: 1.0 within tolerance, 0.0 beyond tolerance + ramp.
    """
    if deviation <= tolerance:
        return 1.0
    if deviation >= tolerance + ramp:
        return 0.0
    return 1.0 - (deviation - tolerance) / ramp


# ─────────────────────────────────────────────────────────────────────────────
# Overlay
# ─────────────────────────────────────────────────────────────────────────────

def draw_attention_overlay(frame_bgr, result: AttentionResult) -> None:
    """
    Render the Phase 5 attention HUD: Attention %, State, Calibration status.
    Positioned top-center to avoid overlap with existing overlays.
    """
    import cv2

    h, w = frame_bgr.shape[:2]
    x = w // 2 - 110
    y_base = 25
    dy = 24

    state_colors = {
        AttentionState.HIGHLY_ATTENTIVE:    (0, 220, 0),
        AttentionState.ATTENTIVE:           (0, 200, 60),
        AttentionState.PARTIALLY_ATTENTIVE: (0, 200, 220),
        AttentionState.DISTRACTED:          (0, 120, 255),
        AttentionState.HIGHLY_DISTRACTED:   (0, 0, 255),
        AttentionState.NO_FACE:             (100, 100, 100),
    }
    color = state_colors.get(result.state, (200, 200, 200))

    calib_text = (
        "Calibrated" if result.is_calibrated
        else f"Calibrating {result.calibration_progress * 100:.0f}%"
    )

    lines = [
        (f"Attention: {result.smoothed_score:.0f}%", color),
        (f"State: {result.state.value}",             color),
        (calib_text,                                  (180, 180, 180)),
    ]

    for i, (text, c) in enumerate(lines):
        cv2.putText(
            frame_bgr, text, (x, y_base + i * dy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, cv2.LINE_AA,
        )
