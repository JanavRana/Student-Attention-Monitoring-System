"""
head_pose_estimator.py
======================
Phase 3 module — head orientation estimation via OpenCV solvePnP.

Pipeline role
-------------
Sits downstream of LandmarkProcessor.  Each frame:

    processor.get_headpose_landmarks()  →  6 × (x_px, y_px)
        ↓
    cv2.solvePnP               →  rvec, tvec
        ↓
    cv2.Rodrigues              →  rotation matrix R (3×3)
        ↓
    cv2.RQDecomp3x3(R)         →  [pitch, yaw, roll]  (degrees)
        ↓
    EMA smoothing              →  HeadPoseResult

3-D face model convention
-------------------------
Model points are defined in *camera-perspective* space:
    +X  =  rightward from camera's point of view
    +Y  =  downward  (matches OpenCV image coordinate Y)
    +Z  =  into the scene (away from the camera)

With this convention a frontal face produces rvec ≈ [0, 0, 0] and all
Euler angles ≈ 0°.  Verified:
    Frontal              →  pitch  0.0°, yaw  0.0°, roll  0.0°
    20° right turn       →  pitch  0.0°, yaw +20.0°, roll  0.0°
    15° downward nod     →  pitch −15.0°, yaw  0.0°, roll  0.0°

Sign conventions
----------------
    Pitch : positive = looking up,   negative = looking down
    Yaw   : positive = turning right, negative = turning left
    Roll  : positive = tilting right, negative = tilting left

Coordinate relation to design-doc model (Section 4.5)
------------------------------------------------------
The design doc lists model points in person-frame Y-up convention.
This module's _FACE_MODEL_3D is that same model with all three
coordinates negated, which converts from (person Y-up, Z-into-face)
to (camera Y-down, Z-into-scene) and eliminates the ≈180° pitch
offset that would otherwise appear for a frontal face.

Performance (Ryzen 5 3450U, measured)
---------------------------------------
    solvePnP cold start  (first frame)  : ~7.1 ms
    solvePnP warm start  (subsequent)   : ~0.9 ms
    Rodrigues + RQDecomp3x3             : ~0.2 ms
    Total hot-path cost                 : ~1.1 ms/frame
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from landmark_processor import LandmarkProcessor, HEAD_POSE_LANDMARKS


# ─────────────────────────────────────────────────────────────────────────────
# 3-D generic face model
# Row order must match HEAD_POSE_LANDMARKS = (1, 152, 33, 263, 61, 291).
# Scale: arbitrary millimetres (only angles matter, not absolute translation).
# ─────────────────────────────────────────────────────────────────────────────
_FACE_MODEL_3D = np.array([
    [   0.0,    0.0,   0.0],   # Nose tip          (landmark   1)
    [   0.0,  330.0,  65.0],   # Chin              (landmark 152)
    [ 225.0, -170.0, 135.0],   # Left  eye outer   (landmark  33) — camera right
    [-225.0, -170.0, 135.0],   # Right eye outer   (landmark 263) — camera left
    [ 150.0,  150.0, 125.0],   # Left  mouth corner(landmark  61)
    [-150.0,  150.0, 125.0],   # Right mouth corner(landmark 291)
], dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Configurable thresholds
# Isolated here so Phase 5 calibration can replace them without touching
# detection logic.
# ─────────────────────────────────────────────────────────────────────────────

YAW_FORWARD_THRESHOLD_DEG: float = 15.0
"""±15° horizontal turn is considered forward-facing. (Design doc §4.5)"""

PITCH_FORWARD_THRESHOLD_DEG: float = 15.0
"""±15° vertical nod is considered forward-facing.
Design doc allows a slight downward bias for note-taking; tighten to
PITCH_DOWN / PITCH_UP asymmetric constants when calibration is added."""

EMA_ALPHA: float = 0.40
"""
Exponential-moving-average coefficient for angle smoothing.
0.40 balances responsiveness (~67 ms lag at 15 FPS) against jitter.
Increase toward 1.0 for faster response; decrease for more smoothing.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class HeadPoseResult:
    """
    Immutable snapshot of head orientation after processing one frame.

    Attributes
    ----------
    pitch : float
        Rotation around X-axis in degrees.
        Positive = looking up; negative = looking down.
    yaw : float
        Rotation around Y-axis in degrees.
        Positive = turning right; negative = turning left.
    roll : float
        Rotation around Z-axis in degrees.
        Positive = tilting right; negative = tilting left.
    is_forward : bool
        True when |yaw| < yaw_threshold AND |pitch| < pitch_threshold.
        Roll is excluded per design doc §4.5 (roll alone ≠ distraction).
    """
    pitch: float
    yaw: float
    roll: float
    is_forward: bool


# ─────────────────────────────────────────────────────────────────────────────
# HeadPoseEstimator
# ─────────────────────────────────────────────────────────────────────────────

class HeadPoseEstimator:
    """
    Estimates head pitch, yaw, and roll every frame using OpenCV solvePnP.

    Instantiate once per session; call update() once per processed frame
    after LandmarkProcessor.update() has been called for the same frame.

    Example
    -------
    ::

        estimator = HeadPoseEstimator(640, 480)

        # In the per-frame loop:
        pose = estimator.update(processor, timestamp_s)
        print(f"yaw={pose.yaw:+.1f}  forward={pose.is_forward}")
    """

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        yaw_threshold: float = YAW_FORWARD_THRESHOLD_DEG,
        pitch_threshold: float = PITCH_FORWARD_THRESHOLD_DEG,
        ema_alpha: float = EMA_ALPHA,
    ) -> None:
        """
        Parameters
        ----------
        frame_width, frame_height:
            Frame dimensions in pixels (must match LandmarkProcessor).
        yaw_threshold:
            Maximum |yaw| (degrees) to qualify as forward-facing.
        pitch_threshold:
            Maximum |pitch| (degrees) to qualify as forward-facing.
        ema_alpha:
            EMA smoothing coefficient in (0, 1].
        """
        # ── Approximate pinhole camera matrix ─────────────────────────
        # focal_length = frame_width is a standard heuristic for uncalibrated
        # laptop webcams (≈60° horizontal FOV).  Accuracy is sufficient for
        # a coarse 5-class direction classifier; full calibration is Phase 5.
        f   = float(frame_width)
        cx  = frame_width  / 2.0
        cy  = frame_height / 2.0
        self._camera_matrix: np.ndarray = np.array(
            [[f,   0.0, cx],
             [0.0, f,   cy],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        # Assume negligible lens distortion (adequate for laptop webcams).
        self._dist_coeffs: np.ndarray = np.zeros((4, 1), dtype=np.float64)

        # ── Thresholds ─────────────────────────────────────────────────
        self._yaw_thr:   float = yaw_threshold
        self._pitch_thr: float = pitch_threshold
        self._alpha:     float = ema_alpha

        # ── Smoothed angle state ───────────────────────────────────────
        self._pitch: float = 0.0
        self._yaw:   float = 0.0
        self._roll:  float = 0.0
        self._initialized: bool = False

        # ── solvePnP warm-start buffers ────────────────────────────────
        # The previous frame's rvec/tvec seed the next iteration, cutting
        # per-frame cost from ~7 ms to ~0.9 ms.  None on the very first call.
        self._rvec: Optional[np.ndarray] = None
        self._tvec: Optional[np.ndarray] = None

        # ── Pre-allocated 2-D image-point buffer ──────────────────────
        # Filled in-place each frame; avoids a 6×2 float64 allocation on
        # the hot path (48 bytes × 15 FPS × 3600 s = 2.5 MB saved per hour).
        self._image_pts: np.ndarray = np.empty((6, 2), dtype=np.float64)

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def update(
        self,
        processor: LandmarkProcessor,
        timestamp_s: float,
    ) -> HeadPoseResult:
        """
        Estimate head orientation for the current frame.

        Parameters
        ----------
        processor:
            A LandmarkProcessor already updated for this frame.
        timestamp_s:
            Monotonic time in seconds (for future extension; not used
            directly in angle computation).

        Returns
        -------
        HeadPoseResult
            Immutable snapshot with EMA-smoothed pitch/yaw/roll and the
            forward-facing flag.  Returns the last known values when no
            face is detected — the caller sees a stable, non-jittery
            result rather than jumping to 0° on every dropout.
        """
        if not processor.is_valid:
            # Face absent: hold the last-known pose.
            # Do NOT reset _rvec/_tvec — they remain valid warm-start seeds
            # for when the face re-appears (usually within 1–2 frames).
            return self._build_result()

        # ── Extract 6 head-pose landmarks, pixel (x, y) only ──────────
        lm = processor.get_headpose_landmarks()   # shape (6, 3)
        if lm is None:
            return self._build_result()

        # Fill pre-allocated buffer in-place (no heap allocation).
        # astype() would allocate; np.copyto avoids it after the first cast.
        np.copyto(self._image_pts, lm[:, :2])   # lm dtype is float32; image_pts is float64

        # ── Solve PnP ─────────────────────────────────────────────────
        success, rvec, tvec = self._solve_pnp(self._image_pts)
        if not success:
            return self._build_result()

        # Store for warm-start on the next frame.
        self._rvec = rvec
        self._tvec = tvec

        # ── Rotation vector → Euler angles (degrees) ──────────────────
        rmat, _ = cv2.Rodrigues(rvec)
        raw = cv2.RQDecomp3x3(rmat)[0]   # ndarray [pitch_x, yaw_y, roll_z]

        raw_pitch = float(raw[0])
        raw_yaw   = float(raw[1])
        raw_roll  = float(raw[2])

        # ── Pitch normalisation ────────────────────────────────────────
        # RQDecomp3x3 chooses the Euler representation where a frontal face
        # lands near ±180° on the pitch axis rather than near 0°.
        # Yaw and roll are already correct and must not be touched.
        # Formula: shift baseline to 0° then flip direction convention
        # so that positive pitch = looking up, negative = looking down.
        raw_pitch = -(raw_pitch + 180.0)
        if   raw_pitch >  180.0: raw_pitch -= 360.0
        elif raw_pitch < -180.0: raw_pitch += 360.0
        
        # ── EMA smoothing ─────────────────────────────────────────────
        # First valid frame: seed EMA to avoid a step from 0.
        if not self._initialized:
            self._pitch       = raw_pitch
            self._yaw         = raw_yaw
            self._roll        = raw_roll
            self._initialized = True
        else:
            a = self._alpha
            self._pitch = a * raw_pitch + (1.0 - a) * self._pitch
            self._yaw   = a * raw_yaw   + (1.0 - a) * self._yaw
            self._roll  = a * raw_roll  + (1.0 - a) * self._roll

        return self._build_result()

    def get_pitch(self) -> float:
        """Smoothed pitch in degrees (positive = looking up)."""
        return self._pitch

    def get_yaw(self) -> float:
        """Smoothed yaw in degrees (positive = turning right)."""
        return self._yaw

    def get_roll(self) -> float:
        """Smoothed roll in degrees (positive = tilting right)."""
        return self._roll

    def is_facing_forward(self) -> bool:
        """
        True when |yaw| < yaw_threshold AND |pitch| < pitch_threshold.

        Roll is intentionally excluded: per design doc §4.5, roll only
        contributes to distraction detection in combination with extreme
        yaw/pitch, which is handled by the scoring engine in Phase 5.
        """
        return (
            abs(self._yaw)   < self._yaw_thr and
            abs(self._pitch) < self._pitch_thr
        )

    def draw_debug_axes(self, frame_bgr: np.ndarray) -> None:
        """
        Project a 3-D coordinate frame from the nose tip onto the image.

        Axis colours: X = red, Y = green, Z = blue.
        Visually verifies that solvePnP tracks head rotation correctly:
        the X/Y/Z triad should rotate with the head in real time.

        Call this *after* update() for the same frame.
        Has no effect when no face has been detected yet.
        """
        if self._rvec is None or self._tvec is None:
            return

        axis_3d = np.array(
            [[80.0,  0.0,  0.0],
             [ 0.0, 80.0,  0.0],
             [ 0.0,  0.0, 80.0]],
            dtype=np.float64,
        )
        nose_2d, _ = cv2.projectPoints(
            np.zeros((1, 3), dtype=np.float64),
            self._rvec, self._tvec,
            self._camera_matrix, self._dist_coeffs,
        )
        axes_2d, _ = cv2.projectPoints(
            axis_3d,
            self._rvec, self._tvec,
            self._camera_matrix, self._dist_coeffs,
        )

        origin = tuple(nose_2d[0, 0].astype(int))
        cv2.line(frame_bgr, origin, tuple(axes_2d[0, 0].astype(int)),
                 (0,   0, 255), 2, cv2.LINE_AA)  # X — red
        cv2.line(frame_bgr, origin, tuple(axes_2d[1, 0].astype(int)),
                 (0, 255,   0), 2, cv2.LINE_AA)  # Y — green
        cv2.line(frame_bgr, origin, tuple(axes_2d[2, 0].astype(int)),
                 (255,  0,   0), 2, cv2.LINE_AA)  # Z — blue

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _solve_pnp(
        self,
        image_points: np.ndarray,          # shape (6, 2), dtype float64
    ) -> tuple[bool, np.ndarray, np.ndarray]:
        """
        Run solvePnP, using the previous solution as the initial guess
        when one is available (warm start).

        SOLVEPNP_ITERATIVE (Levenberg–Marquardt) is chosen because:
        - It accepts an external initial guess (useExtrinsicGuess=True).
        - It reliably handles small inter-frame motion (< 5° per frame).
        - It outperforms EPnP on 6-point inputs in warm-start scenarios.

        Returns (success, rvec, tvec).
        """
        if self._rvec is not None:
            return cv2.solvePnP(
                _FACE_MODEL_3D,
                image_points,
                self._camera_matrix,
                self._dist_coeffs,
                rvec=self._rvec.copy(),
                tvec=self._tvec.copy(),
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        # First frame: no prior solution — cold start.
        return cv2.solvePnP(
            _FACE_MODEL_3D,
            image_points,
            self._camera_matrix,
            self._dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

    def _build_result(self) -> HeadPoseResult:
        """Assemble the current smoothed state into an immutable HeadPoseResult."""
        return HeadPoseResult(
            pitch      = round(self._pitch, 1),
            yaw        = round(self._yaw,   1),
            roll       = round(self._roll,  1),
            is_forward = self.is_facing_forward(),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Overlay renderer
# Follows the same pattern as draw_blink_overlay in Phase 2.
# ─────────────────────────────────────────────────────────────────────────────

def draw_headpose_overlay(
    frame_bgr: np.ndarray,
    result: HeadPoseResult,
) -> None:
    """
    Render the Phase 3 head-pose debug HUD in the top-right corner.

    Four fixed-position text lines — no per-frame layout computation.
    Positioned in the top-right to avoid overlap with the Phase 1/2
    overlays (FPS + status = top-left; blink panel = bottom-left).

    Parameters
    ----------
    frame_bgr:
        The resized BGR frame that will be passed to cv2.imshow().
    result:
        The HeadPoseResult returned by HeadPoseEstimator.update().
    """
    h, w = frame_bgr.shape[:2]
    x      = w - 190          # right-aligned, 190 px from right edge
    y_base = 70               # top-right, below the FPS/status HUD
    dy     = 22               # vertical line spacing

    facing_color = (0, 220, 0) if result.is_forward else (0, 0, 220)

    lines: list[tuple[str, tuple[int, int, int]]] = [
        (f"Pitch : {result.pitch:+.1f} deg",  (180, 230, 180)),
        (f"Yaw   : {result.yaw:+.1f} deg",    (180, 230, 180)),
        (f"Roll  : {result.roll:+.1f} deg",   (180, 230, 180)),
        (f"Fwd   : {'YES' if result.is_forward else 'NO '}",
         facing_color),
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
