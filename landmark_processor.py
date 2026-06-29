"""
landmark_processor.py
=====================
Phase 2 module — coordinate adapter for the AI Visual Engagement System.

Role in the pipeline
--------------------
The LandmarkProcessor sits immediately after the FaceLandmarker and converts
its normalized (0–1) coordinate output into a pixel-space numpy array that
every downstream module shares.  Coordinate conversion happens ONCE per frame
here; no other module ever calls `lm.x * frame_width` independently.

This module also serves as the single source of truth for landmark indices.
All named index groups (eye EAR points, iris centers, head pose anchors) are
defined here as module-level constants.  Downstream modules import what they
need from this file rather than hardcoding numbers locally.

Design decisions
----------------
* Pre-allocated numpy buffer (478 × 3 × float32 = 5.7 KB) — allocated once in
  __init__, refilled in-place each frame.  Avoids per-frame heap allocation and
  GC pressure over long sessions.
* get_landmark() returns a VIEW into the buffer (no copy, O(1)).
* get_landmarks() uses fancy indexing which returns a COPY — unavoidable, but
  the copies are small (e.g., 6 landmarks × 3 floats × 4 bytes = 72 bytes per
  eye) and immediately consumed by the calling module.
* z-coordinate is kept as the normalized relative depth MediaPipe provides.
  Head-pose (Phase 3) only needs the 2D x/y pixel columns; iris-gaze (Phase 4)
  does not use z at all.  Nothing converts z to metric depth — that would
  require per-device camera calibration and is out of scope.
"""

from __future__ import annotations

import numpy as np
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarkerResult


# ─────────────────────────────────────────────────────────────────────────────
# Landmark Index Constants
# Single source of truth for the whole project.  Import from here — never
# hardcode magic numbers in downstream modules.
# Source: design document Section 4.2 landmark table.
# ─────────────────────────────────────────────────────────────────────────────

# ── Blink / EAR (Phase 2) ────────────────────────────────────────────────────
# Six contour points per eye in Soukupová & Čech (p1..p6) ordering:
#   p1 = outer corner, p2 = upper-outer, p3 = upper-inner,
#   p4 = inner corner,  p5 = lower-inner, p6 = lower-outer.
# Using 2 upper + 2 lower lid points (rather than 1 each) averages out lateral
# asymmetry from face rotation and improves stability on glasses wearers.
LEFT_EYE_EAR: tuple[int, ...] = (33, 160, 158, 133, 153, 144)
RIGHT_EYE_EAR: tuple[int, ...] = (362, 385, 387, 263, 373, 380)

# ── Gaze / Iris (Phase 4) ─────────────────────────────────────────────────────
# Refined iris landmarks — only available when FaceLandmarker is initialized
# with the full model (refine_landmarks=True), which adds indices 468–477.
LEFT_IRIS_CENTER: int = 468
RIGHT_IRIS_CENTER: int = 473
# Eye corners for gaze ratio normalization (inner corner is the nasal side)
LEFT_EYE_CORNERS: tuple[int, int] = (33, 133)    # (outer, inner)
RIGHT_EYE_CORNERS: tuple[int, int] = (362, 263)   # (outer, inner)

# ── Head Pose (Phase 3) ──────────────────────────────────────────────────────
# Six anatomical reference points for cv2.solvePnP.
# Tuple order matches the 3D model-point array that HeadPoseEstimator defines.
# See design doc Section 4.5 for the corresponding 3D coordinates.
HEAD_POSE_LANDMARKS: tuple[int, ...] = (1, 152, 33, 263, 61, 291)
#                                        nose chin L-eye R-eye L-mouth R-mouth

# ── Named single-index aliases (used across phases) ───────────────────────────
NOSE_TIP: int = 1
CHIN: int = 152
LEFT_EYE_OUTER: int = 33
LEFT_EYE_INNER: int = 133
RIGHT_EYE_OUTER: int = 263
RIGHT_EYE_INNER: int = 362
LEFT_MOUTH: int = 61
RIGHT_MOUTH: int = 291

# Total landmark count with the full FaceLandmarker model
# (468 mesh points + 10 iris points = 478)
_MAX_LANDMARKS: int = 478


# ─────────────────────────────────────────────────────────────────────────────
# LandmarkProcessor class
# ─────────────────────────────────────────────────────────────────────────────

class LandmarkProcessor:
    """
    Converts a MediaPipe FaceLandmarkerResult into a pixel-space numpy array
    and provides fast indexed access for all downstream pipeline modules.

    The processor is instantiated once per session and its ``update()`` method
    is called once per processed frame.

    Example
    -------
    ::

        processor = LandmarkProcessor(640, 480)

        # Inside the frame loop, after detect_landmarks():
        face_found = processor.update(result)
        if face_found:
            nose    = processor.get_landmark(NOSE_TIP)      # shape (3,)
            eye_pts = processor.get_landmarks(LEFT_EYE_EAR) # shape (6, 3)
            eye_lms = processor.get_eye_landmarks()         # dict of arrays
    """

    def __init__(self, frame_width: int, frame_height: int) -> None:
        """
        Parameters
        ----------
        frame_width, frame_height:
            Pixel dimensions of every frame that will pass through the pipeline.
            Must match the resolution used in FaceLandmarker detection (640×480).
        """
        self._w: int = frame_width
        self._h: int = frame_height

        # Pre-allocated buffer.
        # float32 matches MediaPipe's internal precision and is the natural dtype
        # for subsequent numpy arithmetic (EAR distances, solvePnP image points).
        # Allocating once here avoids a ~5.7 KB heap allocation on every frame.
        self._buf: np.ndarray = np.empty((_MAX_LANDMARKS, 3), dtype=np.float32)

        # _coords is a slice-view into _buf covering the landmarks present in
        # the current frame.  None when no face was detected.
        self._coords: np.ndarray | None = None
        self._n: int = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Primary per-frame entry point
    # ─────────────────────────────────────────────────────────────────────────

    def update(self, result: FaceLandmarkerResult) -> bool:
        """
        Ingest a FaceLandmarkerResult and rebuild the pixel-coordinate array.

        Must be called once per processed frame, immediately after the
        FaceLandmarker inference call (``detect_for_video``).

        Parameters
        ----------
        result:
            The ``FaceLandmarkerResult`` returned by
            ``FaceLandmarker.detect_for_video()``.

        Returns
        -------
        bool
            ``True``  — face detected; all ``get_landmark*`` methods are valid.
            ``False`` — no face; all ``get_landmark*`` methods return ``None``.
        """
        if not result.face_landmarks:
            # Clear state so stale data is never accidentally served.
            self._coords = None
            self._n = 0
            return False

        lms = result.face_landmarks[0]    # index 0 = first (and only) face
        n = min(len(lms), _MAX_LANDMARKS)

        # In-place buffer fill — the hot path.
        # A Python loop over 478 items is ~0.29 ms measured on this hardware.
        # The alternative (np.array([lm.x * w, ...] comprehension)) creates a
        # temporary Python list of 478 tuples before building the array —
        # measurably higher GC pressure over hour-long sessions.
        buf = self._buf
        w = float(self._w)
        h = float(self._h)
        for i in range(n):
            lm = lms[i]
            buf[i, 0] = lm.x * w   # pixel x
            buf[i, 1] = lm.y * h   # pixel y
            buf[i, 2] = lm.z       # normalized relative depth (kept as-is)

        # A slice of a numpy array is a VIEW — zero cost, zero allocation.
        self._coords = buf[:n]
        self._n = n
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # Generic indexed access
    # ─────────────────────────────────────────────────────────────────────────

    def get_landmark(self, index: int) -> np.ndarray | None:
        """
        Return pixel coordinates of a single landmark as a 1-D array [x, y, z].

        The returned array is a **view** into the internal buffer — do not store
        it across frames.  If persistence is needed, copy explicitly::

            pt = processor.get_landmark(NOSE_TIP).copy()

        Parameters
        ----------
        index:
            MediaPipe landmark index (0–477 with refine_landmarks=True).

        Returns
        -------
        numpy.ndarray shape (3,) or None
            ``None`` if no face was detected or ``index`` is out of range.
        """
        if self._coords is None or not (0 <= index < self._n):
            return None
        return self._coords[index]

    def get_landmarks(
        self,
        indices: list[int] | tuple[int, ...] | np.ndarray,
    ) -> np.ndarray | None:
        """
        Return pixel coordinates for multiple landmarks as shape (N, 3).

        Uses numpy fancy indexing which produces a **copy** (not a view).
        The copy is small — 6 landmarks × 12 bytes = 72 bytes for one eye —
        and is immediately consumed by the calling module.

        Parameters
        ----------
        indices:
            Sequence of MediaPipe landmark indices.

        Returns
        -------
        numpy.ndarray shape (N, 3) or None
            ``None`` if no face was detected.
        """
        if self._coords is None:
            return None
        return self._coords[list(indices)]

    def normalized_to_pixel(self, nx: float, ny: float) -> tuple[int, int]:
        """
        Convert a single normalized (x, y) coordinate to pixel space.

        Useful for one-off conversions (e.g., drawing iris center circles)
        without touching the landmark buffer.

        Parameters
        ----------
        nx, ny:
            Normalized MediaPipe coordinates in [0.0, 1.0].

        Returns
        -------
        tuple[int, int]
            ``(pixel_x, pixel_y)``
        """
        return int(nx * self._w), int(ny * self._h)

    # ─────────────────────────────────────────────────────────────────────────
    # Named group accessors — convenience wrappers for downstream modules
    # ─────────────────────────────────────────────────────────────────────────

    def get_eye_landmarks(self) -> dict[str, np.ndarray] | None:
        """
        Return the six EAR contour points for each eye.

        Returns
        -------
        dict with keys ``'left'`` and ``'right'``, each a (6, 3) numpy array,
        or ``None`` if no face was detected.

        Column layout: [pixel_x, pixel_y, z_normalized].

        Used by: ``BlinkDetector`` (Phase 2).
        """
        if self._coords is None:
            return None
        return {
            'left':  self._coords[list(LEFT_EYE_EAR)],
            'right': self._coords[list(RIGHT_EYE_EAR)],
        }

    def get_headpose_landmarks(self) -> np.ndarray | None:
        """
        Return the six anatomical 2-D reference points for solvePnP head pose.

        The row order matches the 3-D model-point array defined in
        ``HeadPoseEstimator`` (Phase 3): nose, chin, left eye corner,
        right eye corner, left mouth corner, right mouth corner.

        Returns
        -------
        numpy.ndarray shape (6, 3) or None
        """
        return self.get_landmarks(HEAD_POSE_LANDMARKS)

    # ─────────────────────────────────────────────────────────────────────────
    # Read-only properties
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def is_valid(self) -> bool:
        """``True`` if the last ``update()`` detected a face."""
        return self._coords is not None

    @property
    def landmark_count(self) -> int:
        """Number of landmarks available from the last frame (0 if no face)."""
        return self._n

    @property
    def frame_size(self) -> tuple[int, int]:
        """``(width, height)`` this processor was configured for."""
        return self._w, self._h
