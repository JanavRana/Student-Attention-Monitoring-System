"""
Phase 1: Webcam Capture + MediaPipe Face Mesh Validation
==========================================================
Target hardware : AMD Ryzen 5 3450U, 8 GB RAM, CPU-only, Windows
MediaPipe       : 0.10.x+ (Tasks API — mp.solutions was removed)
Python          : 3.10+

What changed from the old API (mp.solutions.face_mesh):
  - MediaPipe 0.10 completely replaced mp.solutions with a Tasks API.
  - A separate .task model file must be present (auto-downloaded below).
  - Detector is now  mp_vision.FaceLandmarker  (VIDEO running mode).
  - Connection constants live in  mp_vision.FaceLandmarksConnections.
  - Drawing utilities are in  mp_vision.drawing_utils.

Pipeline this script validates:
    Webcam -> OpenCV capture -> resize 640x480 -> BGR->RGB ->
    mediapipe.Image -> FaceLandmarker.detect_for_video() ->
    draw connections + landmarks -> FPS counter -> imshow -> 'q' to quit

Nothing else is implemented here (no gaze, blink, pose, scoring, logging).
"""

import os
import sys
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import drawing_utils as mp_drawing

# ── Phase 2 additions ────────────────────────────────────────────────
from landmark_processor import LandmarkProcessor
from blink_detector import BlinkDetector, BlinkResult, EyeState, ClosureType

# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────
CAMERA_INDEX = 0            # change to 1, 2 … if your webcam isn't index 0
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480

# Minimum milliseconds to wait between processed frames.
# 66 ms ≈ 15 FPS ceiling — enough for attention monitoring while
# leaving CPU headroom on a Ryzen 5 3450U.
FRAME_DELAY_MS = 66

# MediaPipe FaceLandmarker model — downloaded automatically if absent.
# Place it anywhere you like; just update MODEL_PATH to match.
MODEL_FILENAME = "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

# FaceLandmarker settings
NUM_FACES            = 1    # single-student monitoring
MIN_DETECTION_CONF   = 0.6
MIN_PRESENCE_CONF    = 0.6  # Tasks API uses this instead of min_tracking_confidence
MIN_TRACKING_CONF    = 0.6

WINDOW_NAME = "Phase 1 – Face Mesh Validation  |  press 'q' to quit"


# ─────────────────────────────────────────────────────────────────────
# Connection subsets used for drawing
# Two draw passes: light tessellation (full mesh) then bold contours.
# ─────────────────────────────────────────────────────────────────────
FLC = mp_vision.FaceLandmarksConnections   # shorthand

# Convert frozenset of Connection namedtuples to the list-of-Connection
# format expected by drawing_utils.draw_landmarks.
TESSELATION = list(FLC.FACE_LANDMARKS_TESSELATION)
CONTOURS    = (
    list(FLC.FACE_LANDMARKS_LEFT_EYE)
    + list(FLC.FACE_LANDMARKS_RIGHT_EYE)
    + list(FLC.FACE_LANDMARKS_LEFT_EYEBROW)
    + list(FLC.FACE_LANDMARKS_RIGHT_EYEBROW)
    + list(FLC.FACE_LANDMARKS_LIPS)
    + list(FLC.FACE_LANDMARKS_FACE_OVAL)
)
IRIS_CONTOURS = (
    list(FLC.FACE_LANDMARKS_LEFT_IRIS)
    + list(FLC.FACE_LANDMARKS_RIGHT_IRIS)
)


def download_model(path: str, url: str) -> None:
    """
    Download the FaceLandmarker .task model file if it isn't already present.
    The file is ~2.5 MB (float16 variant) and only needs to be downloaded once.
    """
    if os.path.exists(path):
        return

    print(f"[INFO] Model file '{path}' not found — downloading (~2.5 MB)...")
    print(f"       URL: {url}")

    try:
        # Add a User-Agent header; some CDNs reject plain urllib requests.
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as response, \
             open(path, "wb") as out_file:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 65536  # 64 KB chunks
            while chunk := response.read(chunk_size):
                out_file.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r       {pct:5.1f}%  ({downloaded:,} / {total:,} bytes)", end="", flush=True)
        print(f"\n[INFO] Model saved to '{path}'.")
    except Exception as exc:
        # Clean up a partial file so the next run retries properly.
        if os.path.exists(path):
            os.remove(path)
        raise RuntimeError(
            f"Failed to download model: {exc}\n"
            "       Manual download: go to the URL above and save the file as\n"
            f"       '{os.path.abspath(path)}'"
        ) from exc


def init_webcam(camera_index: int) -> cv2.VideoCapture:
    """
    Open the webcam and confirm it is readable.
    Raises RuntimeError with a helpful message if unavailable.
    """
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam at index {camera_index}.\n"
            "  • Check the camera is connected and not in use by another app.\n"
            "  • Try changing CAMERA_INDEX to 1 or 2 if you have multiple cameras."
        )

    # Request the target resolution; the driver may return something different,
    # which is why we resize every frame explicitly in the main loop.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # Cap the camera's own FPS to our target — avoids the driver buffering
    # extra frames that we'll never use, which wastes memory and adds latency.
    cap.set(cv2.CAP_PROP_FPS, 1000 / FRAME_DELAY_MS)

    return cap


def init_face_landmarker(model_path: str) -> mp_vision.FaceLandmarker:
    """
    Create the FaceLandmarker detector in VIDEO running mode.

    VIDEO mode vs IMAGE mode:
      IMAGE mode  — treats every frame independently (full detection every time).
      VIDEO mode  — uses internal temporal tracking: after the first detection,
                    it seeds subsequent frames with the previous result, which
                    is much cheaper and produces smoother landmark positions.
                    This is the correct mode for a webcam stream.

    Why these confidence values (0.6):
      Too low  (e.g. 0.3) → jittery, low-quality locks in dim lighting.
      Too high (e.g. 0.9) → frequent tracking drops that force expensive
                             full re-detections, hammering CPU.
      0.6 is a balanced starting point for an average laptop webcam indoors.
    """
    base_opts = mp_python.BaseOptions(model_asset_path=model_path)
    options   = mp_vision.FaceLandmarkerOptions(
        base_options                     = base_opts,
        running_mode                     = mp_vision.RunningMode.VIDEO,
        num_faces                        = NUM_FACES,
        min_face_detection_confidence    = MIN_DETECTION_CONF,
        min_face_presence_confidence     = MIN_PRESENCE_CONF,
        min_tracking_confidence          = MIN_TRACKING_CONF,
        output_face_blendshapes          = False,   # not needed in Phase 1–4
        output_facial_transformation_matrixes = False,   # we compute solvePnP ourselves in Phase 3
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


def detect_landmarks(
    detector: mp_vision.FaceLandmarker,
    frame_bgr: np.ndarray,
    timestamp_ms: int,
):
    """
    Resize the frame, convert to RGB, wrap in a mediapipe.Image, and run
    landmark detection.

    Returns (resized_bgr_frame, FaceLandmarkerResult).
    The resized frame is returned so downstream drawing works on the same
    array that was passed to MediaPipe (same dimensions, same content).
    """
    resized_bgr = cv2.resize(frame_bgr, (FRAME_WIDTH, FRAME_HEIGHT))

    # MediaPipe expects RGB; OpenCV delivers BGR.
    rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)

    # mp.Image is a lightweight wrapper — it does NOT copy the array.
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # detect_for_video requires monotonically increasing timestamps (ms).
    result = detector.detect_for_video(mp_img, timestamp_ms)

    return resized_bgr, result


def draw_face_mesh(frame_bgr: np.ndarray, result) -> bool:
    """
    Draw the face mesh on the frame using the Tasks API drawing utilities.

    Two-pass drawing:
      Pass 1 — full tessellation in dim grey (shows all 478 points as a mesh).
      Pass 2 — bold contours for eyes, eyebrows, lips, face oval, and irises.

    Returns True if a face was found and drawn, False otherwise.
    """
    if not result.face_landmarks:
        return False   # no face detected this frame

    # The Tasks API draw_landmarks() operates on a BGR numpy array directly.
    # Unlike the old solutions API, it does NOT need the frame as an mp.Image.
    for face_lms in result.face_landmarks:

        # ── Pass 1: light tessellation ────────────────────────────────
        mp_drawing.draw_landmarks(
            image                  = frame_bgr,
            landmark_list          = face_lms,
            connections            = TESSELATION,
            landmark_drawing_spec  = None,           # hide individual dots in this pass
            connection_drawing_spec= mp_drawing.DrawingSpec(
                color=(50, 50, 50), thickness=1, circle_radius=1
            ),
            is_drawing_landmarks   = False,
        )

        # ── Pass 2: bold contour lines ────────────────────────────────
        mp_drawing.draw_landmarks(
            image                  = frame_bgr,
            landmark_list          = face_lms,
            connections            = CONTOURS,
            landmark_drawing_spec  = mp_drawing.DrawingSpec(
                color=(0, 255, 0), thickness=1, circle_radius=1
            ),
            connection_drawing_spec= mp_drawing.DrawingSpec(
                color=(0, 255, 0), thickness=1, circle_radius=1
            ),
            is_drawing_landmarks   = True,
        )

        # ── Pass 3: iris circles in cyan ──────────────────────────────
        # Iris landmarks (468–477) are only available when the model
        # includes refined landmarks, which the float16 model does.
        mp_drawing.draw_landmarks(
            image                  = frame_bgr,
            landmark_list          = face_lms,
            connections            = IRIS_CONTOURS,
            landmark_drawing_spec  = None,
            connection_drawing_spec= mp_drawing.DrawingSpec(
                color=(255, 255, 0), thickness=2, circle_radius=1
            ),
            is_drawing_landmarks   = False,
        )

    return True


def draw_overlay(frame_bgr: np.ndarray, fps: float, face_found: bool) -> None:
    """
    Render a minimal HUD: FPS counter and face-detection status.
    Using LINE_AA (anti-aliased) keeps text readable at small sizes.
    """
    fps_text  = f"FPS: {fps:.1f}"
    face_text = "Face: DETECTED" if face_found else "Face: NOT FOUND"
    face_color = (0, 255, 0) if face_found else (0, 0, 255)

    cv2.putText(frame_bgr, fps_text,  (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame_bgr, face_text, (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, face_color,  2, cv2.LINE_AA)
    cv2.putText(frame_bgr, "Press 'q' to quit", (10, FRAME_HEIGHT - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)


def draw_blink_overlay(frame_bgr: np.ndarray, blink_result: BlinkResult) -> None:
    """
    Render Phase 2 blink debug HUD in the bottom-left corner.
    Six fixed-position text lines — no per-frame layout computation.
    """
    x        = 10
    y_base   = FRAME_HEIGHT - 135    # anchor: 135 px from bottom edge
    line_gap = 22                    # vertical gap between lines

    is_closed    = blink_result.eye_state is EyeState.CLOSED
    state_color  = (0, 0, 220) if is_closed else (0, 210, 0)

    # Build display lines: (text, color)
    lines = [
        (f"Blinks : {blink_result.blink_count}",            (230, 230, 230)),
        (f"L-EAR  : {blink_result.left_ear:.3f}",           (180, 180, 180)),
        (f"R-EAR  : {blink_result.right_ear:.3f}",          (180, 180, 180)),
        (f"Avg EAR: {blink_result.average_ear:.3f}",        (180, 210, 255)),
        (f"Eyes   : {blink_result.eye_state.value}",        state_color),
    ]

    # Show live closure type + duration only when eyes are closed or
    # immediately after — avoids a stale label cluttering the normal view.
    if is_closed or blink_result.closure_duration_s > 0.0:
        event_text  = blink_result.last_closure_type.value
        dur_text    = f"[{event_text}  {blink_result.closure_duration_s:.2f}s]"
        dur_color   = (0, 140, 255) if is_closed else (130, 130, 130)
        lines.append((dur_text, dur_color))

    for i, (text, color) in enumerate(lines):
        cv2.putText(
            frame_bgr, text, (x, y_base + i * line_gap),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA,
        )


def main() -> None:
    # ── 1. Ensure model file is present ──────────────────────────────
    try:
        download_model(MODEL_FILENAME, MODEL_URL)
    except RuntimeError as err:
        print(f"[ERROR] {err}", file=sys.stderr)
        return

    # ── 2. Open webcam ────────────────────────────────────────────────
    try:
        cap = init_webcam(CAMERA_INDEX)
    except RuntimeError as err:
        print(f"[ERROR] {err}", file=sys.stderr)
        return

    # ── 3. Create detector ────────────────────────────────────────────
    detector = init_face_landmarker(MODEL_FILENAME)

    processor     = LandmarkProcessor(FRAME_WIDTH, FRAME_HEIGHT)
    blink_det     = BlinkDetector()

    print("[INFO] Pipeline initialised — showing live feed.")
    print("       Green mesh = face detected.  Red text = no face found.")

    # Monotonic session start used to build timestamps for detect_for_video().
    session_start_s = time.monotonic()
    prev_time_s     = session_start_s
    consecutive_read_failures = 0

    try:
        while True:
            ret, frame = cap.read()

            if not ret or frame is None:
                consecutive_read_failures += 1
                if consecutive_read_failures >= 30:
                    print("[ERROR] 30 consecutive frame-read failures — webcam likely disconnected.",
                          file=sys.stderr)
                    break
                print(f"[WARN] Frame read failed (attempt {consecutive_read_failures}) — retrying.",
                      file=sys.stderr)
                time.sleep(0.05)
                continue

            consecutive_read_failures = 0   # reset on a successful read

            # Timestamp in ms required by detect_for_video(); must be
            # monotonically increasing — time.monotonic() guarantees this.
            timestamp_ms = int((time.monotonic() - session_start_s) * 1000)

            resized_frame, result = detect_landmarks(detector, frame, timestamp_ms)

            # ── Phase 2: coordinate conversion + blink detection ────────────────
            processor.update(result)
            timestamp_s  = timestamp_ms / 1000.0
            blink_result = blink_det.update(processor, timestamp_s)

            face_found = draw_face_mesh(resized_frame, result)


            # FPS: simple delta-time (no rolling average needed for a
            # live display; a 1-frame delta is stable enough at 10-15 FPS).
            now = time.monotonic()
            fps = 1.0 / max(now - prev_time_s, 1e-6)
            prev_time_s = now

            # Existing overlay + new blink overlay
            draw_overlay(resized_frame, fps, face_found)
            draw_blink_overlay(resized_frame, blink_result) 

            cv2.imshow(WINDOW_NAME, resized_frame)

            # waitKey(1) keeps the GUI event loop alive.
            # 'q' exits cleanly; ESC (27) is also caught as a convenience.
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by Ctrl+C.")

    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()
        print("[INFO] Resources released. Session ended cleanly.")


if __name__ == "__main__":
    main()
