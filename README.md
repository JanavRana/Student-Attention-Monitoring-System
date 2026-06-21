# AI-Based Student Attention Monitoring System
## Software Requirements Specification & Technical Design Document

**Version:** 1.0
**Document Type:** Architecture & Implementation Blueprint
**Target Platform:** Python 3.x, CPU-only, single laptop webcam

---

## 0. Design Philosophy (Read This First)

Before any architecture, one principle governs every decision in this document:

> **The system estimates a proxy for sustained visual engagement with the screen. It does not, and cannot, detect "attention" as a mental or cognitive state.**

A webcam can observe gaze direction, eye openness, head pose, and face presence. It cannot observe working memory, comprehension, motivation, or whether the student is mentally present even while staring at the screen. Every module below is named and scoped honestly: "Gaze Analyzer," not "Mind Reader." This matters for two practical reasons, not just ethics:

1. **It keeps the math honest.** If you scope the system as "visual engagement estimation," your thresholds and weights stay defensible. If you scope it as "attention detection," you will be tempted to over-fit thresholds to make the number look more authoritative than the underlying signal supports.
2. **It keeps false positives manageable.** A student who looks down to take notes, glances at a second monitor, or rests their chin on their hand is not necessarily inattentive. The design below builds in tolerance windows specifically so the system doesn't punish normal human behavior.

Treat the final "attention score" as what it is: a weighted combination of observable proxies, smoothed over time, with stated confidence limitations. The report generator should communicate this to end users in plain language.

---

## 1. Project Overview

### 1.1 Objective
Build a real-time, webcam-based system that:
- Continuously observes a student during an online class.
- Derives a set of observable visual proxies (gaze direction, eye closure, head pose, face presence).
- Combines these proxies into a smoothed, time-windowed **attention score** (0–100).
- Produces a live on-screen indicator and a structured end-of-session report.

### 1.2 Scope
**In scope:** single-face, single-webcam, real-time CPU-only processing; geometric/classical computer vision (no deep gaze models); local session logging and reporting.

**Out of scope:** emotion recognition, multi-student classroom monitoring, cloud processing, biometric identification, audio analysis, keystroke/mouse monitoring, or any claim of detecting sleep, boredom, or comprehension as definite facts.

### 1.3 Key Constraints
- Must run in real time (target ≥15 FPS) on a standard laptop CPU — no GPU dependency.
- Must use only the device's own camera locally; no frames should need to leave the machine for core functionality.
- Must degrade gracefully: poor lighting, glasses, partial occlusion, or face loss should not crash the pipeline or produce wildly incorrect scores.

### 1.4 Primary Users
A single student running the tool on their own machine during a self-paced or live online class, or an educator who wants a personal post-session engagement summary (not a surveillance tool for monitoring others without consent).

---

## 2. System Architecture

### 2.1 Architectural Style
A **modular pipeline architecture** with a central orchestrator. Each pipeline stage is a self-contained class with a narrow interface (one primary `process()`-style method), making the system testable in isolation and easy to extend later (e.g., swapping the gaze module for a deep-learning version without touching anything else).

### 2.2 Module List

| # | Module | Responsibility |
|---|--------|-----------------|
| 1 | `WebcamManager` | Captures frames from the camera in a dedicated thread; handles device errors/disconnects. |
| 2 | `FaceMeshDetector` | Runs MediaPipe Face Mesh; outputs 468 landmark points or "no face" signal. |
| 3 | `LandmarkProcessor` | Converts raw landmark output into normalized, indexed coordinate structures used by downstream modules. |
| 4 | `GazeAnalyzer` | Estimates gaze direction (center/left/right/up/down) from eye/iris landmarks. |
| 5 | `BlinkDetector` | Computes Eye Aspect Ratio (EAR); classifies blink vs. long closure vs. possible sleep. |
| 6 | `HeadPoseEstimator` | Computes yaw/pitch/roll via solvePnP. |
| 7 | `FacePresenceTracker` | Tracks continuous vs. intermittent vs. extended face absence. |
| 8 | `MovementStabilityAnalyzer` (optional) | Tracks frame-to-frame head movement variance as a minor signal. |
| 9 | `AttentionScoringEngine` | Fuses all signals into a smoothed, weighted attention score per time window. |
| 10 | `SessionLogger` | Records per-second/per-window data to memory and disk. |
| 11 | `ReportGenerator` | Produces end-of-session summary, statistics, and charts. |
| 12 | `DashboardUI` | Renders the live video feed plus overlay/status panel. |
| 13 | `SessionController` (orchestrator) | Owns the main loop; wires modules together; manages session lifecycle (start/pause/stop). |

### 2.3 Data Flow Diagram

```
                         ┌─────────────────┐
                         │  WebcamManager   │  (capture thread)
                         └────────┬─────────┘
                                  │ raw BGR frame
                                  ▼
                         ┌─────────────────┐
                         │ FaceMeshDetector │
                         └────────┬─────────┘
                       face found │ no face
                  ┌───────────────┴───────────────┐
                  ▼                                ▼
        ┌──────────────────┐             ┌──────────────────────┐
        │ LandmarkProcessor │             │ FacePresenceTracker  │
        └─────────┬─────────┘             │ (records absence)    │
                   │ normalized landmarks  └───────────┬──────────┘
     ┌─────────────┼──────────────┬────────────────┐    │
     ▼              ▼              ▼                ▼    │
┌──────────┐  ┌─────────────┐ ┌──────────────┐ ┌──────────────────┐
│GazeAnalyzer│ │BlinkDetector│ │HeadPoseEstim.│ │MovementStability  │
└─────┬──────┘  └─────┬───────┘ └──────┬───────┘ └─────────┬─────────┘
      │                │                │                   │
      └────────────────┴────────┬───────┴───────────────────┘
                                  ▼
                     ┌──────────────────────────┐
                     │  AttentionScoringEngine   │ ◄── FacePresenceTracker
                     │ (per-frame → smoothed)    │
                     └────────────┬──────────────┘
                                  │ score + sub-metrics
                    ┌─────────────┼──────────────┐
                    ▼                             ▼
           ┌─────────────────┐           ┌────────────────┐
           │  SessionLogger   │           │   DashboardUI   │
           │ (CSV/JSON write) │           │ (live overlay)  │
           └────────┬─────────┘           └────────────────┘
                    │ on session end
                    ▼
           ┌─────────────────┐
           │ ReportGenerator  │ → final report (PDF/HTML) + charts
           └─────────────────┘
```

### 2.4 Threading Model
- **Thread 1 (Capture):** `WebcamManager` continuously reads frames into a thread-safe single-slot buffer (latest frame wins; older unread frames are dropped to avoid lag build-up).
- **Thread 2 (Processing, main loop):** Pulls the latest frame, runs the CV pipeline, updates the scoring engine, updates the UI.
- **Thread 3 (Logging, optional):** Writes to disk asynchronously so file I/O never blocks frame processing.

This separation keeps webcam I/O latency from stalling the perception pipeline, which is the most common cause of "laggy" CV demos.

---

## 3. Technology Stack

| Purpose | Library | Why |
|---|---|---|
| Video capture | `OpenCV` (`cv2.VideoCapture`) | Mature, fast, cross-platform camera access. |
| Face & landmark detection | `MediaPipe Face Mesh` (`mediapipe.solutions.face_mesh`) | CPU-optimized, 468 3D landmarks including iris points, runs comfortably in real time on laptops without a GPU — far lighter than running a full deep gaze-estimation network. |
| Head pose math | `OpenCV solvePnP` + `numpy` | Standard, well-documented PnP solver; no extra dependency. |
| Numerical processing | `numpy` | Vector math for EAR, ratios, angle calculations. |
| Smoothing/statistics | `collections.deque`, `numpy` (rolling mean), optionally `scipy.signal` | Lightweight moving-average / exponential smoothing without heavy dependencies. |
| Logging/storage | `csv` / `json` (standard library), optionally `sqlite3` for larger sessions | No external DB server needed; CSV/JSON is sufficient and portable. |
| Reporting & charts | `matplotlib`, `pandas` | Standard, reliable for static charts and tabular summaries. |
| GUI / dashboard | `OpenCV `imshow` overlay (simplest)` or `Tkinter` / `PyQt5` for a richer dashboard | See Section 9 for trade-off discussion. |
| Concurrency | `threading`, `queue` | Lightweight, sufficient for this I/O + CPU-bound workload; no need for multiprocessing given MediaPipe's internal C++ efficiency. |

**Why MediaPipe Face Mesh specifically:** It returns 468 landmarks (478 with iris refinement enabled) in a single CPU-efficient pass, including dedicated iris landmarks needed for gaze estimation — eliminating the need for a separate eye-detection model or a deep gaze-estimation network. This is the single most important library choice in the stack because it collapses three traditionally separate problems (face detection, landmark localization, iris localization) into one fast call.

---

## 4. Detailed Algorithm Design

### 4.1 Face Detection & Tracking
- Initialize `FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.6, min_tracking_confidence=0.6)`.
- `refine_landmarks=True` is mandatory — it adds the iris landmarks (indices 468–477) needed for gaze estimation.
- MediaPipe internally tracks the face between frames (using the previous frame's landmarks to seed detection), so explicit external tracking (e.g., a Kalman filter on bounding boxes) is **not required** for a single-face use case. This significantly simplifies Module 2.
- If `results.multi_face_landmarks` is empty for a frame, treat it as "face not found" and route to `FacePresenceTracker` instead of crashing or holding stale landmarks.

### 4.2 Facial Landmark Selection
MediaPipe provides 468–478 points; using all of them is unnecessary and wastes compute. Maintain a curated index map:

| Region | Approx. Landmark Indices | Used By |
|---|---|---|
| Left eye contour | 33, 160, 158, 133, 153, 144 | EAR (blink detection) |
| Right eye contour | 362, 385, 387, 263, 373, 380 | EAR (blink detection) |
| Left iris center | 468 (refined) | Gaze estimation |
| Right iris center | 473 (refined) | Gaze estimation |
| Eye corners (L/R, inner/outer) | 33, 133, 362, 263 | Gaze ratio normalization |
| Nose tip | 1 | Head pose (2D-3D correspondence) |
| Chin | 152 | Head pose |
| Left/right eye outer corners | 33, 263 | Head pose |
| Left/right mouth corners | 61, 291 | Head pose |

Store landmarks as a `numpy` array of shape `(N, 3)` (x, y, z — MediaPipe gives normalized x/y in [0,1] and a relative z), converted to pixel coordinates `(x * frame_width, y * frame_height)` once per frame for all downstream modules to share, rather than each module re-converting independently.

### 4.3 Eye Gaze Estimation

**Design choice: iris-relative-to-eye-corner ratio, not full 3D gaze vector regression.**

Reasoning: full 3D gaze-vector estimation (the kind used in deep-learning gaze papers) requires calibrated camera intrinsics and per-user calibration to be accurate, and is overkill for a 5-class direction estimate (center/left/right/up/down). A 2D ratio-based heuristic, comparable to what's used in many real-time eye-tracking demos, is CPU-cheap and sufficiently accurate for coarse direction classification.

**Horizontal gaze ratio** (per eye):
```
gaze_ratio_x = (iris_x - eye_corner_inner_x) / (eye_corner_outer_x - eye_corner_inner_x)
```
This yields a value roughly in [0, 1]: near 0.5 = looking center, lower = looking toward one side, higher = the other side (sign convention depends on which eye/corner is "inner" vs "outer" for left vs right eye — handle mirroring consistently).

**Vertical gaze ratio:**
```
gaze_ratio_y = (iris_y - eyelid_top_y) / (eyelid_bottom_y - eyelid_top_y)
```

**Direction classification:**
```
if gaze_ratio_x < 0.35:        direction = "LEFT"
elif gaze_ratio_x > 0.65:      direction = "RIGHT"
elif gaze_ratio_y < 0.35:      direction = "UP"
elif gaze_ratio_y > 0.65:      direction = "DOWN"
else:                          direction = "CENTER"
```
(Thresholds are starting points — see Section 4.3.1 on calibration.)

**Why iris tracking over plain eye-region analysis:** Eye-region-only methods (e.g., tracking the white-of-eye to eyelid ratio without iris landmarks) are noisier because they're sensitive to eyelid shape and lighting on the sclera. MediaPipe's refined iris landmarks give a stable, sub-pixel-tracked point, making the ratio calculation much less jittery frame to frame.

**4.3.1 Handling natural eye movement (avoiding false "looking away" flags):**
- Apply a **rolling median filter** (e.g., last 5–8 frames) to `gaze_ratio_x`/`gaze_ratio_y` before classification — this absorbs micro-saccades.
- Require a direction to persist for a **minimum dwell time** (e.g., 0.4–0.6 seconds of consecutive non-center frames) before it's counted as "looking away" for scoring purposes. A single off-center frame is noise, not a behavior.
- **Grace period:** allow up to ~2 seconds of continuous off-center gaze (e.g., glancing at notes) without any score penalty at all. Only sustained looking-away beyond this grace period should begin reducing the attention sub-score. This directly satisfies the requirement to not penalize normal short glances.
- Recommend a **lightweight one-time calibration step** at session start: ask the user to look at 5 points (center, then each edge) for 1–2 seconds each, and use the observed ratios to set personalized thresholds instead of hardcoded population averages. This materially improves accuracy across different eye shapes/camera angles and is cheap to implement.

### 4.4 Blink & Eye Closure Detection

**Eye Aspect Ratio (EAR)** — the standard, well-validated formula (Soukupová & Čech):
```
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
```
where p1..p6 are the six eye-contour landmarks per eye (outer corner, two upper lid points, inner corner, two lower lid points), and `||·||` is Euclidean distance. Compute EAR for both eyes and average them (`avg_EAR = (EAR_left + EAR_right) / 2`) for robustness against single-eye landmark noise.

**Threshold selection:**
- EAR typically sits around 0.25–0.35 for open eyes and drops below ~0.20 for closed eyes — but this varies by person/camera, so **calibrate per-user** during the initial calibration step (ask the user to blink naturally a few times and record the EAR trough) rather than hardcoding a single global threshold.
- Use a working default of `EAR_THRESHOLD = 0.21` if no calibration is performed.

**Time-based classification (state machine):**
```
if avg_EAR < EAR_THRESHOLD:
    closed_frame_counter += 1
else:
    if closed_frame_counter > 0:
        closure_duration = closed_frame_counter / fps
        if closure_duration < 0.4s:        → "normal blink" (log count, no penalty)
        elif 0.4s <= duration < 2.0s:      → "long closure" (minor penalty, flagged)
        elif duration >= 2.0s:             → "possible sleep / inattention" (significant penalty, flagged event)
    closed_frame_counter = 0
```
- The 0.4s lower bound matters because normal voluntary/involuntary blinks last roughly 100–400ms; anything consistently longer is behaviorally different, not a blink.
- Avoid frame-by-frame instantaneous penalties for closed eyes — only the closure *event*, once resolved, should be classified and scored, which inherently filters single-frame noise.

**Handling glasses:**
- Reflections and frame occlusion can corrupt eye-contour landmark precision. Mitigations:
  - Use the per-user calibration step (Section 4.3.1) so the EAR baseline already reflects that user's eyes-with-glasses geometry.
  - If EAR readings become erratic (variance spikes far above the calibrated baseline noise floor) for a sustained period, treat the signal as **low-confidence** rather than asserting "eyes closed" — downweight the blink sub-score's contribution temporarily instead of forcing a classification on noisy data.
  - Advise users in the UI/documentation to ensure even, non-glare lighting if persistent low-confidence flags occur.

### 4.5 Head Pose Estimation

**Method: `cv2.solvePnP`** using a generic 3D face model and six corresponding 2D landmarks.

**3D model points** (approximate, in an arbitrary but consistent unit, generic adult face proportions):
```
NOSE_TIP    = ( 0.0,   0.0,   0.0)
CHIN        = ( 0.0,  -330.0, -65.0)
LEFT_EYE_OC = (-225.0,  170.0, -135.0)   # outer corner
RIGHT_EYE_OC= ( 225.0,  170.0, -135.0)
LEFT_MOUTH  = (-150.0, -150.0, -125.0)
RIGHT_MOUTH = ( 150.0, -150.0, -125.0)
```

**2D image points:** the corresponding pixel coordinates from MediaPipe landmark indices 1 (nose tip), 152 (chin), 33 (left eye outer), 263 (right eye outer), 61 (left mouth corner), 291 (right mouth corner).

**Camera matrix** (approximate, since most laptop webcams aren't individually calibrated):
```
focal_length = frame_width
camera_matrix = [[focal_length, 0, frame_width/2],
                  [0, focal_length, frame_height/2],
                  [0, 0, 1]]
dist_coeffs = zeros(4)   # assume negligible lens distortion
```

**Solve:**
```
success, rotation_vector, translation_vector = cv2.solvePnP(
    model_points, image_points, camera_matrix, dist_coeffs,
    flags=cv2.SOLVEPNP_ITERATIVE
)
rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
```
Decompose `rotation_matrix` into Euler angles (yaw, pitch, roll) using `cv2.RQDecomp3x3` or a manual Euler extraction (e.g., via `cv2.decomposeProjectionMatrix`), producing degrees for each axis.

**Suggested thresholds (calibratable):**
| Axis | "Facing screen" range | "Mild turn" (minor penalty) | "Distracted" (significant penalty) |
|---|---|---|---|
| Yaw | −15° to +15° | 15°–30° / −15°–−30° | beyond ±30° |
| Pitch | −10° to +15° (slight downward bias allowed for note-taking) | 15°–25° down | beyond 25° down, or any sustained upward tilt beyond 15° |
| Roll | −15° to +15° | n/a (roll alone rarely indicates distraction) | only combined with yaw/pitch extremes |

**Distinguishing natural movement from distraction:**
- Apply the same dwell-time logic as gaze: a brief yaw of 25° while reaching for a glass of water shouldn't register as distraction. Require sustained deviation (e.g., >1.5–2 seconds) past threshold before penalizing.
- Track **angle variance** over a short window (e.g., 1 second) — natural micro-adjustments while writing have low variance; if angles are bouncing wildly that's more indicative of restlessness/instability, while a smooth large excursion (turning to talk to someone off-camera) is a clearer distraction signal. This can feed the optional Movement & Stability module rather than head pose directly.

### 4.6 Face Presence Analysis

State machine driven by consecutive frames without a detected face:
```
FACE_VISIBLE        : face detected this frame
FACE_MISSING_BRIEF  : 0 < missing_duration <= 3s   (no penalty — likely camera glitch/adjustment)
FACE_MISSING_EXTENDED: 3s < missing_duration <= 15s (escalating penalty)
FACE_ABSENT_LONG    : missing_duration > 15s        (treat as session paused / heavily penalized, flagged in report)
```
**Reasoning for the 3-second grace window:** brief MediaPipe detection failures from fast head turns, hand-over-face gestures, or transient lighting changes are common and not necessarily inattention; 3 seconds filters most of these without masking real absence.

If absence exceeds a configurable hard limit (e.g., 30–60 seconds), `SessionController` may optionally auto-pause the session clock so a student stepping away doesn't have their average score destroyed by a bathroom break — this should be a user-configurable policy, not a silent assumption, and should be clearly noted in the final report either way.

### 4.7 Movement & Stability Analysis (Optional)

**Recommendation: include as a small-weight signal, not a primary one.**

- **What it measures:** frame-to-frame variance in head position/orientation (e.g., standard deviation of yaw/pitch/roll or nose-tip pixel position over a rolling 2–3 second window).
- **Does it improve accuracy?** Marginally — it's useful mainly for catching restlessness/fidgeting that doesn't cross the head-pose distraction threshold but still suggests disengagement (e.g., constant small movements vs. stable focus).
- **Suggested weight:** ≤5% of total score — it is a weak, indirect signal.
- **Disadvantages:**
  - Risk of penalizing legitimate behaviors: students with ADHD, chronic pain requiring position shifts, or simply different working styles may move more without being less engaged. This is the strongest argument for keeping its weight minimal and never making it a hard cutoff.
  - Adds another noisy signal that can reduce overall system interpretability if over-weighted.
- **Conclusion:** implement it as an optional, clearly-labeled minor modifier that can be disabled entirely via configuration.

---

## 5. Mathematical Models — Attention Scoring Engine

### 5.1 Per-Frame Sub-Scores
Each module outputs a normalized sub-score in [0, 100] per processed frame (not every raw camera frame needs scoring — see Section 8 on processing frequency):

| Sub-score | Derived From | Logic Summary |
|---|---|---|
| `S_gaze` | GazeAnalyzer | 100 if CENTER; decays the longer/further off-center, floored at 0 after sustained off-center beyond grace period |
| `S_blink` | BlinkDetector | 100 normally; drops sharply during "long closure," near 0 during "possible sleep" events |
| `S_headpose` | HeadPoseEstimator | 100 within "facing screen" range; linear decay through "mild turn"; 0 beyond "distracted" threshold |
| `S_presence` | FacePresenceTracker | 100 if visible; 100 still during brief grace window; decays through extended-missing; 0 at long-absent |
| `S_stability` (optional) | MovementStabilityAnalyzer | 100 at low variance; gentle decay at high variance, floored at ~60 (never the dominant cause of a low score) |

**Example decay function (used for gaze/headpose/presence beyond their grace windows), a simple linear ramp is sufficient and explainable:**
```
S = 100                                   if deviation <= grace_threshold
S = 100 * (1 - (t - grace) / ramp_window) if grace_threshold < t <= grace + ramp_window
S = 0                                     if t > grace + ramp_window
```
where `t` is the sustained duration of the deviation in seconds, and `ramp_window` is a tunable value (e.g., 3–5 seconds) controlling how quickly the score falls to zero after the grace period ends.

### 5.2 Weighted Combination (Per-Frame Composite)
```
S_frame = (w_gaze * S_gaze) + (w_blink * S_blink) +
          (w_headpose * S_headpose) + (w_presence * S_presence) +
          (w_stability * S_stability)
```

**Suggested default weights** (sum to 1.0):

| Signal | Weight | Rationale |
|---|---|---|
| Gaze | 0.30 | Most direct proxy for "looking at the screen." |
| Head pose | 0.25 | Strong secondary indicator; correlates with gaze but catches cases where eyes are occluded. |
| Face presence | 0.25 | Binary-ish but critical — no face means no possible attention reading. |
| Blink/eye closure | 0.15 | Important but should not dominate, since normal blinking is frequent and brief. |
| Movement stability (optional) | 0.05 | Weak/secondary, as discussed in 4.7. |

If the optional stability module is disabled, redistribute its 0.05 proportionally across the other four (or simply re-normalize the remaining weights to sum to 1.0).

### 5.3 Temporal Smoothing
Raw `S_frame` values are noisy. Apply **both** of the following:

**(a) Short-window moving average** (e.g., last 1 second of processed frames) to remove frame-level jitter:
```
S_smoothed[t] = mean(S_frame[t - N : t])    # N = frames in ~1 second
```

**(b) Exponential Moving Average (EMA)** for the score displayed live and used for trend calculation, which reacts faster than a long block average but still resists spikes:
```
S_ema[t] = alpha * S_smoothed[t] + (1 - alpha) * S_ema[t-1]
```
Suggested `alpha = 0.2` (slower, more stable display) up to `0.4` (more responsive). Lower alpha = smoother but more lag; this is a tunable UX trade-off, not a fixed constant.

**Update frequency:** Recompute and display `S_ema` once per second is sufficient for a human-readable live indicator — there's no benefit to updating a "you are 73% attentive" number 30 times a second; it would look jittery and convey false precision. Internally, process frames at the full processing rate (Section 8), but only push UI/log updates at ~1 Hz.

### 5.4 Session-Level Aggregate Score
At session end (or for any reporting window):
```
SessionScore = mean(S_ema over the full session duration)
```
Optionally weight by time rather than by sample count if logging intervals are irregular (use a time-weighted average: integrate `S_ema(t)` over session duration and divide by total duration).

### 5.5 Attention Categories
| Category | Score Range | Interpretation |
|---|---|---|
| Highly Attentive | 80–100 | Sustained gaze/pose toward screen, minimal distraction events |
| Moderately Attentive | 60–79 | Generally engaged with periodic, normal lapses |
| Low Attention | 35–59 | Frequent or longer distraction periods |
| Highly Distracted | 0–34 | Mostly absent, looking away, or eyes closed for extended periods |

**Reasoning for thresholds:** They're deliberately wide-banded rather than precise (e.g., not 79.4 vs. 80.1 being a meaningfully different category) because the underlying signal has real measurement uncertainty — fine-grained category boundaries would imply false precision. The bands are also intentionally generous at the top (a wide 80–100 "highly attentive" band) so normal human variability (blinking, occasional note-taking glances) doesn't artificially suppress a genuinely engaged session into a lower-sounding category.

---

## 6. Module / Class Design

> Class skeletons below define structure and contracts only (attributes, method signatures, and responsibilities) — implementation logic should follow Section 4 and 5 during the coding phase.

```python
class WebcamManager:
    def __init__(self, camera_index=0, target_fps=30): ...
    def start(self): ...                 # launches capture thread
    def get_latest_frame(self) -> np.ndarray | None: ...
    def is_connected(self) -> bool: ...
    def stop(self): ...

class FaceMeshDetector:
    def __init__(self, refine_landmarks=True, min_detection_confidence=0.6): ...
    def process(self, frame) -> FaceMeshResult | None: ...   # None if no face

class LandmarkProcessor:
    def __init__(self, frame_width, frame_height): ...
    def to_pixel_coords(self, mediapipe_landmarks) -> np.ndarray: ...
    def get_eye_landmarks(self, coords) -> dict: ...
    def get_headpose_landmarks(self, coords) -> dict: ...

class GazeAnalyzer:
    def __init__(self, calibration: GazeCalibration | None = None): ...
    def update(self, eye_landmarks, timestamp) -> GazeResult: ...
    # GazeResult: direction (enum), confidence, sustained_duration, sub_score

class BlinkDetector:
    def __init__(self, ear_threshold=0.21, fps=30): ...
    def update(self, eye_landmarks, timestamp) -> BlinkResult: ...
    # BlinkResult: ear_value, state (OPEN/BLINK/LONG_CLOSURE/POSSIBLE_SLEEP), sub_score

class HeadPoseEstimator:
    def __init__(self, frame_width, frame_height): ...
    def update(self, pose_landmarks, timestamp) -> HeadPoseResult: ...
    # HeadPoseResult: yaw, pitch, roll, state, sub_score

class FacePresenceTracker:
    def __init__(self, brief_timeout=3.0, extended_timeout=15.0): ...
    def update(self, face_found: bool, timestamp) -> PresenceResult: ...
    # PresenceResult: state, missing_duration, sub_score

class MovementStabilityAnalyzer:
    def __init__(self, window_seconds=2.0): ...
    def update(self, headpose_result, timestamp) -> float: ...  # sub_score

class AttentionScoringEngine:
    def __init__(self, weights: dict, smoothing_alpha=0.3): ...
    def update(self, sub_scores: dict, timestamp) -> AttentionState: ...
    # AttentionState: frame_score, smoothed_score, ema_score, category

class SessionLogger:
    def __init__(self, output_path, format="csv"): ...
    def log_tick(self, timestamp, attention_state, raw_metrics: dict): ...
    def flush(self): ...
    def close(self) -> str: ...   # returns file path

class ReportGenerator:
    def __init__(self, session_log_path): ...
    def generate_summary(self) -> SessionSummary: ...
    def generate_charts(self, output_dir) -> list[str]: ...
    def export(self, format="html") -> str: ...

class DashboardUI:
    def __init__(self, mode="opencv" | "tkinter" | "pyqt"): ...
    def render_frame(self, frame, attention_state, statuses: dict): ...
    def handle_events(self) -> bool: ...   # returns False on quit

class SessionController:
    def __init__(self, config: dict): ...
    def run(self): ...     # main orchestration loop
    def pause(self): ...
    def stop(self) -> SessionSummary: ...
```

---

## 7. Data Flow (Per-Frame Pipeline Summary)

1. `WebcamManager` produces a frame (or signals disconnect).
2. `FaceMeshDetector` runs inference → landmarks or `None`.
3. If `None` → `FacePresenceTracker.update(False, t)`; skip steps 4–7 for this frame; go to step 8 with presence-only data.
4. `LandmarkProcessor` converts to pixel coordinates and slices region-specific landmark sets.
5. `GazeAnalyzer`, `BlinkDetector`, `HeadPoseEstimator` each run `update()` independently (they are mutually independent — can be parallelized if profiling shows benefit, though typically unnecessary given MediaPipe's speed).
6. `MovementStabilityAnalyzer.update()` (optional, fed by head pose history).
7. `FacePresenceTracker.update(True, t)`.
8. All sub-scores collected into a dict and passed to `AttentionScoringEngine.update()`.
9. Resulting `AttentionState` passed to `DashboardUI` (every frame for video smoothness) and `SessionLogger` (throttled to ~1 Hz, see Section 8).
10. Loop continues until stop condition (user quits, session timer ends, or webcam disconnects beyond a recovery timeout).
11. On stop: `SessionLogger.close()` finalizes the log file; `ReportGenerator` consumes it to build the final report.

---

## 8. Data Storage Design

### 8.1 What to Log Per Tick (recommended: once per second, not every processed frame)
```json
{
  "timestamp": "2026-06-20T10:15:32",
  "session_elapsed_sec": 932,
  "attention_score": 78.4,
  "attention_category": "Moderately Attentive",
  "gaze_direction": "CENTER",
  "gaze_subscore": 85.0,
  "eye_state": "OPEN",
  "blink_count_cumulative": 142,
  "long_closure_events_cumulative": 1,
  "head_yaw": -4.2,
  "head_pitch": 7.8,
  "head_roll": 1.1,
  "headpose_subscore": 92.0,
  "face_present": true,
  "presence_subscore": 100.0,
  "stability_subscore": 95.0
}
```

### 8.2 In-Memory Structures
- A `collections.deque(maxlen=N)` per signal for rolling-window calculations (e.g., last 30 frames of EAR for smoothing) — bounded memory, O(1) append/evict.
- A single growing `list[dict]` (or `pandas.DataFrame` built incrementally) for the full-session per-tick log, since a 60-minute session at 1 Hz is only ~3,600 rows — trivially small in memory.

### 8.3 File Format Choice
- **CSV** for the per-tick session log: simplest, human-readable, directly loadable into `pandas` for the report stage, and avoids any schema/versioning overhead. Recommended default.
- **JSON** for the final session summary (totals, categorical breakdowns, metadata like calibration values used) — better suited to nested/structured summary data than flat CSV rows.
- **SQLite** only if you anticipate needing to query across *many* historical sessions (e.g., a teacher reviewing weeks of sessions) — unnecessary complexity for a single-session tool, but worth mentioning as a natural extension point (see Section 13).

### 8.4 Final Report Contents
- Total session duration (and active vs. paused/absent time, if auto-pause is enabled).
- Average attention score (and time-weighted, per Section 5.4).
- % time looking at screen (CENTER gaze) vs. away, broken down by direction (left/right/up/down).
- Total blink count, average blink rate (blinks/minute) — useful as a fatigue indicator on its own.
- Long eye-closure event count and total duration.
- Time with no detected face, broken into brief/extended/long-absent buckets.
- Attention trend over time (the full time series).
- Distribution across the four attention categories (% of session time in each).

### 8.5 Suggested Visualizations
- **Line chart:** attention score over session time (the primary chart).
- **Stacked area or pie chart:** time distribution across gaze directions.
- **Bar chart:** count of blink/long-closure/sleep-possible events over time buckets (e.g., per 5-minute interval), useful for spotting a fatigue trend across the session.
- **Horizontal timeline/Gantt-style strip:** face-present vs. face-absent intervals.

---

## 9. User Interface

### 9.1 Recommendation: start with OpenCV overlay, offer Tkinter as a v2 upgrade
- **OpenCV `imshow` + `cv2.putText`/`cv2.rectangle` overlay** is sufficient for an MVP: draw the live webcam feed with a semi-transparent status panel overlaid (face box, gaze arrow indicator, attention percentage, session timer, eye/head status text). This requires zero extra GUI dependencies and is the fastest path to a working real-time demo.
- **Tkinter** (standard library, no install) is a reasonable upgrade once the core pipeline is stable, for a cleaner dashboard with separate panels (video feed + a live-updating chart using `matplotlib`'s Tkinter backend) and proper buttons (Start/Pause/Stop/Export Report) instead of keyboard shortcuts.
- **PyQt5** is the most polished option if a more "product-like" UI matters, at the cost of a heavier dependency and steeper learning curve — recommended only if the project's grading/scope rewards UI polish specifically.

### 9.2 Live Dashboard Elements
- Webcam feed (primary panel).
- Face detection status indicator (green/red dot + text).
- Current gaze direction (text + small directional arrow/icon).
- Head orientation (numeric yaw/pitch/roll or a simple compass-style icon).
- Eye status (Open / Blinking / Long Closure / Possible Sleep).
- Current attention percentage (large, prominent number, updated at ~1 Hz per Section 5.3).
- Session duration timer.
- A scrolling/rolling mini attention-trend sparkline is a nice-to-have that significantly improves perceived polish for minimal extra effort (just a `matplotlib`-free simple line drawn into the OpenCV frame from the recent score deque).

---

## 10. Performance and Optimization

### 10.1 Expected FPS
- MediaPipe Face Mesh alone typically runs in the 20–40+ FPS range on a modern laptop CPU for single-face processing. With the additional lightweight numpy math in this design (EAR, ratios, solvePnP), expect a realistic sustained **15–25 FPS** end-to-end on an average laptop, which is more than sufficient for this use case — attention doesn't need to be measured at 60 FPS.

### 10.2 Processing Frequency Strategy
- **Capture at native camera rate** (often 30 FPS), but **process every 2nd frame** (i.e., effective ~15 FPS processing) if profiling shows the full pipeline can't sustain real-time at full rate. Attention dynamics change on the order of seconds, not milliseconds, so this is a safe optimization with no meaningful accuracy loss.
- **Decouple display rate from logging rate:** render every processed frame to the UI for visual smoothness, but only write to the session log and update the displayed percentage once per second (Section 5.3) — this avoids both I/O overhead and a flickering, falsely-precise UI number.

### 10.3 Multithreading
- Capture thread (Section 2.4) is the most valuable threading addition — it prevents camera I/O latency from blocking CV processing.
- A separate logging/disk-write thread prevents file flushes from causing visible frame drops, especially relevant if using frequent small CSV writes rather than buffered batch writes.
- Full multiprocessing (separate processes per CV module) is generally **not** necessary or beneficial here: MediaPipe's internal computation is already implemented in optimized C++ and releases the Python GIL during inference, so Python-level multiprocessing overhead (IPC, serialization of frames between processes) would likely cost more than it saves for a single-face pipeline.

### 10.4 Memory Management
- Use bounded `deque`s for all rolling windows (Section 8.2) — never let a buffer grow unbounded across a long session.
- Release/overwrite frame buffers each loop iteration rather than accumulating raw frames; only the lightweight derived metrics need to persist for the full session, not the video frames themselves (don't write raw video to disk unless explicitly required — it's unnecessary for this system's goals and a privacy concern).

### 10.5 Reducing CPU Usage
- Resize frames to a smaller working resolution (e.g., 640×480) for MediaPipe processing even if the camera natively captures higher resolution — landmark accuracy is not meaningfully affected at this scale, but inference cost drops significantly.
- Disable the optional `MovementStabilityAnalyzer` by default if the host machine profiles as borderline on FPS, since it's the lowest-value, lowest-weight signal.

---

## 11. Testing Strategy

| Test Case | Setup | Expected Behavior |
|---|---|---|
| Normal attentive session | User faces screen, occasional natural blinking | Score stays in 80–100 band; no false distraction events logged |
| Looking away (short) | User glances at phone for 1–2s, returns | No score penalty (within grace window); logged as a brief off-center gaze sample but not a "distraction event" |
| Looking away (sustained) | User looks away for 10+ seconds | Score decays per ramp function; counted as a logged distraction interval in the report |
| Normal blinking | Continuous natural blinking over a session | Blink count increments correctly; no penalty to score; EAR state machine returns to OPEN promptly after each blink |
| Simulated sleep/long closure | User keeps eyes closed for 5+ seconds | Classified as "possible sleep" after threshold; significant but not instant-zero score penalty; flagged distinctly from a long blink in the report |
| Wearing glasses | Repeat key tests above with glasses on, varied lighting/glare | EAR baseline still functions after calibration; system flags low-confidence periods rather than misclassifying during glare spikes, instead of asserting false closures |
| Varying lighting (dim / bright / backlit) | Run detection under each condition | Face/landmark detection confidence may drop in poor lighting (expected and should be visible to the user, e.g., a "low confidence" indicator), but the system should not crash or silently log incorrect high-confidence values |
| Multiple faces in frame | A second person enters frame | With `max_num_faces=1`, MediaPipe should continue tracking the originally-locked face if tracking is stable; document this as a known limitation and test that the system doesn't erratically jump between faces |
| Sudden head movement | Fast head turn / look-away and back | Brief detection dropout handled by `FACE_MISSING_BRIEF` grace window; no permanent tracking loss; head pose doesn't report nonsensical extreme angles due to motion blur (clip/reject outlier angle jumps beyond a physically plausible max per-frame delta) |
| Camera disconnection | Unplug/disable webcam mid-session | `WebcamManager` detects failure, surfaces a clear UI error state, session pauses (not crashes); reconnecting resumes cleanly without restarting the whole app |
| No face for entire session start | User not yet in frame when session starts | System waits in a clear "searching for face" state rather than logging garbage data or starting the attention clock prematurely |
| Low-end hardware | Run on a lower-spec laptop | Verify graceful degradation via frame-skipping (Section 10.2) rather than the UI becoming unresponsive |

### 11.1 Expected Limitations & Possible False Positives (state these explicitly in documentation/report output)
- Side profile or extreme yaw can degrade landmark accuracy beyond the geometric model's reliable range — very large genuine head turns may be under- or over-penalized depending on landmark stability at extreme angles.
- Users with ptosis (naturally low/hooded eyelids), certain eye shapes, or strong asymmetric eye conditions may need more aggressive personalized calibration to avoid systematic EAR misclassification.
- Multiple monitors: a student legitimately looking at a second screen will register as "looking away" geometrically even though they may be doing class-related work (e.g., a shared document) — this is a known, unavoidable limitation of single-camera visual proxies and should be disclosed, not hidden.
- The system cannot distinguish "looking at the screen but not comprehending/listening" from genuine attention — this is the central, irreducible limitation and should be stated plainly in any report or UI copy (e.g., a footer note: "This score reflects visual engagement signals, not a measurement of comprehension or mental focus.").

---

## 12. Limitations (Summary)

- **Behavioral, not cognitive:** measures observable visual proxies only; cannot detect comprehension, motivation, or internal mental state.
- **Single-camera geometric constraints:** accuracy degrades at extreme angles, poor lighting, or significant occlusion (masks, hands, hair).
- **Population-general thresholds need calibration:** default thresholds are reasonable starting points, not universally accurate without the per-user calibration step.
- **No ground-truth validation in this scope:** without paired self-report or expert-annotated ground truth, the system's score should be treated as a relative/comparative indicator within a session, not an absolute, validated psychological measurement.
- **Privacy consideration:** even though no video is stored by default (Section 10.4), continuous facial analysis is sensitive; the system should only ever be used with the explicit knowledge and consent of the person being observed, and ideally only by that person on themselves.

---

## 13. Future Scope

Reasonable, technically grounded extensions:

- **Deep-learning gaze estimation** (e.g., appearance-based CNN gaze models) to replace the geometric ratio approach for higher angular accuracy — trade-off is added compute cost and likely needing GPU for real-time use, so this should be optional/configurable, not a default replacement.
- **Personalized calibration profiles** saved across sessions (per-user EAR baseline, gaze ratio thresholds, head-pose neutral position) rather than recalibrating every session.
- **Lightweight ML attention classifier** trained on the engineered features this system already computes (gaze ratio, EAR, head angles, presence duration) as input to a small classical ML model (e.g., gradient boosting) instead of hand-tuned linear weights — feasible without deep learning and could improve scoring accuracy if paired with labeled session data (e.g., self-reported attention checkpoints during test sessions).
- **Session history & trends across multiple sessions** (would justify moving to SQLite per Section 8.3) — e.g., showing a student or teacher how engagement trends across a semester.
- **Aggregate, privacy-conscious classroom analytics** — explicitly only as an opt-in, consent-based feature, never silent monitoring, and ideally reporting only aggregate/anonymized statistics rather than per-student dashboards visible to others.

**Explicitly not recommended:** emotion/mood classification claims, "honesty" or "cheating" detection, or any feature that frames the system's output as a definitive judgment about a student's internal state or character — these go beyond what webcam-based visual analysis can reliably or ethically support.

---

## 14. Suggested Development Phases

| Phase | Deliverable |
|---|---|
| 1 | `WebcamManager` + `FaceMeshDetector` wired together; raw landmark overlay rendering on live feed (validates capture + detection pipeline works end to end). |
| 2 | `LandmarkProcessor` + `BlinkDetector` (EAR) with live EAR value and blink counter displayed — easiest module to validate visually. |
| 3 | `HeadPoseEstimator` with live yaw/pitch/roll readout and a simple 3D axis overlay drawn on the face for sanity-checking solvePnP output. |
| 4 | `GazeAnalyzer` with live direction classification overlay; build and test the calibration routine here. |
| 5 | `FacePresenceTracker` + `MovementStabilityAnalyzer`; integrate all sub-score outputs into a single combined debug overlay. |
| 6 | `AttentionScoringEngine` — implement weighting, smoothing, and category logic; validate against the manual test cases in Section 11. |
| 7 | `SessionLogger` (CSV/JSON) running for full mock sessions; verify schema correctness and bounded memory usage over long runs. |
| 8 | `ReportGenerator` — charts and summary statistics from logged sessions. |
| 9 | `DashboardUI` polish pass (OpenCV → Tkinter upgrade if time allows) and `SessionController` finalization (start/pause/stop, error handling, camera-disconnect recovery). |
| 10 | End-to-end testing against the full Section 11 test matrix; write up limitations/disclosure copy for the final report output. |

---

*End of document.*
