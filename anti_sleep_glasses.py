"""
╔══════════════════════════════════════════════════════════════╗
║        ANTI-SLEEP EYEWEAR DRIVING GLASSES SYSTEM            ║
║        Real-time Drowsiness Detection with EAR Algorithm    ║
╚══════════════════════════════════════════════════════════════╝

REQUIREMENTS:
    pip install opencv-python mediapipe pyttsx3 numpy pygame

HOW IT WORKS:
    - Detects eyes using MediaPipe FaceMesh (468 landmarks)
    - Calculates Eye Aspect Ratio (EAR) per frame
    - If eyes stay closed (EAR < threshold) for 2 seconds → ALARM triggers
    - Draws bounding box around each eye + center dot
    - Visual HUD with status, EAR value, and countdown timer

CONTROLS:
    ESC  → Exit
    R    → Reset alarm manually
"""

import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import threading
import time
import sys

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
EAR_THRESHOLD   = 0.18    # EAR below this = eyes closed (tune: 0.15–0.22)
ALARM_SECONDS   = 2.0     # seconds of closed eyes before alarm
CAMERA_INDEX    = 0       # 0 = default webcam
FRAME_WIDTH     = 900
FRAME_HEIGHT    = 600

# Colors (BGR)
COLOR_GREEN     = (0,   220,  80)
COLOR_RED       = (0,    40, 255)
COLOR_ORANGE    = (0,   165, 255)
COLOR_CYAN      = (255, 220,   0)
COLOR_WHITE     = (255, 255, 255)
COLOR_BLACK     = (0,     0,   0)
COLOR_YELLOW    = (0,   255, 255)

# ─────────────────────────────────────────────
#  MEDIAPIPE EYE LANDMARK INDICES
# ─────────────────────────────────────────────
# Left eye  (from driver's perspective)
LEFT_EYE_TOP        = [159, 158, 157]
LEFT_EYE_BOTTOM     = [145, 153, 144]
LEFT_EYE_LEFT       = 33
LEFT_EYE_RIGHT      = 133
LEFT_EYE_ALL        = [33, 7, 163, 144, 145, 153, 154, 155, 133,
                        173, 157, 158, 159, 160, 161, 246]

# Right eye
RIGHT_EYE_TOP       = [386, 385, 384]
RIGHT_EYE_BOTTOM    = [374, 380, 373]
RIGHT_EYE_LEFT      = 362
RIGHT_EYE_RIGHT     = 263
RIGHT_EYE_ALL       = [362, 382, 381, 380, 374, 373, 390, 249,
                        263, 466, 388, 387, 386, 385, 384, 398]


# ─────────────────────────────────────────────
#  EAR CALCULATION
# ─────────────────────────────────────────────
def compute_ear(landmarks, top_indices, bottom_indices, left_idx, right_idx, w, h):
    """
    Eye Aspect Ratio = vertical_openness / horizontal_width
    Uses average of 3 vertical pairs for accuracy.
    """
    vertical_sum = 0.0
    for t, b in zip(top_indices, bottom_indices):
        ty = landmarks[t].y * h
        by = landmarks[b].y * h
        vertical_sum += abs(ty - by)
    vertical_avg = vertical_sum / len(top_indices)

    lx = landmarks[left_idx].x * w
    rx = landmarks[right_idx].x * w
    horizontal = abs(rx - lx)

    if horizontal < 1e-5:
        return 0.0
    return vertical_avg / horizontal


# ─────────────────────────────────────────────
#  EYE BOUNDING BOX
# ─────────────────────────────────────────────
def get_eye_bbox(landmarks, eye_indices, w, h, padding=8):
    """Returns (x1, y1, x2, y2) bounding box around eye landmarks."""
    xs = [int(landmarks[i].x * w) for i in eye_indices]
    ys = [int(landmarks[i].y * h) for i in eye_indices]
    return (
        max(0, min(xs) - padding),
        max(0, min(ys) - padding),
        min(w, max(xs) + padding),
        min(h, max(ys) + padding)
    )


def get_eye_center(landmarks, top_indices, bottom_indices, left_idx, right_idx, w, h):
    """Returns pixel center (cx, cy) of the eye."""
    cx = int(((landmarks[left_idx].x + landmarks[right_idx].x) / 2) * w)
    cy = int(((landmarks[top_indices[1]].y + landmarks[bottom_indices[1]].y) / 2) * h)
    return cx, cy


# ─────────────────────────────────────────────
#  VOICE ALARM (non-blocking)
# ─────────────────────────────────────────────
class VoiceAlarm:
    def __init__(self):
        self._lock     = threading.Lock()
        self._speaking = False
        try:
            self._engine = pyttsx3.init()
            self._engine.setProperty('rate', 160)
            self._engine.setProperty('volume', 1.0)
            self._available = True
        except Exception:
            self._available = False
            print("[WARN] pyttsx3 not available — voice alarm disabled.")

    def speak(self, text):
        if not self._available:
            return
        with self._lock:
            if self._speaking:
                return
            self._speaking = True

        def _run():
            try:
                e = pyttsx3.init()
                e.setProperty('rate', 155)
                e.setProperty('volume', 1.0)
                e.say(text)
                e.runAndWait()
            finally:
                with self._lock:
                    self._speaking = False

        threading.Thread(target=_run, daemon=True).start()


# ─────────────────────────────────────────────
#  HUD DRAWING HELPERS
# ─────────────────────────────────────────────
def draw_rounded_rect(img, pt1, pt2, color, thickness=2, r=10):
    """Draw a rounded-corner rectangle."""
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.line(img,  (x1+r, y1),    (x2-r, y1),    color, thickness)
    cv2.line(img,  (x1+r, y2),    (x2-r, y2),    color, thickness)
    cv2.line(img,  (x1,   y1+r),  (x1,   y2-r),  color, thickness)
    cv2.line(img,  (x2,   y1+r),  (x2,   y2-r),  color, thickness)
    cv2.ellipse(img, (x1+r, y1+r), (r, r), 180,  0, 90,  color, thickness)
    cv2.ellipse(img, (x2-r, y1+r), (r, r), 270,  0, 90,  color, thickness)
    cv2.ellipse(img, (x1+r, y2-r), (r, r),  90,  0, 90,  color, thickness)
    cv2.ellipse(img, (x2-r, y2-r), (r, r),   0,  0, 90,  color, thickness)


def draw_eye_box(frame, bbox, color, label=None, thickness=2):
    """Draw eye bounding box with corner accents."""
    x1, y1, x2, y2 = bbox
    clen = max(8, (x2 - x1) // 4)

    # Corner brackets
    corners = [
        ((x1, y1+clen), (x1, y1), (x1+clen, y1)),
        ((x2-clen, y1), (x2, y1), (x2, y1+clen)),
        ((x1, y2-clen), (x1, y2), (x1+clen, y2)),
        ((x2-clen, y2), (x2, y2), (x2, y2-clen)),
    ]
    for pts in corners:
        for i in range(len(pts) - 1):
            cv2.line(frame, pts[i], pts[i+1], color, thickness + 1)

    # Subtle full border
    cv2.rectangle(frame, (x1, y1), (x2, y2), (*color[:3],), 1)

    if label:
        cv2.putText(frame, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


def draw_hud(frame, status, ear_val, countdown, alarm_on, face_detected, fps):
    """Draw the full heads-up display overlay."""
    h, w = frame.shape[:2]

    # ── Top status bar ──────────────────────────────────────────
    bar_color = (0, 0, 180) if alarm_on else (20, 20, 20)
    cv2.rectangle(frame, (0, 0), (w, 50), bar_color, -1)
    cv2.rectangle(frame, (0, 0), (w, 50), COLOR_CYAN, 1)

    title = "ANTI-SLEEP DRIVING SYSTEM"
    cv2.putText(frame, title, (12, 32),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, COLOR_CYAN, 1, cv2.LINE_AA)

    fps_text = f"FPS: {fps:.0f}"
    cv2.putText(frame, fps_text, (w - 90, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 1, cv2.LINE_AA)

    # ── Bottom info panel ────────────────────────────────────────
    panel_y = h - 90
    cv2.rectangle(frame, (0, panel_y), (w, h), (15, 15, 15), -1)
    cv2.rectangle(frame, (0, panel_y), (w, panel_y + 1), COLOR_CYAN, -1)

    # Status indicator
    if not face_detected:
        status_color = COLOR_ORANGE
        status_text  = "NO FACE DETECTED"
    elif alarm_on:
        status_color = COLOR_RED
        status_text  = "⚠  WAKE UP! WAKE UP!"
    elif countdown > 0:
        status_color = COLOR_ORANGE
        status_text  = f"EYES CLOSING... {ALARM_SECONDS - countdown:.1f}s"
    else:
        status_color = COLOR_GREEN
        status_text  = "EYES OPEN — SAFE"

    cv2.putText(frame, status_text, (15, panel_y + 35),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, status_color, 2, cv2.LINE_AA)

    # EAR bar
    ear_label = f"EAR: {ear_val:.3f}"
    cv2.putText(frame, ear_label, (15, panel_y + 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 1, cv2.LINE_AA)

    bar_x, bar_y = 115, panel_y + 57
    bar_w, bar_h = 200, 14
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (60, 60, 60), -1)
    filled = int(min(ear_val / 0.35, 1.0) * bar_w)
    bar_fill_color = COLOR_GREEN if ear_val >= EAR_THRESHOLD else COLOR_RED
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h),
                  bar_fill_color, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  COLOR_WHITE, 1)

    threshold_x = bar_x + int((EAR_THRESHOLD / 0.35) * bar_w)
    cv2.line(frame, (threshold_x, bar_y - 3),
             (threshold_x, bar_y + bar_h + 3), COLOR_YELLOW, 2)

    # Timer / countdown
    if countdown > 0 and not alarm_on:
        pct = countdown / ALARM_SECONDS
        cv2.putText(frame, f"TIMER: {countdown:.1f}s / {ALARM_SECONDS}s",
                    (w - 230, panel_y + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_ORANGE, 1, cv2.LINE_AA)
        arc_cx, arc_cy = w - 55, panel_y + 45
        cv2.circle(frame, (arc_cx, arc_cy), 22, (60, 60, 60), -1)
        cv2.ellipse(frame, (arc_cx, arc_cy), (22, 22),
                    -90, 0, int(360 * pct), COLOR_ORANGE, 3)

    cv2.putText(frame, "ESC: Exit  |  R: Reset",
                (w - 220, panel_y + 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 140), 1, cv2.LINE_AA)

    # ── ALARM OVERLAY ────────────────────────────────────────────
    if alarm_on:
        overlay = frame.copy()
        # Red pulsing border
        pulse = int((time.time() * 4) % 2)
        border_color = (0, 0, 255) if pulse == 0 else (0, 80, 255)
        cv2.rectangle(overlay, (0, 0), (w, h), border_color, 20)

        # Semi-transparent red tint
        cv2.rectangle(overlay, (0, 55), (w, panel_y),
                      (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)

        # Big WAKE UP text
        text = "WAKE UP!"
        fs   = 2.8
        thick = 5
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, fs, thick)
        tx = (w - tw) // 2
        ty = (h + th) // 2

        # Shadow
        cv2.putText(frame, text, (tx + 4, ty + 4),
                    cv2.FONT_HERSHEY_DUPLEX, fs, COLOR_BLACK, thick + 2, cv2.LINE_AA)
        # Main
        cv2.putText(frame, text, (tx, ty),
                    cv2.FONT_HERSHEY_DUPLEX, fs, (0, 40, 255), thick, cv2.LINE_AA)


# ─────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────
def main():
    print("═" * 60)
    print("  ANTI-SLEEP EYEWEAR DRIVING GLASSES SYSTEM")
    print("  Starting up...")
    print("═" * 60)

    # Init voice
    alarm = VoiceAlarm()

    # Init MediaPipe
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5
    )

    # Init camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Check CAMERA_INDEX setting.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("  Camera ready. Press ESC to quit.\n")

    # State
    eyes_closed_since = None
    alarm_active      = False
    fps_timer         = time.time()
    fps               = 0
    frame_count       = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame read failed — retrying...")
            time.sleep(0.05)
            continue

        frame = cv2.flip(frame, 1)   # Mirror for natural feel
        h, w  = frame.shape[:2]

        # FPS calculation
        frame_count += 1
        if frame_count % 15 == 0:
            fps = 15 / (time.time() - fps_timer)
            fps_timer = time.time()

        # ── Face landmark detection ──────────────────────────────
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        ear_avg      = 0.0
        face_found   = False
        countdown    = 0.0

        if results.multi_face_landmarks:
            face_found = True
            lms = results.multi_face_landmarks[0].landmark

            # ── Compute EAR ─────────────────────────────────────
            left_ear  = compute_ear(lms,
                                    LEFT_EYE_TOP, LEFT_EYE_BOTTOM,
                                    LEFT_EYE_LEFT, LEFT_EYE_RIGHT, w, h)
            right_ear = compute_ear(lms,
                                    RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM,
                                    RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT, w, h)
            ear_avg   = (left_ear + right_ear) / 2.0

            eyes_closed = ear_avg < EAR_THRESHOLD

            # ── Bounding boxes + dots ────────────────────────────
            box_color = COLOR_RED if eyes_closed else COLOR_GREEN

            left_bbox  = get_eye_bbox(lms, LEFT_EYE_ALL,  w, h, padding=10)
            right_bbox = get_eye_bbox(lms, RIGHT_EYE_ALL, w, h, padding=10)

            left_label  = f"L {left_ear:.2f}"
            right_label = f"R {right_ear:.2f}"

            draw_eye_box(frame, left_bbox,  box_color, left_label)
            draw_eye_box(frame, right_bbox, box_color, right_label)

            # Center dots
            lcx, lcy = get_eye_center(lms,
                                      LEFT_EYE_TOP, LEFT_EYE_BOTTOM,
                                      LEFT_EYE_LEFT, LEFT_EYE_RIGHT, w, h)
            rcx, rcy = get_eye_center(lms,
                                      RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM,
                                      RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT, w, h)

            cv2.circle(frame, (lcx, lcy), 5, COLOR_CYAN,   -1)
            cv2.circle(frame, (lcx, lcy), 7, box_color,     1)
            cv2.circle(frame, (rcx, rcy), 5, COLOR_CYAN,   -1)
            cv2.circle(frame, (rcx, rcy), 7, box_color,     1)

            # ── Drowsiness timer ─────────────────────────────────
            if eyes_closed:
                if eyes_closed_since is None:
                    eyes_closed_since = time.time()
                elapsed   = time.time() - eyes_closed_since
                countdown = elapsed

                if elapsed >= ALARM_SECONDS:
                    alarm_active = True
                    alarm.speak("Wake up! Wake up! You are falling asleep!")
            else:
                # Eyes opened — reset
                eyes_closed_since = None
                alarm_active      = False

        else:
            # No face detected — reset
            eyes_closed_since = None
            alarm_active      = False

        # ── Draw HUD ─────────────────────────────────────────────
        draw_hud(frame, None, ear_avg, countdown,
                 alarm_active, face_found, fps)

        cv2.imshow("Anti-Sleep Driving Glasses", frame)

        # ── Key handling ─────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == 27:          # ESC → quit
            break
        elif key == ord('r') or key == ord('R'):
            eyes_closed_since = None
            alarm_active      = False
            print("  [RESET] Alarm manually reset.")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    print("\n  System shut down. Stay safe on the road!")


if __name__ == "__main__":
    main()
