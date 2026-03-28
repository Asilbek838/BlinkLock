# blinklock.py
# pip install opencv-python mediapipe

import cv2
import time
import math
from collections import deque

import mediapipe as mp

# ----------------------------
# Settings you can tweak
# ----------------------------
EAR_THRESHOLD = 0.22           # Lower = more sensitive
BLINK_MIN_FRAMES = 2           # Must stay closed this many frames to count as a blink
BLINK_WINDOW_SEC = 3.0         # 3 rapid blinks must happen within this window
WINK_MIN_FRAMES = 15           # One eye closed for this many frames = deliberate wink
PIN_CODE = "1234"

# ----------------------------
# States
# ----------------------------
MONITORING = "MONITORING"
LOCKED = "LOCKED"
PIN_ENTRY = "PIN_ENTRY"


def euclidean_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def eye_aspect_ratio(landmarks, eye_indices, w, h):
    """Compute EAR for one eye from 6 landmark indices."""
    pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        pts.append((int(lm.x * w), int(lm.y * h)))

    # EAR formula:
    # (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    A = euclidean_distance(pts[1], pts[5])
    B = euclidean_distance(pts[2], pts[4])
    C = euclidean_distance(pts[0], pts[3])

    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)


def draw_panel(frame, alpha=0.35):
    """Dark overlay for locked / PIN entry states."""
    overlay = frame.copy()
    h, w = frame.shape[:2]
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def put_text(frame, text, org, scale=0.7, color=(255, 255, 255), thickness=2):
    cv2.putText(
        frame,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def main():
    # FaceMesh eye landmark indices commonly used for EAR
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    mp_face_mesh = mp.solutions.face_mesh

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    state = MONITORING
    blink_times = deque()
    pin_buffer = ""

    # Blink / wink debounce counters
    blink_low_frames = 0
    wink_low_frames_left = 0
    wink_low_frames_right = 0

    # UI message
    status_msg = ""
    status_msg_until = 0.0

    def set_status(msg, seconds=1.5):
        nonlocal status_msg, status_msg_until
        status_msg = msg
        status_msg_until = time.time() + seconds

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)  # mirror view
            h, w = frame.shape[:2]
            now = time.time()

            # Default values for display
            left_ear = 0.0
            right_ear = 0.0
            ear_avg = 0.0
            face_found = False

            # FaceMesh wants RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = face_mesh.process(rgb)
            rgb.flags.writeable = True

            if results.multi_face_landmarks:
                face_found = True
                landmarks = results.multi_face_landmarks[0].landmark

                left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
                right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
                ear_avg = (left_ear + right_ear) / 2.0

                left_closed = left_ear < EAR_THRESHOLD
                right_closed = right_ear < EAR_THRESHOLD
                both_closed = left_closed and right_closed
                one_closed = left_closed ^ right_closed  # exactly one eye closed

                # ----------------------------
                # MONITORING: count 3 rapid blinks
                # ----------------------------
                if state == MONITORING:
                    if both_closed:
                        blink_low_frames += 1
                    else:
                        # If the eyes were closed long enough, count one blink on the open transition
                        if blink_low_frames >= BLINK_MIN_FRAMES:
                            blink_times.append(now)

                        blink_low_frames = 0

                        # Remove old blink timestamps outside the window
                        while blink_times and (now - blink_times[0] > BLINK_WINDOW_SEC):
                            blink_times.popleft()

                        if len(blink_times) >= 3:
                            state = LOCKED
                            blink_times.clear()
                            blink_low_frames = 0
                            wink_low_frames_left = 0
                            wink_low_frames_right = 0
                            pin_buffer = ""
                            set_status("LOCKED: 3 rapid blinks detected", 2.0)

                # ----------------------------
                # LOCKED: unlock by deliberate wink
                # ----------------------------
                elif state == LOCKED:
                    if one_closed:
                        if left_closed:
                            wink_low_frames_left += 1
                            wink_low_frames_right = 0
                        else:
                            wink_low_frames_right += 1
                            wink_low_frames_left = 0
                    else:
                        # If one eye stayed closed long enough and then opened, unlock
                        if (
                            wink_low_frames_left >= WINK_MIN_FRAMES
                            or wink_low_frames_right >= WINK_MIN_FRAMES
                        ):
                            state = MONITORING
                            blink_times.clear()
                            blink_low_frames = 0
                            wink_low_frames_left = 0
                            wink_low_frames_right = 0
                            pin_buffer = ""
                            set_status("Unlocked by wink", 2.0)
                        wink_low_frames_left = 0
                        wink_low_frames_right = 0

                # ----------------------------
                # PIN_ENTRY: enter PIN with keyboard
                # ----------------------------
                elif state == PIN_ENTRY:
                    # nothing needed here; keyboard handles it below
                    pass

            else:
                # If no face is found, reset counters that rely on continuous frames
                blink_low_frames = 0
                wink_low_frames_left = 0
                wink_low_frames_right = 0

            # Key handling
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            # Manual unlock shortcut
            if key == ord("u") and state == LOCKED:
                state = MONITORING
                blink_times.clear()
                blink_low_frames = 0
                wink_low_frames_left = 0
                wink_low_frames_right = 0
                pin_buffer = ""
                set_status("Unlocked", 1.5)

            # Enter PIN mode
            if key == ord("p") and state == LOCKED:
                state = PIN_ENTRY
                pin_buffer = ""
                set_status("Enter PIN", 1.5)

            # PIN entry handling
            if state == PIN_ENTRY:
                if ord("0") <= key <= ord("9"):
                    if len(pin_buffer) < 12:
                        pin_buffer += chr(key)
                elif key in (8, 127):  # backspace
                    pin_buffer = pin_buffer[:-1]
                elif key in (13, 10):  # Enter
                    if pin_buffer == PIN_CODE:
                        state = MONITORING
                        blink_times.clear()
                        blink_low_frames = 0
                        wink_low_frames_left = 0
                        wink_low_frames_right = 0
                        set_status("PIN correct. Unlocked.", 2.0)
                    else:
                        set_status("Wrong PIN", 1.5)
                    pin_buffer = ""
                elif key == 27:  # Esc
                    pin_buffer = ""
                    state = LOCKED
                    set_status("PIN canceled", 1.5)

            # Visible lock overlay
            if state in (LOCKED, PIN_ENTRY):
                draw_panel(frame, alpha=0.5)

            # UI
            put_text(frame, "BlinkLock", (20, 35), scale=1.0, color=(0, 255, 255), thickness=2)
            put_text(frame, f"State: {state}", (20, 70), scale=0.8, color=(255, 255, 255), thickness=2)
            put_text(frame, f"EAR Left : {left_ear:.3f}", (20, 110), scale=0.7, color=(255, 255, 0), thickness=2)
            put_text(frame, f"EAR Right: {right_ear:.3f}", (20, 140), scale=0.7, color=(255, 255, 0), thickness=2)
            put_text(frame, f"EAR Avg  : {ear_avg:.3f}", (20, 170), scale=0.7, color=(255, 255, 0), thickness=2)

            if state == MONITORING:
                put_text(frame, f"Blink count in window: {len(blink_times)}/3", (20, 210), scale=0.7, color=(0, 255, 0), thickness=2)
                put_text(frame, "3 rapid blinks -> LOCK", (20, 245), scale=0.7, color=(200, 200, 200), thickness=2)
            elif state == LOCKED:
                put_text(frame, "LOCKED", (20, 210), scale=1.0, color=(0, 0, 255), thickness=3)
                put_text(frame, "Slow wink -> unlock", (20, 245), scale=0.7, color=(200, 200, 200), thickness=2)
                put_text(frame, "Press P for PIN", (20, 275), scale=0.7, color=(200, 200, 200), thickness=2)
                put_text(frame, "Press U for manual unlock", (20, 305), scale=0.7, color=(200, 200, 200), thickness=2)
            elif state == PIN_ENTRY:
                put_text(frame, "PIN ENTRY", (20, 210), scale=1.0, color=(255, 165, 0), thickness=3)
                put_text(frame, "Type digits, Enter to submit, Esc to cancel", (20, 245), scale=0.65, color=(220, 220, 220), thickness=2)
                masked = "*" * len(pin_buffer)
                put_text(frame, f"PIN: {masked}", (20, 280), scale=0.9, color=(255, 255, 255), thickness=2)

            if not face_found:
                put_text(frame, "No face detected", (20, 350), scale=0.8, color=(0, 0, 255), thickness=2)

            # Temporary message
            if time.time() < status_msg_until:
                put_text(frame, status_msg, (20, h - 25), scale=0.8, color=(0, 255, 255), thickness=2)

            cv2.imshow("BlinkLock", frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()