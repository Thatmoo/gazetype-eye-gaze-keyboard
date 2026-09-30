import cv2
import numpy as np
import mediapipe as mp
import pyglet
import time
import json
import os
from collections import deque, Counter
from math import hypot

# =========================================================
# GAZETYPE - ADAPTIVE EYE-GAZE COMMUNICATION KEYBOARD
# =========================================================
# Core interaction:
#   Look LEFT/RIGHT -> choose keyboard
#   Blink once      -> pause current key
#   Blink again     -> select current key
#
# Added project features:
#   1. Personalized calibration saved between sessions
#   2. Adaptive scanning speed
#   3. Word prediction suggestions
#   4. Error-resistant two-stage blink confirmation
#   5. Session performance metrics
#
# Keyboard shortcut:
#   R = recalibrate
#   ESC = exit
# =========================================================

CAMERA_INDEX = 0
MODEL_PATH = "face_landmarker.task"
PROFILE_PATH = "gazetype_profile.json"

# Scanning speed
BASE_FRAMES_PER_LETTER = 25
MIN_FRAMES_PER_LETTER = 15
MAX_FRAMES_PER_LETTER = 50

# Gaze
GAZE_REQUIRED_FRAMES = 12
GAZE_LEFT_THRESHOLD = 0.40
GAZE_RIGHT_THRESHOLD = 0.60

# Blink
BLINK_SMOOTHING = 3
BLINK_FRAMES_REQUIRED = 2
OPEN_FRAMES_REQUIRED = 3
BLINK_COOLDOWN = 0.50
CONFIRMATION_TIMEOUT = 3.0

# Calibration
OPEN_CALIBRATION_SECONDS = 3.0
CALIBRATION_BLINKS_REQUIRED = 3

# =========================================================
# AUDIO
# =========================================================

def load_sound(name):
    try:
        return pyglet.media.load(name, streaming=False)
    except Exception:
        return None

sound = load_sound("sound.wav")
left_sound = load_sound("left.wav")
right_sound = load_sound("right.wav")


def play_audio(audio):
    if audio is not None:
        try:
            audio.play()
        except Exception:
            pass

# =========================================================
# CAMERA
# =========================================================
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError("Could not open camera. Check your camera index.")

# =========================================================
# MEDIAPIPE
# =========================================================
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
)

landmarker = FaceLandmarker.create_from_options(options)

# =========================================================
# KEYBOARDS
# =========================================================
# 15 slots keep the original scanning layout.
keys_left = {
    0: "Q", 1: "W", 2: "E", 3: "R", 4: "T",
    5: "A", 6: "S", 7: "D", 8: "F", 9: "G",
    10: "Z", 11: "X", 12: "C", 13: "V", 14: "BACK",
}

keys_right = {
    0: "Y", 1: "U", 2: "I", 3: "O", 4: "P",
    5: "H", 6: "J", 7: "K", 8: "L", 9: "SPACE",
    10: "B", 11: "N", 12: "M", 13: "PRED", 14: "BACK",
}

# A small control keyboard used only for prediction selection.
prediction_keys = {
    0: "P1", 1: "P2", 2: "P3", 3: "CANCEL",
}

# =========================================================
# COMMON WORD LIST
# =========================================================
# A compact offline dictionary keeps the project self-contained.
COMMON_WORDS = """
i am you we he she it they the a an and or but if is are was were be been
have has had do does did can could will would should may might must this that
these those to of in on for from with by at as about into over after before
hello help need want water food please thank thanks yes no okay good morning
goodbye sorry today tomorrow now here there home college university project
computer keyboard eye eyes gaze blink type typing test system student friend
family doctor call work go come get give take make use look see know think
feel fine not my your our their what where when why how who which can you
please help me i need help i need water i am hungry i am tired
and the of to in for on with at by from is it this that are was be have
more some very just like time day one two three first new people thing
life way make use good know want need look see come think take give find
keep work try tell ask help start stop open close left right select letter
space back please welcome thanks hello world
""".split()

# Keep only useful lowercase alphabetic entries and remove duplicates.
WORD_SET = sorted({w.lower() for w in COMMON_WORDS if w.isalpha()})
WORD_FREQ = Counter()
for rank, word in enumerate(WORD_SET):
    WORD_FREQ[word] = len(WORD_SET) - rank

# =========================================================
# UI HELPERS
# =========================================================
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SMALL = cv2.FONT_HERSHEY_SIMPLEX

BG = (24, 28, 34)
CARD = (38, 44, 52)
CARD2 = (48, 55, 65)
WHITE = (245, 247, 250)
MUTED = (170, 178, 190)
ACCENT = (225, 150, 55)
GREEN = (70, 205, 115)
RED = (80, 90, 235)
BLUE = (220, 150, 60)
DARK = (18, 21, 26)

keyboard = np.zeros((600, 1000, 3), dtype=np.uint8)
board = np.zeros((300, 1400, 3), dtype=np.uint8)

KEY_POSITIONS = {
    0: (0, 0), 1: (200, 0), 2: (400, 0), 3: (600, 0), 4: (800, 0),
    5: (0, 200), 6: (200, 200), 7: (400, 200), 8: (600, 200), 9: (800, 200),
    10: (0, 400), 11: (200, 400), 12: (400, 400), 13: (600, 400), 14: (800, 400),
}


def rounded_rect(img, pt1, pt2, color, radius=18, thickness=-1):
    x1, y1 = pt1
    x2, y2 = pt2
    if thickness == -1:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)
    else:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)


def put_center(img, text, center, scale=1.0, color=WHITE, thickness=2):
    size = cv2.getTextSize(text, FONT, scale, thickness)[0]
    x = int(center[0] - size[0] / 2)
    y = int(center[1] + size[1] / 2)
    cv2.putText(img, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def draw_progress(img, x, y, width, height, progress, color=ACCENT):
    progress = max(0.0, min(1.0, progress))
    rounded_rect(img, (x, y), (x + width, y + height), CARD2, 8)
    if progress > 0:
        rounded_rect(img, (x, y), (x + int(width * progress), y + height), color, 8)

# =========================================================
# CALIBRATION PROFILE
# =========================================================

def load_profile():
    if not os.path.exists(PROFILE_PATH):
        return None
    try:
        with open(PROFILE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "open_baseline" in data and "blink_threshold" in data:
            return data
    except Exception:
        pass
    return None


def save_profile(open_baseline, closed_baseline, blink_threshold, preferred_frames_per_letter=None):
    data = {
        "open_baseline": float(open_baseline),
        "closed_baseline": float(closed_baseline),
        "blink_threshold": float(blink_threshold),
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if preferred_frames_per_letter is not None:
        data["preferred_frames_per_letter"] = int(preferred_frames_per_letter)
    try:
        with open(PROFILE_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

# =========================================================
# EYE MATH
# =========================================================

def distance(p1, p2):
    return hypot(p1.x - p2.x, p1.y - p2.y)


def eye_aspect_ratio(landmarks, indices):
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in indices]
    horizontal = distance(p1, p4)
    if horizontal == 0:
        return 0.0
    return (distance(p2, p6) + distance(p3, p5)) / (2.0 * horizontal)

LEFT_EYE = [33, 159, 158, 133, 153, 145]
RIGHT_EYE = [362, 386, 385, 263, 380, 374]
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]


def iris_center(landmarks, indices):
    xs = [landmarks[i].x for i in indices]
    ys = [landmarks[i].y for i in indices]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def get_gaze(landmarks):
    li = iris_center(landmarks, LEFT_IRIS)
    ri = iris_center(landmarks, RIGHT_IRIS)

    left_outer = landmarks[33]
    left_inner = landmarks[133]
    right_inner = landmarks[362]
    right_outer = landmarks[263]

    left_width = left_inner.x - left_outer.x
    right_width = right_outer.x - right_inner.x

    if abs(left_width) < 0.001 or abs(right_width) < 0.001:
        return 0.5

    left_ratio = (li[0] - left_outer.x) / left_width
    right_ratio = (ri[0] - right_inner.x) / right_width
    return (left_ratio + right_ratio) / 2.0


def draw_eye_points(frame, landmarks, indices, color):
    points = []
    for index in indices:
        lm = landmarks[index]
        points.append([int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])])
    cv2.polylines(frame, [np.array(points, np.int32)], True, color, 2, cv2.LINE_AA)

# =========================================================
# WORD PREDICTION
# =========================================================

def current_partial_word(text):
    if not text:
        return ""
    return text.split(" ")[-1].lower()


def get_predictions(text, limit=3):
    partial = current_partial_word(text)
    if not partial:
        return []
    matches = [w for w in WORD_SET if w.startswith(partial) and w != partial]
    matches.sort(key=lambda w: (-WORD_FREQ[w], len(w)))
    return matches[:limit]

# =========================================================
# TEXT OPERATIONS
# =========================================================
typed_text = ""


def add_text(value):
    global typed_text
    typed_text += value


def backspace():
    global typed_text
    if typed_text:
        typed_text = typed_text[:-1]


def accept_prediction(word):
    global typed_text
    partial = current_partial_word(typed_text)
    if partial:
        typed_text = typed_text[:-len(partial)] + word + " "

# =========================================================
# DRAW KEYBOARD
# =========================================================

def draw_key(index, text, selected=False, paused=False):
    x, y = KEY_POSITIONS[index]
    pad = 7
    x1, y1 = x + pad, y + pad
    x2, y2 = x + 200 - pad, y + 200 - pad

    if paused and selected:
        fill = (60, 72, 78)
        border = GREEN
        text_color = WHITE
    elif selected:
        fill = (74, 88, 105)
        border = ACCENT
        text_color = WHITE
    else:
        fill = CARD
        border = (60, 68, 78)
        text_color = WHITE

    rounded_rect(keyboard, (x1, y1), (x2, y2), fill, 18)
    cv2.rectangle(keyboard, (x1, y1), (x2, y2), border, 3)

    if len(text) == 1:
        scale, thick = 5.5, 3
    elif text == "SPACE":
        scale, thick = 2.0, 2
    elif text == "BACK":
        scale, thick = 2.0, 2
    elif text == "PRED":
        scale, thick = 1.8, 2
    else:
        scale, thick = 2.3, 2

    put_center(keyboard, text, ((x1 + x2) // 2, (y1 + y2) // 2), scale, text_color, thick)


def draw_keyboard_screen(current_keys, letter_index, paused, predictions=None):
    keyboard[:] = BG

    # Header
    cv2.putText(keyboard, "GAZETYPE", (28, 38), FONT, 0.9, WHITE, 2, cv2.LINE_AA)
    cv2.putText(keyboard, "EYE-GAZE KEYBOARD", (28, 65), FONT_SMALL, 0.52, MUTED, 1, cv2.LINE_AA)

    # Keys are drawn in the same 3x5 grid but leave a slim header area.
    for i in range(15):
        draw_key(i, current_keys[i], i == letter_index, paused)

    # Header overlays are kept minimal so the 15-key layout stays familiar.
    if paused:
        status = "PAUSED  •  BLINK AGAIN TO SELECT"
        cv2.putText(keyboard, status, (370, 38), FONT_SMALL, 0.48, GREEN, 1, cv2.LINE_AA)

    if predictions:
        cv2.putText(keyboard, "PREDICTIONS AVAILABLE  •  SELECT PRED", (500, 65),
                    FONT_SMALL, 0.42, ACCENT, 1, cv2.LINE_AA)


def draw_menu(selected_side):
    keyboard[:] = BG
    cv2.putText(keyboard, "GAZETYPE", (38, 58), FONT, 1.2, WHITE, 2, cv2.LINE_AA)
    cv2.putText(keyboard, "Choose a keyboard with your eyes", (38, 88), FONT_SMALL, 0.62, MUTED, 1, cv2.LINE_AA)

    cards = [
        (40, 140, 470, 520, "LEFT", "Q  W  E  R  T", selected_side == "left"),
        (530, 140, 960, 520, "RIGHT", "Y  U  I  O  P", selected_side == "right"),
    ]

    for x1, y1, x2, y2, title, subtitle, active in cards:
        fill = CARD2 if active else CARD
        border = ACCENT if active else (65, 73, 84)
        rounded_rect(keyboard, (x1, y1), (x2, y2), fill, 24)
        cv2.rectangle(keyboard, (x1, y1), (x2, y2), border, 4)
        put_center(keyboard, title, ((x1 + x2) // 2, 275), 2.0, WHITE, 3)
        put_center(keyboard, subtitle, ((x1 + x2) // 2, 350), 0.85, MUTED, 2)
        put_center(keyboard, "LOOK HERE", ((x1 + x2) // 2, 425), 0.62, ACCENT, 2)

    cv2.putText(keyboard, "Hold your gaze on a side to open it", (275, 570), FONT_SMALL, 0.58, MUTED, 1, cv2.LINE_AA)

# =========================================================
# PREDICTION CONTROL
# =========================================================

def draw_prediction_keyboard(predictions, index):
    keyboard[:] = BG
    cv2.putText(keyboard, "WORD PREDICTION", (30, 55), FONT, 1.0, WHITE, 2, cv2.LINE_AA)
    cv2.putText(keyboard, "Look LEFT / RIGHT to move • Blink to select", (30, 88), FONT_SMALL, 0.55, MUTED, 1, cv2.LINE_AA)

    labels = predictions[:3] + ["CANCEL"]
    for i, label in enumerate(labels):
        x1 = 45 + i * 235
        x2 = x1 + 205
        y1, y2 = 170, 430
        active = i == index
        rounded_rect(keyboard, (x1, y1), (x2, y2), CARD2 if active else CARD, 22)
        cv2.rectangle(keyboard, (x1, y1), (x2, y2), ACCENT if active else (65, 73, 84), 4)
        put_center(keyboard, label.upper(), ((x1 + x2) // 2, 295), 1.35 if label != "CANCEL" else 0.9, WHITE, 2)
        put_center(keyboard, f"{i + 1}", ((x1 + x2) // 2, 385), 0.65, MUTED, 2)

# =========================================================
# BOARD
# =========================================================

def draw_board(predictions, speed_value, accuracy_value):
    board[:] = BG

    cv2.putText(board, "GAZETYPE", (35, 42), FONT, 0.75, WHITE, 2, cv2.LINE_AA)
    cv2.putText(board, "COMMUNICATION", (35, 67), FONT_SMALL, 0.45, MUTED, 1, cv2.LINE_AA)

    # Text card
    rounded_rect(board, (25, 85), (1375, 205), CARD, 20)

    if typed_text:
        display = typed_text
        if len(display) > 42:
            display = "..." + display[-39:]
        cv2.putText(board, display, (55, 155), FONT, 1.55, WHITE, 2, cv2.LINE_AA)
    else:
        cv2.putText(board, "Start typing with your eyes...", (55, 155), FONT, 1.1, MUTED, 1, cv2.LINE_AA)

    # Cursor
    if typed_text:
        cv2.line(board, (58, 174), (58, 190), ACCENT, 2)

    # Suggestions
    if predictions:
        cv2.putText(board, "SUGGESTIONS", (35, 238), FONT_SMALL, 0.52, MUTED, 1, cv2.LINE_AA)
        for i, word in enumerate(predictions):
            x1 = 25 + i * 280
            x2 = x1 + 260
            rounded_rect(board, (x1, 250), (x2, 285), CARD2, 12)
            put_center(board, word.upper(), ((x1 + x2) // 2, 268), 0.58, WHITE, 1)

    cv2.putText(board, f"CHARACTERS  {len(typed_text)}", (900, 238), FONT_SMALL, 0.48, MUTED, 1, cv2.LINE_AA)
    cv2.putText(board, f"SCAN  {speed_value:.1f}x", (1080, 238), FONT_SMALL, 0.48, MUTED, 1, cv2.LINE_AA)
    cv2.putText(board, f"CLEAN  {accuracy_value:.0f}%", (1220, 238), FONT_SMALL, 0.43, GREEN, 1, cv2.LINE_AA)

# =========================================================
# APPLICATION STATE
# =========================================================
profile = load_profile()

if profile:
    calibration_stage = "READY"
    open_baseline = float(profile["open_baseline"])
    closed_baseline = float(profile["closed_baseline"])
    blink_threshold = float(profile["blink_threshold"])
    startup_message = "Saved eye profile loaded"
else:
    calibration_stage = "OPEN"
    open_baseline = None
    closed_baseline = None
    blink_threshold = None
    startup_message = ""

calibration_start = time.time()
open_ear_samples = []
blink_closed_samples = []
current_blink_min = None

screen = "MENU"
selected_side = "right"
letter_index = 0
frame_number = 0
letter_paused = False
first_blink_time = 0.0

left_ear_history = deque(maxlen=BLINK_SMOOTHING)
right_ear_history = deque(maxlen=BLINK_SMOOTHING)
blink_closed_frames = 0
eyes_open_frames = 0
blink_armed = True
last_blink_time = 0.0

gaze_history = deque(maxlen=5)
gaze_frames = 0

# Prediction mode
prediction_mode = False
prediction_index = 0
prediction_options = []
last_prediction_move = 0.0

# Adaptive speed
frames_per_letter = BASE_FRAMES_PER_LETTER
frames_per_letter = max(MIN_FRAMES_PER_LETTER, min(MAX_FRAMES_PER_LETTER, frames_per_letter))
recent_scan_times = deque(maxlen=6)
scan_started_at = time.time()

# Session metrics
session_start = time.time()
selection_count = 0
backspace_count = 0
successful_blinks = 0
rejected_blinks = 0
last_selected_time = 0.0

message = startup_message
message_until = time.time() + (2.5 if startup_message else 0)
timestamp_ms = 0

# =========================================================
# ADAPTIVE SPEED
# =========================================================

def update_adaptive_speed(selection_time=None):
    global frames_per_letter

    if selection_time is not None:
        recent_scan_times.append(selection_time)

    if len(recent_scan_times) < 4:
        return

    avg = float(np.median(recent_scan_times))

    # User is reaching selections quickly -> modest speed increase.
    if avg < 2.5:
        frames_per_letter = max(MIN_FRAMES_PER_LETTER, frames_per_letter - 2)

    # User takes a long time -> slow the scanner for comfort.
    elif avg > 5.0:
        frames_per_letter = min(MAX_FRAMES_PER_LETTER, frames_per_letter + 2)


def speed_multiplier():
    return BASE_FRAMES_PER_LETTER / max(1, frames_per_letter)


def correction_free_rate():
    if selection_count == 0:
        return 100.0
    errors = min(backspace_count + rejected_blinks, selection_count)
    return max(0.0, 100.0 * (1.0 - errors / selection_count))

# =========================================================
# MAIN LOOP
# =========================================================
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Camera frame failed.")
            break

        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        new_timestamp = int(time.time() * 1000)
        if new_timestamp <= timestamp_ms:
            timestamp_ms += 1
        else:
            timestamp_ms = new_timestamp

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if not letter_paused and screen == "LETTERS":
            frame_number += 1

        # =====================================================
        # FACE PROCESSING
        # =====================================================
        if result.face_landmarks:
            landmarks = result.face_landmarks[0]

            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)

            left_ear_history.append(left_ear)
            right_ear_history.append(right_ear)

            smooth_left_ear = float(np.median(left_ear_history))
            smooth_right_ear = float(np.median(right_ear_history))
            smooth_ear = (smooth_left_ear + smooth_right_ear) / 2.0

            # -------------------------------------------------
            # CALIBRATION
            # -------------------------------------------------
            if calibration_stage == "OPEN":
                elapsed = time.time() - calibration_start
                if 0.08 < smooth_ear < 0.50:
                    open_ear_samples.append(smooth_ear)

                if elapsed >= OPEN_CALIBRATION_SECONDS:
                    if len(open_ear_samples) >= 30:
                        open_baseline = float(np.median(open_ear_samples))
                        calibration_stage = "BLINK"
                        calibration_start = time.time()
                        blink_closed_samples.clear()
                        current_blink_min = None
                    else:
                        open_ear_samples.clear()
                        calibration_start = time.time()

            elif calibration_stage == "BLINK":
                loose_threshold = open_baseline * 0.85

                if smooth_ear < loose_threshold:
                    if current_blink_min is None:
                        current_blink_min = smooth_ear
                    else:
                        current_blink_min = min(current_blink_min, smooth_ear)
                else:
                    if current_blink_min is not None:
                        if current_blink_min < open_baseline * 0.75:
                            blink_closed_samples.append(current_blink_min)
                        current_blink_min = None

                if len(blink_closed_samples) >= CALIBRATION_BLINKS_REQUIRED:
                    closed_baseline = float(np.median(blink_closed_samples))
                    blink_threshold = (open_baseline + closed_baseline) / 2.0
                    save_profile(open_baseline, closed_baseline, blink_threshold, frames_per_letter)
                    calibration_stage = "READY"
                    blink_armed = True
                    message = "Calibration complete • profile saved"
                    message_until = time.time() + 3.0

            # -------------------------------------------------
            # READY / BLINK DETECTION
            # -------------------------------------------------
            elif calibration_stage == "READY":
                eyes_closed = smooth_ear < blink_threshold

                if eyes_closed:
                    blink_closed_frames += 1
                    eyes_open_frames = 0
                    draw_eye_points(frame, landmarks, LEFT_EYE, GREEN)
                    draw_eye_points(frame, landmarks, RIGHT_EYE, GREEN)
                else:
                    blink_closed_frames = 0
                    eyes_open_frames += 1
                    draw_eye_points(frame, landmarks, LEFT_EYE, RED)
                    draw_eye_points(frame, landmarks, RIGHT_EYE, RED)
                    if eyes_open_frames >= OPEN_FRAMES_REQUIRED:
                        blink_armed = True

                if (
                    eyes_closed
                    and blink_closed_frames >= BLINK_FRAMES_REQUIRED
                    and blink_armed
                    and time.time() - last_blink_time > BLINK_COOLDOWN
                ):
                    last_blink_time = time.time()
                    blink_armed = False
                    blink_closed_frames = 0
                    successful_blinks += 1

                    # -------------------------------------------------
                    # PREDICTION SELECTION
                    # -------------------------------------------------
                    if prediction_mode:
                        if prediction_index < len(prediction_options):
                            accept_prediction(prediction_options[prediction_index])
                            selection_count += 1
                            play_audio(sound)
                            message = "WORD ADDED"
                        else:
                            message = "PREDICTIONS CLOSED"

                        prediction_mode = False
                        prediction_index = 0
                        prediction_options = []
                        screen = "MENU"
                        message_until = time.time() + 1.5
                        gaze_history.clear()
                        gaze_frames = 0

                    # -------------------------------------------------
                    # FIRST BLINK = PAUSE
                    # -------------------------------------------------
                    elif screen == "LETTERS" and not letter_paused:
                        letter_paused = True
                        first_blink_time = time.time()
                        current_keys = keys_left if selected_side == "left" else keys_right
                        current_item = current_keys[letter_index]
                        message = f"PAUSED: {current_item}"
                        message_until = time.time() + 4.0

                    # -------------------------------------------------
                    # SECOND BLINK = SELECT
                    # -------------------------------------------------
                    elif screen == "LETTERS" and letter_paused:
                        if time.time() - first_blink_time <= CONFIRMATION_TIMEOUT:
                            current_keys = keys_left if selected_side == "left" else keys_right
                            selected = current_keys[letter_index]

                            if len(selected) == 1:
                                add_text(selected)
                                selection_count += 1
                            elif selected == "SPACE":
                                add_text(" ")
                                selection_count += 1
                            elif selected == "BACK":
                                backspace()
                                backspace_count += 1
                                selection_count += 1
                            elif selected == "PRED":
                                prediction_options = get_predictions(typed_text)
                                if prediction_options:
                                    prediction_mode = True
                                    prediction_index = 0
                                    last_prediction_move = time.time()
                                    screen = "PREDICTIONS"
                                else:
                                    message = "No predictions for current word"

                            play_audio(sound)

                            # Adaptive speed uses the time from keyboard entry to selection.
                            scan_time = max(0.1, time.time() - scan_started_at)
                            update_adaptive_speed(scan_time)

                            if not prediction_mode:
                                screen = "MENU"

                            letter_paused = False
                            letter_index = 0
                            frame_number = 0
                            gaze_frames = 0
                            gaze_history.clear()

                            message = f"SELECTED: {selected}"
                            message_until = time.time() + 1.5
                            scan_started_at = time.time()

                        else:
                            rejected_blinks += 1
                            first_blink_time = time.time()
                            message = "BLINK AGAIN TO SELECT"
                            message_until = time.time() + 2.0

                # -------------------------------------------------
                # PAUSE TIMEOUT -> SKIP CURRENT LETTER
                # -------------------------------------------------
                # If the user blinks once to pause a letter but does
                # not blink again within the confirmation window,
                # cancel the pause and continue from the NEXT letter.
                if (
                    screen == "LETTERS"
                    and letter_paused
                    and time.time() - first_blink_time > CONFIRMATION_TIMEOUT
                ):
                    letter_paused = False
                    letter_index = (letter_index + 1) % 15
                    frame_number = 0
                    first_blink_time = 0.0
                    message = "SKIPPED • CONTINUING"
                    message_until = time.time() + 1.2

                # -------------------------------------------------
                # GAZE SIDE SELECTION
                # -------------------------------------------------
                if screen == "MENU":
                    gaze = get_gaze(landmarks)
                    gaze_history.append(gaze)
                    smooth_gaze = float(np.mean(gaze_history))

                    if smooth_gaze < GAZE_LEFT_THRESHOLD:
                        selected_side = "left"
                        gaze_frames += 1
                        if gaze_frames >= GAZE_REQUIRED_FRAMES:
                            screen = "LETTERS"
                            letter_index = 0
                            frame_number = 0
                            gaze_frames = 0
                            gaze_history.clear()
                            scan_started_at = time.time()
                            play_audio(left_sound)

                    elif smooth_gaze > GAZE_RIGHT_THRESHOLD:
                        selected_side = "right"
                        gaze_frames += 1
                        if gaze_frames >= GAZE_REQUIRED_FRAMES:
                            screen = "LETTERS"
                            letter_index = 0
                            frame_number = 0
                            gaze_frames = 0
                            gaze_history.clear()
                            scan_started_at = time.time()
                            play_audio(right_sound)
                    else:
                        gaze_frames = 0

                # -------------------------------------------------
                # PREDICTION NAVIGATION
                # -------------------------------------------------
                elif screen == "PREDICTIONS" and prediction_mode:
                    gaze = get_gaze(landmarks)
                    gaze_history.append(gaze)
                    smooth_gaze = float(np.mean(gaze_history))

                    global_last_move = time.time()
                    if global_last_move - last_prediction_move > 0.55:
                        if smooth_gaze < GAZE_LEFT_THRESHOLD:
                            prediction_index = max(0, prediction_index - 1)
                            last_prediction_move = global_last_move
                        elif smooth_gaze > GAZE_RIGHT_THRESHOLD:
                            prediction_index = min(3, prediction_index + 1)
                            last_prediction_move = global_last_move

        # =====================================================
        # DRAW KEYBOARD
        # =====================================================
        predictions = get_predictions(typed_text)

        if calibration_stage != "READY":
            keyboard[:] = BG
            title = "GAZETYPE"
            put_center(keyboard, title, (500, 170), 2.2, WHITE, 3)

            if calibration_stage == "OPEN":
                subtitle = "LOOK AT THE CAMERA WITH YOUR EYES OPEN"
                progress = min(1.0, (time.time() - calibration_start) / OPEN_CALIBRATION_SECONDS)
            else:
                subtitle = f"BLINK NORMALLY  •  {len(blink_closed_samples)}/{CALIBRATION_BLINKS_REQUIRED}"
                progress = min(1.0, len(blink_closed_samples) / CALIBRATION_BLINKS_REQUIRED)

            put_center(keyboard, subtitle, (500, 260), 0.75, MUTED, 2)
            draw_progress(keyboard, 180, 330, 640, 28, progress, ACCENT)
            put_center(keyboard, "Personalized calibration is saved automatically", (500, 430), 0.52, MUTED, 1)
            put_center(keyboard, "R = recalibrate  •  ESC = exit", (500, 500), 0.48, MUTED, 1)

        elif screen == "MENU":
            draw_menu(selected_side)

        elif screen == "LETTERS":
            current_keys = keys_left if selected_side == "left" else keys_right

            if not letter_paused and frame_number >= frames_per_letter:
                letter_index += 1
                frame_number = 0

            if letter_index >= 15:
                letter_index = 0

            draw_keyboard_screen(current_keys, letter_index, letter_paused, predictions)

            # Scanning progress bar for the active key.
            if not letter_paused:
                progress = frame_number / max(1, frames_per_letter)
                draw_progress(keyboard, 170, 575, 660, 10, progress, ACCENT)
            else:
                draw_progress(keyboard, 170, 575, 660, 10, 1.0, GREEN)

        elif screen == "PREDICTIONS":
            draw_prediction_keyboard(prediction_options, prediction_index)

        # =====================================================
        # CAMERA UI
        # =====================================================
        camera_view = frame.copy()

        # Top status bar
        cv2.rectangle(camera_view, (0, 0), (width, 70), DARK, -1)
        cv2.putText(camera_view, "GAZETYPE", (20, 30), FONT, 0.75, WHITE, 2, cv2.LINE_AA)

        if calibration_stage != "READY":
            status = "CALIBRATING"
            status_color = ACCENT
        elif prediction_mode:
            status = "WORD PREDICTION"
            status_color = ACCENT
        elif screen == "MENU":
            status = "CHOOSE SIDE"
            status_color = BLUE
        elif letter_paused:
            status = "LETTER PAUSED"
            status_color = GREEN
        else:
            status = "CYCLING LETTERS"
            status_color = WHITE

        cv2.putText(camera_view, status, (250, 30), FONT, 0.58, status_color, 2, cv2.LINE_AA)

        # Blink indicator
        blink_state = "BLINK READY" if blink_armed else "BLINK PROCESSING"
        cv2.putText(camera_view, blink_state, (width - 230, 30), FONT_SMALL, 0.45, GREEN if blink_armed else MUTED, 1, cv2.LINE_AA)

        # Bottom instruction
        cv2.rectangle(camera_view, (0, height - 58), (width, height), DARK, -1)
        if prediction_mode:
            instruction = "LOOK LEFT / RIGHT = CHOOSE WORD   •   BLINK = SELECT"
        elif screen == "MENU":
            instruction = "LOOK LEFT / RIGHT = CHOOSE KEYBOARD"
        elif letter_paused:
            instruction = "BLINK AGAIN = SELECT   •   WAIT = CANCEL PAUSE"
        else:
            instruction = "BLINK ONCE = PAUSE   •   BLINK AGAIN = SELECT"

        cv2.putText(camera_view, instruction, (20, height - 22), FONT_SMALL, 0.48, WHITE, 1, cv2.LINE_AA)

        # Compact technical readout for demonstration/viva.
        if result.face_landmarks:
            cv2.putText(camera_view, f"EAR {smooth_ear:.3f}", (20, 95), FONT_SMALL, 0.42, MUTED, 1, cv2.LINE_AA)
            cv2.putText(camera_view, f"SCAN {frames_per_letter}f", (20, 118), FONT_SMALL, 0.42, MUTED, 1, cv2.LINE_AA)

        if message and time.time() < message_until:
            rounded_rect(camera_view, (20, 135), (min(width - 20, 480), 180), DARK, 15)
            cv2.putText(camera_view, message, (38, 165), FONT_SMALL, 0.48, WHITE, 1, cv2.LINE_AA)

        # =====================================================
        # BOARD UI
        # =====================================================
        draw_board(predictions, speed_multiplier(), correction_free_rate())

        # =====================================================
        # WINDOWS
        # =====================================================
        cv2.imshow("GAZETYPE - Camera", camera_view)
        cv2.imshow("GAZETYPE - Keyboard", keyboard)
        cv2.imshow("GAZETYPE - Message", board)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break

        if key in (ord("r"), ord("R")):
            calibration_stage = "OPEN"
            calibration_start = time.time()
            open_ear_samples.clear()
            blink_closed_samples.clear()
            current_blink_min = None
            open_baseline = None
            closed_baseline = None
            blink_threshold = None
            message = "Recalibration started"
            message_until = time.time() + 2.0

finally:
    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()

    session_seconds = max(1.0, time.time() - session_start)
    minutes = session_seconds / 60.0
    words = len(typed_text.split())
    wpm = words / minutes if minutes > 0 else 0.0

    # Persist the learned scan speed for the next session.
    if open_baseline is not None and blink_threshold is not None:
        save_profile(open_baseline, closed_baseline, blink_threshold, frames_per_letter)

    # Keep a lightweight local history for project evaluation.
    session_record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "characters": len(typed_text),
        "words": words,
        "selections": selection_count,
        "backspaces": backspace_count,
        "rejected_blinks": rejected_blinks,
        "correction_free_rate": round(correction_free_rate(), 1),
        "final_frames_per_letter": frames_per_letter,
        "estimated_wpm": round(wpm, 1),
    }
    try:
        history = []
        if os.path.exists("gazetype_session_history.json"):
            with open("gazetype_session_history.json", "r", encoding="utf-8") as f:
                history = json.load(f)
        history.append(session_record)
        history = history[-50:]
        with open("gazetype_session_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    except Exception:
        pass

    print("\n================ GAZETYPE SESSION ================")
    print(f"Text: {typed_text}")
    print(f"Characters: {len(typed_text)}")
    print(f"Words: {words}")
    print(f"Selections: {selection_count}")
    print(f"Backspaces: {backspace_count}")
    print(f"Rejected blinks: {rejected_blinks}")
    print(f"Correction-free rate: {correction_free_rate():.1f}%")
    print(f"Final scan setting: {frames_per_letter} frames/letter")
    print(f"Estimated WPM: {wpm:.1f}")
    print("===================================================")
