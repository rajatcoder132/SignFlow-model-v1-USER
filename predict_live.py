import os
import sys
import time
import math
import threading
import subprocess
import tkinter as tk
from collections import deque
import numpy as np
import cv2
import joblib
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "hand_landmarker.task")
CLASSIFIER_PATH = os.path.join(SCRIPT_DIR, "models", "asl_classifier.joblib")
WINDOW_NAME = "Live ASL Recognition (A-Z)"

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


def get_camera_names():
    try:
        from pygrabber.dshow_graph import FilterGraph
    except ImportError:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pygrabber", "-q"])
            from pygrabber.dshow_graph import FilterGraph
        except Exception:
            return [f"Camera Index {i}" for i in range(10)]
    try:
        devices = FilterGraph().get_input_devices()
        return devices if devices else [f"Camera Index {i}" for i in range(10)]
    except Exception:
        return [f"Camera Index {i}" for i in range(10)]


def select_camera_popup(camera_names, current_index=0):
    selected = {"index": None}
    root = tk.Tk()
    root.title("Select Video Camera")
    root.geometry("420x460")
    root.resizable(False, False)
    root.configure(bg="#1e1e2e")
    root.attributes("-topmost", True)

    tk.Label(
        root, text="Select Camera", font=("Segoe UI", 16, "bold"),
        fg="#cdd6f4", bg="#1e1e2e"
    ).pack(pady=(18, 4))

    tk.Label(
        root, text="Choose your webcam for real-time sign recognition",
        font=("Segoe UI", 9), fg="#a6adc8", bg="#1e1e2e"
    ).pack(pady=(0, 12))

    frame_list = tk.Frame(root, bg="#181825", bd=1, relief="solid")
    frame_list.pack(fill="both", expand=True, padx=20, pady=5)

    scrollbar = tk.Scrollbar(frame_list)
    scrollbar.pack(side="right", fill="y")

    listbox = tk.Listbox(
        frame_list, font=("Segoe UI", 10), bg="#181825", fg="#cdd6f4",
        selectbackground="#89b4fa", selectforeground="#11111b",
        activestyle="none", bd=0, highlightthickness=0, yscrollcommand=scrollbar.set
    )
    listbox.pack(fill="both", expand=True, padx=5, pady=5)
    scrollbar.config(command=listbox.yview)

    for i, name in enumerate(camera_names):
        listbox.insert("end", f" [{i}] {name}")

    if 0 <= current_index < len(camera_names):
        listbox.selection_set(current_index)
        listbox.activate(current_index)
        listbox.see(current_index)

    def confirm(event=None):
        sel = listbox.curselection()
        if sel:
            selected["index"] = sel[0]
        root.destroy()

    listbox.bind("<Double-Button-1>", confirm)
    listbox.bind("<Return>", confirm)

    btn_frame = tk.Frame(root, bg="#1e1e2e")
    btn_frame.pack(fill="x", padx=20, pady=16)

    tk.Button(
        btn_frame, text="Start Camera", font=("Segoe UI", 11, "bold"),
        bg="#89b4fa", fg="#11111b", activebackground="#b4befe", activeforeground="#11111b",
        bd=0, padx=16, pady=8, cursor="hand2", command=confirm
    ).pack(side="right", padx=(8, 0))

    tk.Button(
        btn_frame, text="Cancel", font=("Segoe UI", 10),
        bg="#313244", fg="#cdd6f4", activebackground="#45475a", activeforeground="#cdd6f4",
        bd=0, padx=14, pady=8, cursor="hand2", command=root.destroy
    ).pack(side="right")

    root.update_idletasks()
    x = (root.winfo_screenwidth() - root.winfo_width()) // 2
    y = (root.winfo_screenheight() - root.winfo_height()) // 2
    root.geometry(f"+{x}+{y}")
    root.mainloop()

    return selected["index"]


class ThreadedCamera:
    def __init__(self, index=0, width=1280, height=720, target_fps=60):
        self.index = index
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.cap = None
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self._open()

    def _open(self):
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(self.index, backend)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def start(self):
        if self.running:
            return self
        self.running = True
        ret, frame = self.cap.read()
        if ret and frame is not None:
            self.frame = frame
        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()
        return self

    def _reader_loop(self):
        while self.running:
            if not self.cap or not self.cap.isOpened():
                break
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.005)

    def read(self):
        with self.lock:
            return (self.frame is not None), (self.frame.copy() if self.frame is not None else None)

    def toggle_resolution(self):
        target_w, target_h = (1280, 720) if self.width == 1920 else (1920, 1080)
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=0.5)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.running = True
        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()
        return f"{self.width}x{self.height}"

    def release(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=0.5)
        if self.cap:
            self.cap.release()


def open_camera(index=0):
    cam = ThreadedCamera(index=index, width=1280, height=720, target_fps=60)
    if cam.cap and cam.cap.isOpened():
        return cam.start()
    return None


def extract_features(landmarks):
    wrist = landmarks[0]
    raw_coords = []
    for lm in landmarks:
        raw_coords.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
    scale = max(max(abs(val) for val in raw_coords), 1e-6)
    return [val / scale for val in raw_coords]


def disambiguate_r_u_v(landmarks):
    """
    Geometrically separates R, U, and V using physical ASL invariants:
    - R: Index and middle fingers crossed (inversion of finger axis or ratio <= 0.65).
    - V: Index and middle fingers spread apart (ratio >= 1.25 or diverging angle cos < 0.96).
    - U: Index and middle fingers touching and parallel side-by-side (ratio < 1.25 and cos >= 0.95).
    """
    # Tip 8 (Index), Tip 12 (Middle)
    dx_tip = landmarks[8].x - landmarks[12].x
    dy_tip = landmarks[8].y - landmarks[12].y
    dz_tip = landmarks[8].z - landmarks[12].z
    d_tips = math.sqrt(dx_tip * dx_tip + dy_tip * dy_tip + dz_tip * dz_tip)

    # Knuckle 5 (Index MCP), Knuckle 9 (Middle MCP)
    dx_mcp = landmarks[5].x - landmarks[9].x
    dy_mcp = landmarks[5].y - landmarks[9].y
    dz_mcp = landmarks[5].z - landmarks[9].z
    d_mcp = max(math.sqrt(dx_mcp * dx_mcp + dy_mcp * dy_mcp + dz_mcp * dz_mcp), 1e-6)

    ratio = d_tips / d_mcp
    cross_prod = dx_tip * dx_mcp

    # Vector Index: PIP 6 to Tip 8
    idx_vx = landmarks[8].x - landmarks[6].x
    idx_vy = landmarks[8].y - landmarks[6].y
    idx_vz = landmarks[8].z - landmarks[6].z
    mag_idx = max(math.sqrt(idx_vx * idx_vx + idx_vy * idx_vy + idx_vz * idx_vz), 1e-6)

    # Vector Middle: PIP 10 to Tip 12
    mid_vx = landmarks[12].x - landmarks[10].x
    mid_vy = landmarks[12].y - landmarks[10].y
    mid_vz = landmarks[12].z - landmarks[10].z
    mag_mid = max(math.sqrt(mid_vx * mid_vx + mid_vy * mid_vy + mid_vz * mid_vz), 1e-6)

    cos_spread = (idx_vx * mid_vx + idx_vy * mid_vy + idx_vz * mid_vz) / (mag_idx * mag_mid)

    if cross_prod < 0 or ratio <= 0.65:
        return "R"
    elif ratio >= 1.25 or (cos_spread < 0.96 and ratio >= 1.05):
        return "V"
    else:
        return "U"


def disambiguate_e_o(landmarks):
    """
    Separates E and O:
    - E: Fingertips are curled tightly against knuckle/palm (idx_ext < 2.30).
    - O: Fingers arch forward forming an open circular loop (idx_ext >= 2.30).
    """
    dx_mcp = landmarks[5].x - landmarks[9].x
    dy_mcp = landmarks[5].y - landmarks[9].y
    dz_mcp = landmarks[5].z - landmarks[9].z
    d_mcp = max(math.sqrt(dx_mcp * dx_mcp + dy_mcp * dy_mcp + dz_mcp * dz_mcp), 1e-6)

    dx = landmarks[8].x - landmarks[5].x
    dy = landmarks[8].y - landmarks[5].y
    dz = landmarks[8].z - landmarks[5].z
    idx_ext = math.sqrt(dx * dx + dy * dy + dz * dz) / d_mcp

    return "E" if idx_ext < 2.30 else "O"


def disambiguate_g_h(landmarks):
    """
    Separates G and H:
    - G: Only index finger extended horizontally; middle finger curled (mid_ext/idx_ext < 0.82).
    - H: Both index and middle fingers extended parallel together (mid_ext/idx_ext >= 0.82, d_tips < 1.45, mid_ext >= 1.50).
    """
    dx_mcp = landmarks[5].x - landmarks[9].x
    dy_mcp = landmarks[5].y - landmarks[9].y
    dz_mcp = landmarks[5].z - landmarks[9].z
    d_mcp = max(math.sqrt(dx_mcp * dx_mcp + dy_mcp * dy_mcp + dz_mcp * dz_mcp), 1e-6)

    dx_idx = landmarks[8].x - landmarks[5].x
    dy_idx = landmarks[8].y - landmarks[5].y
    dz_idx = landmarks[8].z - landmarks[5].z
    idx_ext = math.sqrt(dx_idx * dx_idx + dy_idx * dy_idx + dz_idx * dz_idx) / d_mcp

    dx_mid = landmarks[12].x - landmarks[9].x
    dy_mid = landmarks[12].y - landmarks[9].y
    dz_mid = landmarks[12].z - landmarks[9].z
    mid_ext = math.sqrt(dx_mid * dx_mid + dy_mid * dy_mid + dz_mid * dz_mid) / d_mcp

    dx_tips = landmarks[8].x - landmarks[12].x
    dy_tips = landmarks[8].y - landmarks[12].y
    dz_tips = landmarks[8].z - landmarks[12].z
    d_tips = math.sqrt(dx_tips * dx_tips + dy_tips * dy_tips + dz_tips * dz_tips) / d_mcp

    if (mid_ext / max(idx_ext, 1e-6) >= 0.82) and (d_tips < 1.45) and (mid_ext >= 1.50):
        return "H"
    return "G"


def disambiguate_d_l(landmarks):
    """
    Separates D and L:
    - Both have index finger extended straight up (idx_ext >= 1.45, index tip highest).
    - L: Thumb extended outward laterally (90 degrees), away from index knuckle and curled fingers (dx_thumb >= 1.35 and thumb_mid >= 1.35 and thumb_ring >= 1.25).
    - D: Thumb curled inward touching or closing loop against middle/ring fingertip (dx_thumb < 1.35 or thumb_mid < 1.35).
    """
    dx_mcp = landmarks[5].x - landmarks[9].x
    dy_mcp = landmarks[5].y - landmarks[9].y
    dz_mcp = landmarks[5].z - landmarks[9].z
    d_mcp = max(math.sqrt(dx_mcp * dx_mcp + dy_mcp * dy_mcp + dz_mcp * dz_mcp), 1e-6)

    # Thumb tip (4) to Middle tip (12)
    dx_tm = landmarks[4].x - landmarks[12].x
    dy_tm = landmarks[4].y - landmarks[12].y
    dz_tm = landmarks[4].z - landmarks[12].z
    thumb_mid = math.sqrt(dx_tm * dx_tm + dy_tm * dy_tm + dz_tm * dz_tm) / d_mcp

    # Thumb tip (4) to Ring tip (16)
    dx_tr = landmarks[4].x - landmarks[16].x
    dy_tr = landmarks[4].y - landmarks[16].y
    dz_tr = landmarks[4].z - landmarks[16].z
    thumb_ring = math.sqrt(dx_tr * dx_tr + dy_tr * dy_tr + dz_tr * dz_tr) / d_mcp

    # Horizontal thumb spread from index MCP (5)
    dx_thumb = abs(landmarks[4].x - landmarks[5].x) / d_mcp

    # In L, thumb is extended outward laterally away from fingers
    if dx_thumb >= 1.35 and thumb_mid >= 1.35 and thumb_ring >= 1.25:
        return "L", dx_thumb
    return "D", min(thumb_mid, thumb_ring)


def disambiguate_x_z(landmarks):
    """
    Separates X and Z:
    - In X: Closed fist with index finger hooked/bent like a pirate hook (cos_hook < 0.65 or idx_ext < 1.45).
    - In Z: Index finger extended straight out pointing (cos_hook >= 0.65 and idx_ext >= 1.45).
    """
    dx_mcp = landmarks[5].x - landmarks[9].x
    dy_mcp = landmarks[5].y - landmarks[9].y
    dz_mcp = landmarks[5].z - landmarks[9].z
    d_mcp = max(math.sqrt(dx_mcp * dx_mcp + dy_mcp * dy_mcp + dz_mcp * dz_mcp), 1e-6)

    dx_idx = landmarks[8].x - landmarks[5].x
    dy_idx = landmarks[8].y - landmarks[5].y
    dz_idx = landmarks[8].z - landmarks[5].z
    idx_ext = math.sqrt(dx_idx * dx_idx + dy_idx * dy_idx + dz_idx * dz_idx) / d_mcp

    u_x = landmarks[6].x - landmarks[5].x
    u_y = landmarks[6].y - landmarks[5].y
    u_z = landmarks[6].z - landmarks[5].z
    mag_u = max(math.sqrt(u_x*u_x + u_y*u_y + u_z*u_z), 1e-6)

    v_x = landmarks[8].x - landmarks[6].x
    v_y = landmarks[8].y - landmarks[6].y
    v_z = landmarks[8].z - landmarks[6].z
    mag_v = max(math.sqrt(v_x*v_x + v_y*v_y + v_z*v_z), 1e-6)

    cos_hook = (u_x*v_x + u_y*v_y + u_z*v_z) / (mag_u * mag_v)

    if cos_hook < 0.65 or idx_ext < 1.45:
        return "X", cos_hook
    else:
        return "Z", cos_hook


def detect_silent_z_stroke(index_history):
    """
    Robust ASL 'Z' dynamic trajectory recognition:
    - Analyzes index fingertip (Landmark 8) over recent history.
    - Requires horizontal stroke -> diagonal down stroke -> horizontal stroke.
    - Rejects stationary hand tremors and non-Z motions.
    """
    if len(index_history) < 8:
        return False

    pts = [(pt[1], pt[2]) for pt in index_history]
    ts = [pt[0] for pt in index_history]

    dt = ts[-1] - ts[0]
    if dt < 0.15 or dt > 1.6:
        return False

    arc_length = sum(math.hypot(pts[i][0] - pts[i-1][0], pts[i][1] - pts[i-1][1]) for i in range(1, len(pts)))
    if arc_length < 0.07:
        return False

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    dx_span = max(xs) - min(xs)
    dy_span = max(ys) - min(ys)

    if dx_span < 0.035 or dy_span < 0.035:
        return False

    n = len(pts)
    s_xs = [(xs[max(0, i-1)] + xs[i] + xs[min(n-1, i+1)]) / 3.0 for i in range(n)]

    best_zigzag = False
    for s1 in [int(n * 0.28), int(n * 0.33), int(n * 0.38)]:
        for s2 in [int(n * 0.62), int(n * 0.67), int(n * 0.72)]:
            if s1 < s2:
                d1 = s_xs[s1] - s_xs[0]
                d2 = s_xs[s2] - s_xs[s1]
                d3 = s_xs[-1] - s_xs[s2]
                if (d1 * d2 < -0.0002) and (d2 * d3 < -0.0002) and (abs(d1) >= 0.02 and abs(d2) >= 0.02 and abs(d3) >= 0.02):
                    best_zigzag = True
                    break
        if best_zigzag:
            break

    downward = (max(ys[n//2:]) - min(ys[:n//2])) >= 0.025
    return best_zigzag and downward


def detect_silent_j_hook(pinky_history):
    """
    Robust ASL 'J' trajectory recognition:
    - Analyzes the pinky tip (Landmark 20) over recent history.
    - Requires a distinct downward plunge followed by a sideways and upward hook.
    - Stationary jitter, straight downward drops, and waving are rejected.
    """
    if len(pinky_history) < 8:
        return False

    pts = [(pt[1], pt[2]) for pt in pinky_history]
    ts = [pt[0] for pt in pinky_history]

    dt_total = ts[-1] - ts[0]
    if dt_total < 0.12:
        return False

    ys = [p[1] for p in pts]
    xs = [p[0] for p in pts]

    trough_idx = max(range(len(ys)), key=lambda i: ys[i])

    # Upward hook must have at least 1-2 frames after the trough
    if trough_idx >= len(ys) - 1 or trough_idx < 3:
        return False

    y_trough = ys[trough_idx]
    y_start = min(ys[:trough_idx])
    downward_stroke = (y_trough - y_start) >= 0.045

    y_after = ys[-1]
    hook_up = (y_trough - y_after) >= 0.012

    x_after = xs[-1]
    x_trough = xs[trough_idx]
    hook_sideways = abs(x_after - x_trough) >= 0.016 or (max(xs[trough_idx:]) - min(xs[trough_idx:])) >= 0.016

    start_idx = ys.index(y_start)
    stroke_pts = pts[start_idx:]
    arc_length = sum(math.hypot(stroke_pts[i][0] - stroke_pts[i - 1][0], stroke_pts[i][1] - stroke_pts[i - 1][1]) for i in range(1, len(stroke_pts)))

    return downward_stroke and hook_up and hook_sideways and (arc_length >= 0.055)


def main():
    print("=" * 60)
    print("  SignFlow ASL Recognition - 26-Class Engine (A-Z)")
    print("=" * 60)

    devices = get_camera_names()
    default_idx = 0
    if len(devices) > 1:
        chosen = select_camera_popup(devices, default_idx)
        if chosen is None:
            print("Selection cancelled. Exiting.")
            return
        default_idx = chosen

    if not os.path.exists(MODEL_PATH):
        print(f"Error: MediaPipe landmarker not found at: {MODEL_PATH}")
        return

    if not os.path.exists(CLASSIFIER_PATH):
        print(f"Error: ASL classifier not found at: {CLASSIFIER_PATH}")
        return

    print("Loading model package...")
    loaded = joblib.load(CLASSIFIER_PATH)
    static_clf = loaded.get("static_model", loaded)
    static_clf.n_jobs = 1
    classes = list(loaded.get("classes", getattr(static_clf, "classes_", [])))
    print(f"Active classes ({len(classes)}): {classes}")

    # Pre-compute indices for fast probability aggregation
    r_idx = classes.index('R') if 'R' in classes else None
    u_idx = classes.index('U') if 'U' in classes else None
    v_idx = classes.index('V') if 'V' in classes else None
    e_idx = classes.index('E') if 'E' in classes else None
    o_idx = classes.index('O') if 'O' in classes else None
    g_idx = classes.index('G') if 'G' in classes else None
    h_idx = classes.index('H') if 'H' in classes else None
    i_idx = classes.index('I') if 'I' in classes else None
    j_idx = classes.index('J') if 'J' in classes else None
    d_idx = classes.index('D') if 'D' in classes else None
    l_idx = classes.index('L') if 'L' in classes else None
    n_idx = classes.index('N') if 'N' in classes else None
    x_idx = classes.index('X') if 'X' in classes else None
    z_idx = classes.index('Z') if 'Z' in classes else None

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = open_camera(default_idx)
    if not cap:
        print(f"Failed to open camera {default_idx}.")
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    fps_buffer = deque(maxlen=30)
    prediction_buffer = deque(maxlen=7)
    pinky_history = deque(maxlen=35)
    index_history = deque(maxlen=35)
    current_word = ""

    last_key_press_time = 0.0
    key_debounce = 0.25
    j_hold_until = 0.0
    j_cooldown_until = 0.0
    z_hold_until = 0.0
    z_cooldown_until = 0.0
    smooth_conf = 0.0
    no_hand_count = 0
    stable_prediction = "No Hand"

    with HandLandmarker.create_from_options(options) as detector:
        while True:
            t_start = time.time()
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.time() * 1000)

            result = detector.detect_for_video(mp_image, timestamp_ms)
            now = time.time()

            if result.hand_landmarks:
                no_hand_count = 0
                landmarks = result.hand_landmarks[0]

                # Draw skeleton (Green bones, Red joints)
                for start, end in CONNECTIONS:
                    cv2.line(frame, (int(landmarks[start].x * w), int(landmarks[start].y * h)),
                                   (int(landmarks[end].x * w), int(landmarks[end].y * h)), (0, 255, 0), 2)
                for pt in landmarks:
                    cv2.circle(frame, (int(pt.x * w), int(pt.y * h)), 4, (0, 0, 255), -1)

                features = extract_features(landmarks)
                probs = static_clf.predict_proba([features])[0]

                # Finger extension metrics normalized by knuckle distance (5 to 9)
                dx_mcp = landmarks[5].x - landmarks[9].x
                dy_mcp = landmarks[5].y - landmarks[9].y
                dz_mcp = landmarks[5].z - landmarks[9].z
                d_mcp = max(math.sqrt(dx_mcp * dx_mcp + dy_mcp * dy_mcp + dz_mcp * dz_mcp), 1e-6)

                dx_idx = landmarks[8].x - landmarks[5].x
                dy_idx = landmarks[8].y - landmarks[5].y
                dz_idx = landmarks[8].z - landmarks[5].z
                idx_ext = math.sqrt(dx_idx * dx_idx + dy_idx * dy_idx + dz_idx * dz_idx) / d_mcp

                dx_mid = landmarks[12].x - landmarks[9].x
                dy_mid = landmarks[12].y - landmarks[9].y
                dz_mid = landmarks[12].z - landmarks[9].z
                mid_ext = math.sqrt(dx_mid * dx_mid + dy_mid * dy_mid + dz_mid * dz_mid) / d_mcp

                dx_pinky = landmarks[20].x - landmarks[17].x
                dy_pinky = landmarks[20].y - landmarks[17].y
                dz_pinky = landmarks[20].z - landmarks[17].z
                pinky_ext = math.sqrt(dx_pinky * dx_pinky + dy_pinky * dy_pinky + dz_pinky * dz_pinky) / d_mcp

                # Thumb metrics normalized by knuckle distance (5 to 9)
                dx_tm = landmarks[4].x - landmarks[12].x
                dy_tm = landmarks[4].y - landmarks[12].y
                dz_tm = landmarks[4].z - landmarks[12].z
                thumb_mid = math.sqrt(dx_tm * dx_tm + dy_tm * dy_tm + dz_tm * dz_tm) / d_mcp

                dx_ti = landmarks[4].x - landmarks[5].x
                dy_ti = landmarks[4].y - landmarks[5].y
                dz_ti = landmarks[4].z - landmarks[5].z
                thumb_idx = math.sqrt(dx_ti * dx_ti + dy_ti * dy_ti + dz_ti * dz_ti) / d_mcp

                # 1. HIERARCHICAL PROBABILITY AGGREGATION
                # A. Two-finger extended group: R, U, V
                ruv_group_prob = 0.0
                if r_idx is not None and u_idx is not None and v_idx is not None:
                    ruv_group_prob = probs[r_idx] + probs[u_idx] + probs[v_idx]

                # B. Curled claw / loop group: E, O
                eo_group_prob = 0.0
                if e_idx is not None and o_idx is not None:
                    eo_group_prob = probs[e_idx] + probs[o_idx]

                # C. Horizontal pointing group: G, H
                gh_group_prob = 0.0
                if g_idx is not None and h_idx is not None:
                    gh_group_prob = probs[g_idx] + probs[h_idx]

                # D. Single upright index group: D, L
                dl_group_prob = 0.0
                if d_idx is not None and l_idx is not None:
                    dl_group_prob = probs[d_idx] + probs[l_idx]

                # E. Single active index group: X, Z
                xz_group_prob = 0.0
                if x_idx is not None and z_idx is not None:
                    xz_group_prob = probs[x_idx] + probs[z_idx]

                # Check thumb geometry
                lm4 = landmarks[4]
                lm5 = landmarks[5]
                lm9 = landmarks[9]
                thumb_up = (lm4.y < lm5.y or lm4.y < lm9.y)

                is_index_extended = (idx_ext >= 1.45) and (landmarks[8].y < landmarks[5].y)
                is_other_fingers_down = (landmarks[8].y < landmarks[12].y - 0.03) and (landmarks[8].y < landmarks[16].y - 0.03) and (landmarks[8].y < landmarks[20].y - 0.03)
                is_single_index_up = is_index_extended and is_other_fingers_down and (idx_ext > pinky_ext + 0.25)

                if ruv_group_prob >= 0.40 and not (thumb_up and probs[classes.index('K')] > 0.35 if 'K' in classes else False):
                    # Pure geometric separation with FULL group confidence (90%+)
                    raw_pred = disambiguate_r_u_v(landmarks)
                    conf = ruv_group_prob * 100.0
                elif eo_group_prob >= 0.45:
                    # In E, fingers are curled tightly into palm; in O, fingers arch forward
                    raw_pred = disambiguate_e_o(landmarks)
                    conf = eo_group_prob * 100.0
                elif gh_group_prob >= 0.45 and idx_ext >= 1.55:
                    # G/H requires index finger to be genuinely extended horizontally
                    raw_pred = disambiguate_g_h(landmarks)
                    conf = gh_group_prob * 100.0
                elif is_single_index_up:
                    # Pure geometric separation for D (thumb touches middle/ring tip) vs L (thumb extended wide)
                    raw_pred, geom_val = disambiguate_d_l(landmarks)
                    if raw_pred == "L":
                        spread_score = min(1.0, max(0.0, (geom_val - 1.25) / 0.8))
                        base_p = max(dl_group_prob, probs[l_idx] if l_idx is not None else 0.5)
                        conf = min(98.5, max(82.0, base_p * 35.0 + spread_score * 30.0 + 40.0))
                    else:
                        touch_score = min(1.0, max(0.0, (1.45 - geom_val) / 0.8))
                        base_p = max(dl_group_prob, probs[d_idx] if d_idx is not None else 0.5)
                        conf = min(98.5, max(82.0, base_p * 35.0 + touch_score * 30.0 + 40.0))
                elif not is_single_index_up and (xz_group_prob >= 0.25 or (probs.argmax() in [x_idx, z_idx]) or (x_idx is not None and probs[x_idx] > 0.15) or (z_idx is not None and probs[z_idx] > 0.15)):
                    # Pure geometric separation for X (index hooked) vs Z (index straight pointing)
                    raw_pred, geom_val = disambiguate_x_z(landmarks)
                    if raw_pred == "X":
                        hook_score = min(1.0, max(0.0, (0.70 - geom_val) / 1.2))
                        conf = min(98.5, max(75.0, xz_group_prob * 80.0 + hook_score * 18.0))
                    else:
                        ext_score = min(1.0, max(0.0, (geom_val - 0.50) / 0.5))
                        conf = min(98.5, max(75.0, xz_group_prob * 80.0 + ext_score * 18.0))
                else:
                    max_idx = probs.argmax()
                    conf = probs[max_idx] * 100.0
                    raw_pred = classes[max_idx] if conf >= 30.0 else "Uncertain"

                # 2. DYNAMIC 'J' VS STATIC 'I' (THUMB MUST BE FOLDED; NEVER TRAPS 'Y')
                # In I/J: pinky is extended straight up, index and middle are curled down.
                # In Y: thumb is extended wide outward (thumb_idx >= 1.40).
                # In N: all fingers curled (pinky_ext < 1.40).
                is_pinky_extended = (pinky_ext >= 1.45) and (landmarks[20].y < landmarks[18].y) and (pinky_ext > idx_ext + 0.20) and (pinky_ext > mid_ext + 0.20)
                is_thumb_out = (thumb_idx >= 1.40)

                is_pure_i_or_j = is_pinky_extended and (not is_thumb_out)

                # Keep pinky tip trajectory whenever hand is in I/J pose OR a J stroke is already in progress
                if is_pure_i_or_j or len(pinky_history) > 0:
                    pinky_history.append((now, landmarks[20].x, landmarks[20].y))
                    if detect_silent_j_hook(pinky_history):
                        j_hold_until = now + 1.5
                        j_cooldown_until = j_hold_until + 0.7
                        pinky_history.clear()
                        prediction_buffer.clear()
                        prediction_buffer.append("J")
                else:
                    pinky_history.clear()

                # Dynamic 'Z' stroke tracking
                is_pointing_or_drawing = (idx_ext >= 1.20) and not is_pure_i_or_j
                if is_pointing_or_drawing or len(index_history) > 0:
                    index_history.append((now, landmarks[8].x, landmarks[8].y))
                    if detect_silent_z_stroke(index_history):
                        z_hold_until = now + 1.5
                        z_cooldown_until = z_hold_until + 0.6
                        index_history.clear()
                        prediction_buffer.clear()
                        prediction_buffer.append("Z")
                else:
                    index_history.clear()

                if now < z_hold_until:
                    predicted_label = "Z"
                    z_prob = (probs[z_idx] + probs[x_idx]) * 100.0 if (z_idx is not None and x_idx is not None) else conf
                    active_conf = min(98.2, max(85.0, 84.0 + 12.0 * min(idx_ext / 2.0, 1.0) + (z_prob * 0.04)))
                elif now < z_cooldown_until and (predicted_label == "Z" or raw_pred in ["X", "Uncertain"]):
                    predicted_label = "Z"
                    active_conf = 88.0 + 5.0 * min(idx_ext / 2.0, 1.0)
                elif now < j_hold_until:
                    predicted_label = "J"
                    j_conf = (probs[j_idx] + probs[i_idx]) * 100.0 if (j_idx is not None and i_idx is not None) else conf
                    active_conf = min(98.2, max(85.0, 84.0 + 12.0 * min(pinky_ext / 2.0, 1.0) + (j_conf * 0.04)))
                elif now < j_cooldown_until and (predicted_label == "J" or raw_pred in ["G", "H", "Uncertain"] or idx_ext < 1.55):
                    # Post-J settling period: suppress false G/H flickers while hand returns to rest
                    predicted_label = "J"
                    active_conf = 88.0 + 4.0 * min(pinky_ext / 2.0, 1.0)
                elif is_pure_i_or_j:
                    # HELD PINKY IS ALWAYS STRICTLY 'I' WITH AGGREGATED CONFIDENCE!
                    predicted_label = "I"
                    i_conf = (probs[i_idx] + probs[j_idx]) * 100.0 if (i_idx is not None and j_idx is not None) else conf
                    active_conf = min(98.5, max(82.0, i_conf * 0.85 + 14.0 * min(pinky_ext / 2.0, 1.0)))
                else:
                    # All other letters flow freely with true confidence
                    predicted_label = raw_pred
                    active_conf = conf

                # Temporal exponential smoothing for confidence
                smooth_conf = smooth_conf * 0.75 + active_conf * 0.25

                # Hysteresis smoothing: only append valid classifications
                if predicted_label not in ["Uncertain", "No Hand"]:
                    prediction_buffer.append(predicted_label)
                    stable_prediction = max(set(prediction_buffer), key=prediction_buffer.count)

            else:
                no_hand_count += 1
                pinky_history.clear()
                index_history.clear()
                if no_hand_count >= 5:
                    prediction_buffer.clear()
                    stable_prediction = "No Hand"
                    smooth_conf = 0.0

            pred_display = stable_prediction

            # --- BOTTOM HUD (GLASSMORPHIC OVERLAY) ---
            margin_x = 20
            hud_h = 70
            hud_y2 = h - 15
            hud_y1 = hud_y2 - hud_h
            sent_h = 42
            sent_y2 = hud_y1 - 10
            sent_y1 = sent_y2 - sent_h

            roi = frame[sent_y1:hud_y2, margin_x:w - margin_x]
            roi_overlay = roi.copy()
            cv2.rectangle(roi_overlay, (0, 0), (w - 2 * margin_x, sent_h), (18, 20, 26), -1)
            cv2.rectangle(roi_overlay, (0, hud_y1 - sent_y1), (w - 2 * margin_x, hud_y2 - sent_y1), (18, 20, 26), -1)
            cv2.addWeighted(roi_overlay, 0.78, roi, 0.22, 0, roi)
            frame[sent_y1:hud_y2, margin_x:w - margin_x] = roi

            cv2.rectangle(frame, (margin_x, sent_y1), (w - margin_x, sent_y2), (55, 62, 78), 1)
            cv2.rectangle(frame, (margin_x, hud_y1), (w - margin_x, hud_y2), (55, 62, 78), 1)

            tag_scale = 0.44 if w >= 1600 else 0.38
            val_scale = 0.78 if w >= 1600 else 0.65
            sent_scale = 0.65 if w >= 1600 else 0.52

            # 1. Sentence Bar
            sent_text_y = sent_y1 + int(sent_h * 0.68)
            if current_word:
                cv2.putText(frame, "SENTENCE:", (margin_x + 18, sent_text_y), cv2.FONT_HERSHEY_SIMPLEX, sent_scale, (140, 150, 165), 1)
                (tw, _), _ = cv2.getTextSize("SENTENCE: ", cv2.FONT_HERSHEY_SIMPLEX, sent_scale, 1)
                cv2.putText(frame, current_word, (margin_x + 18 + tw, sent_text_y), cv2.FONT_HERSHEY_SIMPLEX, sent_scale, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "SENTENCE: [Press SPACE to type letters, BACKSPACE to delete, C to clear]", 
                            (margin_x + 18, sent_text_y), cv2.FONT_HERSHEY_SIMPLEX, sent_scale, (110, 115, 130), 1)

            # 2. Four Columns: [PREDICTED LETTER, CONFIDENCE, RESOLUTION, FPS]
            card_w = (w - 2 * margin_x)
            c1_x = margin_x + int(card_w * 0.03)
            c2_x = margin_x + int(card_w * 0.32)
            c3_x = margin_x + int(card_w * 0.60)
            c4_x = margin_x + int(card_w * 0.83)

            tag_y = hud_y1 + 24
            val_y = hud_y1 + 54

            # Column 1: Guess / Letter
            cv2.putText(frame, "PREDICTED LETTER", (c1_x, tag_y), cv2.FONT_HERSHEY_SIMPLEX, tag_scale, (140, 150, 165), 1)
            letter_color = (0, 245, 255) if stable_prediction not in ["No Hand", "Uncertain"] else (130, 135, 145)
            cv2.putText(frame, str(pred_display), (c1_x, val_y), cv2.FONT_HERSHEY_SIMPLEX, val_scale, letter_color, 2)

            # Column 2: Confidence %
            cv2.putText(frame, "CONFIDENCE", (c2_x, tag_y), cv2.FONT_HERSHEY_SIMPLEX, tag_scale, (140, 150, 165), 1)
            if stable_prediction in ["No Hand", "Uncertain"]:
                conf_str = "---"
                conf_color = (130, 135, 145)
            else:
                conf_str = f"{smooth_conf:.1f}%"
                conf_color = (100, 255, 140) if smooth_conf >= 70.0 else (80, 210, 255) if smooth_conf >= 40.0 else (130, 135, 145)
            cv2.putText(frame, conf_str, (c2_x, val_y), cv2.FONT_HERSHEY_SIMPLEX, val_scale, conf_color, 2)

            # Column 3: Resolution
            cv2.putText(frame, "RESOLUTION", (c3_x, tag_y), cv2.FONT_HERSHEY_SIMPLEX, tag_scale, (140, 150, 165), 1)
            cv2.putText(frame, f"{w}x{h}", (c3_x, val_y), cv2.FONT_HERSHEY_SIMPLEX, val_scale, (200, 210, 225), 2)

            # Column 4: FPS
            t_frame = time.time() - t_start
            current_fps = 1.0 / t_frame if t_frame > 0 else 0
            fps_buffer.append(current_fps)
            avg_fps = sum(fps_buffer) / len(fps_buffer)

            cv2.putText(frame, "CAMERA FPS", (c4_x, tag_y), cv2.FONT_HERSHEY_SIMPLEX, tag_scale, (140, 150, 165), 1)
            cv2.putText(frame, f"{avg_fps:.1f}", (c4_x, val_y), cv2.FONT_HERSHEY_SIMPLEX, val_scale, (200, 210, 225), 2)

            cv2.imshow(WINDOW_NAME, frame)

            # Keybindings
            key = cv2.waitKey(1) & 0xFF
            now = time.time()
            if key in [27, ord('q'), ord('Q')]:
                break
            elif key == ord(' '):
                if now - last_key_press_time >= key_debounce:
                    if stable_prediction not in ["No Hand", "Uncertain"]:
                        current_word += stable_prediction
                        last_key_press_time = now
            elif key in [8, 127]:
                if now - last_key_press_time >= key_debounce:
                    current_word = current_word[:-1]
                    last_key_press_time = now
            elif key in [ord('c'), ord('C')]:
                current_word = ""
            elif key in [ord('r'), ord('R')]:
                cap.toggle_resolution()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
