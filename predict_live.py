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

RESAMPLE_POINTS = 16
MIN_STEP_DIST = 0.006
MIN_ARC_LENGTH = 0.22  # Requires a full stroke before evaluating

STATE_WAIT_STEADY = 0
STATE_ARMED = 1
STATE_DRAWING = 2

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
    root.title("Select Camera")
    root.geometry("420x340")
    root.resizable(False, False)
    root.attributes("-topmost", True)

    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"420x340+{(sw // 2) - 210}+{(sh // 2) - 170}")

    tk.Label(root, text="Select Camera Device:", font=("Arial", 12, "bold")).pack(pady=(15, 5))
    tk.Label(root, text="Double-click a camera or click 'Use Selected Camera'", font=("Arial", 9), fg="#555").pack(pady=(0, 10))

    frame = tk.Frame(root)
    frame.pack(padx=20, pady=5, fill=tk.BOTH, expand=True)

    scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL)
    listbox = tk.Listbox(frame, font=("Arial", 11), yscrollcommand=scrollbar.set, selectmode=tk.SINGLE)
    scrollbar.config(command=listbox.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    for idx, name in enumerate(camera_names):
        listbox.insert(tk.END, f"[{idx}]  {name}")

    if 0 <= current_index < len(camera_names):
        listbox.selection_set(current_index)
        listbox.see(current_index)
    elif camera_names:
        listbox.selection_set(0)

    def confirm(event=None):
        if listbox.curselection():
            selected["index"] = int(listbox.curselection()[0])
            root.destroy()

    listbox.bind("<Double-Button-1>", confirm)
    listbox.bind("<Return>", confirm)
    tk.Button(root, text="Use Selected Camera", font=("Arial", 11, "bold"), bg="#4CAF50", fg="white", command=confirm).pack(pady=15)

    root.mainloop()
    return selected["index"]


class ThreadedCamera:
    """
    High-performance non-blocking camera reader.
    Continuously fetches frames in a dedicated thread to eliminate I/O lag and maximize FPS.
    """
    def __init__(self, index=0, width=1280, height=720, target_fps=60):
        self.index = index
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.cam = None
        self.backend_name = "Default"
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()
        self.thread = None
        self._open()

    def _open(self):
        backends = [("DirectShow", cv2.CAP_DSHOW), ("Media Foundation", cv2.CAP_MSMF), ("Default", cv2.CAP_ANY)]
        for name, backend in backends:
            cam = cv2.VideoCapture(self.index, backend)
            if cam.isOpened():
                # Force MJPG codec and explicitly request target FPS (e.g. 60 / 30)
                cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cam.set(cv2.CAP_PROP_FPS, self.target_fps)
                ret, frame = cam.read()
                if ret and frame is not None:
                    self.cam = cam
                    self.backend_name = name
                    self.frame = frame
                    self.start()
                    return
                cam.release()
        raise RuntimeError(f"Could not open camera index {self.index}.")

    def start(self):
        self.stopped = False
        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()

    def _reader_loop(self):
        while not self.stopped and self.cam is not None and self.cam.isOpened():
            ret, frame = self.cam.read()
            if ret and frame is not None:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.002)

    def read(self):
        with self.lock:
            if self.frame is not None:
                return True, self.frame
            return False, None

    def toggle_resolution(self):
        new_w, new_h = (1920, 1080) if self.width == 1280 else (1280, 720)
        self.width = new_w
        self.height = new_h
        with self.lock:
            if self.cam is not None:
                self.cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, new_w)
                self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, new_h)
                self.cam.set(cv2.CAP_PROP_FPS, self.target_fps)
        print(f"[Camera] Toggled resolution to {new_w}x{new_h}")

    def release(self):
        self.stopped = True
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=0.3)
        if self.cam is not None:
            self.cam.release()
            self.cam = None


def open_camera(index=0):
    camera = ThreadedCamera(index=index, width=1280, height=720, target_fps=60)
    return camera, camera.backend_name


def extract_features(landmarks):
    wrist = landmarks[0]
    raw_coords = []
    for lm in landmarks:
        raw_coords.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
    scale = max(max(abs(val) for val in raw_coords), 1e-6)
    return [val / scale for val in raw_coords]


def path_arc_length(points):
    if len(points) < 2:
        return 0.0
    return sum(
        math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1])
        for i in range(1, len(points))
    )


def resample_and_normalize_path(points, n=RESAMPLE_POINTS):
    if len(points) < 2:
        return [0.0] * (n * 2)

    dists = [0.0]
    for i in range(1, len(points)):
        dists.append(dists[-1] + math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1]))

    total_len = dists[-1]
    if total_len < 1e-6:
        return [0.0] * (n * 2)

    resampled = []
    for k in range(n):
        target = (total_len * k) / (n - 1)
        idx = 0
        while idx < len(dists) - 2 and dists[idx + 1] < target:
            idx += 1
        seg_len = dists[idx + 1] - dists[idx]
        if seg_len < 1e-6:
            rx, ry = points[idx]
        else:
            ratio = (target - dists[idx]) / seg_len
            rx = points[idx][0] + ratio * (points[idx + 1][0] - points[idx][0])
            ry = points[idx][1] + ratio * (points[idx + 1][1] - points[idx][1])
        resampled.append((rx, ry))

    ox, oy = resampled[0]
    rel_coords = []
    for rx, ry in resampled:
        rel_coords.extend([rx - ox, ry - oy])

    max_abs = max(max(abs(v) for v in rel_coords), 1e-6)
    return [v / max_abs for v in rel_coords]


def build_dynamic_feature_vector(index_path, pinky_path, pose_list):
    norm_index = resample_and_normalize_path(index_path, RESAMPLE_POINTS)
    norm_pinky = resample_and_normalize_path(pinky_path, RESAMPLE_POINTS)
    mean_pose = np.mean(np.array(pose_list, dtype=np.float64), axis=0)
    max_pose = max(np.max(np.abs(mean_pose)), 1e-6)
    norm_pose = (mean_pose / max_pose).tolist()
    return norm_index + norm_pinky + norm_pose


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"MediaPipe model missing at: {MODEL_PATH}")
    if not os.path.exists(CLASSIFIER_PATH):
        raise FileNotFoundError(f"Trained classifier missing at: {CLASSIFIER_PATH}\nRun train_model.py first!")

    print("Loading unified ASL classifier...")
    loaded = joblib.load(CLASSIFIER_PATH)
    if isinstance(loaded, dict) and loaded.get("type") == "unified_asl_model":
        static_clf = loaded["static_model"]
        dynamic_clf = loaded.get("dynamic_model", None)
    else:
        static_clf = loaded
        dynamic_clf = None

    camera_names = get_camera_names()
    chosen_index = select_camera_popup(camera_names, current_index=0)
    if chosen_index is None:
        return

    current_cam_index = chosen_index
    camera, backend_name = open_camera(current_cam_index)

    detector = HandLandmarker.create_from_options(
        HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=1
        )
    )

    # Resizable window that auto-fits display without cropping edge of webcam
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    last_timestamp_ms = -1
    prev_frame_time = time.time()
    fps_smooth = 0.0
    prediction_buffer = deque(maxlen=8)
    current_word = ""

    pen_state = STATE_WAIT_STEADY
    prev_idx_pt = None
    prev_pnk_pt = None
    steady_frames = 0
    stroke_pause_frames = 0

    live_index_path = []
    live_pinky_path = []
    live_poses = []

    dynamic_hold_label = None
    dynamic_hold_conf = 0.0
    dynamic_hold_until = 0.0
    locked_trail_px = []

    def reset_motion():
        nonlocal pen_state, prev_idx_pt, prev_pnk_pt, steady_frames, stroke_pause_frames
        pen_state = STATE_WAIT_STEADY
        prev_idx_pt = None
        prev_pnk_pt = None
        steady_frames = 0
        stroke_pause_frames = 0
        live_index_path.clear()
        live_pinky_path.clear()
        live_poses.clear()

    print("\n--- Unified Live ASL Recognition (A-Z) Started ---")
    print("SPACEBAR : Add current letter to sentence")
    print("BACKSPACE: Delete last letter")
    print("C        : Clear sentence")
    print("R        : Toggle Resolution (720p / 1080p)")
    print("TAB      : Switch Camera")
    print("Q        : Quit\n")

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                continue

            cur_time = time.time()
            dt = cur_time - prev_frame_time
            prev_frame_time = cur_time
            if dt > 0:
                inst_fps = 1.0 / dt
                fps_smooth = inst_fps if fps_smooth == 0.0 else (fps_smooth * 0.9 + inst_fps * 0.1)

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # High-FPS Optimization: Subsample frame for MediaPipe processing
            # MediaPipe calculates normalized coords in [0, 1], so landmark tracking is 100% resolution-independent
            # while running 3x faster than processing full 1080p pixels on CPU.
            target_proc_w = 640
            if w > target_proc_w:
                proc_h = int(target_proc_w * (h / w))
                proc_frame = cv2.resize(frame, (target_proc_w, proc_h), interpolation=cv2.INTER_LINEAR)
                rgb_frame = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
            else:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            timestamp_ms = int(time.time() * 1000)
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms

            result = detector.detect_for_video(mp_image, timestamp_ms)

            predicted_label = "No Hand"
            confidence_pct = 0.0
            now = time.time()

            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]

                for start, end in CONNECTIONS:
                    cv2.line(frame, (int(landmarks[start].x * w), int(landmarks[start].y * h)),
                                   (int(landmarks[end].x * w), int(landmarks[end].y * h)), (0, 255, 0), 2)
                for pt in landmarks:
                    cv2.circle(frame, (int(pt.x * w), int(pt.y * h)), 4, (0, 0, 255), -1)

                features = extract_features(landmarks)
                
                # Check static prediction (40% threshold for D)
                probs = static_clf.predict_proba([features])[0]
                max_idx = probs.argmax()
                static_conf = probs[max_idx] * 100.0
                static_pred = static_clf.classes_[max_idx] if static_conf >= 40.0 else "Uncertain"

                idx_pt = (landmarks[8].x, landmarks[8].y)
                pnk_pt = (landmarks[20].x, landmarks[20].y)

                # If Z was triggered, HOLD it on screen for 2 full seconds and draw PURPLE trail!
                if now < dynamic_hold_until and dynamic_hold_label is not None:
                    predicted_label = dynamic_hold_label
                    confidence_pct = dynamic_hold_conf
                else:
                    # STRICT RULE: Only activate smart pen if holding D (for Z) or I (for J)
                    is_valid_base = (static_pred == "D" or static_pred == "I")

                    if not is_valid_base:
                        reset_motion()
                        predicted_label = static_pred
                        confidence_pct = static_conf
                    else:
                        if pen_state == STATE_WAIT_STEADY:
                            if prev_idx_pt is not None:
                                move = max(
                                    math.hypot(idx_pt[0] - prev_idx_pt[0], idx_pt[1] - prev_idx_pt[1]),
                                    math.hypot(pnk_pt[0] - prev_pnk_pt[0], pnk_pt[1] - prev_pnk_pt[1])
                                )
                                if move < MIN_STEP_DIST:
                                    steady_frames += 1
                                else:
                                    steady_frames = 0
                                if steady_frames >= 3:
                                    pen_state = STATE_ARMED
                                    live_index_path = [idx_pt]
                                    live_pinky_path = [pnk_pt]
                                    live_poses = [features]
                            prev_idx_pt = idx_pt
                            prev_pnk_pt = pnk_pt
                            predicted_label = static_pred
                            confidence_pct = static_conf

                        elif pen_state == STATE_ARMED:
                            step = max(
                                math.hypot(idx_pt[0] - live_index_path[-1][0], idx_pt[1] - live_index_path[-1][1]),
                                math.hypot(pnk_pt[0] - live_pinky_path[-1][0], pnk_pt[1] - live_pinky_path[-1][1])
                            )
                            if step >= MIN_STEP_DIST:
                                pen_state = STATE_DRAWING
                                live_index_path.append(idx_pt)
                                live_pinky_path.append(pnk_pt)
                                live_poses.append(features)
                            predicted_label = static_pred
                            confidence_pct = static_conf

                        elif pen_state == STATE_DRAWING:
                            step = max(
                                math.hypot(idx_pt[0] - live_index_path[-1][0], idx_pt[1] - live_index_path[-1][1]),
                                math.hypot(pnk_pt[0] - live_pinky_path[-1][0], pnk_pt[1] - live_pinky_path[-1][1])
                            )
                            if step >= 0.003:
                                live_index_path.append(idx_pt)
                                live_pinky_path.append(pnk_pt)
                                live_poses.append(features)
                                stroke_pause_frames = 0
                            else:
                                stroke_pause_frames += 1
                                live_index_path.append(idx_pt)
                                live_pinky_path.append(pnk_pt)
                                live_poses.append(features)

                            # EVALUATE ONCE FULL Z STROKE ARC LENGTH IS REACHED
                            arc_len = max(path_arc_length(live_index_path), path_arc_length(live_pinky_path))
                            # Wait until full stroke is drawn AND user pauses at end
                            if dynamic_clf is not None and arc_len >= 0.18 and len(live_index_path) >= 12 and stroke_pause_frames >= 4:
                                dyn_vec = build_dynamic_feature_vector(live_index_path, live_pinky_path, live_poses)
                                d_probs = dynamic_clf.predict_proba([dyn_vec])[0]
                                d_idx = d_probs.argmax()
                                d_label = dynamic_clf.classes_[d_idx]
                                d_conf = d_probs[d_idx] * 100.0

                                if d_label in ["J", "Z"] and d_conf >= 40.0:
                                    dynamic_hold_label = d_label
                                    dynamic_hold_conf = d_conf
                                    dynamic_hold_until = now + 2.0  # Lock Z on screen for 2 full seconds!
                                    active_path = live_pinky_path if d_label == "J" else live_index_path
                                    locked_trail_px = [(int(px * w), int(py * h)) for px, py in active_path]
                                    prediction_buffer.clear()
                                    reset_motion()
                                else:
                                    reset_motion()

                            predicted_label = static_pred
                            confidence_pct = static_conf

                # Draw yellow active trail or purple locked trail
                if now < dynamic_hold_until and locked_trail_px:
                    for i in range(1, len(locked_trail_px)):
                        cv2.line(frame, locked_trail_px[i - 1], locked_trail_px[i], (255, 0, 255), 4)
                elif pen_state == STATE_DRAWING:
                    trail_pts = [(int(px * w), int(py * h)) for px, py in live_index_path]
                    for i in range(1, len(trail_pts)):
                        cv2.line(frame, trail_pts[i - 1], trail_pts[i], (0, 255, 255), 3)
            else:
                reset_motion()

            if now < dynamic_hold_until and dynamic_hold_label is not None:
                stable_prediction = dynamic_hold_label
                rem_sec = dynamic_hold_until - now
                pred_display = f"{stable_prediction} (Locked {rem_sec:.1f}s)"
            else:
                prediction_buffer.append(predicted_label)
                stable_prediction = max(set(prediction_buffer), key=prediction_buffer.count) if prediction_buffer else "No Hand"
                pred_display = stable_prediction

            # --- SLEEK BOTTOM HUD (AESTHETIC GLASSMORPHIC OVERLAY) ---
            margin_x = 20
            hud_h = 70
            hud_y2 = h - 15
            hud_y1 = hud_y2 - hud_h
            sent_h = 42
            sent_y2 = hud_y1 - 10
            sent_y1 = sent_y2 - sent_h

            # Fast ROI-only alpha blend (avoids full-frame copy/blend of millions of pixels)
            roi = frame[sent_y1:hud_y2, margin_x:w - margin_x]
            roi_overlay = roi.copy()
            cv2.rectangle(roi_overlay, (0, 0), (w - 2 * margin_x, sent_h), (18, 20, 26), -1)
            cv2.rectangle(roi_overlay, (0, hud_y1 - sent_y1), (w - 2 * margin_x, hud_y2 - sent_y1), (18, 20, 26), -1)
            cv2.addWeighted(roi_overlay, 0.78, roi, 0.22, 0, roi)
            frame[sent_y1:hud_y2, margin_x:w - margin_x] = roi

            # High-tech subtle borders
            cv2.rectangle(frame, (margin_x, sent_y1), (w - margin_x, sent_y2), (55, 62, 78), 1)
            cv2.rectangle(frame, (margin_x, hud_y1), (w - margin_x, hud_y2), (55, 62, 78), 1)

            # Responsive font scaling based on camera width
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

            # 2. Four Columns in Bottom HUD: [LETTER, CONFIDENCE, RESOLUTION, FPS]
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
                conf_str = f"{confidence_pct:.1f}%"
                conf_color = (100, 255, 140) if confidence_pct >= 70.0 else (80, 210, 255) if confidence_pct >= 40.0 else (130, 135, 145)
            cv2.putText(frame, conf_str, (c2_x, val_y), cv2.FONT_HERSHEY_SIMPLEX, val_scale, conf_color, 2)

            # Column 3: Resolution
            cv2.putText(frame, "RESOLUTION", (c3_x, tag_y), cv2.FONT_HERSHEY_SIMPLEX, tag_scale, (140, 150, 165), 1)
            cv2.putText(frame, f"{w} x {h}", (c3_x, val_y), cv2.FONT_HERSHEY_SIMPLEX, val_scale, (225, 230, 240), 2)

            # Column 4: FPS
            cv2.putText(frame, "FRAME RATE", (c4_x, tag_y), cv2.FONT_HERSHEY_SIMPLEX, tag_scale, (140, 150, 165), 1)
            cv2.putText(frame, f"{fps_smooth:.1f} FPS", (c4_x, val_y), cv2.FONT_HERSHEY_SIMPLEX, val_scale, (120, 255, 180), 2)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in [ord('q'), ord('Q')]:
                break
            elif key == 32:
                if stable_prediction not in ["No Hand", "Uncertain", "NONE"]:
                    current_word += stable_prediction
                    dynamic_hold_until = 0.0
            elif key in [8, 127]:
                current_word = current_word[:-1]
            elif key in [ord('c'), ord('C')]:
                current_word = ""
            elif key in [ord('r'), ord('R')]:
                camera.toggle_resolution()
            elif key == 9:
                new_idx = select_camera_popup(camera_names, current_index=current_cam_index)
                if new_idx is not None and new_idx != current_cam_index:
                    camera.release()
                    try:
                        camera, backend_name = open_camera(new_idx)
                        current_cam_index = new_idx
                    except Exception:
                        camera, backend_name = open_camera(current_cam_index)
                    reset_motion()

    finally:
        camera.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()