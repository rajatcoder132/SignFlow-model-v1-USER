# SignFlow Model v1 (User Model)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.14-orange.svg)](https://developers.google.com/mediapipe)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-RandomForest-green.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/status-active-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Official User Release for SignFlow**: A lightweight, production-ready American Sign Language (ASL) recognition model package optimized for real-time live webcam translation. Contains only the required inference models and prediction runtime for seamless end-user performance.

---

## 🌟 Overview

* **Real-Time Live Translation**: Instant recognition of static alphabet signs and dynamic gestures (Z motion).
* **High-Performance (30–60 FPS)**: Powered by a non-blocking threaded camera reader and subsampled MediaPipe inference.
* **Modern Glassmorphic HUD**: Displays live predicted letters, confidence scores, capture resolution, real-time FPS, and a sentence builder.
* **Zero Edge Cropping**: Automatically scales to fit any screen resolution without clipping your hand at the borders.
* **Plug-and-Play**: Ready to run standalone or within the **SignFlow Studio** desktop harness.

---

## 📁 Repository Structure

```
SignFlow-model-v1-USER/
├── models/
│   ├── asl_classifier.joblib      # Pre-trained unified Random Forest model
│   └── hand_landmarker.task       # MediaPipe Vision Task landmarker bundle
├── predict_live.py                # Standalone live inference HUD with 30-60 FPS
├── requirements.txt               # Minimal runtime dependencies
└── README.md                      # Documentation
```

---

## 🚀 Quickstart

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/rajatcoder132/SignFlow-model-v1-USER.git
cd SignFlow-model-v1-USER

# Install runtime dependencies
pip install -r requirements.txt
```

### 2. Launch Translation

```bash
python predict_live.py
```

---

## 🎮 Keyboard Controls

| Key | Action | Description |
| :---: | :--- | :--- |
| <kbd>Spacebar</kbd> | **Add Letter** | Appends the currently recognized letter to your active sentence |
| <kbd>Backspace</kbd> | **Delete** | Removes the last character from your sentence |
| <kbd>C</kbd> | **Clear** | Clears the entire current sentence |
| <kbd>R</kbd> | **Toggle Resolution** | Switches webcam capture between **720p** (fastest) and **1080p** |
| <kbd>Tab</kbd> | **Switch Camera** | Opens a device popup to select another connected webcam |
| <kbd>Q</kbd> | **Quit** | Safely exits the live recognition stream |

---

## 🖥️ SignFlow Studio Harness Integration

To use this model inside **SignFlow Studio** (`signlanguageapp`):
1. Open SignFlow Studio.
2. Click **+ Add Model** on the Predict page.
3. Select **SignFlow-model-v1 (USER)** from the Cloud Models tab to automatically download and register it, or click **Browse Folder** and point to this directory.

---

## 📄 License

This project is licensed under the MIT License.
