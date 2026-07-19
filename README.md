# ✍️ Smart Pen — Handwriting Quality Analyzer

> **Real-time handwriting quality scoring powered by ESP32, MPU6050 IMU sensor, and XGBoost Machine Learning.**

[![Arduino](https://img.shields.io/badge/Arduino-ESP32-00979D?logo=arduino&logoColor=white)](https://www.arduino.cc/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-ML-EB5A46?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A complete end-to-end **IoT + ML** system that transforms an ordinary pen into a **smart handwriting coach**. By capturing 6-axis motion data from an MPU6050 accelerometer/gyroscope mounted on a custom-designed pen, the system extracts biomechanical features and uses an optimized **XGBoost classifier** to score handwriting quality in real time via a Streamlit dashboard.

Built from scratch in **4 weeks** as a university capstone project — hardware, firmware, dataset collection, feature engineering, model training, and UI.

---

## 🎯 What It Does

| Feature | Description |
|---------|-------------|
| 🖊️ **Motion Capture** | Collects 6-axis IMU data (accelerometer + gyroscope) from ESP32 via serial communication |
| 📊 **Feature Engineering** | Extracts **14 raw + derived features** from gyroscope magnitude signals using sliding windows |
| 🤖 **ML Prediction** | XGBoost classifier with **91% accuracy** predicting 3 quality classes: **Perfect**, **Medium**, **Bad** |
| 🌐 **Live Dashboard** | Streamlit web interface for real-time recording, feature visualization, and instant quality feedback |
| ⚡ **Real-Time** | End-to-end pipeline: sensor → feature extraction → prediction in under 2 seconds |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              HARDWARE LAYER                                  │
│  ┌─────────────┐     I2C      ┌─────────────┐     Serial/USB     ┌───────┐ │
│  │   MPU6050   │◄────────────►│    ESP32    │◄──────────────────►│   PC  │ │
│  │  (Accel+Gyro)│              │  (Arduino)  │   Raw IMU Data     │       │ │
│  └─────────────┘              └─────────────┘                    └───────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            SOFTWARE LAYER                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     FEATURE EXTRACTION PIPELINE                      │   │
│  │  Gyro Magnitude → Sliding Windows → 14 Features (speed, vibration,   │   │
│  │  pause_ratio, smoothness, gyro_std, gyro_max, gyro_range, jerk,     │   │
│  │  energy, gyro_energy_ratio, jerk_speed_ratio, stability, peak)      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                      │
│                                      ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    XGBOOST CLASSIFIER (91% Acc)                      │   │
│  │  • 3-class: Bad (0) | Medium (1) | Perfect (2)                      │   │
│  │  • RandomizedSearchCV hyperparameter tuning                          │   │
│  │  • Top-7 feature selection via importance ranking                    │   │
│  │  • 5-Fold Stratified Cross-Validation                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                      │
│                                      ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    STREAMLIT LIVE DASHBOARD                          │   │
│  │  • Start Recording  • Real-time Analysis  • Feature Breakdown       │   │
│  │  • Quality Prediction (Perfect ✨ / Medium / Bad)                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| **Hardware** | ESP32 DevKit v1, MPU6050 (6-DOF IMU), Custom 3D-printed pen housing |
| **Firmware** | Arduino C++ (I2C communication, data streaming over Serial) |
| **Backend / ML** | Python, XGBoost, scikit-learn, NumPy, pandas, joblib |
| **Frontend** | Streamlit (Python-based web UI) |
| **Data Collection** | Custom Python scripts for automated session recording & CSV export |

---

## 📸 Screenshots

### 🎛️ Live Analysis Dashboard
The Streamlit interface enables real-time handwriting session recording, instant feature extraction, and ML-powered quality scoring.

![Smart Pen Live Analysis](https://raw.githubusercontent.com/serinenadabenmissi-design/smart-pen-/refs/heads/main/screenshots/smart%20pen%20analyze.png)

> **Dashboard highlights:**
> - ▶️ **Start Recording** — Capture live IMU data stream from ESP32 via serial port
> - 🔍 **Analyze** — Trigger feature extraction pipeline and XGBoost inference
> - 📊 **Extracted Features** — Visual breakdown of 7 selected biomechanical metrics
> - 🧠 **Prediction** — Instant quality classification with confidence indicator

---

## 🚀 Getting Started

### Prerequisites
- ESP32 DevKit v1 development board
- MPU6050 sensor module (GY-521 breakout)
- USB cable for ESP32
- Python 3.10+
- Arduino IDE 2.x

### 1. Hardware Setup

Connect the MPU6050 to the ESP32 via I2C:

```
MPU6050 VCC → ESP32 3.3V
MPU6050 GND → ESP32 GND
MPU6050 SCL → ESP32 GPIO22 (D22)
MPU6050 SDA → ESP32 GPIO21 (D21)
```

> **💡 Design Note:** The MPU6050 and ESP32 are housed inside a custom 3D-printed oversized pen casing. The real pen is inserted into the casing so the user writes naturally while the sensor captures motion data. Ergonomics were critical — the pen had to feel comfortable during extended writing sessions to capture authentic handwriting patterns.

### 2. Flash the Firmware

```bash
# 1. Open Arduino IDE
# 2. Install ESP32 board support (Tools → Board → Boards Manager)
# 3. Install MPU6050 library (Sketch → Include Library → Manage Libraries)
# 4. Select board: "ESP32 Dev Module"
# 5. Upload firmware/smart_pen.ino to your ESP32
```

The firmware streams raw 6-axis IMU data (`ax, ay, az, gx, gy, gz`) over Serial at ~100 Hz.

### 3. Install Python Dependencies

```bash
pip install numpy pandas scikit-learn xgboost streamlit joblib
```

### 4. Run the Streamlit Dashboard

```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`.

---

## 🔬 Feature Extraction Pipeline

Raw gyroscope magnitude is computed from the 3-axis gyro data, then processed through **sliding windows** (default 50 samples) to extract robust temporal features:

### Raw Features (per window)

| Feature | Formula / Description |
|---------|----------------------|
| `speed` | Mean gyro magnitude — overall writing velocity |
| `vibration` | `0.7 × mean(|Δgyro|) + 0.3 × std(gyro)` — shakiness metric |
| `pause_ratio` | Fraction of samples below `0.3 × speed` threshold — detects micro-pauses |
| `smoothness` | `speed / (vibration + ε)` — fluidity of strokes |

### Global Features (full signal)

| Feature | Description |
|---------|-------------|
| `gyro_std` | Standard deviation of gyro magnitude — motion variability |
| `gyro_max` | Peak gyro magnitude — maximum intensity |
| `gyro_range` | `max - min` of gyro magnitude — dynamic range |
| `jerk_mean` | Mean absolute derivative of gyro magnitude — acceleration changes |
| `jerk_max` | Peak absolute jerk — sudden motion spikes |
| `energy` | Mean squared gyro magnitude — total signal energy |

### Derived Features

| Feature | Formula | Intuition |
|---------|---------|-----------|
| `gyro_energy_ratio` | `energy / (speed + 1)` | Efficiency of energy vs. speed |
| `jerk_speed_ratio` | `jerk_mean / (speed + 1)` | Jerk normalized by velocity |
| `stability_score` | `1 / (vibration + gyro_std + 1)` | Inverse of total instability |
| `peak_ratio` | `gyro_max / (gyro_std + 1)` | Peak intensity vs. variability |
| `control_score` | `smoothness / (pause_ratio + 0.01)` | Fluidity penalized by pauses |

### Selected Features (Top-7 via XGBoost Importance)

The final model uses only the **7 most informative features** for optimal performance:

```
1. energy
2. control_score
3. gyro_energy_ratio
4. smoothness
5. pause_ratio
6. jerk_mean
7. speed
```

---

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| **Algorithm** | XGBoost Classifier (`multi:softprob`) |
| **Accuracy** | **91%** (5-Fold Stratified CV) |
| **Classes** | 3 — Bad (0) / Medium (1) / Perfect (2) |
| **Dataset Size** | 127 handwriting samples |
| **Features** | 7 (selected from 14 engineered features) |
| **Hyperparameter Tuning** | RandomizedSearchCV (40 iterations, 3-fold inner CV) |
| **Scoring Metric** | `f1_macro` (balanced across classes) |
| **Preprocessing** | Log-transform for skewed features + StandardScaler |

### Training Pipeline

```
Raw CSV (127 samples)
    │
    ├──→ Label encoding: bad→0, medium→1, perfect→2
    ├──→ Feature engineering: 14 features from gyro magnitude
    ├──→ Log transform for skewed features (skew > 1.0)
    ├──→ RandomizedSearchCV hyperparameter optimization
    ├──→ Top-7 feature selection via XGBoost importance
    ├──→ StandardScaler normalization
    └──→ Final XGBoost training + 5-Fold CV evaluation
```

### Saved Artifacts

| File | Description |
|------|-------------|
| `xgb_best.joblib` | Trained XGBoost model |
| `scaler_selected.joblib` | StandardScaler fitted on selected features |
| `selected_features.txt` | List of 7 selected feature names |

---

## 📁 Project Structure

```
smart-pen/
│
├── firmware/
│   └── smart_pen.ino              # Arduino C++: ESP32 + MPU6050 I2C data streaming
│
├── data_collection/
│   ├── collect_bad.py             # Script to record "bad" handwriting sessions
│   ├── collect_medium.py          # Script to record "medium" handwriting sessions
│   ├── collect_perfect.py         # Script to record "perfect" handwriting sessions
│   └── sessions_summary.csv       # Final labeled dataset (127 samples)
│
├── ml_pipeline/
│   ├── train_model.py             # Full training pipeline (feature eng → XGBoost → CV)
│   ├── extract_features.py        # Feature extraction from raw gyro data
│   ├── xgb_best.joblib            # Saved trained model
│   ├── scaler_selected.joblib     # Saved StandardScaler
│   └── selected_features.txt      # Top-7 selected feature names
│
├── app.py                         # Streamlit live dashboard
│
├── screenshots/
│   ├── smart pen analyze.png      # Live analysis dashboard
│   ├── IMG_20260617_163659.jpg    # Hardware: pen + ESP32 + wiring
│   ├── IMG_20260617_163741.jpg    # Hardware: close-up of components
│   └── IMG_20260718_233857.jpg    # Hardware: pen connected to PC
│
└── README.md
```

---

## 💡 Key Challenges & Solutions

### 1. 🎚️ Sensor Noise & Data Quality
**Challenge:** MPU6050 raw data is extremely noisy. Poor data quality leads to unreliable model predictions.

**Solution:**
- Re-collected the entire dataset **3+ times** over 3 weeks until signal quality was consistent
- Implemented sliding window aggregation (50-sample windows) to smooth out high-frequency noise
- Used derived ratios (`gyro_energy_ratio`, `jerk_speed_ratio`) to normalize against speed variations

### 2. 🏗️ Hardware Ergonomics
**Challenge:** The sensor board and ESP32 are bulky. Mounting them on a pen without affecting natural handwriting was critical — the user's true skill level must be captured, not their discomfort.

**Solution:**
- Designed and **3D-printed a custom oversized pen casing** that houses all electronics
- The real pen sits inside the casing, so the user grips and writes normally
- Iterated on weight distribution and grip comfort across multiple prototypes

### 3. ⚖️ Class Imbalance & Labeling
**Challenge:** Manual labeling of 127 samples into 3 classes is subjective. Uneven class distribution could bias the model.

**Solution:**
- Collected **~42 samples per class** (127 ÷ 3) for balanced representation
- Used `StratifiedKFold` to maintain class proportions during cross-validation
- Optimized for `f1_macro` to ensure fair performance across all quality levels

### 4. ⏱️ Time Constraint
**Challenge:** Only **4 weeks** to build the entire system from scratch — hardware, firmware, data collection, ML pipeline, and UI.

**Solution:**
- Parallelized hardware prototyping with data collection script development
- Used `RandomizedSearchCV` instead of GridSearch for efficient hyperparameter tuning
- Leveraged Streamlit for rapid UI prototyping without frontend framework overhead

---

## 🎓 Key Learnings

- **Sensor Fusion in Practice:** Combining accelerometer and gyroscope data, then deriving ratio-based features, creates robust motion signatures invariant to writing speed
- **Domain-Driven Feature Engineering:** Designing features with handwriting biomechanics in mind (jerk, smoothness, pause ratio) outperformed generic statistical features
- **Iterative Data Collection:** In IoT/ML projects, data quality is more important than quantity — 127 clean samples beat 1000 noisy ones
- **End-to-End System Thinking:** Every design choice (pen ergonomics → sensor placement → sampling rate → feature window size) cascades into model performance

---

## 🙋‍♀️ Author

**Serine Nada Ben Missi**

Embedded Systems & Machine Learning Enthusiast

> *University capstone project — built in 4 weeks from zero to working prototype.*

---

## 📄 License

This project is licensed under the MIT License.

---

> *"Turning motion into insight — one stroke at a time."* ✍️✨
