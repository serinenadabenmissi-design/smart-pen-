# ✍️ Smart Pen — Handwriting Quality Analyzer

> **Real-time handwriting quality scoring powered by ESP32, MPU6050 IMU sensor, and XGBoost Machine Learning.**

[![Arduino](https://img.shields.io/badge/Arduino-ESP32-00979D?logo=arduino&logoColor=white)](https://www.arduino.cc/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-ML-EB5A46?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A complete end-to-end **IoT + ML** system that turns an ordinary pen into a **smart handwriting coach**. A custom-built pen captures 6-axis motion data (MPU6050 accelerometer/gyroscope) while writing; the signal is turned into biomechanical features and scored in real time by an XGBoost classifier through a live Streamlit dashboard.

Built solo in **4 weeks** as a university capstone — hardware, firmware, dataset collection, feature engineering, model training, and UI.

<!-- 🎥 Add a short demo GIF or video link here — this is the single best thing you can add to this README. -->

---

## 📋 Table of Contents
- [What It Does](#-what-it-does)
- [Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Screenshots](#-screenshots)
- [Getting Started](#-getting-started)
- [Feature Engineering](#-feature-extraction-pipeline)
- [Model Performance](#-model-performance)
- [Project Structure](#-project-structure)
- [Challenges & Solutions](#-key-challenges--solutions)
- [Key Learnings](#-key-learnings)
- [Roadmap](#-roadmap)
- [Author](#-author)

---

## 🎯 What It Does

| Feature | Description |
|---------|-------------|
| 🖊️ **Motion Capture** | Streams 6-axis IMU data (accel + gyro) from ESP32 over serial |
| 📊 **Feature Engineering** | 14 raw + derived features extracted from gyro-magnitude signals via sliding windows |
| 🤖 **ML Prediction** | XGBoost classifier, **91% accuracy**, 3 classes: **Perfect / Medium / Bad** |
| 🌐 **Live Dashboard** | Streamlit UI for recording, feature visualization, and instant feedback |
| ⚡ **Real-Time** | Sensor → features → prediction in under 2 seconds |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                            HARDWARE LAYER                            │
│  MPU6050 ──I2C──► ESP32 ──Serial/USB──► PC                          │
│ (Accel+Gyro)      (Arduino)          Raw IMU Data                   │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                            SOFTWARE LAYER                             │
│  Gyro Magnitude → Sliding Windows → 14 Features                      │
│         │                                                             │
│         ▼                                                             │
│  XGBoost Classifier (91% acc) — Bad(0) / Medium(1) / Perfect(2)      │
│         │                                                             │
│         ▼                                                             │
│  Streamlit Live Dashboard — record, analyze, predict                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| **Hardware** | ESP32 DevKit v1, MPU6050 (6-DOF IMU), custom 3D-printed pen housing |
| **Firmware** | Arduino C++ (I2C communication, serial data streaming) |
| **Backend / ML** | Python, XGBoost, scikit-learn, NumPy, pandas, joblib |
| **Frontend** | Streamlit |
| **Data Collection** | Custom Python scripts for session recording & CSV export |

---

## 📸 Screenshots ### 🎛️ Live Analysis DashboardThe Streamlit interface enables real-time handwriting session recording, instant feature extraction, and ML-powered quality scoring. ![Smart Pen Live Analysis](https://raw.githubusercontent.com/serinenadabenmissi-design/smart-pen-/refs/heads/main/screenshots/smart%20pen%20analyze.png) 
> ▶️ Record a live session → 🔍 extract features → 🧠 get an instant quality prediction with a feature breakdown.

<!--
### Hardware
Add your hardware photos here (pen + ESP32 + wiring) — they're already in /screenshots,
just embed 2-3 of the clearest ones so reviewers don't have to dig for them:
![Hardware setup](screenshots/hardware-1.jpg)
-->

---

## 🚀 Getting Started

### Prerequisites
- ESP32 DevKit v1 + MPU6050 (GY-521 breakout)
- USB cable, Arduino IDE 2.x, Python 3.10+

### 1. Hardware Setup
```
MPU6050 VCC → ESP32 3.3V
MPU6050 GND → ESP32 GND
MPU6050 SCL → ESP32 GPIO22 (D22)
MPU6050 SDA → ESP32 GPIO21 (D21)
```
> **Design note:** the MPU6050 and ESP32 are housed in a custom 3D-printed oversized pen casing, with a real pen inserted inside — so the user writes naturally while the sensor captures authentic motion.

### 2. Flash the Firmware
1. Install ESP32 board support (Tools → Board → Boards Manager)
2. Install the MPU6050 library (Sketch → Include Library → Manage Libraries)
3. Select board **"ESP32 Dev Module"**
4. Upload `firmware/smart_pen.ino`

The firmware streams `ax, ay, az, gx, gy, gz` over Serial at ~100 Hz.

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Dashboard
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`.

---

## 🔬 Feature Extraction Pipeline

Gyro magnitude is computed from the 3-axis gyro signal, then processed in **50-sample sliding windows**.

| Category | Features |
|---|---|
| **Per-window** | `speed`, `vibration`, `pause_ratio`, `smoothness` |
| **Global** | `gyro_std`, `gyro_max`, `gyro_range`, `jerk_mean`, `jerk_max`, `energy` |
| **Derived** | `gyro_energy_ratio`, `jerk_speed_ratio`, `stability_score`, `peak_ratio`, `control_score` |

**Top-7 selected features** (by XGBoost importance): `energy`, `control_score`, `gyro_energy_ratio`, `smoothness`, `pause_ratio`, `jerk_mean`, `speed`

---

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| Algorithm | XGBoost (`multi:softprob`) |
| Accuracy | **91%** (5-fold stratified CV) |
| Classes | Bad(0) / Medium(1) / Perfect(2) |
| Dataset | 127 handwriting samples (~42 per class) |
| Features | 7 of 14 engineered features |
| Tuning | RandomizedSearchCV, 40 iterations, 3-fold inner CV |
| Scoring | `f1_macro` |
| Preprocessing | Log-transform on skewed features + StandardScaler |

```
Raw CSV (127 samples) → label encoding → feature engineering (14 features)
→ log transform (skew > 1.0) → RandomizedSearchCV → top-7 feature selection
→ StandardScaler → final XGBoost + 5-fold CV
```

**Saved artifacts:** `xgb_best.joblib`, `scaler_selected.joblib`, `selected_features.txt`

---

## 📁 Project Structure

```
smart-pen/
├── firmware/
│   └── smart_pen.ino
├── data_collection/
│   ├── collect_bad.py
│   ├── collect_medium.py
│   ├── collect_perfect.py
│   └── sessions_summary.csv
├── ml_pipeline/
│   ├── train_model.py
│   ├── extract_features.py
│   ├── xgb_best.joblib
│   ├── scaler_selected.joblib
│   └── selected_features.txt
├── app.py
├── requirements.txt
├── screenshots/
└── README.md
```

---

## 💡 Key Challenges & Solutions

**1. Sensor noise & data quality**
MPU6050 raw data is very noisy. Re-collected the dataset 3+ times over 3 weeks, applied 50-sample sliding-window aggregation, and used normalized ratio features (`gyro_energy_ratio`, `jerk_speed_ratio`) to reduce speed-dependent bias.

**2. Hardware ergonomics**
Bulky electronics risked distorting natural handwriting. Designed and 3D-printed a custom oversized pen casing so the real pen sits inside and the user writes normally — iterated on weight and grip across multiple prototypes.

**3. Class imbalance & labeling**
Manual 3-class labeling is subjective. Balanced collection to ~42 samples/class, used `StratifiedKFold`, and optimized for `f1_macro` for fair cross-class performance.

**4. Time constraint (4 weeks)**
Parallelized hardware prototyping with data-collection tooling, used `RandomizedSearchCV` over grid search, and used Streamlit to skip frontend framework overhead.

---

## 🎓 Key Learnings

- Combining accel + gyro into ratio-based features creates motion signatures that are robust to writing-speed variation.
- Domain-driven features (jerk, smoothness, pause ratio) outperformed generic statistical ones.
- In IoT/ML projects, data quality beats data quantity — 127 clean samples outperformed larger noisy sets.
- Every design choice — pen ergonomics, sensor placement, sampling rate, window size — cascades directly into model performance.

---

## 🗺️ Roadmap

- [ ] Expand the dataset beyond 127 samples for better generalization
- [ ] On-device inference (run the model directly on ESP32 or a companion mobile app)
- [ ] Support left-handed grip calibration
- [ ] Add a short demo video/GIF to this README

---

## 🙋‍♀️ Author

**Serine Nada Ben Missi**
Embedded Systems & Machine Learning Enthusiast

<!-- Add your LinkedIn / portfolio / email here so reviewers can reach you directly -->

> University capstone project — built in 4 weeks, from zero to working prototype.

---

## 📄 License

MIT License — see [LICENSE](LICENSE).
