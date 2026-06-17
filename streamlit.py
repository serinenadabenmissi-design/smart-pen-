import streamlit as st
import serial, time
import numpy as np
import joblib

# ---------------- CONFIG ----------------
PORT = "COM4"
BAUD = 115200

MODEL_PATH = "xgb_best.joblib"
SCALER_PATH = "scaler_selected.joblib"

FEATURES_ORDER = [
    "energy",
    "control_score",
    "gyro_energy_ratio",
    "smoothness",
    "pause_ratio",
    "jerk_mean",
    "speed"
]

# ---------------- LOAD MODEL ----------------
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ---------------- SESSION STATE ----------------
if "ser" not in st.session_state:
    st.session_state.ser = None
if "rows" not in st.session_state:
    st.session_state.rows = []

# ---------------- FEATURE EXTRACTION ----------------
def extract_features(gx, gy, gz):
    gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)

    speed = gyro_mag.mean()
    dg = np.abs(np.diff(gyro_mag))
    vibration = 0.7 * dg.mean() + 0.3 * gyro_mag.std()
    pause_ratio = (gyro_mag < 0.3 * speed).mean()
    smoothness = speed / (vibration + 1e-6)

    jerk_mean = np.abs(np.diff(gyro_mag)).mean()
    energy = np.mean(gyro_mag**2)

    gyro_energy_ratio = energy / (speed + 1)
    control_score = smoothness / (pause_ratio + 0.01)

    return {
        "energy": energy,
        "control_score": control_score,
        "gyro_energy_ratio": gyro_energy_ratio,
        "smoothness": smoothness,
        "pause_ratio": pause_ratio,
        "jerk_mean": jerk_mean,
        "speed": speed,
    }

# ---------------- UI ----------------
st.set_page_config(page_title="✍️ Smart Pen Live", layout="wide")

st.markdown("<h1 style='text-align:center; color:#4B0082;'>✍️ Smart Pen – Live Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#666;'>Collect handwriting data from Arduino and predict quality in real-time</p>", unsafe_allow_html=True)
st.markdown("---")

col1, col2 = st.columns([1,1])

# ---------------- START BUTTON ----------------
with col1:
    st.markdown("### ▶️ Start Recording")
    if st.button("Start ✍️", key="start_btn"):
        try:
            ser = serial.Serial(PORT, BAUD, timeout=1)
            time.sleep(2)
            while True:
                line = ser.readline().decode(errors="ignore").strip()
                if line.startswith("t_ms"):
                    break
            ser.write(b"s")
            st.session_state.ser = ser
            st.session_state.rows = []
            st.success("Recording started! Write now ✍️")
        except Exception as e:
            st.error(f"Serial error: {e}")

# ---------------- ANALYZE BUTTON ----------------
with col2:
    st.markdown("### 🔍 Analyze")
    if st.button("Analyze 🧠", key="analyze_btn"):
        if st.session_state.ser is None:
            st.warning("Start recording first! ▶️")
        else:
            ser = st.session_state.ser
            ser.write(b"p")
            time.sleep(0.5)
            while ser.in_waiting:
                line = ser.readline().decode(errors="ignore").strip()
                if line and not line.startswith("t_ms"):
                    p = line.split(",")
                    if len(p) == 7:
                        st.session_state.rows.append(p)
            ser.close()
            st.session_state.ser = None

            rows = st.session_state.rows
            if len(rows) < 10:
                st.error("Not enough data! ❌")
            else:
                gx = np.array([int(r[4]) for r in rows])
                gy = np.array([int(r[5]) for r in rows])
                gz = np.array([int(r[6]) for r in rows])

                features = extract_features(gx, gy, gz)

                # -------- FEATURES CARD --------
                st.markdown("### 📊 Extracted Features")
                st.markdown("<div style='background-color:#F0F8FF; padding:10px; border-radius:10px;'>", unsafe_allow_html=True)
                for k,v in features.items():
                    st.markdown(f"**{k}**: `{v:.3f}`")
                st.markdown("</div>", unsafe_allow_html=True)

                # -------- PREDICTION CARD --------
                X = np.array([features[f] for f in FEATURES_ORDER]).reshape(1, -1)
                Xs = scaler.transform(X)
                proba = model.predict_proba(Xs)[0]
                pred = np.argmax(proba)
                labels = ["BAD 😞", "MEDIUM 🙂", "PERFECT ✨"]

                st.markdown("### 🧠 Prediction")
                st.markdown(f"<div style='background-color:#024d02; padding:15px; border-radius:10px; font-size:20px; text-align:center;'>{labels[pred]}</div>", unsafe_allow_html=True)
                st.progress(float(np.max(proba)))

                st.session_state.rows = []

st.markdown("---")
st.markdown("<p style='text-align:center; color:#999;'>Smart Pen Live – Powered by Python & Streamlit 💜</p>", unsafe_allow_html=True)
