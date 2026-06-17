import serial, time, csv, math, os
import numpy as np
from datetime import datetime

PORT = "COM4"
BAUD = 115200
DURATION_SEC = 12

LABEL = "bad"          # good / medium / bad (يدوي)
SUMMARY_CSV = "sessions_summary.csv"
WRITER_ID = "p01"
TEXT_ID = "word01"

# ---------------- SERIAL ----------------
ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)

while True:
    line = ser.readline().decode(errors="ignore").strip()
    if line.startswith("t_ms,ax,ay,az,gx,gy,gz"):
        break

ser.write(b"s")
print(f"Recording {LABEL.upper()}... ✍️")

rows = []
t0 = time.time()
while time.time() - t0 < DURATION_SEC:
    line = ser.readline().decode(errors="ignore").strip()
    if line and not line.startswith("t_ms"):
        p = line.split(",")
        if len(p) == 7:
            rows.append(p)

ser.write(b"p")
ser.close()

# ---------------- FEATURES ----------------
gx = np.array([int(r[4]) for r in rows])
gy = np.array([int(r[5]) for r in rows])
gz = np.array([int(r[6]) for r in rows])

gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)

# 1. Speed
speed_score = float(gyro_mag.mean())

# 2. Vibration
dg = np.abs(np.diff(gyro_mag))
vibration_score = float(0.7 * dg.mean() + 0.3 * gyro_mag.std())

# 3. Pause
pause_thr = 0.30 * speed_score
pause_ratio = float((gyro_mag < pause_thr).mean())

# 4. Smoothness
smoothness_score = float(speed_score / (vibration_score + 1e-6))

# -------- NEW STRONG FEATURES --------
gyro_std = float(gyro_mag.std())
gyro_max = float(gyro_mag.max())
gyro_range = float(gyro_mag.max() - gyro_mag.min())

jerk = np.diff(gyro_mag)
jerk_mean = float(np.mean(np.abs(jerk)))
jerk_max = float(np.max(np.abs(jerk)))

energy = float(np.mean(gyro_mag**2))

# ---------------- DISPLAY ----------------
print("\n===== FEATURES =====")
print(f"speed        : {speed_score:.2f}")
print(f"vibration    : {vibration_score:.2f}")
print(f"pause_ratio  : {pause_ratio:.3f}")
print(f"smoothness   : {smoothness_score:.2f}")
print(f"gyro_std     : {gyro_std:.2f}")
print(f"gyro_max     : {gyro_max:.2f}")
print(f"gyro_range   : {gyro_range:.2f}")
print(f"jerk_mean    : {jerk_mean:.2f}")
print(f"jerk_max     : {jerk_max:.2f}")
print(f"energy       : {energy:.2f}")
print("====================\n")

# ---------------- SAVE CSV ----------------
session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
file_exists = os.path.exists(SUMMARY_CSV)

with open(SUMMARY_CSV, "a", newline="") as f:
    w = csv.writer(f)
    if not file_exists:
        w.writerow([
            "session_id","label",
            "speed_score","vibration_score","pause_ratio","smoothness_score",
            "gyro_std","gyro_max","gyro_range",
            "jerk_mean","jerk_max","energy",
            "writer_id","text_id"
        ])
    w.writerow([
        session_id, LABEL,
        speed_score, vibration_score, pause_ratio, smoothness_score,
        gyro_std, gyro_max, gyro_range,
        jerk_mean, jerk_max, energy,
        WRITER_ID, TEXT_ID
    ])

print("✅ Saved to", SUMMARY_CSV)
