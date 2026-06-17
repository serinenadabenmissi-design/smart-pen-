import serial, time, csv, os
import numpy as np
from datetime import datetime

# ---------------- CONFIG ----------------
PORT = "COM4"
BAUD = 115200
DURATION_SEC = 12
SUMMARY_CSV = "sessions_new.csv"

# ---------------- SERIAL READING ----------------
ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)

# انتظار header من Arduino
while True:
    line = ser.readline().decode(errors="ignore").strip()
    if line.startswith("t_ms,ax,ay,az,gx,gy,gz"):
        break

ser.write(b"s")
print("Recording ... ✍️")

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

# ---------------- EXTRACT FEATURES ----------------
gx = np.array([int(r[4]) for r in rows])
gy = np.array([int(r[5]) for r in rows])
gz = np.array([int(r[6]) for r in rows])

gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)

# speed
speed = float(gyro_mag.mean())

# vibration
dg = np.abs(np.diff(gyro_mag))
vibration = float(0.7 * dg.mean() + 0.3 * gyro_mag.std())

# pause_ratio
pause_ratio = float((gyro_mag < 0.3 * speed).mean())

# smoothness
smoothness = float(speed / (vibration + 1e-6))

# gyro_std, gyro_max
gyro_std = float(gyro_mag.std())
gyro_max = float(gyro_mag.max())

# jerk
jerk = np.abs(np.diff(gyro_mag))
jerk_mean = float(jerk.mean())

# energy
energy = float(np.mean(gyro_mag**2))

# derived features
gyro_energy_ratio = energy / (speed + 1)
control_score = smoothness / (pause_ratio + 0.01)

# ---------------- SAVE TO CSV ----------------
session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
file_exists = os.path.exists(SUMMARY_CSV)

selected_features = ["energy","control_score","gyro_energy_ratio","smoothness",
                     "pause_ratio","jerk_mean","speed"]

with open(SUMMARY_CSV, "a", newline="") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["session_id"] + selected_features)
    writer.writerow([session_id, energy, control_score, gyro_energy_ratio,
                     smoothness, pause_ratio, jerk_mean, speed])

print("✅ Tentative saved to", SUMMARY_CSV)
print("\nFeatures:")
for k, v in zip(selected_features, [energy, control_score, gyro_energy_ratio,
                                   smoothness, pause_ratio, jerk_mean, speed]):
    print(f"{k}: {v:.3f}")
