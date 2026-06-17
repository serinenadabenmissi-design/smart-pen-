import numpy as np

def extract_features(gx, gy, gz, window_size=50):
    """
    Compute features from gyro signals (gx, gy, gz)
    Returns a dictionary with all features needed by the model.
    """
    gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)
    n = len(gyro_mag)
    speed_scores = []
    vibration_scores = []
    pause_ratios = []
    smoothness_scores = []

    for start in range(0, n, window_size):
        end = min(start + window_size, n)
        window = gyro_mag[start:end]
        if len(window) < 2:
            continue

        speed = window.mean()
        speed_scores.append(speed)
        dg = np.abs(np.diff(window))
        vibration = 0.7 * dg.mean() + 0.3 * window.std()
        vibration_scores.append(vibration)
        pause_thr = 0.3 * speed
        pause_ratio = (window < pause_thr).mean()
        pause_ratios.append(pause_ratio)
        smoothness = speed / (vibration + 1e-6)
        smoothness_scores.append(smoothness)

    speed_score = float(np.mean(speed_scores))
    vibration_score = float(np.mean(vibration_scores))
    pause_ratio = float(np.mean(pause_ratios))
    smoothness_score = float(np.mean(smoothness_scores))

    gyro_std = float(gyro_mag.std())
    gyro_max = float(gyro_mag.max())
    gyro_range = float(gyro_mag.max() - gyro_mag.min())
    jerk = np.diff(gyro_mag)
    jerk_mean = float(np.mean(np.abs(jerk)))
    jerk_max = float(np.max(np.abs(jerk)))
    energy = float(np.mean(gyro_mag**2))

    features = {
        "speed": speed_score,
        "vibration": vibration_score,
        "pause_ratio": pause_ratio,
        "smoothness": smoothness_score,
        "gyro_std": gyro_std,
        "gyro_max": gyro_max,
        "gyro_range": gyro_range,
        "jerk_mean": jerk_mean,
        "jerk_max": jerk_max,
        "energy": energy,
        "gyro_energy_ratio": energy / (speed_score + 1),
        "jerk_speed_ratio": jerk_mean / (speed_score + 1),
        "stability_score": 1 / (vibration_score + gyro_std + 1),
        "peak_ratio": gyro_max / (gyro_std + 1),
    }

    return features
