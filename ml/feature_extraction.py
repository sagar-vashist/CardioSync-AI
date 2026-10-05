import numpy as np
from scipy.signal import find_peaks


def basic_statistics(signal, prefix):
    """
    Extract basic statistical features from a signal.
    """

    signal = np.asarray(signal, dtype=float)

    if len(signal) == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_range": 0.0,
        }

    return {
        f"{prefix}_mean": float(np.mean(signal)),
        f"{prefix}_std": float(np.std(signal)),
        f"{prefix}_min": float(np.min(signal)),
        f"{prefix}_max": float(np.max(signal)),
        f"{prefix}_range": float(np.ptp(signal)),
    }


def detect_peaks(signal, fs, minimum_distance_seconds=0.3):
    """
    Detect peaks in a physiological signal.

    minimum_distance_seconds prevents unrealistically close peaks.
    """

    signal = np.asarray(signal, dtype=float)

    if len(signal) < 5:
        return np.array([], dtype=int)

    distance = max(
        int(fs * minimum_distance_seconds),
        1
    )

    # Adaptive prominence
    std = np.std(signal)

    if std <= 1e-10:
        return np.array([], dtype=int)

    prominence = 0.3 * std

    peaks, _ = find_peaks(
        signal,
        distance=distance,
        prominence=prominence
    )

    return peaks


def calculate_rr_intervals(ecg, ecg_fs):
    """
    Estimate RR intervals from ECG peaks.

    Returns RR intervals in milliseconds.
    """

    peaks = detect_peaks(
        ecg,
        ecg_fs,
        minimum_distance_seconds=0.3
    )

    if len(peaks) < 2:
        return np.array([])

    rr_seconds = np.diff(peaks) / float(ecg_fs)

    # Keep physiologically plausible intervals
    rr_seconds = rr_seconds[
        (rr_seconds >= 0.3) &
        (rr_seconds <= 2.0)
    ]

    return rr_seconds * 1000.0


def calculate_hrv_features(ecg, ecg_fs):
    """
    Calculate basic HRV features from RR intervals.
    """

    rr = calculate_rr_intervals(ecg, ecg_fs)

    if len(rr) < 2:
        return {
            "mean_rr": 0.0,
            "sdnn": 0.0,
            "rmssd": 0.0,
            "rr_count": 0
        }

    mean_rr = np.mean(rr)

    sdnn = np.std(
        rr,
        ddof=1
    ) if len(rr) > 1 else 0.0

    successive_differences = np.diff(rr)

    rmssd = np.sqrt(
        np.mean(successive_differences ** 2)
    ) if len(successive_differences) > 0 else 0.0

    return {
        "mean_rr": float(mean_rr),
        "sdnn": float(sdnn),
        "rmssd": float(rmssd),
        "rr_count": int(len(rr))
    }


def calculate_ppg_features(ppg_ir, ppg_red, ppg_fs):
    """
    Extract basic PPG pulse features.
    """

    features = {}

    features.update(
        basic_statistics(ppg_ir, "ppg_ir")
    )

    features.update(
        basic_statistics(ppg_red, "ppg_red")
    )

    # Pulse detection on IR channel
    peaks = detect_peaks(
        ppg_ir,
        ppg_fs,
        minimum_distance_seconds=0.3
    )

    features["ppg_peak_count"] = int(len(peaks))

    if len(peaks) >= 2:
        intervals = np.diff(peaks) / float(ppg_fs)

        valid_intervals = intervals[
            (intervals >= 0.3) &
            (intervals <= 2.0)
        ]

        if len(valid_intervals) > 0:
            pulse_rate = 60.0 / np.mean(valid_intervals)
        else:
            pulse_rate = 0.0
    else:
        pulse_rate = 0.0

    features["ppg_pulse_rate"] = float(pulse_rate)

    return features


def extract_features(
    ecg,
    ppg_ir,
    ppg_red,
    heart_rate,
    spo2,
    ecg_fs,
    ppg_fs
):
    """
    Extract complete feature vector for ML.
    """

    features = {}

    # ECG statistical features
    features.update(
        basic_statistics(ecg, "ecg")
    )

    # HRV features
    features.update(
        calculate_hrv_features(
            ecg,
            ecg_fs
        )
    )

    # PPG features
    features.update(
        calculate_ppg_features(
            ppg_ir,
            ppg_red,
            ppg_fs
        )
    )

    # Direct physiological measurements
    features["heart_rate"] = float(heart_rate)
    features["spo2"] = float(spo2)

    return features