import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    """
    Apply Butterworth bandpass filtering.

    signal : 1-D numpy array
    lowcut : lower cutoff frequency in Hz
    highcut: upper cutoff frequency in Hz
    fs     : sampling frequency in Hz
    """

    signal = np.asarray(signal, dtype=float)

    if len(signal) < 20:
        return signal

    nyquist = 0.5 * fs

    # Keep cutoff frequencies valid
    low = max(lowcut / nyquist, 0.001)
    high = min(highcut / nyquist, 0.99)

    if low >= high:
        return signal

    b, a = butter(order, [low, high], btype="band")

    try:
        return filtfilt(b, a, signal)
    except ValueError:
        return signal


def normalize_signal(signal):
    """
    Z-score normalization.
    """

    signal = np.asarray(signal, dtype=float)

    mean = np.mean(signal)
    std = np.std(signal)

    if std < 1e-10:
        return signal - mean

    return (signal - mean) / std


def preprocess_ecg(ecg, fs):
    """
    ECG preprocessing.

    Typical ECG band:
    approximately 0.5 - 40 Hz.
    """

    ecg = np.asarray(ecg, dtype=float)

    if len(ecg) == 0:
        return ecg

    filtered = bandpass_filter(
        ecg,
        lowcut=0.5,
        highcut=40.0,
        fs=fs
    )

    return normalize_signal(filtered)


def preprocess_ppg(ppg, fs):
    """
    PPG preprocessing.

    Keeps the main pulse-related frequency content.
    """

    ppg = np.asarray(ppg, dtype=float)

    if len(ppg) == 0:
        return ppg

    filtered = bandpass_filter(
        ppg,
        lowcut=0.5,
        highcut=8.0,
        fs=fs
    )

    return normalize_signal(filtered)


def preprocess_signals(ecg, ppg_ir, ppg_red, ecg_fs, ppg_fs):
    """
    Preprocess ECG and both PPG channels.
    """

    clean_ecg = preprocess_ecg(ecg, ecg_fs)
    clean_ppg_ir = preprocess_ppg(ppg_ir, ppg_fs)
    clean_ppg_red = preprocess_ppg(ppg_red, ppg_fs)

    return {
        "ecg": clean_ecg,
        "ppg_ir": clean_ppg_ir,
        "ppg_red": clean_ppg_red
    }