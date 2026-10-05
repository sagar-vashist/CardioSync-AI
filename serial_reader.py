import serial
import time
from collections import deque

import numpy as np
from scipy.signal import find_peaks


class SerialSensorReader:

    def __init__(
        self,
        port='COM6',
        baudrate=115200,
        mock_fallback=False
    ):
        self.port = port
        self.baudrate = baudrate
        self.mock_fallback = mock_fallback

        self.serial_conn = None
        self.is_running = False

        self.latest_ppg_ir = 0.0
        self.latest_ppg_red = 0.0

        self.latest_ecg = 0.0
        self.latest_hr = 0.0
        self.latest_spo2 = 0.0
        self.latest_lead_off = 0

        self.ecg_buffer = deque(maxlen=1200)
        self.ppg_ir_buffer = deque(maxlen=1200)
        self.ppg_red_buffer = deque(maxlen=1200)
        self.ppg_time_buffer = deque(maxlen=1200)

        self.last_feature_time = 0.0
        self.feature_interval = 2.0
        self.latest_features = {}

        self._connect()

    def _connect(self):
        try:
            self.serial_conn = serial.Serial(
                self.port,
                self.baudrate,
                timeout=1
            )
            time.sleep(2)
            self.serial_conn.reset_input_buffer()
            print(f"Successfully connected to {self.port}")
        except serial.SerialException as e:
            print(f"Failed to connect to {self.port}: {e}")
            if self.mock_fallback:
                print("Mock mode is not implemented.")
            else:
                raise e

    @staticmethod
    def _stats(values):
        values = np.asarray(values, dtype=float)
        if len(values) == 0:
            return {"mean": 0.0, "std": 0.0, "range": 0.0}
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "range": float(np.ptp(values))
        }

    @staticmethod
    def _normalize(values):
        values = np.asarray(values, dtype=float)
        if len(values) == 0:
            return values
        mean = np.mean(values)
        std = np.std(values)
        if std < 1e-10:
            return values - mean
        return (values - mean) / std

    def _detect_ppg_peaks(self):
        """Robust engineering PPG beat detector for the 10 Hz prototype stream."""
        if len(self.ppg_ir_buffer) < 50:
            return []

        signal = np.asarray(self.ppg_ir_buffer, dtype=float)
        timestamps = np.asarray(self.ppg_time_buffer, dtype=float)

        if len(signal) != len(timestamps):
            return []

        # Estimate the actual sampling rate from received R:IR,RED packets.
        dt = np.diff(timestamps)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if len(dt) < 5:
            return []

        fs = 1.0 / float(np.median(dt))
        fs = float(np.clip(fs, 5.0, 50.0))

        if np.std(signal) < 0.5:
            return []

        # Remove the DC component and slow baseline drift. The prototype is
        # around 10 Hz, so 0.5--3 Hz covers normal pulse frequencies.
        try:
            from scipy.signal import butter, sosfiltfilt

            low = 0.5 / (fs / 2.0)
            high = min(3.0 / (fs / 2.0), 0.95)
            if low >= high:
                raise ValueError("invalid filter band")

            sos = butter(2, [low, high], btype="bandpass", output="sos")
            filtered = sosfiltfilt(sos, signal)
        except Exception:
            # Fallback when the buffer is too short for filtering.
            filtered = signal - np.mean(signal)

        filtered = np.asarray(filtered, dtype=float)
        std = float(np.std(filtered))
        if std <= 1e-8:
            return []

        min_distance = max(2, int(round(fs * 0.35)))
        prominence = max(0.03 * std, 1e-8)

        candidates = []
        for candidate_signal in (filtered, -filtered):
            peaks, _ = find_peaks(
                candidate_signal,
                distance=min_distance,
                prominence=prominence
            )

            valid_peaks = []
            for index in peaks:
                if not valid_peaks:
                    valid_peaks.append(int(index))
                    continue

                interval = timestamps[index] - timestamps[valid_peaks[-1]]
                if 0.30 <= interval <= 2.00:
                    valid_peaks.append(int(index))

            candidates.append(valid_peaks)

        best = max(candidates, key=len) if candidates else []
        return [(int(i), float(timestamps[i])) for i in best]

    def _calculate_ppg_hrv(self):
        peaks = self._detect_ppg_peaks()

        if len(peaks) < 3:
            return {
                "mean_rr": 0.0,
                "sdnn": 0.0,
                "rmssd": 0.0,
                "rr_count": 0
            }

        rr_intervals = []
        for i in range(1, len(peaks)):
            rr_ms = (peaks[i][1] - peaks[i - 1][1]) * 1000.0
            if 300 <= rr_ms <= 2000:
                rr_intervals.append(rr_ms)

        if len(rr_intervals) < 2:
            return {
                "mean_rr": 0.0,
                "sdnn": 0.0,
                "rmssd": 0.0,
                "rr_count": len(rr_intervals)
            }

        rr = np.asarray(rr_intervals, dtype=float)
        mean_rr = float(np.mean(rr))
        sdnn = float(np.std(rr, ddof=1)) if len(rr) > 1 else 0.0
        differences = np.diff(rr)
        rmssd = float(np.sqrt(np.mean(differences ** 2))) if len(differences) else 0.0

        return {
            "mean_rr": mean_rr,
            "sdnn": sdnn,
            "rmssd": rmssd,
            "rr_count": int(len(rr))
        }

    def _calculate_features(self):
        if len(self.ecg_buffer) < 20:
            return {}

        if len(self.ppg_ir_buffer) < 30:
            return {}

        ecg = np.asarray(
            self.ecg_buffer,
            dtype=float
        )

        ppg_ir = np.asarray(
            self.ppg_ir_buffer,
            dtype=float
        )

        ppg_red = np.asarray(
            self.ppg_red_buffer,
            dtype=float
        )

        ecg_norm = self._normalize(ecg)
        ppg_ir_norm = self._normalize(ppg_ir)
        ppg_red_norm = self._normalize(ppg_red)

        ecg_stats = self._stats(ecg_norm)
        ir_stats = self._stats(ppg_ir_norm)
        red_stats = self._stats(ppg_red_norm)

        hrv = self._calculate_ppg_hrv()
        peaks = self._detect_ppg_peaks()

        peak_count = len(peaks)
        pulse_rate = 0.0

        if len(peaks) >= 2:
            intervals = []

            for i in range(1, len(peaks)):
                interval = (
                    peaks[i][1] -
                    peaks[i - 1][1]
                )

                if 0.3 <= interval <= 2.0:
                    intervals.append(interval)

            if intervals:
                pulse_rate = 60.0 / np.mean(intervals)

        return {
            "ecg_mean": ecg_stats["mean"],
            "ecg_std": ecg_stats["std"],
            "ecg_range": ecg_stats["range"],
            "mean_rr": hrv["mean_rr"],
            "sdnn": hrv["sdnn"],
            "rmssd": hrv["rmssd"],
            "rr_count": hrv["rr_count"],
            "ppg_mean": ir_stats["mean"],
            "ppg_std": ir_stats["std"],
            "ppg_range": ir_stats["range"],
            "ppg_peak_count": peak_count,
            "ppg_pulse_rate": float(pulse_rate),
            "heart_rate": float(self.latest_hr),
            "spo2": float(self.latest_spo2),
            "lead_off": int(self.latest_lead_off)
        }

    def parse_line(self, line):
        line = line.strip()

        if not line:
            return None

        if line.startswith("R:"):
            try:
                parts = line[2:].split(",")

                if len(parts) != 2:
                    return None

                ir = float(parts[0])
                red = float(parts[1])

                self.latest_ppg_ir = ir
                self.latest_ppg_red = red

                if ir > 0:
                    self.ppg_ir_buffer.append(ir)
                    self.ppg_red_buffer.append(red)
                    self.ppg_time_buffer.append(
                        time.monotonic()
                    )

                now = time.monotonic()

                if (
                    now - self.last_feature_time
                    >= self.feature_interval
                ):
                    self.latest_features = (
                        self._calculate_features()
                    )
                    self.last_feature_time = now

                return {
                    "ecg_value": self.latest_ecg,
                    "ppg_value": ir,
                    "ppg_ir": ir,
                    "ppg_red": red,
                    "vitals": {
                        "hr": self.latest_hr,
                        "spo2": self.latest_spo2
                    },
                    "lead_off": self.latest_lead_off,
                    "features": self.latest_features
                }

            except (ValueError, TypeError):
                return None

        if line.startswith("ECG"):
            return None

        parts = line.split(",")

        if len(parts) != 4:
            return None

        try:
            ecg = float(parts[0])
            heart_rate = float(parts[1])
            spo2 = float(parts[2])
            lead_off = int(parts[3])

            self.latest_ecg = ecg
            self.latest_hr = heart_rate
            self.latest_spo2 = spo2
            self.latest_lead_off = lead_off

            self.ecg_buffer.append(ecg)

            now = time.monotonic()

            if (
                now - self.last_feature_time
                >= self.feature_interval
            ):
                self.latest_features = (
                    self._calculate_features()
                )
                self.last_feature_time = now

            return {
                "ecg_value": ecg,
                "ppg_value": self.latest_ppg_ir,
                "ppg_ir": self.latest_ppg_ir,
                "ppg_red": self.latest_ppg_red,
                "vitals": {
                    "hr": heart_rate,
                    "spo2": spo2
                },
                "lead_off": lead_off,
                "features": self.latest_features
            }

        except (ValueError, TypeError):
            return None

    def read_stream(self, callback):
        self.is_running = True

        print("Starting ESP32 data stream...")
        print(
            f"Reading from {self.port} "
            f"at {self.baudrate} baud"
        )
        print("Waiting for ECG + PPG data...")

        while self.is_running:
            try:
                line = (
                    self.serial_conn
                    .readline()
                    .decode(
                        "utf-8",
                        errors="ignore"
                    )
                    .strip()
                )

                if not line:
                    continue

                parsed_data = self.parse_line(line)

                if parsed_data is not None:
                    callback(parsed_data)

            except serial.SerialException as e:
                print(
                    f"Serial connection lost: {e}"
                )
                self.is_running = False
                break

            except Exception as e:
                print(
                    f"Serial read error: {e}"
                )

    def stop(self):
        self.is_running = False

        if (
            self.serial_conn
            and self.serial_conn.is_open
        ):
            self.serial_conn.close()

        print("Serial port closed.")
