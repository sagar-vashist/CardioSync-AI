from flask import Flask
from flask_socketio import SocketIO

from serial_reader import SerialSensorReader
from ml.classifier import ModelPredictor

import threading
import time
import numpy as np


app = Flask(__name__, static_folder='.', static_url_path='')
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

reader_thread = None
sensor_reader = None

health_predictor = ModelPredictor()

FEATURE_COLUMNS = [
    "ecg_mean", "ecg_std", "ecg_range",
    "mean_rr", "sdnn", "rmssd", "rr_count",
    "ppg_mean", "ppg_std", "ppg_range",
    "ppg_peak_count", "ppg_pulse_rate",
    "heart_rate", "spo2"
]

# --------------------------------------------------
# FINAL ASSESSMENT SETTINGS
# --------------------------------------------------

ASSESSMENT_DURATION_SECONDS = 30.0

assessment_values = []
assessment_start_time = None
assessment_complete = False
assessment_lock = threading.Lock()


def reset_assessment():
    global assessment_values, assessment_start_time, assessment_complete

    with assessment_lock:
        assessment_values = []
        assessment_start_time = None
        assessment_complete = False


def safe_float(value, default=0.0):
    try:
        value = float(value)
        if np.isfinite(value):
            return value
    except (ValueError, TypeError):
        pass
    return default


def add_assessment_sample(features, signal_quality=100.0, lead_off=0):
    global assessment_values, assessment_start_time, assessment_complete

    now = time.monotonic()

    with assessment_lock:
        if assessment_complete:
            return None

        if assessment_start_time is None:
            assessment_start_time = now

        sample = {
            "heart_rate": safe_float(features.get("heart_rate", 0.0)),
            "spo2": safe_float(features.get("spo2", 0.0)),
            "mean_rr": safe_float(features.get("mean_rr", 0.0)),
            "rmssd": safe_float(features.get("rmssd", 0.0)),
            "sdnn": safe_float(features.get("sdnn", 0.0)),
            "rr_count": safe_float(features.get("rr_count", 0)),
            "signal_quality": safe_float(signal_quality),
            "lead_off": int(lead_off)
        }

        if sample["heart_rate"] > 0:
            assessment_values.append(sample)

        elapsed = now - assessment_start_time

        return {
            "elapsed": elapsed,
            "remaining": max(0.0, ASSESSMENT_DURATION_SECONDS - elapsed),
            "count": len(assessment_values),
            "complete": elapsed >= ASSESSMENT_DURATION_SECONDS
        }


def calculate_final_assessment():
    with assessment_lock:
        samples = list(assessment_values)

    if not samples:
        return None

    def avg(key):
        values = [
            safe_float(sample.get(key, 0.0))
            for sample in samples
            if safe_float(sample.get(key, 0.0)) > 0
        ]
        return float(np.mean(values)) if values else 0.0

    averaged_features = {
        "heart_rate": avg("heart_rate"),
        "spo2": avg("spo2"),
        "mean_rr": avg("mean_rr"),
        "rmssd": avg("rmssd"),
        "sdnn": avg("sdnn"),
        "rr_count": avg("rr_count"),
        "signal_quality": avg("signal_quality"),
        "lead_off": 1 if any(
            sample.get("lead_off", 0) == 1 for sample in samples
        ) else 0
    }

    result = health_predictor.calculate_health_score(averaged_features)

    result["averages"] = {
        "heart_rate": round(averaged_features["heart_rate"], 1),
        "spo2": round(averaged_features["spo2"], 1),
        "mean_rr": round(averaged_features.get("mean_rr", 0.0), 1),
        "rmssd": round(averaged_features["rmssd"], 1),
        "sdnn": round(averaged_features["sdnn"], 1),
        "signal_quality": round(averaged_features["signal_quality"], 1)
    }

    result["sample_count"] = len(samples)
    result["assessment_duration"] = ASSESSMENT_DURATION_SECONDS

    return result


def background_reader_task():
    global sensor_reader

    try:
        sensor_reader = SerialSensorReader(
            port="COM6",
            baudrate=115200,
            mock_fallback=False
        )

        print("Starting ESP32 serial reader...")
        print("Starting ESP32 data stream...")
        print("Reading from COM6 at 115200 baud")
        print("Final assessment window: 30 seconds")

        reset_assessment()

        def on_new_data(data):
            global assessment_complete

            # ECG/PPG/vitals continue streaming live.
            socketio.emit("sensor_data", data)

            if assessment_complete:
                return

            features = data.get("features", {})
            if not features:
                return

            signal_quality = safe_float(
                features.get("signal_quality", 100.0)
            )

            lead_off = int(
                data.get(
                    "lead_off",
                    features.get("lead_off", 0)
                )
            )

            progress = add_assessment_sample(
                features,
                signal_quality=signal_quality,
                lead_off=lead_off
            )

            if progress is None:
                return

            socketio.emit(
                "assessment_progress",
                {
                    "elapsed": round(progress["elapsed"], 1),
                    "remaining": round(progress["remaining"], 1),
                    "sample_count": progress["count"],
                    "duration": ASSESSMENT_DURATION_SECONDS
                }
            )

            if not progress["complete"]:
                return

            with assessment_lock:
                if assessment_complete:
                    return
                assessment_complete = True

            final_result = calculate_final_assessment()

            if final_result is None:
                print("Final assessment could not be calculated.")
                return

            socketio.emit(
                "health_prediction",
                {
                    "score": final_result["score"],
                    "status": final_result["status"],
                    "components": final_result["components"],
                    "averages": final_result["averages"],
                    "sample_count": final_result["sample_count"],
                    "assessment_duration": final_result["assessment_duration"],
                    "locked": True
                }
            )

            print("=" * 60)
            print("FINAL CARDIOVASCULAR HEALTH ASSESSMENT")
            print("=" * 60)
            print(f"Status : {final_result['status']}")
            print(f"Score  : {final_result['score']}/100")
            print(f"HR Avg : {final_result['averages']['heart_rate']} BPM")
            print(f"SpO2 Avg : {final_result['averages']['spo2']}%")
            print(f"Mean RR Avg : {final_result['averages']['mean_rr']} ms")
            print(f"RMSSD Avg : {final_result['averages']['rmssd']} ms")
            print(f"SDNN Avg : {final_result['averages']['sdnn']} ms")
            print(
                "Signal Quality Avg : "
                f"{final_result['averages']['signal_quality']}"
            )
            print(f"Samples : {final_result['sample_count']}")
            print("=" * 60)

        sensor_reader.read_stream(callback=on_new_data)

    except Exception as e:
        print(f"Serial reader error: {e}")


@app.route("/")
def index():
    return app.send_static_file("index.html")


@socketio.on("connect")
def handle_connect():
    global reader_thread

    print("Client connected to WebSocket")

    if reader_thread is None or not reader_thread.is_alive():
        print("Starting ESP32 serial reader thread...")
        reader_thread = threading.Thread(
            target=background_reader_task,
            daemon=True
        )
        reader_thread.start()


@socketio.on("disconnect")
def handle_disconnect():
    print("Client disconnected")


if __name__ == "__main__":
    print("=" * 60)
    print("CARDIOVASCULAR MONITORING DASHBOARD")
    print("=" * 60)
    print("ESP32 Port : COM6")
    print("Baud Rate  : 115200")
    print("Health AI  : Final 30-second Cardiovascular Health Assessment")
    print("Dashboard  : http://localhost:5001")
    print("Socket mode: threading")
    print("=" * 60)

    socketio.run(
        app,
        host="0.0.0.0",
        port=5001,
        debug=False,
        allow_unsafe_werkzeug=True
    )
