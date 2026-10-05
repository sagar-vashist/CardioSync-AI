import os
import numpy as np
import pandas as pd

from ml.preprocessing import preprocess_ecg, preprocess_ppg
from ml.feature_extraction import extract_features


# BIDMC Signals.csv sampling rate
FS = 125


def process_record(record_number, data_dir):
    record_id = f"{record_number:02d}"

    signals_path = os.path.join(
        data_dir,
        f"bidmc_{record_id}_Signals.csv"
    )

    numerics_path = os.path.join(
        data_dir,
        f"bidmc_{record_id}_Numerics.csv"
    )

    if not os.path.exists(signals_path):
        print(f"Skipping {record_id}: Signals file missing")
        return None

    if not os.path.exists(numerics_path):
        print(f"Skipping {record_id}: Numerics file missing")
        return None

    # Load files
    signals = pd.read_csv(signals_path)
    numerics = pd.read_csv(numerics_path)

    # Clean column names
    signals.columns = signals.columns.str.strip()
    numerics.columns = numerics.columns.str.strip()

    # ---------------------------------------------------------
    # BIDMC signal selection
    # ECG  = Lead II
    # PPG  = PLETH
    # ---------------------------------------------------------

    ecg = signals["II"].values.astype(float)
    ppg = signals["PLETH"].values.astype(float)

    # ---------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------

    clean_ecg = preprocess_ecg(
        ecg,
        FS
    )

    clean_ppg = preprocess_ppg(
        ppg,
        FS
    )

    # ---------------------------------------------------------
    # Numerical physiological measurements
    # ---------------------------------------------------------

    hr_values = pd.to_numeric(
        numerics["HR"],
        errors="coerce"
    ).dropna()

    spo2_values = pd.to_numeric(
        numerics["SpO2"],
        errors="coerce"
    ).dropna()

    if len(hr_values) == 0:
        print(f"Skipping {record_id}: no valid HR")
        return None

    if len(spo2_values) == 0:
        print(f"Skipping {record_id}: no valid SpO2")
        return None

    heart_rate = float(hr_values.mean())
    spo2 = float(spo2_values.mean())

    # ---------------------------------------------------------
    # Feature extraction
    #
    # BIDMC has ONE PPG waveform.
    # Therefore we do NOT pretend that it has separate
    # MAX30102 IR and RED channels.
    # ---------------------------------------------------------

    features = {}

    # ECG statistical features
    features["ecg_mean"] = float(np.mean(clean_ecg))
    features["ecg_std"] = float(np.std(clean_ecg))
    features["ecg_min"] = float(np.min(clean_ecg))
    features["ecg_max"] = float(np.max(clean_ecg))
    features["ecg_range"] = float(np.ptp(clean_ecg))

    # HRV features from ECG
    from ml.feature_extraction import calculate_hrv_features

    hrv_features = calculate_hrv_features(
        clean_ecg,
        FS
    )

    features.update(hrv_features)

    # ---------------------------------------------------------
    # PPG features
    # ---------------------------------------------------------

    features["ppg_mean"] = float(np.mean(clean_ppg))
    features["ppg_std"] = float(np.std(clean_ppg))
    features["ppg_min"] = float(np.min(clean_ppg))
    features["ppg_max"] = float(np.max(clean_ppg))
    features["ppg_range"] = float(np.ptp(clean_ppg))

    from ml.feature_extraction import detect_peaks

    ppg_peaks = detect_peaks(
        clean_ppg,
        FS,
        minimum_distance_seconds=0.3
    )

    features["ppg_peak_count"] = int(len(ppg_peaks))

    if len(ppg_peaks) >= 2:

        intervals = np.diff(ppg_peaks) / float(FS)

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

    # ---------------------------------------------------------
    # Direct physiological values
    # ---------------------------------------------------------

    features["heart_rate"] = heart_rate
    features["spo2"] = spo2

    # Keep record ID for tracking
    features["record_id"] = record_id

    return features


def build_feature_dataset(
    data_dir,
    output_file,
    start_record=1,
    end_record=53
):

    all_features = []

    for record_number in range(
        start_record,
        end_record + 1
    ):

        print(
            f"Processing BIDMC record "
            f"{record_number:02d}..."
        )

        try:

            features = process_record(
                record_number,
                data_dir
            )

            if features is not None:
                all_features.append(features)

        except Exception as e:

            print(
                f"Error in record "
                f"{record_number:02d}: {e}"
            )

    if not all_features:
        raise RuntimeError(
            "No valid records were processed."
        )

    dataframe = pd.DataFrame(
        all_features
    )

    os.makedirs(
        os.path.dirname(output_file),
        exist_ok=True
    )

    dataframe.to_csv(
        output_file,
        index=False
    )

    print("\n================================")
    print("Clean feature dataset created")
    print("================================")

    print(
        f"Records processed: "
        f"{len(dataframe)}"
    )

    print(
        f"Features: "
        f"{len(dataframe.columns)}"
    )

    print(
        f"Saved to: "
        f"{output_file}"
    )

    print("\nFeature columns:")

    print(
        dataframe.columns.tolist()
    )

    print("\nPreview:")

    print(
        dataframe.head()
    )


if __name__ == "__main__":

    DATA_DIR = "data/raw/bidmc_csv"

    OUTPUT_FILE = (
        "data/processed/"
        "bidmc_features.csv"
    )

    build_feature_dataset(
        data_dir=DATA_DIR,
        output_file=OUTPUT_FILE,
        start_record=1,
        end_record=53
    )