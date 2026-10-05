import os
import pandas as pd


def load_bidmc_record(record_number, data_dir):
    """
    Load one BIDMC recording.

    Example:
        record_number = 1
    """

    record_id = f"{record_number:02d}"

    signals_file = os.path.join(
        data_dir,
        f"bidmc_{record_id}_Signals.csv"
    )

    numerics_file = os.path.join(
        data_dir,
        f"bidmc_{record_id}_Numerics.csv"
    )

    if not os.path.exists(signals_file):
        raise FileNotFoundError(
            f"Signals file not found: {signals_file}"
        )

    if not os.path.exists(numerics_file):
        raise FileNotFoundError(
            f"Numerics file not found: {numerics_file}"
        )

    signals = pd.read_csv(signals_file)
    numerics = pd.read_csv(numerics_file)

    # Remove accidental whitespace from column names
    signals.columns = signals.columns.str.strip()
    numerics.columns = numerics.columns.str.strip()

    return signals, numerics


if __name__ == "__main__":

    DATA_DIR = "data/raw/bidmc_csv"

    signals, numerics = load_bidmc_record(
        record_number=1,
        data_dir=DATA_DIR
    )

    print("\n===== SIGNALS =====")
    print(signals.columns.tolist())

    print("\n===== NUMERICS =====")
    print(numerics.columns.tolist())

    print("\n===== SIGNAL SAMPLE =====")
    print(signals.head())

    print("\n===== NUMERICS SAMPLE =====")
    print(numerics.head())