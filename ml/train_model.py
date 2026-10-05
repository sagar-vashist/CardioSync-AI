import os
import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "bidmc_features.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "ml"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "model.pkl"
)


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

print("=" * 60)
print("TRAINING PHYSIOLOGICAL PATTERN MODEL")
print("=" * 60)

print(f"Dataset: {DATA_PATH}")

if not os.path.exists(DATA_PATH):

    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_PATH}"
    )


df = pd.read_csv(DATA_PATH)

print(
    f"Loaded {len(df)} records"
)


# --------------------------------------------------
# Select features
# --------------------------------------------------

FEATURE_COLUMNS = [
    "ecg_mean",
    "ecg_std",
    "ecg_range",

    "mean_rr",
    "sdnn",
    "rmssd",
    "rr_count",

    "ppg_mean",
    "ppg_std",
    "ppg_range",

    "ppg_peak_count",
    "ppg_pulse_rate",

    "heart_rate",
    "spo2"
]


missing = [
    col
    for col in FEATURE_COLUMNS
    if col not in df.columns
]


if missing:

    raise ValueError(
        "Missing feature columns:\n" +
        "\n".join(missing)
    )


X = df[FEATURE_COLUMNS].copy()


# --------------------------------------------------
# Clean data
# --------------------------------------------------

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(
    X.median()
)


print(
    f"Using {len(FEATURE_COLUMNS)} features"
)


# --------------------------------------------------
# Scale features
# --------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# --------------------------------------------------
# K-Means clustering
# --------------------------------------------------

print(
    "Training K-Means with 3 physiological patterns..."
)


model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=20
)

labels = model.fit_predict(
    X_scaled
)


# --------------------------------------------------
# Print cluster information
# --------------------------------------------------

df["cluster"] = labels

print()
print("Cluster distribution:")

print(
    df["cluster"]
    .value_counts()
    .sort_index()
)


print()
print("Cluster feature means:")

print(
    df.groupby("cluster")[
        FEATURE_COLUMNS
    ].mean().round(2)
)


# --------------------------------------------------
# Save model
# --------------------------------------------------

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


joblib.dump(
    {
        "model": model,
        "scaler": scaler,
        "features": FEATURE_COLUMNS
    },
    MODEL_PATH
)


print()
print(
    f"Model saved successfully:"
)

print(MODEL_PATH)

print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)