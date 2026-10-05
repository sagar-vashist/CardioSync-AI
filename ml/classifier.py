import numpy as np


class ModelPredictor:

    """
    Explainable cardiovascular health scoring engine.

    IMPORTANT:
    This is an engineering/prototype score.
    It is NOT a medical diagnosis or clinically validated
    cardiovascular risk prediction model.
    """

    def __init__(self, model_path=None):
        self.model_path = model_path

        print(
            "Cardiovascular Health Scoring Engine loaded."
        )

    # ========================================================
    # Helper
    # ========================================================

    @staticmethod
    def clamp(value, minimum=0.0, maximum=100.0):

        return max(
            minimum,
            min(maximum, float(value))
        )

    # ========================================================
    # Heart Rate Score
    #
    # Reference:
    # Typical resting adult HR: 60-100 BPM.
    #
    # This score assumes the user is measuring at rest.
    # ========================================================

    def heart_rate_score(self, hr):

        if hr is None or not np.isfinite(hr):
            return 0.0

        hr = float(hr)

        if 60 <= hr <= 100:
            return 100.0

        if 50 <= hr < 60:
            return 85.0

        if 100 < hr <= 110:
            return 85.0

        if 40 <= hr < 50:
            return 65.0

        if 110 < hr <= 120:
            return 65.0

        if 30 <= hr < 40:
            return 45.0

        if 120 < hr <= 140:
            return 45.0

        return 25.0

    # ========================================================
    # SpO2 Score
    #
    # FDA:
    # 95%-100% is typical for most healthy individuals.
    # ========================================================

    def spo2_score(self, spo2):

        if spo2 is None or not np.isfinite(spo2):
            return 0.0

        spo2 = float(spo2)

        if spo2 >= 95:
            return 100.0

        if 93 <= spo2 < 95:
            return 80.0

        if 90 <= spo2 < 93:
            return 60.0

        return 30.0

    # ========================================================
    # HRV Score
    #
    # We do NOT claim a universal normal RMSSD threshold.
    #
    # Instead, this component evaluates whether the current
    # rolling-window HRV measurement is usable and reasonably
    # stable for this prototype.
    # ========================================================

    def hrv_score(self, rmssd, sdnn, rr_count):

        if (
            rmssd is None
            or sdnn is None
            or rr_count is None
        ):
            return 0.0

        try:
            rmssd = float(rmssd)
            sdnn = float(sdnn)
            rr_count = int(rr_count)
        except (ValueError, TypeError):
            return 0.0

        if rr_count < 2:
            return 40.0

        if rmssd <= 0 or sdnn <= 0:
            return 40.0

        # Engineering reference band.
        # This is NOT a clinical cutoff.
        if 20 <= rmssd <= 200:
            rmssd_component = 100.0

        elif 10 <= rmssd < 20:
            rmssd_component = 75.0

        elif 200 < rmssd <= 300:
            rmssd_component = 75.0

        else:
            rmssd_component = 50.0

        if 20 <= sdnn <= 150:
            sdnn_component = 100.0

        elif 10 <= sdnn < 20:
            sdnn_component = 75.0

        elif 150 < sdnn <= 250:
            sdnn_component = 75.0

        else:
            sdnn_component = 50.0

        return (
            0.6 * rmssd_component
            + 0.4 * sdnn_component
        )

    # ========================================================
    # Signal Quality Score
    #
    # ECG/PPG raw amplitudes are NOT treated as health values.
    # We only use signal quality / lead status.
    # ========================================================

    def signal_quality_score(
        self,
        signal_quality=100.0,
        lead_off=0
    ):

        try:
            quality = float(signal_quality)
        except (ValueError, TypeError):
            quality = 0.0

        quality = self.clamp(
            quality,
            0.0,
            100.0
        )

        if int(lead_off) == 1:
            quality *= 0.5

        return quality

    # ========================================================
    # Overall Cardiovascular Health Score
    # ========================================================

    def calculate_health_score(self, features):

        if not features:
            return {
                "score": 0.0,
                "status": "Waiting for data",
                "components": {}
            }

        hr = features.get(
            "heart_rate",
            0.0
        )

        spo2 = features.get(
            "spo2",
            0.0
        )

        rmssd = features.get(
            "rmssd",
            0.0
        )

        sdnn = features.get(
            "sdnn",
            0.0
        )

        rr_count = features.get(
            "rr_count",
            0
        )

        lead_off = features.get(
            "lead_off",
            0
        )

        # ----------------------------------------------------
        # Component scores
        # ----------------------------------------------------

        hr_score = self.heart_rate_score(hr)

        spo2_score = self.spo2_score(spo2)

        hrv_score = self.hrv_score(
            rmssd,
            sdnn,
            rr_count
        )

        # Current dashboard already calculates a quality
        # percentage separately. If unavailable, use 100
        # when the leads are connected.
        signal_quality = features.get(
            "signal_quality",
            100.0
        )

        quality_score = self.signal_quality_score(
            signal_quality,
            lead_off
        )

        # ----------------------------------------------------
        # Weighted overall score
        #
        # HR      = 30%
        # SpO2    = 30%
        # HRV     = 25%
        # Quality = 15%
        # ----------------------------------------------------

        overall_score = (
            0.30 * hr_score
            + 0.30 * spo2_score
            + 0.25 * hrv_score
            + 0.15 * quality_score
        )

        overall_score = self.clamp(
            overall_score
        )

        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        if overall_score >= 85:
            status = "Good"

        elif overall_score >= 70:
            status = "Fair"

        elif overall_score >= 50:
            status = "Needs Attention"

        else:
            status = "Poor"

        return {
            "score": round(
                overall_score,
                1
            ),

            "status": status,

            "components": {
                "heart_rate": round(
                    hr_score,
                    1
                ),

                "spo2": round(
                    spo2_score,
                    1
                ),

                "hrv": round(
                    hrv_score,
                    1
                ),

                "signal_quality": round(
                    quality_score,
                    1
                )
            }
        }

    # ========================================================
    # Existing app.py compatibility
    # ========================================================

    def predict(self, feature_array):

        """
        Kept for compatibility with the existing application.

        Converts the model feature array into the health-score
        output instead of returning Pattern A/B/C.
        """

        try:

            X = np.asarray(
                feature_array,
                dtype=float
            )

            if X.ndim == 1:
                X = X.reshape(
                    1,
                    -1
                )

            row = X[0]

            if len(row) < 14:
                return (
                    "Waiting for data",
                    0.0
                )

            features = {

                "ecg_mean": row[0],
                "ecg_std": row[1],
                "ecg_range": row[2],

                "mean_rr": row[3],
                "sdnn": row[4],
                "rmssd": row[5],
                "rr_count": row[6],

                "ppg_mean": row[7],
                "ppg_std": row[8],
                "ppg_range": row[9],

                "ppg_peak_count": row[10],
                "ppg_pulse_rate": row[11],

                "heart_rate": row[12],
                "spo2": row[13]
            }

            result = self.calculate_health_score(
                features
            )

            return (
                result["status"],
                result["score"]
            )

        except Exception as e:

            print(
                f"Health scoring error: {e}"
            )

            return (
                "Score unavailable",
                0.0
            )