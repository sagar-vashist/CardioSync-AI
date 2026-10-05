# CardioSync AI

A hybrid hardware–software cardiovascular monitoring prototype that streams ECG and PPG sensor readings to a live dashboard for signal visualization and feature analysis.

## Our project

CardioSync AI was our **minor project in the final year of engineering**. Our four-member team worked together on both the hardware and software, integrating sensor acquisition, serial communication, signal processing, and the dashboard.

## What it does

- Reads ECG, heart rate, SpO2, and PPG data from an ESP32 over USB serial.
- Displays live ECG and PPG waveforms in a browser dashboard.
- Extracts signal features and averages measurements over a 30-second assessment.
- Shows a prototype cardiovascular health score based on the averaged readings.

The live score is calculated by weighted scoring logic. The included K-means training script and `model.pkl` are exploratory and are not used to generate the live score.

## Run locally

**You’ll need:** Python, an ESP32 with compatible firmware, and sensors connected to the computer running the app. The default serial port is **COM6** at **115200 baud**; update it in `app.py` if needed.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open [http://localhost:5001](http://localhost:5001). The ESP32 firmware should send newline-terminated readings in these formats:

```text
<ecg_value>,<heart_rate>,<spo2>,<lead_off>
R:<infrared_value>,<red_value>
```

For example: `512,72,98,0` and `R:12345,11800`.

## Project files

- `app.py` — Flask and Socket.IO server
- `serial_reader.py` — serial input and signal features
- `index.html`, `css/`, `js/` — live dashboard
- `ml/` — health scoring, signal processing, and exploratory model training
- `hardware_test.py` — serial connection test
- `hardware_ecg_plot.py` — standalone ECG plot

Dataset files are not included. The dataset-processing and model-training scripts require the expected BIDMC files under `data/`.

> **Disclaimer:** This is an engineering and educational prototype, not a clinically validated system. Its score is not a diagnosis or medical advice. Do not use it to make medical decisions.

## License

See [LICENSE](./LICENSE).
