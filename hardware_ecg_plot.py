import serial
import matplotlib.pyplot as plt
from collections import deque

PORT = "COM6"
BAUD_RATE = 115200

ser = serial.Serial(PORT, BAUD_RATE, timeout=1)

ECG_BUFFER_SIZE = 100

ecg_data = deque(maxlen=ECG_BUFFER_SIZE)

plt.ion()

fig, ax = plt.subplots()

line, = ax.plot([], [])

ax.set_title("Live ECG - ESP32 + AD8232")
ax.set_xlabel("Samples")
ax.set_ylabel("ECG Raw Value")
ax.grid(True)

try:
    while True:

        raw = ser.readline().decode(
            "utf-8",
            errors="ignore"
        ).strip()

        if not raw:
            continue

        if raw.startswith("ECG"):
            continue

        parts = raw.split(",")

        if len(parts) != 4:
            continue

        try:
            ecg = float(parts[0])
            heart_rate = float(parts[1])
            spo2 = float(parts[2])
            lead_off = int(parts[3])
        except ValueError:
            continue

        # Ignore disconnected electrodes
        if lead_off != 0:
            continue

        ecg_data.append(ecg)

        line.set_xdata(range(len(ecg_data)))
        line.set_ydata(list(ecg_data))

        ax.relim()
        ax.autoscale_view()

        ax.set_title(
            f"Live ECG | HR: {heart_rate:.1f} BPM | "
            f"SpO2: {spo2:.1f}%"
        )

        fig.canvas.draw()
        fig.canvas.flush_events()

except KeyboardInterrupt:
    print("\nStopping ECG graph...")

finally:
    ser.close()
    plt.close("all")