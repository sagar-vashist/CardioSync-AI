import serial
import time

PORT = "COM6"
BAUD_RATE = 115200

print("=" * 50)
print("ESP32 HARDWARE SERIAL TEST")
print("=" * 50)
print(f"Port : {PORT}")
print(f"Baud : {BAUD_RATE}")
print()
print("Connecting...")

ser = serial.Serial(
    PORT,
    BAUD_RATE,
    timeout=1
)

time.sleep(2)

print("Connected!")
print("Reading ESP32 data...")
print("-" * 50)

try:
    while True:

        line = ser.readline().decode(
            "utf-8",
            errors="ignore"
        ).strip()

        if not line:
            continue

        print("RAW:", line)

        # Skip header
        if line.startswith("ECG"):
            continue

        parts = line.split(",")

        if len(parts) != 4:
            print("Invalid data format")
            continue

        try:
            ecg = float(parts[0])
            heart_rate = float(parts[1])
            spo2 = float(parts[2])
            lead_off = int(parts[3])

            print(
                f"ECG={ecg:.0f} | "
                f"HR={heart_rate:.1f} BPM | "
                f"SpO2={spo2:.1f}% | "
                f"LeadOff={lead_off}"
            )

        except ValueError:
            print("Could not parse:", parts)

except KeyboardInterrupt:

    print("\nStopping...")

finally:

    ser.close()
    print("Serial port closed.")