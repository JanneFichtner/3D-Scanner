#!/usr/bin/env python3
import os
import time
from datetime import datetime

import cv2
import numpy as np
from picamera2 import Picamera2

# --- I2C / AS5600 ---
import smbus
SAVE_DIR = "/home/janne/Desktop/Masterprojekt/Scan/scan_maßband_01"
DEVICE_AS5600 = 0x36  # Default I2C address
bus = smbus.SMBus(1)  # I2C Bus 1 auf Raspberry Pi

def read_raw_angle():
    """Liest den Rohwinkel (0..4095) vom AS5600."""
    data = bus.read_i2c_block_data(DEVICE_AS5600, 0x0C, 2)
    return (data[0] << 8) | data[1]

def read_magnitude():
    """Optional: Magnetfeld-Magnitude (0..4095)."""
    data = bus.read_i2c_block_data(DEVICE_AS5600, 0x1B, 2)
    return (data[0] << 8) | data[1]

def raw_to_deg(raw, raw_zero):
    """Konvertiert Rohwinkel (0..4095) in Grad relativ zu raw_zero (0..360)."""
    rel = (raw + 4096 - raw_zero) & 0x0FFF
    return (rel * 360.0) / 4096.0

# --- Speicherpfade ---

os.makedirs(SAVE_DIR, exist_ok=True)
CSV_PATH = os.path.join(SAVE_DIR, "angles_capture_log.csv")

# --- Kamera-Parameter ---
RESOLUTION = (1296, 972)  # stabiler Modus für OV5647; bei Bedarf (640, 480)
PREVIEW_WINDOW = "Scan Preview"

def main():
    print("Starte Kamera & Sensor …")
    cam = Picamera2()
    config = cam.create_preview_configuration(main={"size": RESOLUTION})
    cam.configure(config)
    cam.start()
    time.sleep(0.5)

    # Auto-Exposure / Auto-Whitebalance EIN (gewünscht)
    cam.set_controls({
        "AeEnable": True,
        "AwbEnable": True
    })

    # Winkel-Nullpunkt setzen (beim Start)
    raw_zero = read_raw_angle()
    print(f"AS5600 Nullpunkt gesetzt. raw_zero={raw_zero}")

    # CSV-Header anlegen, falls neu
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w") as f:
            f.write("timestamp,filename,angle_deg,raw_angle,magnitude\n")

    print("Live-Preview geöffnet.")
    print("Bedienung: [Enter] = Foto speichern  |  [q] = Beenden")

    # kleines Debounce, damit ein Enter nicht mehrfach triggert
    last_capture_time = 0.0
    debounce_sec = 0.25

    try:
        while True:
            frame = cam.capture_array()

            # aktuellen Winkel lesen
            raw = read_raw_angle()
            mag = read_magnitude()
            angle_deg = raw_to_deg(raw, raw_zero)

            # Overlay ins Preview-Bild
            overlay_text = f"Angle: {angle_deg:7.2f} deg   (raw={raw:4d}, mag={mag:4d})"
            cv2.putText(frame, overlay_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow(PREVIEW_WINDOW, frame)
            k = cv2.waitKey(1) & 0xFF

            # Quit
            if k == ord('q'):
                print("Beende …")
                break

            # Enter -> Bild + CSV speichern
            if k == 13:  # Enter
                now = time.time()
                if now - last_capture_time < debounce_sec:
                    continue
                last_capture_time = now

                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"scan_angle_{angle_deg:07.2f}_deg_{ts}.jpg"
                filepath = os.path.join(SAVE_DIR, filename)

                # Foto als Datei speichern (direkt aus Kamera, nicht aus Frame)
                cam.capture_file(filepath)

                # CSV-Log
                with open(CSV_PATH, "a") as f:
                    f.write(f"{ts},{filename},{angle_deg:.6f},{raw},{mag}\n")

                print(f"Gespeichert: {filepath}")

    except KeyboardInterrupt:
        print("\nAbbruch durch Benutzer.")
    finally:
        # Aufräumen
        try:
            cam.stop()
        except Exception:
            pass
        try:
            bus.close()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("Fertig.")
if __name__ == "__main__":
    main()
