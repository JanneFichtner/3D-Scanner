#!/usr/bin/env python3
from picamera2 import Picamera2
import cv2
import os
from time import sleep
from datetime import datetime

# Zielordner
SAVE_DIR = "/home/janne/Desktop/Masterprojekt/Scan"
os.makedirs(SAVE_DIR, exist_ok=True)

# ► Belichtungsstaffelung (Mikrosekunden) – passe gern an
EXPOSURE_TIMES_US = [
    1000,      # 1/1000 s
    5000,      # 1/200 s
    10000,     # 1/100 s
    20000,     # 1/50 s
    50000,     # 1/20 s
    100000,    # 1/10 s
    200000,    # 1/5 s
    400000,    # 0.4 s
]

# ► ISO-Verstärkung (AnalogueGain); ca. 1.0 ~ ISO100
GAINS = [1.0, 2.0, 4.0, 8.0]

# Vorschaufenster aktiv (True) oder headless (False)
SHOW_PREVIEW = True

# Auflösung: Der Sensor-Modus (aus rpicam-hello) zeigt 1296x972 an – gute Basis
MAIN_SIZE = (640, 480)   # (Breite, Höhe), z. B. (640, 480) falls erwünscht

# Zeit für Stabilisierung nach Parameteränderung (Sekunden)
SETTLE_SEC = 0.35

def main():
    print("Starte Kamera …")
    cam = Picamera2()
    config = cam.create_preview_configuration(main={"size": MAIN_SIZE})
    cam.configure(config)
    cam.start()
    sleep(0.5)

    # Automatik abschalten → manuelle Belichtung/Verstärkung
    cam.set_controls({
        "AeEnable": False,
        "AwbEnable": True,   # Weißabgleich Automatik an lassen; bei Laser-Setups ggf. False + ColorGains
    })

    idx = 1
    total = len(EXPOSURE_TIMES_US) * len(GAINS)
    print(f"Nehme {total} Bilder in einer Belichtungsreihe auf …")
    print("Drücke 'q' im Preview-Fenster für Abbruch.")

    
    cam.set_controls({
        "ExposureTime": int(20000),   # µs
        "AnalogueGain": float(2),
    })
    sleep(SETTLE_SEC)  # Stabilisierung

    # optional: Live-Preview
    frame = cam.capture_array()
    if SHOW_PREVIEW:
        overlay_text = f"Exp: {10000} µs  Gain: {2:.2f}  ({idx}/{total})"
        cv2.putText(frame, overlay_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Belichtungsreihe – Vorschau", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            print("Abbruch durch Benutzer.")
            cam.stop()
            cv2.destroyAllWindows()
            return

        # Datei schreiben
        fname = f"img_exp{10000}_gain{2:.2f}_idx{idx:02d}.jpg"
        fpath = os.path.join(SAVE_DIR, fname)
        cam.capture_file(fpath)
        print(f"[{idx:02d}/{total}] gespeichert: {fpath}")
        idx += 1

        # kleine Pause zwischen Captures
        sleep(0.15)

    cam.stop()
    if SHOW_PREVIEW:
        cv2.destroyAllWindows()
    print("Fertig!")

if __name__ == "__main__":
    main()
