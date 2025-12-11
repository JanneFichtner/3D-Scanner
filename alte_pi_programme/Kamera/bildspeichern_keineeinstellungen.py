#!/usr/bin/env python3
from picamera2 import Picamera2
from time import sleep

# Dateiname
SAVE_PATH = "/home/janne/Desktop/auto_capture.jpg"

# Kamera starten (ohne manuelle Settings)
cam = Picamera2()
cam.configure(cam.create_still_configuration())
cam.start()

sleep(1.0)  # kurze Wartezeit, damit Belichtung & Weißabgleich stabil sind

# Bild speichern
cam.capture_file(SAVE_PATH)
cam.stop()

print(f"Bild gespeichert unter: {SAVE_PATH}")
