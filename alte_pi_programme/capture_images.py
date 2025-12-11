from picamera2 import Picamera2
import cv2
import os
from time import sleep

# Zielordner
dir1 = "/home/janne/Desktop/Masterprojekt/Bilder Kalibrierung/checkerboard_light"
os.makedirs(dir1, exist_ok=True)
dir2 = "/home/janne/Desktop/Masterprojekt/Bilder Kalibrierung/laser_dark"
os.makedirs(dir2, exist_ok=True)

# Manuelle Belichtungswerte für die Laser-Aufnahme (dir2)
LASER_EXPOSURE_TIME = 40000      # in Mikrosekunden (z. B. 20000 = 0.02 s)
LASER_ANALOG_GAIN = 2           # ISO-Verstärkung (z. B. 1.0 = normal, 2.0 = doppelt so hell)


# Kamera initialisieren
cam = Picamera2()
config = cam.create_preview_configuration(main={"size": (640, 480)})
cam.configure(config)
cam.start()
sleep(0.5)

meta = cam.capture_metadata()
rg, bg = 1.0, 1.0
if meta and "ColourGains" in meta and meta["ColourGains"]:
    rg, bg = meta["ColourGains"]
FIXED_COLOUR_GAINS = (float(rg), float(bg))  # oder (1.0, 1.0)


print("Livebild läuft. Drücke [Enter], um Bilder aufzunehmen (je 2 pro Durchgang, insgesamt 10 Durchgänge).")
print("→ 1. Enter: Bild in checkerboard_light")
print("→ 2. Enter: Bild in laser_dark")
print("Drücke 'q' im Fenster, um das Programm vorzeitig zu beenden.")

i = 1
while i <= 10:
    frame = cam.capture_array()
    cv2.imshow("Live-Preview", frame)
    key = cv2.waitKey(1) & 0xFF

    # Abbruch
    if key == ord('q'):
        print("Abbruch durch Benutzer.")
        break

    # Erstes Enter → Bild in dir1 speichern
    if key == 13:  # Enter
        
        cam.set_controls({"AeEnable": True, "AwbEnable": True})
        sleep(0.3)  # kurze Stabilisierung

        filename1 = os.path.join(dir1, f"{i:02d}.jpg")
        cam.capture_file(filename1)
        print(f"→ Bild {i}/10 gespeichert in checkerboard_light")
        sleep(0.5)  # kleine Pause, damit Bild stabil ist
        cam.set_controls({
    "AeEnable": False,
    "AwbEnable": False,
    "ColourGains": FIXED_COLOUR_GAINS,
    "ExposureTime": LASER_EXPOSURE_TIME,
    "AnalogueGain": LASER_ANALOG_GAIN,
    # optional: Rauschunterdrückung weicher stellen
    # "NoiseReductionMode": "off",  # je nach Build verfügbar
})
        # Auf zweites Enter warten
        print("Drücke erneut [Enter] für das Laser-Bild")
        while True:
            key2 = cv2.waitKey(1) & 0xFF
            if key2 == ord('q'):
                print("Abbruch durch Benutzer.")
                cam.stop()
                cv2.destroyAllWindows()
                exit()
            if key2 == 13:
                filename2 = os.path.join(dir2, f"{i:02d}.jpg")
                cam.capture_file(filename2)
                print(f"→ Bild {i}/10 gespeichert in laser_dark ")
                i += 1
                sleep(0.5)
                break

cam.stop()
cv2.destroyAllWindows()
print("Fertig!")
