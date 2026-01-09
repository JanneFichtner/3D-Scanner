import time
import threading
from hardware.cameras import Cameras
from hardware.encoder import Encoder
from hardware.Nano import Nano
from logik.kalibrierung import Kalibrierung

def main():
 
    # Hardware initialisieren   
    cameras = Cameras()
    cameras.get_parameters()
    encoder = Encoder()
    nano = Nano()

    #lokig initialisieren
    kalibrierung = Kalibrierung(nano, cameras, encoder)
    #triangulation = Triangulation(nano, cameras, encoder)
    #nano.move_to_angle(encoder.get_angle(), 180)
    kalibrierung.get_calibration_data()
    print("Hauptprogramm: Lese Winkel aus. Drücke Strg+C zum Beenden.")

    try:
        while True:
            # Während die Laser-Routine im Hintergrund läuft,
            # aktualisiert dieser Loop permanent den Winkel im Terminal.
            angle = encoder.get_angle()
            print(f" Aktueller Winkel: {angle:7.2f}° ", end="\r")
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nAbbruch durch Benutzer...")
        # Optional: Laser beim Abbruch sicher ausschalten
        nano.send_data(4, 0)
        nano.send_data(6, 0)

if __name__ == "__main__":
    main()