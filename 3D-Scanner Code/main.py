import time
from hardware.cameras import Cameras
from hardware.encoder import Encoder
import threading
def main():

    #Kameras initialisieren
    cameras = Cameras()
    cameras.get_parameters()

    #Encoder initialisieren
    encoder = Encoder()
    while True:
        angle = encoder.get_angle()
        
        # \r sorgt dafür, dass die Zeile im Terminal überschrieben wird
        print(f" {angle:7.2f}° ", end="\r")
        
        time.sleep(0.05) # 20 Hz Update-Rate

    #I2C verbindung zum Nano initialisieren
if __name__ == "__main__":
    main()


