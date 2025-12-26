from hardware.cameras import Cameras
from hardware.encoder import Encoder
import threading
def main():

    #Kameras initialisieren
    cameras = Cameras()
    cameras.get_parameters()

    #Encoder initialisieren
    encoder = Encoder()

    #I2C verbindung zum Nano initialisieren



