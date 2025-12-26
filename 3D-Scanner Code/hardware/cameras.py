#from picamera2 import Picamera2
import time
import config

class Cameras:
    def __init__(self):
        self.ISO = config.ISO
        self.Belichtungszeit = 1000
        #Kameras initialisieren
        """cam = Picamera2()
        cam.configure(cam.create_preview_configuration(main={"size": RESOLUTION}))
        cam.start();
        time.sleep(0.5)"""
    def take_pic(self, kamera):
        return kamera
    def get_parameters(self):
        print(f"Iso: {self.ISO} Belichtungszeit: {self.Belichtungszeit}")