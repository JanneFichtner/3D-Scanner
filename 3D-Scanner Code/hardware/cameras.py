#from picamera2 import Picamera2
import time
import config
from picamera2 import Picamera2

class Cameras:
    def __init__(self):
        self.ISO = config.ISO
        self.Belichtungszeit = config.BELICHTUNGSZEIT
        #Kameras initialisieren
        self.cam = Picamera2()
        self.cam.configure(self.cam.create_still_configuration(
            main={"format": "RGB888", "size": config.CAM_RESOLUTION}
        ))        
        self.cam.start();
        time.sleep(0.5)
    def get_parameters(self):
        print(f"Iso: {self.ISO} Belichtungszeit: {self.Belichtungszeit}")
    def take_pic(self):
        #Nimmt ein Bild als rgb array auf
        img_rgb = self.cam.capture_array()
        return(img_rgb)