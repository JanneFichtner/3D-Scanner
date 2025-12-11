from picamera2 import Picamera2
from time import sleep

cam = Picamera2()

cam.start()


cam.capture_file("erstes_laserbild.jpg")

cam.stop()
print("Bild gespeichert als erstes_laserbild.jpg")
