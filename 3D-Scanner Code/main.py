import time
from hardware.cameras import Cameras
from hardware.encoder import Encoder
from hardware.stepper import Stepper
import threading
def main():
                                                                                                                                                                                                                                                                                                                                                                                                                          
    #Kameras initialisieren
    cameras = Cameras()
    cameras.get_parameters()

    #Encoder initialisieren
    encoder = Encoder()
    stepper = Stepper()
    #stepper.send_data(2, 200 * 16 * 5 * 4)
    for i in range(3):
        stepper.send_data(3, 200 * 16 * 5 * 4)
        stepper.send_data(5, 200 * 16 * 5 * 4)
        print("laser an")
        time.sleep(2)
        stepper.send_data(4, 1000)
        stepper.send_data(6, 1000)
        print("laser aus")
        time.sleep(2)
    
    while True:
        angle = encoder.get_angle()
        
        # \r sorgt dafür, dass die Zeile im Terminal überschrieben wird
        print(f" {angle:7.2f}° ", end="\r")
        
        time.sleep(0.05) # 20 Hz Update-Rate
        
    #I2C verbindung zum Nano initialisieren
if __name__ == "__main__":
    main()
 

