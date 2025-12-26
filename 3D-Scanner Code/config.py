# config.py

# Kamera Einstellungen
CAM_RESOLUTION = (1920, 1080)
ISO = 800           # Analog Gain (ungefähr)
BELICHTUNGSZEIT = 20000 # In Mikrosekunden (20ms)
CAM_ID_1 = 0
CAM_ID_2 = 1

#Stepper Einstellungen
MAX_SPEED = 3000
SCHRITTE_UMDREHUNG = 200 * 5 * 2 * 16
I2C_ADRESSE_NANO = 0x31  #für arduino

#Encoder
I2C_ADRESSE_ENCODER = 0x36
I2C_BUS_ENCODER = 1
