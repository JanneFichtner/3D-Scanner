import smbus
import struct
import time
import config


class Stepper:
    def __init__(self):
        self.I2C_ADRESSE_NANO = config.I2C_ADRESSE_NANO
        self.I2C_BUS_NANO = config.I2C_BUS_NANO
        try:
            self.bus = smbus.SMBus(self.I2C_BUS_NANO)
        except Exception as e:
            print(f"Fehler: Encoder nicht gefunden. {e}")
            exit(1)
    def send_data(self, command, value):
        
        #wert in 2 bytes umwandeln
        high_byte = (value >> 8) & 0xFF
        low_byte = value & 0xFF
        #daten senden
        self.bus.write_i2c_block_data(self.I2C_ADRESSE_NANO, command, [high_byte, low_byte])