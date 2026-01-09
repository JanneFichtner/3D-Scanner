import smbus
import struct
import time
import config


class Nano:
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
    def move_to_angle(self, current_angle, angle):
        print(f"aktiv! current_angle: {current_angle}, angle: {angle}")
        full_rot_steps = 200 * 16 * 5 * 4
        d_angle = angle - current_angle
        
        if d_angle > 180:
         d_angle -= 360
        if d_angle < -180:
            d_angle += 360

        steps = int(abs((d_angle / 360) * full_rot_steps))
        if d_angle == 0:
            cmd = None
        elif d_angle > 0:
            cmd = 2
        elif d_angle < 0:
            cmd = 1
        
        if cmd is not None:
            self.send_data(cmd, steps)