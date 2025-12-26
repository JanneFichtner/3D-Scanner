import smbus
import config

class Encoder:
    def __init__(self):
        self.I2C_ADRESS_ENCODER = config.I2C_ADRESSE
        self.I2C_BUS_ENCODER = config.I2C_BUS_ENCODER
        self.bus = smbus.SMBus(self.I2C_BUS_ENCODER)
        self.ANGLE_BIT_ZERO = self.get_angle_bit()
    def get_angle_bit(self):
        d = self.bus.read_i2c_block_data(I2C_ADRESS_ENCODER, 0x0C, 2)
        return ((d[0] << 8) | d[1]) & 0x0FFF

    def read_magnitude(self):
        d = self.bus.read_i2c_block_data(I2C_ADRESS_ENCODER, 0x1B, 2)
        return ((d[0] << 8) | d[1]) & 0x0FFF

    def bit_to_deg(self, raw):
        rel = (raw + 4096 - self.ANGLE_BIT_ZERO) & 0x0FFF
        return (rel * 360.0) / 4096.0
    def get_angle(self):
        return self.bit_to_deg(self.get_angle_bit())
