import smbus2
import struct
import time
import config


class Stepper:
    def __init__(self):
        self.address = config.I2C_ADDR_MOTOR  # z.B. 0x08
        self.bus_num = 1  # Pi Standard I2C Bus
        self.bus = smbus2.SMBus(self.bus_num)

    def _send_packet(self, command_char, value):
        """
        Verpackt Daten binär für den Arduino.
        Format: '<ci'
        < = Little Endian (Standard bei ESP32/Arduino)
        c = char (1 Byte)
        i = int (4 Bytes)
        """
        try:
            # command_char muss bytes sein (z.B. b'M')
            if isinstance(command_char, str):
                command_char = command_char.encode('ascii')

            payload = struct.pack('<ci', command_char, int(value))

            # Konvertiere bytes zu einer Liste von Integern für smbus
            data_list = list(payload)

            # Senden an Register 0 (Arduino ignoriert Register oft, aber I2C braucht eins)
            self.bus.write_i2c_block_data(self.address, 0x00, data_list)

        except Exception as e:
            print(f"I2C Fehler (Stepper): {e}")

    def move_rel(self, steps):
        """Bewegt den Motor relativ um X Schritte"""
        print(f"Stepper: Bewege um {steps} Schritte")
        self._send_packet('M', steps)

    def move_abs(self, position):
        """Fährt zu absoluter Position"""
        self._send_packet('G', position)

    def set_speed(self, speed):
        """Setzt die Geschwindigkeit in Steps/Sek"""
        self._send_packet('S', speed)

    def enable(self, state=True):
        """Motor an (True) oder aus (False)"""
        val = 1 if state else 0
        self._send_packet('E', val)


# --- Test Bereich (nur wenn Datei direkt ausgeführt wird) ---
if __name__ == "__main__":
    motor = Stepper()
    motor.enable(True)
    time.sleep(0.5)
    motor.move_rel(1600)  # Halbe Umdrehung bei 1/16 Microstepping (3200 total)
    print("Befehl gesendet.")
