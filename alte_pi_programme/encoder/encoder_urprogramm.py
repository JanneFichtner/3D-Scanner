#!/usr/bin/env python3
import smbus
import time
import os

# --- Konfiguration ---
DEVICE_AS5600 = 0x36  # Die Adresse, die dein i2cdetect gefunden hat
BUS_NR = 1            # Standard I2C Bus beim Pi

class EncoderTest:
    def __init__(self):
        try:
            self.bus = smbus.SMBus(BUS_NR)
            # Test-Lesezugriff, um Verbindung zu prüfen
            self.bus.read_byte(DEVICE_AS5600)
            print(f"Encoder auf Adresse {hex(DEVICE_AS5600)} gefunden!")
        except Exception as e:
            print(f"Fehler: Encoder nicht gefunden. {e}")
            exit(1)
            
        # Nullpunkt beim Start setzen
        self.raw_zero = self.read_raw_angle()
        print(f"Nullpunkt kalibriert auf: {self.raw_zero}")

    def read_raw_angle(self):
        """Liest den 12-Bit Rohwert (0-4095)."""
        # Register 0x0C ist das High-Byte, 0x0D das Low-Byte
        data = self.bus.read_i2c_block_data(DEVICE_AS5600, 0x0C, 2)
        return ((data[0] << 8) | data[1]) & 0x0FFF

    def read_magnitude(self):
        """Gibt die Magnetstärke zurück (hilft beim Ausrichten des Magneten)."""
        data = self.bus.read_i2c_block_data(DEVICE_AS5600, 0x1B, 2)
        return ((data[0] << 8) | data[1]) & 0x0FFF

    def get_angle(self):
        """Berechnet den Winkel in Grad relativ zum Startpunkt."""
        raw = self.read_raw_angle()
        # Überlauf-Korrektur (Modulo 4096)
        rel = (raw + 4096 - self.raw_zero) & 0x0FFF
        return (rel * 360.0) / 4096.0

def main():
    test = EncoderTest()
    
    print("\n--- Encoder Live-Daten (Strg+C zum Beenden) ---")
    print("Winkel [°] | Rohwert | Magnetstärke")
    print("-" * 40)
    
    try:
        while True:
            angle = test.get_angle()
            raw = test.read_raw_angle()
            mag = test.read_magnitude()
            
            # \r sorgt dafür, dass die Zeile im Terminal überschrieben wird
            print(f" {angle:7.2f}°  |  {raw:4d}   |   {mag:4d}  ", end="\r")
            
            time.sleep(0.05) # 20 Hz Update-Rate
            
    except KeyboardInterrupt:
        print("\n\nTest beendet.")

if __name__ == "__main__":
    main()