import smbus2
import time

# I2C Konfiguration
I2C_BUS = 1         # Standard I2C Bus am Raspberry Pi
I2C_ADDR = 0x08     # Muss exakt die Adresse aus dem Arduino-Sketch sein

def main():
    try:
        # Bus öffnen
        bus = smbus2.SMBus(I2C_BUS)
        print(f"Sende Befehl an Adresse {hex(I2C_ADDR)}...")

        # Wir senden einfach eine Zahl (1 byte). 
        # 1 = Blinken
        befehl = 1
        
        # write_byte(Adresse, Wert)
        bus.write_byte(I2C_ADDR, befehl)
        
        print("Befehl gesendet! Der Nano sollte jetzt blinken.")
        
        # Bus schließen
        bus.close()
        
    except OSError as e:
        print("Fehler: I2C Gerät nicht gefunden!")
        print("1. Ist der Nano angeschlossen?")
        print("2. Stimmen SDA/SCL?")
        print("3. Haben beide GND verbunden?")
        print(f"System-Meldung: {e}")

if __name__ == "__main__":
    main()